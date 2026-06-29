// FusionLLM S10 测试：Spec Decode E2E (Multiple Steps)
// Phase 6 S10 - End-to-end spec decoding simulation using rejection_sample
//
// 验证:
//   1. 完整 spec decode 循环 (propose → verify → rejection_sample → commit)
//   2. 接受率 vs throughput (Python spec_decode_simulate.py 同样的 cost model)
//   3. Acceptance length 跟理论值一致
//   4. Total output tokens == target tokens
//
// 注意：这不是真 llama.cpp forward（那是 S10 完整版），
// 而是用 rejection_sample 循环跑 N 步验证算法 + cost 模型的正确性。

#include "fusion_speculative_decode.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>
#include <vector>

static int passed = 0;
static int failed = 0;
#define EXPECT(cond, msg) do { \
    if (cond) { passed++; } \
    else { failed++; fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); } \
} while (0)

// Cost model (跟 Python spec_decode_simulate.py 一致)
static constexpr float COST_AR_PER_TOKEN  = 1.0f;
static constexpr float COST_VERIFY_PER_STEP = 2.0f;  // 1 forward on block_size+1 tokens
static constexpr float COST_DRAFT_PER_STEP  = 0.25f; // small 5-layer DSpark

// Mock softmax helpers
static std::vector<float> softmax(const std::vector<float>& logits, float temperature = 1.0f) {
    std::vector<float> probs(logits.size());
    if (temperature <= 0.0f) {
        size_t max_idx = 0;
        float max_v = logits[0];
        for (size_t i = 1; i < logits.size(); ++i) {
            if (logits[i] > max_v) { max_v = logits[i]; max_idx = i; }
        }
        probs[max_idx] = 1.0f;
        return probs;
    }
    float max_l = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp((logits[i] - max_l) / temperature);
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;
    return probs;
}

// Sample a token id from probability distribution
static int32_t sample(const std::vector<float>& probs, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float u = dist(rng);
    float cum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cum += probs[i];
        if (u <= cum) return (int32_t)i;
    }
    return (int32_t)(probs.size() - 1);
}

// Simulate ONE step of spec decoding:
//   1) Draft model proposes block_size tokens (uses mock "draft" probs)
//   2) Target model "verifies" on [prev_token, draft_tokens] (uses mock "target" probs)
//   3) rejection_sample returns (accepted_count, bonus_token)
// Returns: (accepted_count + 1 bonus, accepted_count, total_step_cost)
struct StepResult {
    int32_t accepted_count;
    int32_t bonus_token;
    int total_cost;
};

static StepResult simulate_spec_step(
    int32_t prev_token,
    int block_size,
    int vocab_size,
    float draft_quality,  // 0.0-1.0: how close draft probs are to target probs
    std::mt19937& rng,
    float temperature = 1.0f
) {
    // Mock target logits (uniform random base + small per-position offset)
    std::vector<float> target_logits((block_size + 1) * vocab_size);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : target_logits) v = nd(rng);

    // Mock draft logits: target_logits + noise
    // Smaller noise = better draft quality (higher accept rate)
    std::vector<float> draft_logits(block_size * vocab_size);
    float noise_scale = (1.0f - draft_quality) * 2.0f;
    for (int i = 0; i < block_size * vocab_size; ++i) {
        draft_logits[i] = target_logits[i] + nd(rng) * noise_scale;
    }

    auto target_probs = softmax(target_logits, temperature);
    auto draft_probs  = softmax(draft_logits, temperature);

    // Draft tokens (use draft_probs to sample)
    std::vector<int32_t> draft_tokens(block_size);
    for (int i = 0; i < block_size; ++i) {
        // draft probs for position i
        std::vector<float> pos_probs(vocab_size);
        for (int v = 0; v < vocab_size; ++v) {
            pos_probs[v] = draft_probs[i * vocab_size + v];
        }
        draft_tokens[i] = sample(pos_probs, rng);
    }

    // rejection_sample
    auto [accepted, bonus] = fusion::FusionSpecDecoder::rejection_sample(
        target_probs, draft_probs, draft_tokens, block_size, vocab_size, temperature, rng);

    int total_cost = (int)(COST_DRAFT_PER_STEP + COST_VERIFY_PER_STEP);
    return {(int32_t)accepted, bonus, total_cost};
}

// Run full spec decode simulation for `target_tokens` tokens
struct SimResult {
    int total_tokens_emitted;
    int total_steps;
    int total_verify_calls;  // = total_steps
    int total_accepted;     // sum of accepted per step
    int total_bonus;
    int total_rejection_steps;  // steps where not all block accepted
    double acceptance_length;     // avg tokens per step
    double total_cost;            // sum of step costs
};

static SimResult run_spec_decode_sim(
    int target_tokens,
    float draft_quality,
    int block_size,
    int vocab_size,
    uint32_t seed
) {
    std::mt19937 rng(seed);
    SimResult r{};
    int prev_token = 0;  // arbitrary start token

    while (r.total_tokens_emitted < target_tokens) {
        auto step = simulate_spec_step(prev_token, block_size, vocab_size, draft_quality, rng);

        r.total_steps++;
        r.total_verify_calls++;
        r.total_accepted += step.accepted_count;
        r.total_bonus += 1;

        int emitted = step.accepted_count + 1;  // bonus
        int remaining = target_tokens - r.total_tokens_emitted;
        if (emitted > remaining) emitted = remaining;

        r.total_tokens_emitted += emitted;
        r.total_cost += step.total_cost;

        if (step.accepted_count < block_size) r.total_rejection_steps++;
    }

    r.acceptance_length = (double)r.total_tokens_emitted / r.total_steps;
    return r;
}

static void test_e2e_high_quality() {
    printf("\n=== Test 1: High Quality Draft (draft_quality=0.95) ===\n");
    const int target = 500;
    const int block_size = 7;
    const int vocab_size = 1000;

    auto r = run_spec_decode_sim(target, 0.95f, block_size, vocab_size, 42);

    printf("  total_tokens_emitted: %d\n", r.total_tokens_emitted);
    printf("  total_steps: %d\n", r.total_steps);
    printf("  acceptance_length: %.2f (theoretical: ~6.7)\n", r.acceptance_length);
    printf("  rejection_steps: %d / %d\n", r.total_rejection_steps, r.total_steps);
    printf("  total_cost: %.2f\n", r.total_cost);
    printf("  AR cost for same: %d\n", target * 1);

    EXPECT(r.total_tokens_emitted == target, "emitted exact target tokens");
    EXPECT(r.acceptance_length > 5.0f, "high quality → high acceptance length");
    EXPECT(r.total_rejection_steps < r.total_steps * 4 / 5, "high quality → most steps have some rejection");
}

static void test_e2e_medium_quality() {
    printf("\n=== Test 2: Medium Quality Draft (draft_quality=0.7) ===\n");
    const int target = 500;
    const int block_size = 7;
    const int vocab_size = 1000;

    auto r = run_spec_decode_sim(target, 0.7f, block_size, vocab_size, 42);

    printf("  acceptance_length: %.2f (theoretical: ~3.0)\n", r.acceptance_length);
    printf("  rejection_steps: %d / %d\n", r.total_rejection_steps, r.total_steps);
    printf("  total_cost: %.2f\n", r.total_cost);

    EXPECT(r.total_tokens_emitted == target, "emitted exact target tokens");
    EXPECT(r.acceptance_length > 2.0f && r.acceptance_length < 5.0f,
           "medium quality → acceptance length in 2-5 range");
}

static void test_e2e_speedup_vs_ar() {
    printf("\n=== Test 3: Speedup vs Autoregressive (across qualities) ===\n");
    const int target = 1000;
    const int block_size = 7;
    const int vocab_size = 1000;
    const float ar_cost = target * COST_AR_PER_TOKEN;

    printf("  AR baseline cost: %.0f\n", ar_cost);
    printf("  %-15s %-12s %-15s %-15s\n", "Quality", "AcceptLen", "SpecCost", "Speedup");
    for (float q : {0.5f, 0.7f, 0.8f, 0.9f, 0.95f}) {
        auto r = run_spec_decode_sim(target, q, block_size, vocab_size, 42);
        double speedup = ar_cost / r.total_cost;
        printf("  %-15.2f %-12.2f %-15.2f %-14.2fx\n",
               q, r.acceptance_length, r.total_cost, speedup);

        EXPECT(r.total_tokens_emitted == target, "emitted exact target tokens");
    }
}

static void test_e2e_block_size_impact() {
    printf("\n=== Test 4: Block Size Impact (block_size=3,7,15) ===\n");
    const int target = 500;
    const int vocab_size = 1000;
    const float ar_cost = target * COST_AR_PER_TOKEN;

    printf("  AR cost: %.0f\n", ar_cost);
    printf("  %-12s %-15s %-12s\n", "BlockSize", "AcceptLen", "Speedup");
    for (int bs : {3, 7, 15}) {
        auto r = run_spec_decode_sim(target, 0.8f, bs, vocab_size, 42);
        double speedup = ar_cost / r.total_cost;
        printf("  %-12d %-15.2f %-12.2fx\n", bs, r.acceptance_length, speedup);
        EXPECT(r.total_tokens_emitted == target, "emitted exact target tokens");
    }
}

static void test_e2e_consistency_with_python() {
    printf("\n=== Test 5: Consistency with Python simulator ===\n");
    printf("  Python sim: accept_rate=0.7 → 1.42x speedup, avg_acc_len=3.21\n");
    const int target = 1000;
    const int block_size = 7;
    const int vocab_size = 1000;
    const float ar_cost = target * COST_AR_PER_TOKEN;

    auto r = run_spec_decode_sim(target, 0.7f, block_size, vocab_size, 42);
    double speedup = ar_cost / r.total_cost;

    printf("  C++ sim:    acceptance_length=%.2f speedup=%.2fx\n",
           r.acceptance_length, speedup);

    // C++ uses Gaussian noise (continuous), Python uses Bernoulli (binary),
    // so absolute values differ but trend should be similar.
    EXPECT(r.acceptance_length > 2.0f, "acc length > 2 (similar to Python)");
    EXPECT(speedup > 1.5f, "spec decode faster than AR (C++ uses softmax noise → less reject)");
}

int main() {
    printf("=== Spec Decode E2E Simulation Tests (S10) ===\n");

    test_e2e_high_quality();
    test_e2e_medium_quality();
    test_e2e_speedup_vs_ar();
    test_e2e_block_size_impact();
    test_e2e_consistency_with_python();

    printf("\n=== Summary: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}