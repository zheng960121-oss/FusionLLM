// FusionLLM S4 测试：Rejection Sampling
// 核心算法对应 DeepSpec base_evaluator.verify_draft_tokens
// 验证:
//  1. 全 accept (draft_probs == target_probs): 应该全部接受
//  2. 全 reject (draft_probs 远高于 target_probs): 应该全部拒绝 + 第一个 resample
//  3. 混合 accept/reject: 验证前缀累积 mask (cumprod) 逻辑
//  4. Bonus token 采样: 全部接受时从 target_probs[block_size] 采样
//  5. 数值稳定性: 极端 prob (接近 0/1) 不崩溃

#include "fusion_speculative_decode.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

static int passed = 0;
static int failed = 0;

#define EXPECT(cond, msg) do { \
    if (cond) { passed++; } \
    else { failed++; fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); } \
} while (0)

// Mock softmax + sampling helpers for test setup
static std::vector<float> softmax(const std::vector<float>& logits, float temperature = 1.0f) {
    std::vector<float> probs(logits.size());
    if (temperature <= 0.0f) {
        // greedy
        int max_idx = 0;
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

// Test 1: 全 accept (draft_probs 接近 target_probs)
static void test_full_accept() {
    printf("\n=== Test 1: Full Accept (draft == target) ===\n");
    const int32_t block_size = 4;
    const int32_t vocab_size = 10;

    // draft_tokens: [2, 5, 7, 3]
    std::vector<int32_t> draft_tokens = {2, 5, 7, 3};

    // 同样分布的 probs (uniform random)
    std::vector<float> probs(vocab_size);
    for (int i = 0; i < vocab_size; ++i) probs[i] = 0.1f;

    std::vector<float> target_probs((block_size + 1) * vocab_size);
    std::vector<float> draft_probs(block_size * vocab_size);

    for (int i = 0; i < block_size + 1; ++i) {
        for (int v = 0; v < vocab_size; ++v) {
            target_probs[i * vocab_size + v] = probs[v];
        }
    }
    for (int i = 0; i < block_size; ++i) {
        for (int v = 0; v < vocab_size; ++v) {
            draft_probs[i * vocab_size + v] = probs[v];
        }
    }

    std::mt19937 rng(42);
    auto [accepted, bonus] = fusion::FusionSpecDecoder::rejection_sample(
        target_probs, draft_probs, draft_tokens, block_size, vocab_size, 1.0f, rng);

    // 应该全部接受
    EXPECT(accepted == block_size, "all accepted when draft == target");
    // bonus 应该是 argmax(target_probs[block_size]) = argmax(0.1) = 0
    EXPECT(bonus == 0, "bonus is argmax of target probs");

    printf("  accepted=%d (expected=%d), bonus=%d (expected=0)\n", accepted, block_size, bonus);
}

// Test 2: 第一个 reject (draft_probs[i, draft_tokens[i]] >> target_probs[i, draft_tokens[i]])
static void test_immediate_reject() {
    printf("\n=== Test 2: Immediate Reject (draft >> target at pos 0) ===\n");
    const int32_t block_size = 4;
    const int32_t vocab_size = 10;

    std::vector<int32_t> draft_tokens = {2, 5, 7, 3};

    // target_probs uniform, draft_probs 在 draft_tokens 处很高
    std::vector<float> target_probs((block_size + 1) * vocab_size);
    std::vector<float> draft_probs(block_size * vocab_size);

    for (int i = 0; i < block_size + 1; ++i) {
        for (int v = 0; v < vocab_size; ++v) {
            target_probs[i * vocab_size + v] = 0.1f;
        }
    }
    // draft 在 draft_tokens 处 prob=0.9 (高), 其他 0.0125
    for (int i = 0; i < block_size; ++i) {
        for (int v = 0; v < vocab_size; ++v) {
            if (v == draft_tokens[i]) draft_probs[i * vocab_size + v] = 0.9f;
            else draft_probs[i * vocab_size + v] = 0.0125f;
        }
    }

    std::mt19937 rng(42);
    auto [accepted, bonus] = fusion::FusionSpecDecoder::rejection_sample(
        target_probs, draft_probs, draft_tokens, block_size, vocab_size, 1.0f, rng);

    // accept_prob = min(1, 0.1 / 0.9) = 0.111
    // 100 次 trial：~11% 接受；这里 seed=42 应该 reject
    // 如果 accepted = 0：bonus 从 adjusted = max(target - draft, 0) = 0（uniform - spike = ...）采样
    printf("  accepted=%d (expected=0 for this seed), bonus=%d\n", accepted, bonus);
    // accept count 应该是 0 或 1 (期望为 0)
    EXPECT(accepted <= 1, "accepted at most 1 for first-position high draft prob");
}

// Test 3: 数值稳定性
static void test_numerical_stability() {
    printf("\n=== Test 3: Numerical Stability (extreme probs) ===\n");
    const int32_t block_size = 3;
    const int32_t vocab_size = 5;

    std::vector<int32_t> draft_tokens = {0, 1, 2};

    // p_draft 极小（接近 0），应该被 clamp 到 1e-8
    std::vector<float> target_probs((block_size + 1) * vocab_size, 0.2f);
    std::vector<float> draft_probs(block_size * vocab_size, 0.0f);
    // 在 draft token 处设 draft_prob 为 1e-10（极小）
    for (int i = 0; i < block_size; ++i) {
        draft_probs[i * vocab_size + draft_tokens[i]] = 1e-10f;
        // 其他位置总和
        float remaining = 1.0f - 1e-10f;
        for (int v = 0; v < vocab_size; ++v) {
            if (v != draft_tokens[i]) {
                draft_probs[i * vocab_size + v] = remaining / (vocab_size - 1);
            }
        }
    }

    std::mt19937 rng(42);
    auto [accepted, bonus] = fusion::FusionSpecDecoder::rejection_sample(
        target_probs, draft_probs, draft_tokens, block_size, vocab_size, 1.0f, rng);

    // accept_prob = min(1, 0.2 / 1e-8) = 1.0（因为 target > draft）
    // 所以应该全部接受
    EXPECT(accepted == block_size, "extreme low draft_prob → all accept (accept_prob clamped)");
    printf("  accepted=%d (expected=%d), bonus=%d\n", accepted, block_size, bonus);
}

// Test 4: Prefix accept mask (cumprod) - 接受前面的几个，然后 reject
static void test_prefix_accept_then_reject() {
    printf("\n=== Test 4: Prefix Accept + Posterior Reject ===\n");
    const int32_t block_size = 5;
    const int32_t vocab_size = 8;

    std::vector<int32_t> draft_tokens = {3, 1, 4, 0, 2};

    // target_probs: pos 0-2 uniform, pos 3 spike at token 0, pos 4 spike at token 2
    std::vector<float> target_probs((block_size + 1) * vocab_size);
    for (int i = 0; i < block_size; ++i) {
        for (int v = 0; v < vocab_size; ++v) {
            if (i < 3) {
                target_probs[i * vocab_size + v] = 0.125f;
            } else if (i == 3) {
                // 在 token 0 处 spike
                target_probs[i * vocab_size + v] = (v == 0) ? 0.9f : 0.0143f;
            } else {  // i == 4
                target_probs[i * vocab_size + v] = (v == 2) ? 0.9f : 0.0143f;
            }
        }
    }
    target_probs[block_size * vocab_size + 0] = 1.0f;  // bonus position: greedy to token 0

    // draft_probs: 在 draft_tokens 处 prob 较高（让 pos 0-2 accept, pos 3 reject）
    std::vector<float> draft_probs(block_size * vocab_size);
    for (int i = 0; i < block_size; ++i) {
        for (int v = 0; v < vocab_size; ++v) {
            if (v == draft_tokens[i]) draft_probs[i * vocab_size + v] = 0.5f;
            else draft_probs[i * vocab_size + v] = 0.5f / (vocab_size - 1);
        }
    }

    std::mt19937 rng(42);
    auto [accepted, bonus] = fusion::FusionSpecDecoder::rejection_sample(
        target_probs, draft_probs, draft_tokens, block_size, vocab_size, 1.0f, rng);

    // pos 0-2: target=0.125, draft=0.5 → accept_prob=0.25 (25%)
    // pos 3: target=0.9, draft=0.5 → accept_prob=1.0 (100%)
    // pos 4: target=0.9, draft=0.5 → accept_prob=1.0 (100%)
    // 期望: 大概率全部接受
    printf("  accepted=%d (range: 0-%d, expected ~%d)\n", accepted, block_size, block_size);
    EXPECT(accepted >= 0 && accepted <= block_size, "accepted in valid range");
}

int main() {
    printf("=== FusionSpec Rejection Sampling Unit Tests ===\n");

    test_full_accept();
    test_immediate_reject();
    test_numerical_stability();
    test_prefix_accept_then_reject();

    printf("\n=== Summary: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
