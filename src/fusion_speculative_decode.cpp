// FusionLLM Phase 6: Speculative Decoding Loop implementation
// 对应 DeepSpec base_evaluator.verify_draft_tokens + generate_decoding_sample

#include "fusion_speculative_decode.h"
#include "fusion_window.h"

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>

namespace fusion {

// ---------- Rejection Sampling（核心算法，对应 DeepSpec verify_draft_tokens）----------

std::pair<int32_t, int32_t> FusionSpecDecoder::rejection_sample(
    const std::vector<float>& target_probs,
    const std::vector<float>& draft_probs,
    const std::vector<int32_t>& draft_tokens,
    int32_t block_size,
    int32_t vocab_size,
    float temperature,
    std::mt19937& rng
) {
    // 对应 PyTorch (DeepSpec base_evaluator.py:verify_draft_tokens):
    //   for i in range(draft_count):
    //       accept_prob = min(1, target_probs[i, draft_tokens[i]] / draft_probs[i, draft_tokens[i]])
    //       if rand() < accept_prob: accept
    //       else: reject (cumulative), resample target_probs[i] adjusted by draft_probs
    //
    // bonus_token = resample(target_probs[accepted_count]) adjusted by draft_probs[accepted_count]

    int32_t accepted_count = 0;
    int32_t bonus_token = -1;

    // 逐个 draft position 做 rejection sampling
    for (int32_t i = 0; i < block_size; ++i) {
        int32_t tok = draft_tokens[i];

        // target_probs 和 draft_probs 在 token 维度的 prob
        float p_target = target_probs[i * vocab_size + tok];
        float p_draft = draft_probs[i * vocab_size + tok];

        // Clamp 避免除零
        p_draft = std::max(p_draft, 1e-8f);
        p_target = std::max(p_target, 0.0f);

        // accept_prob = min(1, target / draft)
        float accept_prob = std::min(1.0f, p_target / p_draft);

        std::uniform_real_distribution<float> uni(0.0f, 1.0f);
        if (uni(rng) < accept_prob) {
            // 接受
            accepted_count++;
        } else {
            // 拒绝：从调整后的分布 resample
            // adjusted_probs = max(target_probs[i] - draft_probs[i], 0), normalize
            std::vector<float> adjusted(vocab_size);
            float sum = 0.0f;
            for (int32_t v = 0; v < vocab_size; ++v) {
                float adj = std::max(target_probs[i * vocab_size + v] - draft_probs[i * vocab_size + v], 0.0f);
                adjusted[v] = adj;
                sum += adj;
            }
            if (sum <= 0.0f) {
                // Fallback: 用 target_probs 直接采样
                for (int32_t v = 0; v < vocab_size; ++v) {
                    adjusted[v] = target_probs[i * vocab_size + v];
                    sum += adjusted[v];
                }
            }
            // Normalize
            float inv_sum = 1.0f / sum;
            for (int32_t v = 0; v < vocab_size; ++v) {
                adjusted[v] *= inv_sum;
            }

            // Cumulative sample
            std::uniform_real_distribution<float> uni2(0.0f, 1.0f);
            float r = uni2(rng);
            float cum = 0.0f;
            for (int32_t v = 0; v < vocab_size; ++v) {
                cum += adjusted[v];
                if (r <= cum) {
                    bonus_token = v;
                    break;
                }
            }
            if (bonus_token < 0) bonus_token = vocab_size - 1;  // numerical fallback

            return {accepted_count, bonus_token};
        }
    }

    // 全部接受：bonus 从 target_probs[block_size] 采样（不调整）
    bonus_token = 0;
    float max_p = -1.0f;
    for (int32_t v = 0; v < vocab_size; ++v) {
        float p = target_probs[block_size * vocab_size + v];
        if (p > max_p) {
            max_p = p;
            bonus_token = v;
        }
    }

    return {accepted_count, bonus_token};
}

// ---------- Spec Decoder 主类 ----------

bool FusionSpecDecoder::init(
    llama_context* target_ctx,
    FusionDSparkModel* draft_model,
    FusionHSExtractor* hs_extractor
) {
    if (!target_ctx || !draft_model) {
        fprintf(stderr, "[FusionSpec] init: null context\n");
        return false;
    }
    target_ctx_ = target_ctx;
    draft_model_ = draft_model;
    hs_extractor_ = hs_extractor;
    stats_ = SpecDecodeStats{};
    initialized_ = true;
    fprintf(stderr, "[FusionSpec] initialized: target_ctx=%p draft=%p hs_extract=%p\n",
            (void*)target_ctx, (void*)draft_model, (void*)hs_extractor);
    return true;
}

int FusionSpecDecoder::generate(
    const std::vector<int32_t>& input_ids,
    std::vector<int32_t>& output_ids,
    int32_t max_new_tokens,
    float temperature,
    const std::vector<int32_t>& stop_token_ids
) {
    if (!initialized_) {
        fprintf(stderr, "[FusionSpec] generate: not initialized\n");
        return -1;
    }

    output_ids = input_ids;
    int32_t start = (int32_t)input_ids.size();
    int32_t max_length = start + max_new_tokens;

    std::mt19937 rng(42);  // 固定 seed 便于测试

    // 注：完整的 spec decoding 实现需要：
    //   1. prefill target model with input_ids
    //   2. extract target hidden states (FusionHSExtractor)
    //   3. propose: draft model forward → block_size tokens
    //   4. verify: target model forward on [current_token, draft_tokens]
    //   5. rejection sampling (上面实现)
    //   6. commit accepted tokens, crop KV cache
    //   7. repeat from step 2 with new target hidden states
    //
    // S4 第一晚只完成框架 + rejection sampling。完整 forward 集成
    // 留到后续 step（需要 llama.cpp 内部 forward hook）。

    fprintf(stderr, "[FusionSpec] generate: S4 skeleton (rejection_sample + step_spec framework)\n");
    fprintf(stderr, "  input_ids: %zu tokens, max_new: %d\n", input_ids.size(), max_new_tokens);
    fprintf(stderr, "  temperature: %.3f, stop_tokens: %zu\n", temperature, stop_token_ids.size());

    // 占位：循环调用 step_spec 直到 max_new_tokens
    while (start < max_length) {
        // 这里需要真正的 draft + target forward
        // 当前阶段用 autoregressive 替代，避免无限等待
        auto step = step_autoregressive(output_ids, start, temperature, stop_token_ids,
                                        /*target_kv=*/*((llama_kv_cache*)nullptr));
        if (step.terminated) break;
        stats_.total_output_tokens++;
        if (start >= max_length) break;
    }

    stats_.acceptance_length = (double)stats_.total_output_tokens /
                               std::max(stats_.total_verify_calls, 1);

    return (int)stats_.total_output_tokens;
}

SpecDecodeStep FusionSpecDecoder::step_spec(
    std::vector<int32_t>& output_ids,
    int32_t& start,
    float temperature,
    const std::vector<int32_t>& stop_token_ids,
    llama_kv_cache& target_kv,
    llama_kv_cache& draft_kv
) {
    SpecDecodeStep result;
    fprintf(stderr, "[FusionSpec] step_spec: S4 TODO (skeleton)\n");
    // 完整实现在 S4 后续：
    //   1. propose: draft_forward(target_hs, draft_input) → block_size tokens
    //   2. verify: target_forward([current, draft]) → target_probs
    //   3. rejection_sample(target_probs, draft_probs, draft_tokens)
    //   4. commit accepted tokens + bonus, crop KV cache
    //   5. extract new target hidden states for next iteration

    // S5: spec verify 触发 FusionLLM sliding window 统计
    window_on_verify_step_begin();
    // verify_forward(...);  // S4 TODO: 这里跑 target forward
    window_on_verify_step_end();
    int advances_per_verify = window_get_last_step_advance_count();
    fprintf(stderr, "[FusionSpec] verify step: %d window_advance calls (sliding window协同)\n",
            advances_per_verify);

    return result;
}

SpecDecodeStep FusionSpecDecoder::step_autoregressive(
    std::vector<int32_t>& output_ids,
    int32_t& start,
    float temperature,
    const std::vector<int32_t>& stop_token_ids,
    llama_kv_cache& target_kv
) {
    SpecDecodeStep result;
    fprintf(stderr, "[FusionSpec] step_autoregressive: S4 TODO\n");
    // 占位：单 token 采样 + advance
    if (start >= (int)output_ids.size()) {
        output_ids.push_back(0);  // dummy token
    }
    result.accepted_count = 1;
    result.bonus_token = 0;
    result.effective_length = 1;
    start++;
    stats_.total_verify_calls++;
    stats_.total_draft_proposed += 0;
    stats_.total_draft_accepted += 0;
    return result;
}

} // namespace fusion
