// FusionLLM Phase 6: Speculative Decoding Loop
// 对应 DeepSpec base_evaluator.generate_decoding_sample + verify_draft_tokens
// 见 docs/DSpark_FusionLLM_detailed_spec.md §2.3 + §3.5

#pragma once

#include "fusion_draft_model.h"
#include "fusion_hs_extract.h"

#include "llama.h"
#include "ggml.h"

#include <vector>
#include <cstdint>
#include <memory>
#include <random>

struct llama_context;

namespace fusion {

// 一步 spec decoding 的结果
struct SpecDecodeStep {
    int32_t accepted_count = 0;         // 接受的 draft tokens 数
    int32_t bonus_token = -1;           // resample 的 bonus token
    int32_t effective_length = 0;       // accepted_count + 1（bonus）
    bool terminated = false;             // 命中 EOS
    int32_t n_draft_proposed = 0;       // 这次 propose 了多少 draft tokens
    int32_t n_verify_calls = 0;         // verify 调用次数（= 1 每次 step）
};

// 完整 generate 结果
struct SpecDecodeStats {
    int32_t total_draft_proposed = 0;
    int32_t total_draft_accepted = 0;
    int32_t total_verify_calls = 0;
    int32_t total_output_tokens = 0;
    double acceptance_length = 0.0;    // 平均每个 verify step 接受的 tokens（含 bonus）
    double acceptance_rate = 0.0;       // total_accepted / total_proposed
};

// Spec decoder 主类
class FusionSpecDecoder {
public:
    FusionSpecDecoder() = default;
    ~FusionSpecDecoder() = default;

    // 初始化：绑定 target model context + draft model + hidden state extractor
    bool init(
        llama_context* target_ctx,
        FusionDSparkModel* draft_model,
        FusionHSExtractor* hs_extractor
    );

    // 完整 generate：propose + verify + commit 直到 max_new_tokens 或 EOS
    // input_ids:    prompt tokens
    // output_ids:   输出（接受的所有 tokens，含 prompt）
    // max_new_tokens: 最大生成数
    // temperature:  采样温度（0 = greedy）
    // stop_token_ids: 终止 token 列表
    int generate(
        const std::vector<int32_t>& input_ids,
        std::vector<int32_t>& output_ids,
        int32_t max_new_tokens,
        float temperature = 0.0f,
        const std::vector<int32_t>& stop_token_ids = {}
    );

    // 单步 spec decode（内部接口，也可以外部调用测试）
    // target_kv/draft_kv 留作未来扩展（目前 KV 都在 llama_context 内部管理）
    SpecDecodeStep step_spec(
        std::vector<int32_t>& output_ids,           // 当前已接受的 tokens
        int32_t& start,                              // 当前生成位置
        float temperature,
        const std::vector<int32_t>& stop_token_ids = {},
        struct llama_kv_cache* target_kv = nullptr,
        struct llama_kv_cache* draft_kv = nullptr
    );

    // Fallback autoregressive（draft model 不可用或全部拒绝时）
    SpecDecodeStep step_autoregressive(
        std::vector<int32_t>& output_ids,
        int32_t& start,
        float temperature,
        const std::vector<int32_t>& stop_token_ids = {},
        struct llama_kv_cache* target_kv = nullptr
    );

    // Rejection sampling（核心算法，可独立测试）
    // 对应 DeepSpec verify_draft_tokens 的 rejection sampling 逻辑
    // target_probs: [block_size + 1, vocab]  target model 在 verify 时的 probs
    // draft_probs:  [block_size, vocab]      draft model 的 probs
    // draft_tokens: [block_size]              draft 生成的 tokens
    // 返回: (accepted_count, bonus_token)
    static std::pair<int32_t, int32_t> rejection_sample(
        const std::vector<float>& target_probs,
        const std::vector<float>& draft_probs,
        const std::vector<int32_t>& draft_tokens,
        int32_t block_size,
        int32_t vocab_size,
        float temperature,
        std::mt19937& rng
    );

    const SpecDecodeStats& stats() const { return stats_; }

private:
    llama_context* target_ctx_ = nullptr;
    FusionDSparkModel* draft_model_ = nullptr;
    FusionHSExtractor* hs_extractor_ = nullptr;

    // 累积状态
    SpecDecodeStats stats_;
    bool initialized_ = false;
};

} // namespace fusion
