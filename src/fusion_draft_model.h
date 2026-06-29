// FusionLLM Phase 6: DSpark Draft Model
// 对应 DeepSpec Qwen3DSparkModel + DSparkDraftProposal
// 见 docs/DSpark_FusionLLM_detailed_spec.md

#pragma once

#include "llama.h"
#include "ggml.h"

#include <vector>
#include <cstdint>
#include <memory>
#include <string>

struct llama_context;
struct ggml_context;

namespace fusion {

// 配置（从 GGUF metadata 读取）
struct FusionDSparkConfig {
    int32_t block_size = 7;
    int32_t num_draft_layers = 5;
    int32_t markov_rank = 256;
    std::string markov_head_type = "vanilla";  // vanilla / gated / rnn
    int32_t mask_token_id = 151669;
    bool enable_confidence_head = false;

    // target_layer_ids: 要抽 target model 哪些层的 hidden state
    // -1 表示 embedding output，>=0 表示 layer i 的 output
    std::vector<int32_t> target_layer_ids = {1, 9, 17, 25, 33};

    // 目标模型信息（用于共享 embeddings）
    int32_t n_embd = 0;          // hidden_size
    int32_t n_head = 0;
    int32_t n_head_kv = 0;
    int32_t n_ff = 0;
    int32_t n_vocab = 0;
    int32_t head_dim = 0;
    int32_t rope_mode = 0;       // RoPE type

    // 计算 target_layer_ids 在拼接后的维度
    int32_t concat_dim() const { return (int32_t)target_layer_ids.size() * n_embd; }
};

// Markov head 类型
enum class MarkovHeadType {
    NONE = 0,       // markov_rank == 0
    VANILLA = 1,    // Embedding(vocab, rank) + Linear(rank, vocab)
    GATED = 2,      // 加 gate_proj(hidden + prev_emb)
    RNN = 3,        // 加 GRU-like recurrent state
};

// 单个 decoder layer 权重
struct DSparkLayerWeights {
    // Attention (4 projections + 2 RMSNorms)
    struct ggml_tensor* wq = nullptr;
    struct ggml_tensor* wk = nullptr;
    struct ggml_tensor* wv = nullptr;
    struct ggml_tensor* wo = nullptr;
    struct ggml_tensor* attn_q_norm = nullptr;
    struct ggml_tensor* attn_k_norm = nullptr;

    // FFN
    struct ggml_tensor* ffn_gate = nullptr;
    struct ggml_tensor* ffn_up = nullptr;
    struct ggml_tensor* ffn_down = nullptr;

    // Norms
    struct ggml_tensor* input_layernorm = nullptr;
    struct ggml_tensor* post_attention_layernorm = nullptr;
};

// Markov head 权重
struct MarkovHeadWeights {
    struct ggml_tensor* markov_w1 = nullptr;  // [vocab, rank]
    struct ggml_tensor* markov_w2 = nullptr;  // [rank, vocab]
    // Gated 模式
    struct ggml_tensor* gate_proj = nullptr;  // [rank, hidden + rank]
    // RNN 模式
    struct ggml_tensor* joint_proj = nullptr; // [3*rank, 2*rank + hidden]
    MarkovHeadType type = MarkovHeadType::NONE;
    int32_t rank = 0;
};

// Confidence head 权重（可选）
struct ConfidenceHeadWeights {
    struct ggml_tensor* proj = nullptr;  // [1, input_dim]
    bool with_markov = false;
};

// 完整 draft model 权重集合
struct FusionDSparkWeights {
    // Embedding + LM head（可与 target 共享）
    struct ggml_tensor* embed_tokens = nullptr;
    struct ggml_tensor* lm_head = nullptr;

    // fc: 把多个 target layer hidden state 拼成一个 hidden_size
    // [hidden_size, len(target_layer_ids) * hidden_size]
    struct ggml_tensor* fc = nullptr;
    struct ggml_tensor* hidden_norm = nullptr;  // RMSNorm [hidden_size]

    // Layers (num_draft_layers 个)
    std::vector<DSparkLayerWeights> layers;

    // Final norm
    struct ggml_tensor* final_norm = nullptr;

    // Rotary embedding（从 target 共享）

    // Heads
    MarkovHeadWeights markov_head;
    ConfidenceHeadWeights confidence_head;
};

// DSpark draft model 主类
class FusionDSparkModel {
public:
    FusionDSparkModel() = default;
    ~FusionDSparkModel() = default;

    // 加载 draft GGUF（独立文件，例如 qwen3-4b-dspark-q4.gguf）
    bool load_from_gguf(const std::string& path);

    // 共享 target model 的 embedding 和 lm_head（避免重复存储）
    void share_target_embeddings(ggml_tensor* target_embed, ggml_tensor* target_lm_head);

    // 推理 forward
    // target_hs:        [1, ctx_len, concat_dim()]  从 target 抽的多层 hidden state 拼接
    // draft_input_ids:  [1, block_size]
    // position_ids:     [1, ctx_len + block_size]
    // past_kv_draft:    之前的 draft KV cache（spec decode 用）
    // 返回: [1, block_size, vocab] logits
    struct ggml_tensor* forward(
        struct ggml_context* ctx,
        struct ggml_tensor* target_hs,
        struct ggml_tensor* draft_input_ids,
        struct ggml_tensor* position_ids,
        struct llama_kv_cache* past_kv_draft
    );

    // 采样 block_size 个 draft tokens（基于 Markov head 调整）
    // 返回: (sampled_token_ids [1, block_size], probs [1, block_size, vocab])
    // 实际 llama.cpp 端采样由调用方用 llama_sample_token 完成
    void apply_markov_head(
        struct ggml_context* ctx,
        struct ggml_tensor* base_logits,         // [1, block_size, vocab]
        struct ggml_tensor* hidden_states,       // [1, block_size, n_embd]
        struct ggml_tensor* prev_token_ids,      // [1, block_size]
        struct ggml_tensor* output_logits        // 输出: [1, block_size, vocab]
    );

    // 配置 + 权重访问器
    const FusionDSparkConfig& config() const { return cfg_; }
    const FusionDSparkWeights& weights() const { return weights_; }
    bool is_loaded() const { return loaded_; }

private:
    FusionDSparkConfig cfg_;
    FusionDSparkWeights weights_;
    bool loaded_ = false;
    std::string model_path_;

    // 内部 helper
    bool parse_metadata(const struct gguf_context* gctx);
    bool load_weights(const struct ggml_context* ctx, const struct gguf_context* gctx);
    void free_weights();
};

// Markov head forward（对应 PyTorch markov_head.apply_block_logits）
// base_logits: [B, num_blocks, block_size, vocab]
// token_ids:   [B, num_blocks, block_size]
// hidden_states: [B, num_blocks, block_size, n_embd]
// 返回: 调整后的 logits（同 shape）
struct ggml_tensor* markov_head_forward(
    struct ggml_context* ctx,
    const MarkovHeadWeights& head,
    struct ggml_tensor* base_logits,
    struct ggml_tensor* token_ids,
    struct ggml_tensor* hidden_states,
    int32_t n_embd
);

} // namespace fusion
