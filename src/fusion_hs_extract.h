// FusionLLM Phase 6: Hidden State Extraction for DSpark
// 利用 llama.cpp 内置的 embeddings_layer_inp API 暴露 per-layer hidden states
// 对应 DeepSpec 的 extract_context_feature(hidden_states, layer_ids)

#pragma once

#include <vector>
#include <cstdint>

struct llama_context;

namespace fusion {

// 抽取配置
struct HSExtractConfig {
    // target_layer_ids: DSpark 要抽 hidden state 的 layer
    // -1 表示 embedding output（hidden_states[0] in HuggingFace）
    // i >= 0 表示 layer i 的 output（hidden_states[i+1] in HuggingFace）
    // 内部转换为 llama.cpp 的 t_layer_inp 索引：
    //   layer_id == -1 -> t_layer_inp[0] (embedding)
    //   layer_id >= 0  -> t_layer_inp[layer_id + 1]
    std::vector<int32_t> target_layer_ids = {};

    // 输出维度（per layer hidden state size）
    // 由 model.n_embd 决定，这里只用于 sanity check
    int64_t hidden_dim = 0;
};

// 抽取的 hidden state buffer
// 对应 PyTorch: extract_context_feature 的输出
// shape: [n_tokens, len(target_layer_ids) * hidden_dim]
// storage: row-major float32, 与 PyTorch 兼容（可直接 torch.from_numpy）
struct HSBuffer {
    std::vector<float> data;          // shape: [n_tokens * concat_dim]
    int64_t n_tokens = 0;
    int64_t concat_dim = 0;           // = len(target_layer_ids) * hidden_dim
    int64_t hidden_dim = 0;           // 单 layer hidden dim
    std::vector<int32_t> layer_ids;   // 记录的 layer_ids（顺序 = data 列顺序）

    // 转 PyTorch tensor 友好的 raw pointer
    const float* ptr() const { return data.data(); }
    size_t size() const { return data.size(); }
};

class FusionHSExtractor {
public:
    FusionHSExtractor() = default;
    ~FusionHSExtractor() = default;

    // 注册要抽取的 target layer_ids
    // 在 llama_context 初始化前调用，启用 llama.cpp 的 embeddings_layer_inp 机制
    bool configure(llama_context* ctx, const HSExtractConfig& cfg);

    // 抽取最近一次 forward 的 hidden states
    // 返回 shape: [n_tokens, len(target_layer_ids) * hidden_dim]
    HSBuffer extract(llama_context* ctx) const;

    // 获取 layer_id 在 t_layer_inp 数组里的索引
    // layer_id == -1 -> 0 (embedding)
    // layer_id >= 0  -> layer_id + 1
    static int32_t to_layer_inp_index(int32_t layer_id) {
        return (layer_id < 0) ? 0 : (layer_id + 1);
    }

    const HSExtractConfig& config() const { return cfg_; }

private:
    HSExtractConfig cfg_;
    bool configured_ = false;
};

} // namespace fusion
