// FusionLLM Phase 6: Hidden State Extraction implementation
// 对应 DeepSpec extract_context_feature：
//   def extract_context_feature(hidden_states, layer_ids):
//       return torch.cat(
//           [hidden_states[0 if layer_id == -1 else layer_id + 1] for layer_id in layer_ids],
//           dim=-1,
//       )

#include "fusion_hs_extract.h"

// llama-context.h 是内部头，FusionHSExtractor 作为 fork 内部模块可以直接使用
#include "llama.h"
#include "llama-context.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace fusion {

bool FusionHSExtractor::configure(llama_context* ctx, const HSExtractConfig& cfg) {
    if (!ctx) {
        fprintf(stderr, "[FusionHS] configure: ctx is null\n");
        return false;
    }
    if (cfg.target_layer_ids.empty()) {
        fprintf(stderr, "[FusionHS] configure: target_layer_ids is empty\n");
        return false;
    }

    cfg_ = cfg;
    configured_ = false;

    // 启用 llama.cpp 的 per-layer embeddings output
    // embeddings_layer_inp[idx] == true 时，t_layer_inp[idx] 被标记为 graph output
    // forward 后用 get_embeddings_layer_inp(idx) 拿到 raw float* 指针
    for (int32_t layer_id : cfg.target_layer_ids) {
        int32_t idx = to_layer_inp_index(layer_id);
        fprintf(stderr, "[FusionHS] configure: enabling t_layer_inp[%d] (from layer_id=%d)\n",
                idx, layer_id);
        ctx->set_embeddings_layer_inp(idx, true);
    }

    configured_ = true;
    fprintf(stderr, "[FusionHS] configure: enabled %zu target_layer_ids: [",
            cfg.target_layer_ids.size());
    for (size_t i = 0; i < cfg.target_layer_ids.size(); ++i) {
        fprintf(stderr, "%s%d", i ? "," : "", cfg.target_layer_ids[i]);
    }
    fprintf(stderr, "] (mapped to t_layer_inp indices [");
    for (size_t i = 0; i < cfg.target_layer_ids.size(); ++i) {
        fprintf(stderr, "%s%d", i ? "," : "",
                to_layer_inp_index(cfg.target_layer_ids[i]));
    }
    fprintf(stderr, "])\n");

    return true;
}

HSBuffer FusionHSExtractor::extract(llama_context* ctx) const {
    HSBuffer out;
    if (!configured_ || !ctx) {
        fprintf(stderr, "[FusionHS] extract: not configured or ctx null\n");
        return out;
    }

    out.layer_ids = cfg_.target_layer_ids;
    out.hidden_dim = cfg_.hidden_dim;
    out.concat_dim = (int64_t)cfg_.target_layer_ids.size() * cfg_.hidden_dim;

    // 第一遍：先抽 layer 0 (idx=0)，确定 n_tokens
    // 因为所有 target layer 的 n_tokens 必须相同（同一次 forward）
    int32_t first_idx = to_layer_inp_index(cfg_.target_layer_ids[0]);
    const float* first_ptr = ctx->get_embeddings_layer_inp(first_idx);
    if (!first_ptr) {
        fprintf(stderr, "[FusionHS] extract: layer_inp[%d] is null (not computed?)\n", first_idx);
        return out;
    }

    // t_layer_inp 的 buffer 大小是 n_embd * n_batch（见 llama-context.cpp:2101）
    // n_tokens 是当前 batch 的实际 token 数（context_decode 时为 n_tokens）
    // 我们用第一个 layer 的数据初始化 buffer 并 probe n_tokens
    // 假设：buffer 长度 = hidden_dim * n_tokens
    // 由于 llama.cpp 没暴露 n_tokens，我们假设 hidden_dim 已知，data 长度 = hidden_dim * n_tokens
    // 但拿不到 data 长度，只能假设 hidden_dim 在 cfg 里
    if (cfg_.hidden_dim <= 0) {
        fprintf(stderr, "[FusionHS] extract: hidden_dim not set (%ld)\n",
                cfg_.hidden_dim);
        return out;
    }

    // 探测 n_tokens：扫描 llama_batch 的逻辑位置数
    // 简化方案：第一次 extract 时假设 n_tokens = 1（prefill-only），后续 caller 调整
    // 实际生产：调用方传 n_tokens 进来
    out.n_tokens = 1;  // TODO: probe from llama_batch

    // 分配 output buffer
    out.data.resize(out.n_tokens * out.concat_dim);

    // 第二遍：抽每个 layer 并拼接
    // 顺序按 cfg.target_layer_ids 升序（与 PyTorch extract_context_feature 一致）
    for (size_t li = 0; li < cfg_.target_layer_ids.size(); ++li) {
        int32_t idx = to_layer_inp_index(cfg_.target_layer_ids[li]);
        const float* layer_data = ctx->get_embeddings_layer_inp(idx);
        if (!layer_data) {
            fprintf(stderr, "[FusionHS] extract: layer_inp[%d] is null\n", idx);
            out.data.clear();
            out.n_tokens = 0;
            return out;
        }
        // 拷贝到 output 对应位置
        // out[:, li * hidden_dim : (li + 1) * hidden_dim] = layer_data
        for (int64_t t = 0; t < out.n_tokens; ++t) {
            std::memcpy(
                out.data.data() + t * out.concat_dim + li * cfg_.hidden_dim,
                layer_data + t * cfg_.hidden_dim,
                cfg_.hidden_dim * sizeof(float)
            );
        }
    }

    fprintf(stderr, "[FusionHS] extract: %lld tokens × %lld concat_dim (hidden_dim=%lld, %zu layers)\n",
            (long long)out.n_tokens, (long long)out.concat_dim,
            (long long)out.hidden_dim, cfg_.target_layer_ids.size());

    return out;
}

} // namespace fusion
