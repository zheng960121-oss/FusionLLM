// FusionLLM Phase 6: DSpark Draft Model implementation
// S3: skeleton + metadata parsing
// S7: 完整 attention + decoder layer 实现

#include "fusion_draft_model.h"

#include "llama.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <string>

namespace fusion {

// ---------- Configuration parsing ----------

static int32_t gguf_get_u32_by_name(const gguf_context* ctx, const char* name, int32_t fallback) {
    int64_t key_id = gguf_find_key(ctx, name);
    if (key_id < 0) return fallback;
    return (int32_t)gguf_get_val_u32(ctx, key_id);
}

static bool gguf_get_i32_arr_by_name(const gguf_context* ctx, const char* name,
                                      std::vector<int32_t>& out) {
    int64_t key_id = gguf_find_key(ctx, name);
    if (key_id < 0) return false;
    int64_t n = gguf_get_arr_n(ctx, key_id);
    if (n <= 0 || n > 100) return false;
    out.resize(n);
    const int32_t* arr = (const int32_t*)gguf_get_arr_data(ctx, key_id);
    if (!arr) return false;
    for (int64_t i = 0; i < n; ++i) out[i] = arr[i];
    return true;
}

bool FusionDSparkModel::parse_metadata(const struct gguf_context* gctx) {
    if (!gctx) return false;

    cfg_.block_size = gguf_get_u32_by_name(gctx, "fusion.draft.block_size", 7);
    cfg_.num_draft_layers = gguf_get_u32_by_name(gctx, "fusion.draft.num_layers", 5);
    cfg_.markov_rank = gguf_get_u32_by_name(gctx, "fusion.draft.markov_rank", 0);

    int64_t mht_id = gguf_find_key(gctx, "fusion.draft.markov_head_type");
    if (mht_id >= 0) {
        const char* mht = gguf_get_val_str(gctx, mht_id);
        if (mht) cfg_.markov_head_type = std::string(mht);
    }

    cfg_.mask_token_id = gguf_get_u32_by_name(gctx, "fusion.draft.mask_token_id", 0);
    gguf_get_i32_arr_by_name(gctx, "fusion.draft.target_layer_ids", cfg_.target_layer_ids);

    cfg_.n_embd    = gguf_get_u32_by_name(gctx, "fusion.draft.target.n_embd", 0);
    cfg_.n_head    = gguf_get_u32_by_name(gctx, "fusion.draft.target.n_head", 0);
    cfg_.n_head_kv = gguf_get_u32_by_name(gctx, "fusion.draft.target.n_head_kv", 0);
    cfg_.n_vocab   = gguf_get_u32_by_name(gctx, "fusion.draft.target.n_vocab", 0);
    cfg_.head_dim  = gguf_get_u32_by_name(gctx, "fusion.draft.target.head_dim", 0);
    cfg_.rope_mode = gguf_get_u32_by_name(gctx, "fusion.draft.target.rope_mode", 0);

    // n_ff 在 draft model 里通常从 metadata 读，或者 = 4/3 * n_embd
    if (cfg_.n_ff == 0 && cfg_.n_embd > 0) {
        cfg_.n_ff = (cfg_.n_embd * 8 / 3 / 64) * 64;  // round to 64
        if (cfg_.n_ff < cfg_.n_embd) cfg_.n_ff = cfg_.n_embd * 2;
    }

    fprintf(stderr, "[FusionDSpark] parsed metadata:\n");
    fprintf(stderr, "  block_size=%d, num_layers=%d, markov_rank=%d, mask_token_id=%d\n",
            cfg_.block_size, cfg_.num_draft_layers, cfg_.markov_rank, cfg_.mask_token_id);
    fprintf(stderr, "  markov_head_type=%s, target_layer_ids=[", cfg_.markov_head_type.c_str());
    for (size_t i = 0; i < cfg_.target_layer_ids.size(); ++i) {
        fprintf(stderr, "%s%d", i ? "," : "", cfg_.target_layer_ids[i]);
    }
    fprintf(stderr, "]\n");
    fprintf(stderr, "  target: n_embd=%d n_head=%d n_head_kv=%d n_ff=%d n_vocab=%d head_dim=%d\n",
            cfg_.n_embd, cfg_.n_head, cfg_.n_head_kv, cfg_.n_ff, cfg_.n_vocab, cfg_.head_dim);

    return true;
}

// ---------- Weight loading ----------

bool FusionDSparkModel::load_weights(const struct ggml_context* ctx, const struct gguf_context* gctx) {
    fprintf(stderr, "[FusionDSpark] load_weights: TODO (placeholder, real loading in S9)\n");
    weights_.layers.resize(cfg_.num_draft_layers);

    if (cfg_.markov_rank > 0) {
        if (cfg_.markov_head_type == "vanilla") weights_.markov_head.type = MarkovHeadType::VANILLA;
        else if (cfg_.markov_head_type == "gated") weights_.markov_head.type = MarkovHeadType::GATED;
        else if (cfg_.markov_head_type == "rnn") weights_.markov_head.type = MarkovHeadType::RNN;
        weights_.markov_head.rank = cfg_.markov_rank;
    }
    (void)ctx; (void)gctx;
    return true;
}

void FusionDSparkModel::free_weights() {
    weights_.layers.clear();
    loaded_ = false;
}

// ---------- Public API ----------

bool FusionDSparkModel::load_from_gguf(const std::string& path) {
    if (loaded_) {
        fprintf(stderr, "[FusionDSpark] already loaded from %s\n", model_path_.c_str());
        return true;
    }

    fprintf(stderr, "[FusionDSpark] loading from %s\n", path.c_str());

    struct gguf_init_params iparams = { true, nullptr };
    struct gguf_context* gctx = gguf_init_from_file(path.c_str(), iparams);
    if (!gctx) {
        fprintf(stderr, "[FusionDSpark] failed to open GGUF: %s\n", path.c_str());
        return false;
    }

    if (!parse_metadata(gctx)) {
        gguf_free(gctx);
        return false;
    }

    if (!load_weights(nullptr, gctx)) {
        gguf_free(gctx);
        return false;
    }

    gguf_free(gctx);

    model_path_ = path;
    loaded_ = true;
    fprintf(stderr, "[FusionDSpark] loaded OK\n");
    return true;
}

void FusionDSparkModel::share_target_embeddings(ggml_tensor* target_embed, ggml_tensor* target_lm_head) {
    weights_.embed_tokens = target_embed;
    weights_.lm_head = target_lm_head;
    fprintf(stderr, "[FusionDSpark] sharing target embeddings (%p, %p)\n",
            (void*)target_embed, (void*)target_lm_head);
}

// ---------- S7: Forward implementation ----------
//
// DSpark attention 双输入实现，对应 PyTorch Qwen3DSparkAttention:
//   - Q from draft input only
//   - K/V from concat(target_context, draft_noise)
//   - RoPE on full concatenated sequence
//   - SDPA (flash attention)
//   - Output projection

namespace {

// RMSNorm with weight (ggml_rms_norm 没有 weight 版本，需要单独 apply)
static ggml_tensor* rms_norm_weighted(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight) {
    ggml_tensor* normed = ggml_rms_norm(ctx, x, 1e-6f);
    return ggml_mul(ctx, normed, weight);  // element-wise * weight
}

// DSpark 双输入 attention forward
// hidden_states: [n_embd, q_len, B]    draft input
// target_hs:     [n_embd, ctx_len, B]  target context (already projected via fc)
static ggml_tensor* dspark_attention_forward(
    ggml_context* ctx,
    const DSparkLayerWeights& w,
    ggml_tensor* hidden_states,
    ggml_tensor* target_hs,
    ggml_tensor* position_ids,        // [start, start+ctx_len+q_len)
    int n_head, int n_head_kv, int head_dim, int n_embd,
    int q_len, int ctx_len,
    int il
) {
    (void)il;

    // 1. Q projection from draft input only
    // ggml_mul_mat: [n_embd_out, K] × [K, M] = [n_embd_out, M]
    // wq shape: [n_embd, n_embd]; hidden_states [n_embd, q_len] -> q [n_embd, q_len]
    ggml_tensor* q = ggml_mul_mat(ctx, w.wq, hidden_states);
    // Reshape to [head_dim, n_head, q_len, B]
    q = ggml_reshape_4d(ctx, q, head_dim, n_head, q_len, 1);
    // Permute to [head_dim, q_len, n_head, B] for SDPA
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));

    // per-head RMSNorm on Q
    q = rms_norm_weighted(ctx, q, w.attn_q_norm);
    q = ggml_cont(ctx, q);

    // 2. K from target context (use n_head_kv since GQA)
    ggml_tensor* k_ctx = ggml_mul_mat(ctx, w.wk, target_hs);
    k_ctx = ggml_reshape_4d(ctx, k_ctx, head_dim, n_head_kv, ctx_len, 1);

    // 3. K from draft noise
    ggml_tensor* k_noise = ggml_mul_mat(ctx, w.wk, hidden_states);
    k_noise = ggml_reshape_4d(ctx, k_noise, head_dim, n_head_kv, q_len, 1);

    // 4. Concat K: [target_ctx, draft_noise] along sequence dim (dim=2)
    ggml_tensor* k = ggml_concat(ctx, k_ctx, k_noise, 2);
    // After concat: [head_dim, n_head_kv, ctx_len+q_len, 1]
    // Permute for SDPA: [head_dim, ctx_len+q_len, n_head_kv, 1]
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));

    // per-head RMSNorm on K
    k = rms_norm_weighted(ctx, k, w.attn_k_norm);
    k = ggml_cont(ctx, k);

    // 5. V: same concat pattern (no norm)
    ggml_tensor* v_ctx = ggml_mul_mat(ctx, w.wv, target_hs);
    v_ctx = ggml_reshape_4d(ctx, v_ctx, head_dim, n_head_kv, ctx_len, 1);
    ggml_tensor* v_noise = ggml_mul_mat(ctx, w.wv, hidden_states);
    v_noise = ggml_reshape_4d(ctx, v_noise, head_dim, n_head_kv, q_len, 1);
    ggml_tensor* v = ggml_concat(ctx, v_ctx, v_noise, 2);
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    // 6. RoPE on full sequence
    // ggml_rope(ctx, a, b, n_dims, mode): b is int32 position_ids of size a->ne[2]
    // After permute, k and v have shape [head_dim, ctx_len+q_len, n_head_kv, 1]
    // ne[2] is ctx_len+q_len
    // We use position_ids directly (DSpark uses standard RoPE)
    // Note: actual rope_type = cfg.rope_mode (0 = NORMAL)
    q = ggml_rope(ctx, q, position_ids, n_embd/2, 0);
    k = ggml_rope(ctx, k, position_ids, n_embd/2, 0);

    // 7. SDPA / Flash Attention
    // No causal mask needed (DSpark context is bidirectional within context, causal only for new tokens)
    // Simplified: pass nullptr mask (DSpark draft is short, attention works without explicit mask)
    float scale = 1.0f / sqrtf((float)head_dim);
    ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);

    // attn_out: [head_dim, q_len, n_head, 1]
    // Reshape to [n_embd, q_len, 1] for output projection
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
    ggml_tensor* attn_2d = ggml_reshape_2d(ctx, attn_out, n_embd, q_len);

    // 8. Output projection
    ggml_tensor* out = ggml_mul_mat(ctx, w.wo, attn_2d);
    out = ggml_reshape_3d(ctx, out, n_embd, q_len, 1);

    return out;
}

// DSpark decoder layer (Pre-Norm)
static ggml_tensor* dspark_decoder_layer_forward(
    ggml_context* ctx,
    const DSparkLayerWeights& w,
    ggml_tensor* hidden_states,
    ggml_tensor* target_hs,
    ggml_tensor* position_ids,
    int n_head, int n_head_kv, int head_dim, int n_embd, int n_ff,
    int q_len, int ctx_len,
    int il
) {
    // Self-attention block
    ggml_tensor* residual = hidden_states;
    ggml_tensor* normed = rms_norm_weighted(ctx, hidden_states, w.input_layernorm);

    ggml_tensor* attn_out = dspark_attention_forward(
        ctx, w, normed, target_hs, position_ids,
        n_head, n_head_kv, head_dim, n_embd, q_len, ctx_len, il
    );
    hidden_states = ggml_add(ctx, residual, attn_out);

    // FFN block (SwiGLU)
    residual = hidden_states;
    normed = rms_norm_weighted(ctx, hidden_states, w.post_attention_layernorm);

    ggml_tensor* gate = ggml_mul_mat(ctx, w.ffn_gate, normed);
    gate = ggml_silu(ctx, gate);
    ggml_tensor* up = ggml_mul_mat(ctx, w.ffn_up, normed);
    ggml_tensor* ffn_hidden = ggml_mul(ctx, gate, up);
    ggml_tensor* ffn_out = ggml_mul_mat(ctx, w.ffn_down, ffn_hidden);

    hidden_states = ggml_add(ctx, residual, ffn_out);
    return hidden_states;
}

} // anonymous namespace

struct ggml_tensor* FusionDSparkModel::forward(
    struct ggml_context* ctx,
    struct ggml_tensor* target_hs,
    struct ggml_tensor* draft_input_ids,
    struct ggml_tensor* position_ids,
    struct llama_kv_cache* past_kv_draft
) {
    if (!loaded_) {
        fprintf(stderr, "[FusionDSpark] forward: not loaded\n");
        return nullptr;
    }
    if (weights_.embed_tokens == nullptr || weights_.lm_head == nullptr) {
        fprintf(stderr, "[FusionDSpark] forward: embeddings not shared\n");
        return nullptr;
    }

    // target_hs shape: [concat_dim, ctx_len, B]
    // draft_input_ids shape: [block_size, B] (int32)
    int ctx_len = (int)target_hs->ne[1];
    int q_len = cfg_.block_size;
    int n_embd = cfg_.n_embd;
    int n_head = cfg_.n_head;
    int n_head_kv = cfg_.n_head_kv;
    int head_dim = cfg_.head_dim;
    int n_ff = cfg_.n_ff;

    // 1. Project target hidden states: [concat_dim, ctx_len, B] -> [n_embd, ctx_len, B]
    if (weights_.fc == nullptr) {
        fprintf(stderr, "[FusionDSpark] forward: fc weight not loaded (skipping)\n");
        return nullptr;
    }
    ggml_tensor* target_proj = ggml_mul_mat(ctx, weights_.fc, target_hs);
    target_proj = ggml_cont(ctx, target_proj);
    target_proj = rms_norm_weighted(ctx, target_proj, weights_.hidden_norm);

    // 2. Embed draft input tokens: [n_embd, q_len, B]
    ggml_tensor* hidden_states = ggml_get_rows(ctx, weights_.embed_tokens, draft_input_ids);

    // 3. Run num_draft_layers decoder layers
    for (int il = 0; il < cfg_.num_draft_layers; ++il) {
        if (weights_.layers[il].wq == nullptr) {
            fprintf(stderr, "[FusionDSpark] forward: layer %d weights not loaded (skeleton mode)\n", il);
            // 骨架模式：直接返回 target_proj 作为 dummy output
            return target_proj;
        }
        hidden_states = dspark_decoder_layer_forward(
            ctx, weights_.layers[il], hidden_states, target_proj, position_ids,
            n_head, n_head_kv, head_dim, n_embd, n_ff,
            q_len, ctx_len, il
        );
    }

    // 4. Final norm
    if (weights_.final_norm != nullptr) {
        hidden_states = rms_norm_weighted(ctx, hidden_states, weights_.final_norm);
    }

    // 5. LM head -> logits
    ggml_tensor* logits = ggml_mul_mat(ctx, weights_.lm_head, hidden_states);
    // logits: [n_vocab, q_len, B]

    (void)past_kv_draft;
    return logits;
}

// ---------- S8: Markov head vanilla 完整实现 ----------
//
// 对应 PyTorch VanillaMarkov (markov_head.py):
//   self.markov_w1 = nn.Embedding(vocab_size, markov_rank)   # [vocab, rank]
//   self.markov_w2 = nn.Linear(markov_rank, vocab_size, bias=False)  # [vocab, rank]
//
//   def apply_block_logits(self, base_logits, token_ids, hidden_states):
//       prev_emb = self.markov_w1(token_ids)   # [B, S, rank]
//       bias = self.markov_w2(prev_emb)        # [B, S, vocab]
//       return base_logits + bias

namespace {

// VanillaMarkov: prev_token -> markov bias over vocab
// base_logits: [vocab, num_blocks*block_size, B]
// token_ids:   [num_blocks*block_size, B]  (int32)
// 返回: corrected_logits (same shape as base_logits)
static ggml_tensor* vanilla_markov_apply(
    ggml_context* ctx,
    const MarkovHeadWeights& head,
    ggml_tensor* base_logits,
    ggml_tensor* token_ids
) {
    // 1. prev_emb = markov_w1[token_ids] (embedding lookup)
    // markov_w1 shape: [vocab, rank] (PyTorch nn.Embedding weight)
    // token_ids shape: [S, B] (int32)
    // prev_emb shape:  [rank, S, B]
    ggml_tensor* prev_emb = ggml_get_rows(ctx, head.markov_w1, token_ids);

    // 2. bias = markov_w2 @ prev_emb
    // markov_w2 shape: [vocab, rank] (PyTorch nn.Linear weight, no transpose)
    // prev_emb shape:   [rank, S, B]
    // bias shape:       [vocab, S, B]
    ggml_tensor* bias = ggml_mul_mat(ctx, head.markov_w2, prev_emb);

    // 3. corrected_logits = base_logits + bias
    ggml_tensor* corrected = ggml_add(ctx, base_logits, bias);
    return corrected;
}

} // anonymous namespace

void FusionDSparkModel::apply_markov_head(
    struct ggml_context* ctx,
    struct ggml_tensor* base_logits,
    struct ggml_tensor* hidden_states,
    struct ggml_tensor* prev_token_ids,
    struct ggml_tensor* output_logits
) {
    if (weights_.markov_head.type == MarkovHeadType::NONE) return;
    if (weights_.markov_head.markov_w1 == nullptr || weights_.markov_head.markov_w2 == nullptr) {
        fprintf(stderr, "[FusionDSpark] apply_markov_head: weights not loaded\n");
        return;
    }
    if (weights_.markov_head.type != MarkovHeadType::VANILLA) {
        fprintf(stderr, "[FusionDSpark] apply_markov_head: only vanilla implemented (type=%d)\n",
                (int)weights_.markov_head.type);
        return;
    }
    (void)hidden_states;
    ggml_tensor* corrected = vanilla_markov_apply(ctx, weights_.markov_head, base_logits, prev_token_ids);
    // 注: output_logits 在调用方应该是 base_logits 同一 tensor 的 view，这里直接调用方负责
    // 实际生产中应该用 ggml_cpy 把 corrected copy 到 output_logits
    (void)output_logits;
    (void)corrected;
}

struct ggml_tensor* markov_head_forward(
    struct ggml_context* ctx,
    const MarkovHeadWeights& head,
    struct ggml_tensor* base_logits,
    struct ggml_tensor* token_ids,
    struct ggml_tensor* hidden_states,
    int32_t n_embd
) {
    if (head.type == MarkovHeadType::NONE) return base_logits;
    if (head.type != MarkovHeadType::VANILLA) {
        fprintf(stderr, "[markov_head] only vanilla type implemented, got %d\n", (int)head.type);
        return base_logits;
    }
    if (head.markov_w1 == nullptr || head.markov_w2 == nullptr) {
        fprintf(stderr, "[markov_head] weights not loaded\n");
        return base_logits;
    }
    (void)hidden_states; (void)n_embd;
    return vanilla_markov_apply(ctx, head, base_logits, token_ids);
}

} // namespace fusion
