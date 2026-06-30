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
    fprintf(stderr, "[FusionDSpark] load_weights: loading %d draft layers + heads\n",
            cfg_.num_draft_layers);
    weights_.layers.resize(cfg_.num_draft_layers);

    if (cfg_.markov_rank > 0) {
        if (cfg_.markov_head_type == "vanilla") weights_.markov_head.type = MarkovHeadType::VANILLA;
        else if (cfg_.markov_head_type == "gated") weights_.markov_head.type = MarkovHeadType::GATED;
        else if (cfg_.markov_head_type == "rnn") weights_.markov_head.type = MarkovHeadType::RNN;
        weights_.markov_head.rank = cfg_.markov_rank;
    }

    if (cfg_.enable_confidence_head) {
        // Note: config has enable_confidence_head but no with_markov flag in struct
        // Markov coupling is handled inside apply_markov_head caller
    }

    if (!ctx) {
        fprintf(stderr, "[FusionDSpark] load_weights: no ggml ctx provided\n");
        return false;
    }

    // Lookup helper: get a tensor from ggml_context by name. Returns nullptr
    // if the tensor doesn't exist in the GGUF (e.g. optional tensors).
    auto get = [&](const char* name) -> ggml_tensor* {
        return ggml_get_tensor((ggml_context*) ctx, name);
    };

    int n_found = 0, n_missing = 0;

    // 1) Embedding + LM head (note: usually shared with target, can be nullptr)
    weights_.embed_tokens = get("embed_tokens.weight");
    if (weights_.embed_tokens) n_found++; else n_missing++;

    weights_.lm_head = get("lm_head.weight");
    if (weights_.lm_head) n_found++; else n_missing++;

    // 2) FC + hidden_norm + final_norm
    weights_.fc = get("fc.weight");
    if (weights_.fc) n_found++; else n_missing++;

    weights_.hidden_norm = get("hidden_norm.weight");
    if (weights_.hidden_norm) n_found++; else n_missing++;

    weights_.final_norm = get("norm.weight");
    if (weights_.final_norm) n_found++; else n_missing++;

    // 3) Per-layer weights
    for (int il = 0; il < cfg_.num_draft_layers; ++il) {
        DSparkLayerWeights& w = weights_.layers[il];
        char name[256];

        // Attention projections + norms
        snprintf(name, sizeof(name), "layers.%d.attn_q.weight", il);
        w.wq = get(name);
        snprintf(name, sizeof(name), "layers.%d.attn_k.weight", il);
        w.wk = get(name);
        snprintf(name, sizeof(name), "layers.%d.attn_v.weight", il);
        w.wv = get(name);
        snprintf(name, sizeof(name), "layers.%d.attn_output.weight", il);
        w.wo = get(name);
        snprintf(name, sizeof(name), "layers.%d.attn_q_norm.weight", il);
        w.attn_q_norm = get(name);
        snprintf(name, sizeof(name), "layers.%d.attn_k_norm.weight", il);
        w.attn_k_norm = get(name);

        // FFN
        snprintf(name, sizeof(name), "layers.%d.ffn_gate.weight", il);
        w.ffn_gate = get(name);
        snprintf(name, sizeof(name), "layers.%d.ffn_up.weight", il);
        w.ffn_up = get(name);
        snprintf(name, sizeof(name), "layers.%d.ffn_down.weight", il);
        w.ffn_down = get(name);

        // Norms
        snprintf(name, sizeof(name), "layers.%d.input_layernorm.weight", il);
        w.input_layernorm = get(name);
        snprintf(name, sizeof(name), "layers.%d.post_attention_layernorm.weight", il);
        w.post_attention_layernorm = get(name);

        if (w.wq && w.wk && w.wv && w.wo &&
            w.attn_q_norm && w.attn_k_norm &&
            w.ffn_gate && w.ffn_up && w.ffn_down &&
            w.input_layernorm && w.post_attention_layernorm) {
            n_found += 11;
        } else {
            fprintf(stderr, "[FusionDSpark] WARNING: layer %d has missing weights (running in skeleton mode)\n", il);
            n_missing += 11;
        }
    }

    // 4) Markov head
    if (cfg_.markov_rank > 0 && weights_.markov_head.type == MarkovHeadType::VANILLA) {
        weights_.markov_head.markov_w1 = get("markov_head.markov_w1.weight");
        weights_.markov_head.markov_w2 = get("markov_head.markov_w2.weight");
        if (weights_.markov_head.markov_w1) n_found++; else n_missing++;
        if (weights_.markov_head.markov_w2) n_found++; else n_missing++;
    }

    // 5) Confidence head
    if (cfg_.enable_confidence_head) {
        weights_.confidence_head.proj = get("confidence_head.proj.weight");
        if (weights_.confidence_head.proj) n_found++; else n_missing++;
    }

    fprintf(stderr, "[FusionDSpark] weights loaded: %d found, %d missing\n", n_found, n_missing);
    (void)gctx;  // kept for API compat; tensors live in weights_ctx_ now
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

    // Allocate a ggml context via GGUF init (will auto-load + register all tensors).
    // params.ctx = &weights_ctx_ means GGUF creates weights_ctx_ and stores all
    // tensor metadata + data pointers inside it.
    struct gguf_init_params iparams = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &weights_ctx_,
    };
    struct gguf_context* gctx = gguf_init_from_file(path.c_str(), iparams);
    if (!gctx) {
        fprintf(stderr, "[FusionDSpark] failed to open GGUF: %s\n", path.c_str());
        return false;
    }

    if (!parse_metadata(gctx)) {
        ggml_free(weights_ctx_); weights_ctx_ = nullptr;
        gguf_free(gctx);
        return false;
    }

    if (!load_weights(weights_ctx_, gctx)) {
        ggml_free(weights_ctx_); weights_ctx_ = nullptr;
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

    // ==== Q branch ====
    // 1. Q projection from draft input only: wq ne=[q_dim, n_embd], hidden_states ne=[n_embd, q_len, B]
    //    mul_mat yields ne = [q_dim, q_len*B] which we then reshape.
    ggml_tensor* q = ggml_mul_mat(ctx, w.wq, hidden_states);
    // Reshape to [head_dim, n_head, q_len, B] (ne[2] = q_len so ggml_rope matches)
    q = ggml_reshape_4d(ctx, q, head_dim, n_head, q_len, 1);

    // per-head RMSNorm on Q (apply on ne=[head_dim, n_head, q_len, B])
    q = rms_norm_weighted(ctx, q, w.attn_q_norm);
    q = ggml_cont(ctx, q);

    // RoPE BEFORE permute so that ne[2] = q_len matches position_ids view size.
    // Q is from draft only, so we take the LAST q_len positions of position_ids.
    ggml_tensor* pos_q = ggml_view_1d(ctx, position_ids, q_len, ctx_len * sizeof(int32_t));
    q = ggml_rope(ctx, q, pos_q, head_dim, 0);
    // Now permute to [head_dim, q_len, n_head, B] for SDPA.
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));

    // ==== K branch (target context + draft noise concat along seq dim) ====
    ggml_tensor* k_ctx = ggml_mul_mat(ctx, w.wk, target_hs);
    k_ctx = ggml_reshape_4d(ctx, k_ctx, head_dim, n_head_kv, ctx_len, 1);

    ggml_tensor* k_noise = ggml_mul_mat(ctx, w.wk, hidden_states);
    k_noise = ggml_reshape_4d(ctx, k_noise, head_dim, n_head_kv, q_len, 1);

    // Concat along sequence dim: ne = [head_dim, n_head_kv, ctx_len+q_len, B]
    ggml_tensor* k = ggml_concat(ctx, k_ctx, k_noise, 2);

    // per-head RMSNorm on K (apply on ne=[head_dim, n_head_kv, ctx_len+q_len, B])
    k = rms_norm_weighted(ctx, k, w.attn_k_norm);
    k = ggml_cont(ctx, k);

    // RoPE on full concat sequence; position_ids covers ctx_len+q_len entries.
    // K ne=[head_dim, n_head_kv, ctx_len+q_len, B], so we view position_ids
    // to length ctx_len+q_len (already same as total, so just use full).
    ggml_tensor* pos_kv = ggml_view_1d(ctx, position_ids, ctx_len + q_len, 0);
    k = ggml_rope(ctx, k, pos_kv, head_dim, 0);

    // Permute to [head_dim, ctx_len+q_len, n_head_kv, B] for SDPA.
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));

    // ==== V branch (same concat pattern, no RoPE/norm) ====
    ggml_tensor* v_ctx = ggml_mul_mat(ctx, w.wv, target_hs);
    v_ctx = ggml_reshape_4d(ctx, v_ctx, head_dim, n_head_kv, ctx_len, 1);
    ggml_tensor* v_noise = ggml_mul_mat(ctx, w.wv, hidden_states);
    v_noise = ggml_reshape_4d(ctx, v_noise, head_dim, n_head_kv, q_len, 1);
    ggml_tensor* v = ggml_concat(ctx, v_ctx, v_noise, 2);
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    // 7. SDPA / Flash Attention
    // No causal mask needed (DSpark context is bidirectional within context, causal only for new tokens)
    // Simplified: pass nullptr mask (DSpark draft is short, attention works without explicit mask)
    float scale = 1.0f / sqrtf((float)head_dim);
    ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);

    // attn_out: [head_dim, q_len, n_head, 1] (from ggml_flash_attn_ext output shape)
    // Permute to [head_dim, n_head, q_len, 1] for reshape to 2D
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
    // Reshape to [q_dim, q_len] where q_dim = n_head * head_dim (handles GQA where q_dim != n_embd)
    const int64_t q_dim = (int64_t) n_head * head_dim;
    ggml_tensor* attn_2d = ggml_reshape_2d(ctx, attn_out, q_dim, q_len);

    // 8. Output projection: wo has shape [n_embd, q_dim]
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

    // Skeleton / no-alloc guard: if the caller's ggml_context is in no_alloc mode
    // (e.g. metadata-only / shape-test path like test-fusion-dspart-attention),
    // we cannot actually run any ggml_mul_mat — placeholders have no data buffer
    // and ggml will abort on the first matmul.  Return nullptr so the test sees
    // the documented "skeleton mode" contract instead of crashing the process.
    if (ggml_get_no_alloc(ctx)) {
        fprintf(stderr, "[FusionDSpark] forward: ggml context is no_alloc (skeleton mode), returning nullptr\n");
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
