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

// ---------- Helpers ----------

// Softmax: in-place probs[i] = exp(logits[i]) / sum
static void softmax_inplace(std::vector<float>& probs, float temperature = 1.0f) {
    if (temperature <= 0.0f) {
        // greedy: 1.0 at argmax, 0 elsewhere
        size_t max_idx = 0;
        float max_v = probs[0];
        for (size_t i = 1; i < probs.size(); ++i) {
            if (probs[i] > max_v) { max_v = probs[i]; max_idx = i; }
        }
        std::fill(probs.begin(), probs.end(), 0.0f);
        probs[max_idx] = 1.0f;
        return;
    }
    float max_l = *std::max_element(probs.begin(), probs.end());
    float sum = 0.0f;
    for (auto& v : probs) {
        v = std::exp((v - max_l) / temperature);
        sum += v;
    }
    float inv_sum = 1.0f / std::max(sum, 1e-30f);
    for (auto& v : probs) v *= inv_sum;
}

// Argmax sample
static int32_t sample_argmax(const float* logits, int32_t vocab_size) {
    int32_t max_idx = 0;
    float max_v = logits[0];
    for (int32_t v = 1; v < vocab_size; ++v) {
        if (logits[v] > max_v) { max_v = logits[v]; max_idx = v; }
    }
    return max_idx;
}

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
            float inv_sum = 1.0f / std::max(sum, 1e-30f);
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
    if (!target_ctx) {
        fprintf(stderr, "[FusionSpec] init: null target context\n");
        return false;
    }
    // Allow null draft/hs_extractor for AR fallback path
    target_ctx_ = target_ctx;
    draft_model_ = draft_model;
    hs_extractor_ = hs_extractor;
    stats_ = SpecDecodeStats{};
    initialized_ = true;
    fprintf(stderr, "[FusionSpec] initialized: target_ctx=%p draft=%p hs_extract=%p (spec_decode=%s)\n",
            (void*)target_ctx, (void*)draft_model, (void*)hs_extractor,
            (draft_model && draft_model->is_loaded()) ? "ON" : "OFF");
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

    fprintf(stderr, "[FusionSpec] generate: input=%zu tokens, max_new=%d, temp=%.3f, spec_decode=%s\n",
            input_ids.size(), max_new_tokens, temperature,
            draft_model_ && draft_model_->is_loaded() ? "ON" : "OFF");

    // 1) Prefill target model with input_ids (chunked to avoid OOM)
    int chunk_size = 512;
    llama_kv_cache* target_kv = nullptr;  // not used directly; ctx owns KV
    for (int32_t c = 0; c * chunk_size < (int)input_ids.size(); ++c) {
        int32_t s_start = c * chunk_size;
        int32_t s_end   = std::min(s_start + chunk_size, (int)input_ids.size());
        int32_t n_tok   = s_end - s_start;
        llama_batch batch = llama_batch_init(n_tok, 0, 1);
        batch.n_tokens = n_tok;
        for (int32_t i = 0; i < n_tok; ++i) {
            batch.token[i]    = input_ids[s_start + i];
            batch.pos[i]      = s_start + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]   = (s_start + i == (int)input_ids.size() - 1);
        }
        int rc = llama_decode(target_ctx_, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "[FusionSpec] generate: prefill chunk %d failed (rc=%d)\n", c, rc);
            return -1;
        }
    }
    fprintf(stderr, "[FusionSpec] prefill done: %zu tokens\n", input_ids.size());

    // 2) Generate loop: prefer spec_decode when draft_model loaded
    int total_calls = 0;
    while (start < max_length && total_calls < 200) {  // safety cap
        total_calls++;
        SpecDecodeStep step;
        if (draft_model_ && draft_model_->is_loaded() && hs_extractor_) {
            step = step_spec(output_ids, start, temperature, stop_token_ids);
        } else {
            step = step_autoregressive(output_ids, start, temperature, stop_token_ids);
        }
        if (step.terminated) break;
        stats_.total_output_tokens++;
    }

    stats_.acceptance_length = (double)stats_.total_output_tokens /
                               std::max(stats_.total_verify_calls, 1);

    fprintf(stderr, "[FusionSpec] generate done: %d output tokens, %d verify calls, "
            "acceptance=%.2f, proposed=%d, accepted=%d\n",
            stats_.total_output_tokens, stats_.total_verify_calls,
            stats_.acceptance_length, stats_.total_draft_proposed, stats_.total_draft_accepted);

    return (int)stats_.total_output_tokens;
}

// ===========================================================================
// step_autoregressive (real impl): single-token target decode + sample
// ===========================================================================
SpecDecodeStep FusionSpecDecoder::step_autoregressive(
    std::vector<int32_t>& output_ids,
    int32_t& start,
    float temperature,
    const std::vector<int32_t>& stop_token_ids,
    struct llama_kv_cache* target_kv /*unused - ctx owns KV*/
) {
    SpecDecodeStep result;
    stats_.total_verify_calls++;

    if (!target_ctx_) {
        fprintf(stderr, "[FusionSpec] step_ar: no target ctx\n");
        return result;
    }

    // 1) Last token
    int32_t last_token = (start > 0 && start <= (int)output_ids.size())
                          ? output_ids[start - 1] : 0;

    // 2) Build batch = [last_token], logits=true
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 1;
    batch.token[0]    = last_token;
    batch.pos[0]      = start;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0]   = true;

    // 3) llama_decode
    int rc = llama_decode(target_ctx_, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        fprintf(stderr, "[FusionSpec] step_ar: llama_decode failed rc=%d\n", rc);
        result.terminated = true;
        return result;
    }

    // 4) Sample from logits
    float* logits = llama_get_logits_ith(target_ctx_, 0);
    int32_t vocab_size = (int32_t)llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(target_ctx_)));
    int32_t sampled = (temperature <= 0.0f)
                      ? sample_argmax(logits, vocab_size)
                      : [&]() {
                            std::vector<float> probs(logits, logits + vocab_size);
                            softmax_inplace(probs, temperature);
                            std::mt19937 rng(42 + start);  // deterministic per-step
                            std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
                            return dist(rng);
                        }();

    // 5) Check stop tokens
    if (!stop_token_ids.empty()) {
        for (int32_t s : stop_token_ids) {
            if (sampled == s) {
                result.terminated = true;
                result.accepted_count = 0;
                result.bonus_token = sampled;
                result.effective_length = 0;
                return result;
            }
        }
    }

    // 6) Commit
    output_ids.push_back(sampled);
    start++;
    result.accepted_count = 1;
    result.bonus_token = sampled;
    result.effective_length = 1;
    return result;
}

// ===========================================================================
// step_spec (real impl): draft + target verify + rejection_sample
// ===========================================================================
SpecDecodeStep FusionSpecDecoder::step_spec(
    std::vector<int32_t>& output_ids,
    int32_t& start,
    float temperature,
    const std::vector<int32_t>& stop_token_ids,
    struct llama_kv_cache* target_kv /*unused*/,
    struct llama_kv_cache* draft_kv /*unused*/
) {
    SpecDecodeStep result;
    stats_.total_verify_calls++;

    if (!target_ctx_ || !draft_model_ || !hs_extractor_) {
        fprintf(stderr, "[FusionSpec] step_spec: missing target/draft/hs\n");
        return result;
    }

    int32_t block_size = draft_model_->config().block_size;
    if (block_size <= 0) block_size = 7;  // DSpark default

    int32_t n_vocab = (int32_t)llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(target_ctx_)));

    // -----------------------------------------------------------------
    // Step 1: extract target hidden states (from previous target forward)
    // -----------------------------------------------------------------
    HSBuffer hs_buf;
    try {
        hs_buf = hs_extractor_->extract(target_ctx_);
    } catch (...) {
        fprintf(stderr, "[FusionSpec] step_spec: extract hidden states failed\n");
        return result;
    }
    int32_t ctx_len = (int32_t)hs_buf.data.size() / (int32_t)hs_buf.concat_dim;
    if (ctx_len <= 0) ctx_len = 1;

    // -----------------------------------------------------------------
    // Step 2: build draft input (last block_size target tokens)
    // -----------------------------------------------------------------
    int32_t draft_start = std::max(0, start - block_size);
    std::vector<int32_t> draft_input_ids(block_size, 0);
    for (int32_t i = 0; i < block_size; ++i) {
        int32_t src_idx = draft_start + i;
        if (src_idx < start && src_idx < (int32_t)output_ids.size()) {
            draft_input_ids[i] = output_ids[src_idx];
        } else {
            draft_input_ids[i] = 0;  // pad
        }
    }

    // -----------------------------------------------------------------
    // Step 3: draft forward → draft_logits
    // -----------------------------------------------------------------
    // Allocate ggml context for draft forward graph
    size_t draft_ctx_size = 256 * 1024 * 1024;  // 256 MB
    ggml_context* draft_ctx = ggml_init({draft_ctx_size, nullptr, false});
    if (!draft_ctx) {
        fprintf(stderr, "[FusionSpec] step_spec: ggml_init failed\n");
        return result;
    }

    int32_t concat_dim = (int32_t)draft_model_->config().concat_dim();
    ggml_tensor* target_hs = ggml_new_tensor_3d(draft_ctx, GGML_TYPE_F32,
                                                  concat_dim, ctx_len, 1);
    ggml_tensor* draft_inp = ggml_new_tensor_2d(draft_ctx, GGML_TYPE_I32,
                                                  block_size, 1);
    ggml_tensor* pos_ids   = ggml_new_tensor_2d(draft_ctx, GGML_TYPE_I32,
                                                  ctx_len + block_size, 1);

    // Fill target_hs from hs_buf (concat_dim × ctx_len)
    if (target_hs && hs_buf.data.size() >= (size_t)concat_dim * ctx_len) {
        std::memcpy(target_hs->data, hs_buf.data.data(),
                    concat_dim * ctx_len * sizeof(float));
    }
    // Fill draft_inp
    if (draft_inp) {
        std::memcpy(draft_inp->data, draft_input_ids.data(),
                    block_size * sizeof(int32_t));
    }
    // Fill pos_ids
    if (pos_ids) {
        int32_t* p = (int32_t*)pos_ids->data;
        for (int32_t i = 0; i < ctx_len + block_size; ++i) p[i] = i;
    }

    // Call draft forward
    ggml_tensor* draft_logits = draft_model_->forward(draft_ctx, target_hs,
                                                       draft_inp, pos_ids, nullptr);
    if (!draft_logits) {
        fprintf(stderr, "[FusionSpec] step_spec: draft forward failed\n");
        ggml_free(draft_ctx);
        return result;
    }

    // Compute draft graph
    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_backend) {
        fprintf(stderr, "[FusionSpec] step_spec: no CPU backend\n");
        ggml_free(draft_ctx);
        return result;
    }

    // Copy draft_input_ids tensor data into a safe buffer (gallocr reuse trap)
    std::vector<int32_t> draft_inp_copy = draft_input_ids;
    std::vector<int32_t> pos_ids_copy(ctx_len + block_size);
    for (int32_t i = 0; i < ctx_len + block_size; ++i) pos_ids_copy[i] = i;
    std::vector<float> hs_copy;
    if (target_hs) {
        hs_copy.assign((float*)target_hs->data,
                       (float*)target_hs->data + concat_dim * ctx_len);
    }

    ggml_cgraph* gf = ggml_new_graph(draft_ctx);
    ggml_build_forward_expand(gf, draft_logits);
    ggml_gallocr_t allo = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!ggml_gallocr_alloc_graph(allo, gf)) {
        fprintf(stderr, "[FusionSpec] step_spec: alloc_graph failed\n");
        ggml_gallocr_free(allo);
        ggml_free(draft_ctx);
        ggml_backend_free(cpu_backend);
        return result;
    }

    // Restore input data (gallocr may have moved tensors)
    if (target_hs) std::memcpy(target_hs->data, hs_copy.data(), hs_copy.size() * sizeof(float));
    if (draft_inp) std::memcpy(draft_inp->data, draft_inp_copy.data(), draft_inp_copy.size() * sizeof(int32_t));
    if (pos_ids)   std::memcpy(pos_ids->data, pos_ids_copy.data(), pos_ids_copy.size() * sizeof(int32_t));

    ggml_status cs = ggml_backend_graph_compute(cpu_backend, gf);
    if (cs != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[FusionSpec] step_spec: draft compute failed (%d)\n", cs);
        ggml_gallocr_free(allo);
        ggml_free(draft_ctx);
        ggml_backend_free(cpu_backend);
        return result;
    }

    // -----------------------------------------------------------------
    // Step 4: sample draft tokens from draft_logits
    // -----------------------------------------------------------------
    // draft_logits shape: [vocab_size, block_size, 1]
    std::vector<int32_t> draft_tokens(block_size);
    {
        float* logits_ptr = (float*)draft_logits->data;
        for (int32_t i = 0; i < block_size; ++i) {
            const float* row = logits_ptr + i * n_vocab;
            draft_tokens[i] = (temperature <= 0.0f)
                              ? sample_argmax(row, n_vocab)
                              : [&]() {
                                    std::vector<float> probs(row, row + n_vocab);
                                    softmax_inplace(probs, temperature);
                                    std::mt19937 rng(7 + start * 17 + i);
                                    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
                                    return dist(rng);
                                }();
        }
    }

    // -----------------------------------------------------------------
    // Step 5: target verify forward
    //   Build batch = [last_target_token] + draft_tokens (block_size+1 tokens)
    //   all positions have logits=true for verify
    // -----------------------------------------------------------------
    int32_t last_token = output_ids[start - 1];
    int32_t verify_n   = block_size + 1;
    llama_batch vbatch = llama_batch_init(verify_n, 0, 1);
    vbatch.n_tokens = verify_n;
    vbatch.token[0]    = last_token;
    vbatch.pos[0]      = start;
    vbatch.n_seq_id[0] = 1;
    vbatch.seq_id[0][0] = 0;
    vbatch.logits[0]   = true;  // will be used for bonus if all accepted
    for (int32_t i = 0; i < block_size; ++i) {
        vbatch.token[i + 1]    = draft_tokens[i];
        vbatch.pos[i + 1]      = start + 1 + i;
        vbatch.n_seq_id[i + 1] = 1;
        vbatch.seq_id[i + 1][0] = 0;
        vbatch.logits[i + 1]   = true;
    }

    // S5: spec verify 触发 FusionLLM sliding window 统计
    window_on_verify_step_begin();

    int rc = llama_decode(target_ctx_, vbatch);
    llama_batch_free(vbatch);

    window_on_verify_step_end();
    int advances_per_verify = window_get_last_step_advance_count();

    if (rc != 0) {
        fprintf(stderr, "[FusionSpec] step_spec: target verify failed rc=%d\n", rc);
        ggml_gallocr_free(allo);
        ggml_free(draft_ctx);
        ggml_backend_free(cpu_backend);
        return result;
    }

    // -----------------------------------------------------------------
    // Step 6: extract target_probs from verify logits
    // -----------------------------------------------------------------
    std::vector<float> target_probs(verify_n * n_vocab);
    for (int32_t i = 0; i < verify_n; ++i) {
        float* row = llama_get_logits_ith(target_ctx_, i);
        std::memcpy(target_probs.data() + i * n_vocab, row,
                    n_vocab * sizeof(float));
        // Apply softmax to get probs (for i < block_size, used in rejection_sample;
        // for i == block_size, used for bonus)
        std::vector<float> probs(row, row + n_vocab);
        softmax_inplace(probs, temperature);
        std::memcpy(target_probs.data() + i * n_vocab, probs.data(),
                    n_vocab * sizeof(float));
    }

    // -----------------------------------------------------------------
    // Step 7: extract draft_probs from draft_logits (softmax)
    // -----------------------------------------------------------------
    std::vector<float> draft_probs(block_size * n_vocab);
    {
        float* logits_ptr = (float*)draft_logits->data;
        for (int32_t i = 0; i < block_size; ++i) {
            std::vector<float> probs(logits_ptr + i * n_vocab,
                                     logits_ptr + (i + 1) * n_vocab);
            softmax_inplace(probs, temperature);
            std::memcpy(draft_probs.data() + i * n_vocab, probs.data(),
                        n_vocab * sizeof(float));
        }
    }

    // -----------------------------------------------------------------
    // Step 8: rejection sampling
    // -----------------------------------------------------------------
    std::mt19937 rng(42 + start * 1009);  // deterministic per-spec-step
    auto [accepted_count, bonus_token] = rejection_sample(
        target_probs, draft_probs, draft_tokens,
        block_size, n_vocab, temperature, rng);

    // -----------------------------------------------------------------
    // Step 9: commit accepted tokens + bonus
    // -----------------------------------------------------------------
    // Crop KV cache: remove positions [start + 1 + accepted_count, start + 1 + block_size)
    //   (those were the rejected draft positions)
    if (accepted_count < block_size) {
        llama_memory_t mem = llama_get_memory(target_ctx_);
        int32_t p_keep_lo = start;  // keep up to (but not including) start (the last committed position)
        int32_t p_rm_lo   = start + 1 + accepted_count;  // first rejected position
        // llama_memory_seq_rm signature: (mem, seq_id, p0, p1) — removes [p0, p1)
        // We need to keep [0, p_rm_lo) and remove [p_rm_lo, +inf)
        // Use seq_keep or seq_rm carefully:
        llama_memory_seq_rm(mem, 0, p_rm_lo, -1);  // remove rejected positions
    }

    // Append accepted draft tokens + bonus to output_ids
    for (int32_t i = 0; i < accepted_count; ++i) {
        output_ids.push_back(draft_tokens[i]);
    }
    output_ids.push_back(bonus_token);

    start += accepted_count + 1;  // accepted_count draft tokens + 1 bonus

    // -----------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------
    ggml_gallocr_free(allo);
    ggml_free(draft_ctx);
    ggml_backend_free(cpu_backend);

    // Stats
    stats_.total_draft_proposed += block_size;
    stats_.total_draft_accepted += accepted_count;

    result.accepted_count = accepted_count;
    result.bonus_token = bonus_token;
    result.effective_length = accepted_count + 1;

    if ((int)stats_.total_verify_calls % 5 == 1 || accepted_count == 0) {
        fprintf(stderr, "[FusionSpec] step: start=%d accepted=%d bonus=%d window_adv=%d\n",
                start - accepted_count - 1, accepted_count, bonus_token, advances_per_verify);
    }

    return result;
}

} // namespace fusion