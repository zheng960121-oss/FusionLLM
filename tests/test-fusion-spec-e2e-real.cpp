// FusionLLM Phase 6 S11: Real E2E spec decode driver
// Loads Qwen3-8B Q4_K_M (target) + DSpark Qwen3-4B (draft), runs spec decode,
// measures acceptance rate and tokens/sec vs autoregressive.
//
// Usage: test-fusion-spec-e2e-real <target_gguf> <draft_gguf> <prompt_tokens> <gen_tokens> [temperature]
//
// Example:
//   test-fusion-spec-e2e-real \
//     ~/Desktop/models/Qwen3-8B-Q4_K_M.gguf \
//     /tmp/dspark_qwen3_4b_real.gguf \
//     1024 30 0.0

#include "llama.h"
#include "fusion_speculative_decode.h"
#include "fusion_draft_model.h"
#include "fusion_hs_extract.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>
#include <string>

// Load token_embd + output from a target model GGUF for shared embeddings
struct target_embeds {
    ggml_context* ctx = nullptr;
    ggml_tensor* embed_tokens = nullptr;
    ggml_tensor* output = nullptr;
};

static bool load_target_embeddings(const char* path, target_embeds& out) {
    struct gguf_init_params iparams = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &out.ctx,
    };
    struct gguf_context* gctx = gguf_init_from_file(path, iparams);
    if (!gctx) { fprintf(stderr, "FAIL: open target GGUF\n"); return false; }
    out.embed_tokens = ggml_get_tensor(out.ctx, "token_embd.weight");
    out.output       = ggml_get_tensor(out.ctx, "output.weight");
    gguf_free(gctx);
    return (out.embed_tokens != nullptr);
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <target_gguf> <draft_gguf> <prompt_tokens> <gen_tokens> [temperature]\n", argv[0]);
        return 1;
    }
    const char* target_path = argv[1];
    const char* draft_path  = argv[2];
    int n_prompt = atoi(argv[3]);
    int n_gen    = atoi(argv[4]);
    float temperature = (argc > 5) ? (float)atof(argv[5]) : 0.0f;

    fprintf(stderr, "=== Phase 6 S11: Real E2E Spec Decode ===\n");
    fprintf(stderr, "  target: %s\n", target_path);
    fprintf(stderr, "  draft:  %s\n", draft_path);
    fprintf(stderr, "  prompt: %d tokens, gen: %d, temp: %.2f\n",
            n_prompt, n_gen, temperature);

    // ----------------------------------------------------------------
    // 1. Load target model + context
    // ----------------------------------------------------------------
    auto t0 = std::chrono::steady_clock::now();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model* model = llama_model_load_from_file(target_path, mp);
    if (!model) { fprintf(stderr, "FAIL: model load\n"); return 1; }

    int n_layers = llama_model_n_layer(model);
    int n_embd   = llama_model_n_embd(model);
    (void)n_embd;  // unused now (DSpark config drives hidden_dim)

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = n_prompt + n_gen + 256;
    cp.n_batch   = 512;
    cp.n_threads = 4;
    // Enable embeddings_layer_inp for hidden state extraction
    // (will be configured via FusionHSExtractor)
    llama_context* ctx = llama_new_context_with_model(model, cp);
    if (!ctx) { fprintf(stderr, "FAIL: ctx\n"); llama_model_free(model); return 1; }

    int n_vocab = (int)llama_vocab_n_tokens(llama_model_get_vocab(model));
    fprintf(stderr, "  loaded target: layers=%d embd=%d vocab=%d (%.0f ms)\n",
            n_layers, n_embd, n_vocab, ms_since(t0));

    // ----------------------------------------------------------------
    // 2. Load target embeddings (for shared embeddings with draft)
    // ----------------------------------------------------------------
    target_embeds target;
    if (!load_target_embeddings(target_path, target)) {
        fprintf(stderr, "FAIL: load target embeddings\n");
        llama_free(ctx); llama_model_free(model); return 1;
    }

    // ----------------------------------------------------------------
    // 3. Load draft model (DSpark Qwen3-4B)
    // ----------------------------------------------------------------
    auto t_d0 = std::chrono::steady_clock::now();
    fusion::FusionDSparkModel draft;
    if (!draft.load_from_gguf(draft_path)) {
        fprintf(stderr, "FAIL: draft load\n");
        return 1;
    }
    draft.share_target_embeddings(target.embed_tokens, target.output);
    fprintf(stderr, "  loaded draft: block_size=%d n_embd=%d concat_dim=%d (%.0f ms)\n",
            draft.config().block_size, draft.config().n_embd,
            draft.config().concat_dim(), ms_since(t_d0));

    // ----------------------------------------------------------------
    // 4. Configure hidden state extractor
    // ----------------------------------------------------------------
    fusion::FusionHSExtractor hs_ext;
    fusion::HSExtractConfig hs_cfg;
    // Use DSpark's expected layer IDs but clamp to model's n_layer.
    // DSpark was trained on a smaller model; when target has fewer layers,
    // we use the LAST few layers instead of DSpark's defaults.
    std::vector<int32_t> cfg_layers = draft.config().target_layer_ids;
    std::vector<int32_t> safe_layers;
    for (int32_t lid : cfg_layers) {
        if (lid < n_layers) safe_layers.push_back(lid);
    }
    if (safe_layers.empty()) {
        // Fallback: use last 5 layers
        int32_t start = std::max(0, n_layers - 5);
        for (int32_t i = start; i < n_layers; ++i) safe_layers.push_back(i);
    }
    hs_cfg.target_layer_ids = safe_layers;
    hs_cfg.hidden_dim = draft.config().n_embd;
    if (!hs_ext.configure(ctx, hs_cfg)) {
        fprintf(stderr, "FAIL: hs_ext configure\n");
        return 1;
    }
    fprintf(stderr, "  hs_ext: %zu layers (clamped from DSpark's %zu), hidden_dim=%d\n",
            hs_cfg.target_layer_ids.size(), cfg_layers.size(), hs_cfg.hidden_dim);

    // ----------------------------------------------------------------
    // 5. Initialize spec decoder
    // ----------------------------------------------------------------
    fusion::FusionSpecDecoder decoder;
    // For wire test: set skip_spec to AR path (draft not used even if loaded)
    int skip_spec = (argc > 6) ? atoi(argv[6]) : 0;
    if (skip_spec) {
        fprintf(stderr, "  skip_spec=1 (AR path only)\n");
        decoder.init(ctx, nullptr, nullptr);
    } else {
        if (!decoder.init(ctx, &draft, &hs_ext)) {
            fprintf(stderr, "FAIL: decoder init\n");
            return 1;
        }
    }

    // ----------------------------------------------------------------
    // 6. Build prompt
    // ----------------------------------------------------------------
    std::mt19937 rng(42);
    std::vector<int32_t> input_ids(n_prompt);
    for (int i = 0; i < n_prompt; ++i) {
        input_ids[i] = rng() % n_vocab;
    }

    // ----------------------------------------------------------------
    // 7. Run spec decode generate
    // ----------------------------------------------------------------
    fprintf(stderr, "\n--- Running spec decode generate ---\n");
    std::vector<int32_t> output_ids;
    auto t_g0 = std::chrono::steady_clock::now();
    int rc = decoder.generate(input_ids, output_ids, n_gen, temperature, {});
    double gen_ms = ms_since(t_g0);
    fprintf(stderr, "  spec decode: rc=%d output_size=%zu (%.0f ms, %.2f t/s)\n",
            rc, output_ids.size(), gen_ms,
            (output_ids.size() - input_ids.size()) / (gen_ms / 1000.0));

    // ----------------------------------------------------------------
    // 8. Compare autoregressive (run again with no draft)
    // ----------------------------------------------------------------
    fprintf(stderr, "\n--- Running AR baseline ---\n");
    // Reset context
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_clear(mem, true);

    fusion::FusionSpecDecoder ar_decoder;
    // init without draft → falls back to AR
    ar_decoder.init(ctx, nullptr, nullptr);

    std::vector<int32_t> ar_output;
    auto t_ar0 = std::chrono::steady_clock::now();
    int rc_ar = ar_decoder.generate(input_ids, ar_output, n_gen, temperature, {});
    double ar_ms = ms_since(t_ar0);
    fprintf(stderr, "  AR baseline: rc=%d output_size=%zu (%.0f ms, %.2f t/s)\n",
            rc_ar, ar_output.size(), ar_ms,
            (ar_output.size() - input_ids.size()) / (ar_ms / 1000.0));

    // ----------------------------------------------------------------
    // 9. Compare
    // ----------------------------------------------------------------
    fprintf(stderr, "\n=== Summary ===\n");
    fprintf(stderr, "  spec decode: %d new tokens in %.0f ms (%.2f t/s)\n",
            (int)(output_ids.size() - input_ids.size()), gen_ms,
            (output_ids.size() - input_ids.size()) / (gen_ms / 1000.0));
    fprintf(stderr, "  AR baseline: %d new tokens in %.0f ms (%.2f t/s)\n",
            (int)(ar_output.size() - input_ids.size()), ar_ms,
            (ar_output.size() - input_ids.size()) / (ar_ms / 1000.0));
    if (ar_ms > 0) {
        fprintf(stderr, "  speedup: %.2fx\n", gen_ms > 0 ? ar_ms / gen_ms : 1.0);
    }

    // Cleanup
    llama_free(ctx);
    llama_model_free(model);
    if (target.ctx) ggml_free(target.ctx);
    return 0;
}