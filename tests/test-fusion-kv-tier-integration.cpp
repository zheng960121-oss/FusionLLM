// FusionLLM W1 D4 integration test: real llama_context + FusionKVTierManager
//
// Loads Qwen 0.5B Q4, creates a llama_context, attaches a FusionKVTierManager,
// runs llama_decode once, and verifies that the hook fires (n_gpu_copies > 0).
//
// This is the end-to-end test that the llama.cpp integration code path
// actually invokes our tier manager.  Without this, we'd only be testing
// the data path in isolation, not the wiring.

#include "llama.h"
#include "fusion_kv_tier.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static int g_passed = 0;
static int g_failed = 0;
#define EXPECT(cond, msg) do { \
    if (cond) { g_passed++; printf("  PASS: %s\n", msg); } \
    else      { g_failed++; fprintf(stderr, "  FAIL: %s (line %d)\n", msg, __LINE__); } \
} while (0)

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* model_path = (argc > 1) ? argv[1]
        : "/Users/jk/Desktop/llama.cpp-fusionllm/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    printf("=== W1 D4 Integration: llama_context + FusionKVTierManager ===\n");
    printf("    model: %s\n", model_path);

    // 1. Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;  // offload everything to GPU
    llama_model* model = llama_model_load_from_file(model_path, model_params);
    EXPECT(model != nullptr, "model loaded");
    if (!model) return 1;

    // 2. Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;  // small context for quick test
    ctx_params.n_threads = 4;
    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    EXPECT(ctx != nullptr, "context created");
    if (!ctx) { llama_model_free(model); return 1; }

    // 3. Create tier manager.
    //    Match the model's actual KV layout so ensure_for_attention does
    //    meaningful work.  We use the model's reported n_embd_head_k + n_head_kv
    //    as a proxy for head_dim, and n_layer as n_layers.
    int n_layers = llama_model_n_layer(model);
    int n_embd   = llama_model_n_embd(model);
    int n_head   = llama_model_n_head(model);
    int n_head_kv = llama_model_n_head_kv(model);
    int head_dim = n_embd / n_head;  // standard Qwen layout
    int kv_size = 512;  // matches ctx_params.n_ctx
    fprintf(stderr, "[Integration] model: n_layers=%d n_embd=%d n_head=%d n_head_kv=%d head_dim=%d\n",
            n_layers, n_embd, n_head, n_head_kv, head_dim);

    // The tier manager needs a ggml_context for L0 allocation.  Create a
    // simple one (large enough for the L0 staging buffers).
    size_t ctx_size = 256 * 1024 * 1024;  // 256 MB
    struct ggml_init_params iparams = { ctx_size, nullptr, false /* no_alloc=false */ };
    ggml_context* ggml_ctx = ggml_init(iparams);
    EXPECT(ggml_ctx != nullptr, "ggml_init OK");
    if (!ggml_ctx) { llama_free(ctx); llama_model_free(model); return 1; }

    // 4. Construct the manager
    fusion::FusionKVTierManager mgr(ggml_ctx, n_layers, kv_size, head_dim,
                                     GGML_TYPE_F16, /*block_size=*/128);

    // 5. Set SSD path (use /tmp) so promote can succeed end-to-end
    mgr.set_ssd_path("/tmp/fusion_kv_tier_integration");
    fprintf(stderr, "[Integration] manager constructed; layers=%d kv_size=%d head_dim=%d\n",
            n_layers, kv_size, head_dim);

    // 6. Attach
    EXPECT(fusion_kv_tier_is_attached(ctx) == 0, "not attached before attach");
    fusion_kv_tier_attach(ctx, reinterpret_cast<fusion_kv_tier_t*>(&mgr));
    EXPECT(fusion_kv_tier_is_attached(ctx) == 1, "attached after fusion_kv_tier_attach");

    auto s_before = mgr.get_stats();
    fprintf(stderr, "[Integration] stats before decode: gpu_copies=%zu\n", s_before.n_gpu_copies);

    // 7. Run llama_decode with a single token
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 1;
    batch.token[0] = 1;  // arbitrary token
    batch.pos[0]   = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    int rc = llama_decode(ctx, batch);
    EXPECT(rc == 0, "llama_decode returned 0");
    llama_batch_free(batch);

    auto s_after = mgr.get_stats();
    fprintf(stderr, "[Integration] stats after decode:  gpu_copies=%zu promotions=%zu ssd_reads=%zu\n",
            s_after.n_gpu_copies, s_after.n_promotions, s_after.n_ssd_reads);

    // The hook should have fired for at least one layer, causing n_gpu_copies
    // to increase.  With W1 D4 calling ensure_for_attention(0, 0, kv_size) for
    // each layer, we'd expect n_gpu_copies to grow by n_layers (one per layer
    // for the initial L2->L0 promotion).  Be conservative: just check > 0.
    EXPECT(s_after.n_gpu_copies > s_before.n_gpu_copies,
           "hook fired (n_gpu_copies increased)");

    // 8. Detach
    fusion_kv_tier_attach(ctx, nullptr);
    EXPECT(fusion_kv_tier_is_attached(ctx) == 0, "detached");

    // 9. Cleanup
    llama_free(ctx);
    llama_model_free(model);
    ggml_free(ggml_ctx);
    // Don't delete SSD dir — useful for debugging
    fprintf(stderr, "[Integration] SSD files left at /tmp/fusion_kv_tier_integration/ for inspection\n");

    printf("\n=== Summary: %d passed, %d failed ===\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}
