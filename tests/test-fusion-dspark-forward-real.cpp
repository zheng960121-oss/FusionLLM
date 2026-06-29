// FusionLLM S7+ Forward Integration Test (Real DSpark Qwen3-4B)
// Phase 6 S10-complete: load real weights, share target embeddings, run forward.
//
// 验证:
//   1. Real DSpark GGUF loads 60 tensors (load_weights 实装)
//   2. Target Qwen3-4B GGUF loads + token_embd/output 可拿到
//   3. share_target_embeddings 正确连接
//   4. forward() 不返回 nullptr (真跑 forward, 不是 skeleton)
//   5. forward() 输出 logits shape 正确: [vocab, block_size, 1]

#include "fusion_draft_model.h"

#include <cstdio>
#include <vector>
#include <random>
#include <cstring>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

static int passed = 0;
static int failed = 0;
#define EXPECT(cond, msg) do { \
    if (cond) { passed++; printf("  PASS: %s\n", msg); } \
    else { failed++; fprintf(stderr, "  FAIL: %s (line %d)\n", msg, __LINE__); } \
} while (0)


// Helper: load just token_embd + output (lm_head) from a target model GGUF.
// Returns true on success. Outputs are ggml_tensor pointers into a new ctx.
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
    if (!gctx) {
        fprintf(stderr, "FAIL: cannot open target GGUF %s\n", path);
        return false;
    }

    // Qwen3-4B GGUF tensor names (llama.cpp convention):
    //   token_embd.weight  (embedding)
    //   output.weight      (lm head) — may be tied (= token_embd.weight) for some models
    out.embed_tokens = ggml_get_tensor(out.ctx, "token_embd.weight");
    out.output       = ggml_get_tensor(out.ctx, "output.weight");

    // Fallback for tied embeddings (Qwen3-4B: tie_word_embeddings=false in HF config
    // but llama.cpp quantization may skip output if no separate weight).
    if (out.embed_tokens == nullptr) {
        fprintf(stderr, "FAIL: target GGUF missing token_embd.weight\n");
        gguf_free(gctx);
        return false;
    }
    if (out.output == nullptr) {
        fprintf(stderr, "[target] output.weight missing → using tied token_embd.weight\n");
        out.output = out.embed_tokens;  // tied embeddings
    }

    gguf_free(gctx);
    return true;
}


int main(int argc, char** argv) {
    const char* dspark_path = "/tmp/dspark_qwen3_4b_real.gguf";
    const char* target_path = "/tmp/qwen3_4b/Qwen3-4B-Q4_K_M.gguf";

    if (argc >= 2) dspark_path = argv[1];
    if (argc >= 3) target_path = argv[2];

    printf("=== Real DSpark Forward Integration Test ===\n");
    printf("  DSpark GGUF: %s\n", dspark_path);
    printf("  Target GGUF: %s\n", target_path);
    printf("\n");

    // ---------- 1. Load DSpark draft model (real weights) ----------
    fusion::FusionDSparkModel draft;
    if (!draft.load_from_gguf(dspark_path)) {
        fprintf(stderr, "FAIL: load DSpark GGUF\n");
        return 1;
    }
    EXPECT(draft.is_loaded(), "DSpark model loaded");
    EXPECT(draft.config().block_size == 7, "block_size = 7");
    EXPECT(draft.config().num_draft_layers == 5, "num_draft_layers = 5");
    EXPECT(draft.config().n_embd == 2560, "target.n_embd = 2560");
    EXPECT(draft.config().n_vocab == 151936, "vocab_size = 151936");
    EXPECT(draft.config().concat_dim() == 5 * 2560, "concat_dim = 12800");

    // ---------- 2. Load target embeddings ----------
    target_embeds target;
    if (!load_target_embeddings(target_path, target)) {
        fprintf(stderr, "FAIL: load target embeddings\n");
        return 1;
    }
    EXPECT(target.embed_tokens != nullptr, "target.embed_tokens loaded");
    EXPECT(target.output != nullptr, "target.output (lm_head) loaded");
    EXPECT(target.embed_tokens->ne[0] == 2560, "target embed n_embd = 2560");
    EXPECT(target.embed_tokens->ne[1] == 151936, "target embed vocab = 151936");

    // ---------- 3. Share target embeddings with draft ----------
    draft.share_target_embeddings(target.embed_tokens, target.output);
    // We can't directly check draft.weights() but we can verify forward doesn't bail
    // on missing embeddings anymore.

    // ---------- 4. Prepare input tensors ----------
    size_t ctx_size = 256 * 1024 * 1024;  // 256 MB
    struct ggml_init_params iparams = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context* ctx = ggml_init(iparams);
    EXPECT(ctx != nullptr, "ggml_init OK (256 MB)");

    int ctx_len = 8;
    int block_size = draft.config().block_size;
    int concat_dim = draft.config().concat_dim();
    int n_embd = draft.config().n_embd;

    // target_hs: [concat_dim, ctx_len, 1]
    struct ggml_tensor* target_hs = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, concat_dim, ctx_len, 1);
    // draft_input_ids: [block_size, 1] (I32)
    struct ggml_tensor* draft_input_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, block_size, 1);
    // position_ids: [ctx_len + block_size, 1]
    struct ggml_tensor* position_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, ctx_len + block_size, 1);

    EXPECT(target_hs != nullptr, "target_hs allocated");
    EXPECT(draft_input_ids != nullptr, "draft_input_ids allocated");
    EXPECT(position_ids != nullptr, "position_ids allocated");

    // ---------- 5. Fill with simple values ----------
    // Target hidden states (concat_dim, ctx_len, 1) — random values
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!cpu) {
        ggml_free(ctx); ggml_free(target.ctx);
        fprintf(stderr, "FAIL: no CPU backend\n");
        return 1;
    }
    ggml_gallocr_t allo = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    ggml_cgraph* gf = ggml_new_graph(ctx);

    // Build the forward graph (target_hs + draft_input_ids + position_ids are inputs)
    struct ggml_tensor* out = draft.forward(ctx, target_hs, draft_input_ids, position_ids, nullptr);

    if (out == nullptr) {
        fprintf(stderr, "FAIL: forward returned nullptr (still in skeleton mode?)\n");
        ggml_gallocr_free(allo);
        ggml_free(ctx); ggml_free(target.ctx);
        ggml_backend_free(cpu);
        return 1;
    }
    EXPECT(out != nullptr, "forward returns non-null (REAL forward executed)");

    // Output shape: [n_vocab, block_size, 1]
    fprintf(stderr, "  out shape: ne=[%lld, %lld, %lld]\n",
            (long long)out->ne[0], (long long)out->ne[1], (long long)out->ne[2]);
    EXPECT(out->ne[0] == draft.config().n_vocab,
           "out ne[0] = vocab_size (151936)");
    EXPECT(out->ne[1] == block_size,
           "out ne[1] = block_size (7)");
    EXPECT(out->ne[2] == 1,
           "out ne[2] = 1 (batch)");

    // Build forward graph + allocate
    ggml_build_forward_expand(gf, out);
    if (!ggml_gallocr_alloc_graph(allo, gf)) {
        fprintf(stderr, "FAIL: alloc graph failed\n");
        ggml_gallocr_free(allo);
        ggml_free(ctx); ggml_free(target.ctx);
        ggml_backend_free(cpu);
        return 1;
    }

    // Fill input tensors with simple values
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 0.1f);

    // target_hs: small non-zero values
    float* p_target_hs = (float*) target_hs->data;
    for (int i = 0; i < concat_dim * ctx_len; ++i) p_target_hs[i] = nd(rng);

    // draft_input_ids: token ids in [0, vocab)
    std::uniform_int_distribution<int32_t> uniform_t(0, 151935);
    int32_t* p_draft_ids = (int32_t*) draft_input_ids->data;
    for (int i = 0; i < block_size; ++i) p_draft_ids[i] = uniform_t(rng);

    // position_ids: 0..ctx_len+block_size-1
    int32_t* p_pos_ids = (int32_t*) position_ids->data;
    for (int i = 0; i < ctx_len + block_size; ++i) p_pos_ids[i] = i;

    // Compute forward
    enum ggml_status st = ggml_backend_graph_compute(cpu, gf);
    EXPECT(st == GGML_STATUS_SUCCESS, "forward graph compute success");

    // Read output and sanity check (should not all be NaN/Inf)
    float* p_out = (float*) out->data;
    int n_total = (int)(out->ne[0] * out->ne[1]);
    int n_finite = 0, n_nan = 0, n_inf = 0;
    float max_abs = 0.0f;
    for (int i = 0; i < n_total; ++i) {
        float v = p_out[i];
        if (std::isnan(v)) n_nan++;
        else if (std::isinf(v)) n_inf++;
        else n_finite++;
        float a = std::fabs(v);
        if (a > max_abs && !std::isnan(v) && !std::isinf(v)) max_abs = a;
    }
    fprintf(stderr, "  output stats: %d finite, %d NaN, %d Inf, max|f| = %.4f\n",
            n_finite, n_nan, n_inf, max_abs);

    EXPECT(n_nan == 0, "no NaN in output");
    EXPECT(n_inf == 0, "no Inf in output");
    EXPECT(n_finite > 0, "at least some finite values");

    ggml_gallocr_free(allo);
    ggml_free(ctx);
    // Don't free target.ctx — its tensors are referenced by draft.weights_
    // (they'll be freed when FusionDSparkModel destructor runs since target.ctx
    //  owns them). Actually no — target.ctx is owned by us separately; if we
    // free it here, draft.weights_ will dangle. We need to keep it alive
    // for the duration of draft. Let's not free it in this test.
    // ggml_free(target.ctx);
    (void)target.ctx;  // suppress unused warning
    ggml_backend_free(cpu);

    printf("\n=== Summary: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}