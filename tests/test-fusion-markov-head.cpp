// Test: fusion_markov_head_forward (Phase 6 S8)
//
// Compares ggml Markov head forward against a naive C++ reference (PyTorch order).
//
//   PyTorch reference:  base_logits + W2 @ W1[token_ids]
//   ggml impl:          same, expressed in ggml ops
//
// Pass criterion: max abs diff < 1e-3 across all elements (FP32).
//
// Tensors are stored in ggml's ne order.  We always fill the ggml data buffer
// using the formula: data[i, j, k, ...] at offset i + j*ne[0] + k*ne[0]*ne[1] + ...
// The naive reference works in PyTorch shape [B, S, V] row-major; we bridge
// between layouts at compare time.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include "../src/fusion_markov_head.h"

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define FAIL(msg) do { \
    fprintf(stderr, "  ❌ FAIL: %s\n", msg); \
    g_tests_failed++; \
    return 1; \
} while (0)

// Naive VanillaMarkov reference in PyTorch shape [B, S, V].
static void naive_markov_forward(
    const float* base_logits_pt,  // PyTorch [B, S, V] row-major
    const int32_t* token_ids_pt,  // PyTorch [B, S] row-major
    const float* w1_pt,           // PyTorch [V, rank] row-major (embedding)
    const float* w2_pt,           // PyTorch [V, rank] row-major (linear)
    float* out_pt,                // PyTorch [B, S, V] row-major
    int B, int S, int V, int rank) {
    for (int b = 0; b < B; b++) {
        for (int s = 0; s < S; s++) {
            int32_t tok = token_ids_pt[b * S + s];
            if (tok < 0 || tok >= V) {
                fprintf(stderr, "naive: token id %d out of range [0, %d)\n", tok, V);
                std::exit(2);
            }
            for (int v = 0; v < V; v++) {
                float bias = 0.f;
                for (int r = 0; r < rank; r++) {
                    // w2[v, r] @ w1[tok, r]
                    bias += w2_pt[v * rank + r] * w1_pt[tok * rank + r];
                }
                out_pt[(b * S + s) * V + v] = base_logits_pt[(b * S + s) * V + v] + bias;
            }
        }
    }
}

// ggml ne = [V, S, B]: convert flat index (in ggml row-major order) to PyTorch
// [B, S, V] row-major index.  Equivalent: pt_idx = b*S*V + s*V + v.
static inline size_t pt_idx(int b, int s, int v, int S, int V) {
    return (size_t)b * S * V + (size_t)s * V + (size_t)v;
}

// ggml ne = [V, S, B]: data[v, s, b] at offset v + s*V + b*V*S.
static inline size_t ggml_idx_3d(int v, int s, int b, int V, int S) {
    return (size_t)v + (size_t)s * V + (size_t)b * V * S;
}

// ggml ne = [V, S]: data[v, s] at offset v + s*V.
static inline size_t ggml_idx_2d(int v, int s, int V) {
    return (size_t)v + (size_t)s * V;
}

// ggml ne = [S, B]: data[s, b] at offset s + b*S.
static inline size_t ggml_idx_2d_sb(int s, int b, int S) {
    return (size_t)s + (size_t)b * S;
}

// ggml ne = [rank, V]: data[r, v] at offset r + v*rank.
static inline size_t ggml_idx_2d_rv(int r, int v, int rank) {
    return (size_t)r + (size_t)v * rank;
}

// Run one test case with the given shape.
static int run_case(const char* name,
                    int B, int S, int V, int rank,
                    uint32_t seed) {
    fprintf(stderr, "\n── case: %s  (B=%d S=%d V=%d rank=%d seed=%u)\n",
            name, B, S, V, rank, seed);

    // ---- Build ggml context & tensors ----
    // All tensors share one ggml context, with na_alloc=true (we allocate separately).
    struct ggml_init_params params = {
        /* .mem_size   = */ 64 * 1024 * 1024,  // 64 MB, plenty
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) FAIL("ggml_init failed");

    // Create tensors in ggml ne order.
    // base_logits:  ne = [V, S, B] (PyTorch [B, S, V])
    // token_ids:    ne = [S, B]   (PyTorch [B, S]) I32
    // w1, w2:       ne = [rank, V] (PyTorch [V, rank])
    struct ggml_tensor* base_logits = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, V, S, B);
    struct ggml_tensor* token_ids   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, S, B);
    struct ggml_tensor* w1          = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, V);
    struct ggml_tensor* w2          = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, V);

    // Build the graph.
    struct ggml_tensor* out = fusion_markov_head_forward(ctx, base_logits, token_ids, w1, w2);
    if (!out) { ggml_free(ctx); FAIL("fusion_markov_head_forward returned null"); }

    if (out->ne[0] != V || out->ne[1] != S || out->ne[2] != B) {
        fprintf(stderr,
                "  ❌ FAIL: out shape ne=[%lld, %lld, %lld], expected [%d, %d, %d]\n",
                (long long) out->ne[0], (long long) out->ne[1], (long long) out->ne[2],
                V, S, B);
        ggml_free(ctx);
        g_tests_failed++;
        return 1;
    }

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // Allocate backed storage.
    ggml_gallocr_t allo = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!ggml_gallocr_alloc_graph(allo, gf)) {
        ggml_gallocr_free(allo);
        ggml_free(ctx);
        FAIL("ggml_gallocr_alloc_graph failed");
    }

    // ---- Fill input tensors with deterministic random data ----
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist_w(-0.5f, 0.5f);
    std::uniform_real_distribution<float> dist_l(-2.0f, 2.0f);
    std::uniform_int_distribution<int32_t> dist_t(0, V - 1);

    // base_logits  ggml ne=[V,S,B]: fill in ggml order.
    float* p_base = (float*) base_logits->data;
    for (int b = 0; b < B; b++)
        for (int s = 0; s < S; s++)
            for (int v = 0; v < V; v++)
                p_base[ggml_idx_3d(v, s, b, V, S)] = dist_l(rng);

    // token_ids  ggml ne=[S,B]: fill in ggml order.
    int32_t* p_tok = (int32_t*) token_ids->data;
    for (int b = 0; b < B; b++)
        for (int s = 0; s < S; s++)
            p_tok[ggml_idx_2d_sb(s, b, S)] = dist_t(rng);

    // w1, w2  ggml ne=[rank,V]: fill in ggml order.
    float* p_w1 = (float*) w1->data;
    float* p_w2 = (float*) w2->data;
    for (int v = 0; v < V; v++) {
        for (int r = 0; r < rank; r++) {
            p_w1[ggml_idx_2d_rv(r, v, rank)] = dist_w(rng);
            p_w2[ggml_idx_2d_rv(r, v, rank)] = dist_w(rng);
        }
    }

    // ---- Save copies of inputs BEFORE the compute, because ggml_gallocr may
    //      reuse the input tensor memory for intermediate tensors, destroying
    //      the originals. ----
    std::vector<float>  base_copy(p_base, p_base + (size_t)B * S * V);
    std::vector<int32_t> tok_copy(p_tok,  p_tok  + (size_t)B * S);
    std::vector<float>  w1_copy(p_w1, p_w1 + (size_t)V * rank);
    std::vector<float>  w2_copy(p_w2, p_w2 + (size_t)V * rank);

    // ---- Compute graph with CPU backend ----
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend) {
        ggml_gallocr_free(allo);
        ggml_free(ctx);
        FAIL("ggml_backend_init_by_type(CPU) returned null");
    }
    enum ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_backend_free(backend);
        ggml_gallocr_free(allo);
        ggml_free(ctx);
        fprintf(stderr, "  ❌ FAIL: ggml_backend_graph_compute status=%d\n", (int) st);
        g_tests_failed++;
        return 1;
    }

    // ---- Build the PyTorch-shaped reference inputs (in PyTorch row-major) ----
    // Use the saved copies (w1_copy, w2_copy, base_copy, tok_copy) because the
    // ggml-backed p_* pointers may have been overwritten by the compute.
    std::vector<int32_t> tok_pt((size_t)B * S);
    for (int b = 0; b < B; b++)
        for (int s = 0; s < S; s++)
            tok_pt[(size_t)b * S + s] = tok_copy[ggml_idx_2d_sb(s, b, S)];

    std::vector<float> base_pt((size_t)B * S * V);
    for (int b = 0; b < B; b++)
        for (int s = 0; s < S; s++)
            for (int v = 0; v < V; v++)
                base_pt[(size_t)b * S * V + (size_t)s * V + v] = base_copy[ggml_idx_3d(v, s, b, V, S)];

    std::vector<float> ref_pt((size_t)B * S * V);
    naive_markov_forward(
        base_pt.data(), tok_pt.data(),
        w1_copy.data(), w2_copy.data(),
        ref_pt.data(),
        B, S, V, rank);

    // ---- Compare ggml output (ggml ne=[V,S,B]) to reference (PyTorch [B,S,V]) ----
    float* p_out = (float*) out->data;
    double max_abs_diff = 0.0;
    double sum_abs_diff = 0.0;
    int worst[3] = {0, 0, 0};
    for (int b = 0; b < B; b++) {
        for (int s = 0; s < S; s++) {
            for (int v = 0; v < V; v++) {
                float ggml_val = p_out[ggml_idx_3d(v, s, b, V, S)];
                float ref_val  = ref_pt[(size_t)b * S * V + (size_t)s * V + v];
                double d = std::fabs((double) ggml_val - (double) ref_val);
                if (d > max_abs_diff) {
                    max_abs_diff = d;
                    worst[0] = b;
                    worst[1] = s;
                    worst[2] = v;
                }
                sum_abs_diff += d;
            }
        }
    }
    // DEBUG: dump worst-case data
    {
        int bw = worst[0], sw = worst[1], vw = worst[2];
        fprintf(stderr, "  DEBUG worst (b=%d s=%d v=%d):\n", bw, sw, vw);
        fprintf(stderr, "    base_pt[b=%d,s=%d,v=%d] = %.6f\n", bw, sw, vw, base_pt[bw*S*V + sw*V + vw]);
        fprintf(stderr, "    tok_pt[b=%d,s=%d] = %d\n", bw, sw, tok_pt[bw*S + sw]);
        int tw = tok_pt[bw*S + sw];
        fprintf(stderr, "    w1_PT[%d, :rank] (first 8): ", tw);
        for (int r = 0; r < std::min(8, rank); r++) fprintf(stderr, "%.3f ", w1_copy[r + tw*rank]);
        fprintf(stderr, "\n    w2_PT[%d, :rank] (first 8): ", vw);
        for (int r = 0; r < std::min(8, rank); r++) fprintf(stderr, "%.3f ", w2_copy[r + vw*rank]);
        fprintf(stderr, "\n    ggml out at (b=%d,s=%d,v=%d) = %.6f\n", bw, sw, vw, p_out[ggml_idx_3d(vw, sw, bw, V, S)]);
        fprintf(stderr, "    ref_pt at (b=%d,s=%d,v=%d) = %.6f\n", bw, sw, vw, ref_pt[bw*S*V + sw*V + vw]);

        // Dump entire s=sw row of ggml output vs reference
        fprintf(stderr, "  DEBUG ggml out for s=%d: ", sw);
        for (int v = 0; v < std::min(V, 20); v++) fprintf(stderr, "%.3f ", p_out[ggml_idx_3d(v, sw, bw, V, S)]);
        fprintf(stderr, "\n  DEBUG ref_pt for s=%d: ", sw);
        for (int v = 0; v < std::min(V, 20); v++) fprintf(stderr, "%.3f ", ref_pt[bw*S*V + sw*V + v]);
        fprintf(stderr, "\n  DEBUG base_pt for s=%d: ", sw);
        for (int v = 0; v < std::min(V, 20); v++) fprintf(stderr, "%.3f ", base_pt[bw*S*V + sw*V + v]);
        fprintf(stderr, "\n  DEBUG ref - base for s=%d: ", sw);
        for (int v = 0; v < std::min(V, 20); v++) fprintf(stderr, "%.3f ", ref_pt[bw*S*V + sw*V + v] - base_pt[bw*S*V + sw*V + v]);
        fprintf(stderr, "\n  DEBUG ggml - base for s=%d: ", sw);
        for (int v = 0; v < std::min(V, 20); v++) fprintf(stderr, "%.3f ", p_out[ggml_idx_3d(v, sw, bw, V, S)] - base_pt[bw*S*V + sw*V + v]);
        fprintf(stderr, "\n  DEBUG w2_PT[v=0, :rank]: ");
        for (int r = 0; r < rank; r++) fprintf(stderr, "%.3f ", p_w2[r + 0*rank]);
        fprintf(stderr, "\n  DEBUG w2_PT[v=%d, :rank]: ", vw);
        for (int r = 0; r < rank; r++) fprintf(stderr, "%.3f ", p_w2[r + vw*rank]);
        fprintf(stderr, "\n  DEBUG w1_PT[tok=%d, :rank]: ", tw);
        for (int r = 0; r < rank; r++) fprintf(stderr, "%.3f ", p_w1[r + tw*rank]);
        fprintf(stderr, "\n");
    }
    size_t n = (size_t)B * S * V;
    double mean_abs_diff = sum_abs_diff / (double) n;

    fprintf(stderr, "  max |diff| = %.3e\n", max_abs_diff);
    fprintf(stderr, "  mean|diff| = %.3e\n", mean_abs_diff);
    fprintf(stderr, "  worst@ (b=%d, s=%d, v=%d) ggml=%.6f ref=%.6f\n",
            worst[0], worst[1], worst[2],
            p_out[ggml_idx_3d(worst[2], worst[1], worst[0], V, S)],
            ref_pt[(size_t)worst[0] * S * V + (size_t)worst[1] * V + worst[2]]);

    int rc = 0;
    const double TOL = 1e-3;
    if (max_abs_diff >= TOL) {
        fprintf(stderr, "  ❌ FAIL: max|diff|=%.3e >= tol %.3e\n", max_abs_diff, TOL);
        g_tests_failed++;
        rc = 1;
    } else {
        fprintf(stderr, "  ✅ PASS\n");
        g_tests_passed++;
    }

    ggml_backend_free(backend);
    ggml_gallocr_free(allo);
    ggml_free(ctx);
    return rc;
}

int main(int argc, char** argv) {
    (void) argc;
    (void) argv;
    fprintf(stderr, "=== test-fusion-markov-head (Phase 6 S8) ===\n");

    int rc = 0;
    rc |= run_case("small random",        1,    7,    64,   16,  0xC0FFEE);
    rc |= run_case("dspark-block7-vocab", 1,    7, 151936, 256, 0xDEEDBE);
    rc |= run_case("multi-batch",         2,    7,  1024,   32, 0xA11CE);
    rc |= run_case("tall block",          1,   64,  4096,  128, 0xBA0BAB);
    rc |= run_case("single token",        1,    1,    32,    4, 0x1234);
    rc |= run_case("2d base (S only)",    1,    8,   128,   16, 0xC0DE);

    fprintf(stderr, "\n=== summary ===\n");
    fprintf(stderr, "passed: %d\n", g_tests_passed);
    fprintf(stderr, "failed: %d\n", g_tests_failed);
    if (g_tests_failed == 0) {
        fprintf(stderr, "🎉 all markov head tests passed\n");
    }
    return rc == 0 ? 0 : 1;
}
