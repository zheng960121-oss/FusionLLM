#!/bin/bash
# Phase 3 W1 D5: 8B + FusionKVTierManager end-to-end validation
#
# Loads Qwen3-8B Q4_K_M, attaches FusionKVTierManager to a llama_context,
# runs prefill + generation, and reports tier manager stats.  The goal is
# to verify that on a real model (not a unit test):
#   1. W1 D3 SSD offload works (L0/L1/L2 byte counts make sense)
#   2. W1 D4 llama_context hook fires (n_gpu_copies > 0)
#   3. Performance is acceptable (no 10x regression vs D0 baseline)
#
# Usage:
#   ./benchmarks/phase3_w1_d5_8b_kv_tier.sh [prompt_tokens] [gen_tokens]
#   defaults: prompt=4096, gen=10
#
# D0 baseline (no tier manager, n_ctx=32K):
#   4K prefill: 455 t/s, generation: 24 t/s
#   32K prefill: 11+ min

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FUSION_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_DIR="$HOME/Desktop/llama.cpp-fusionllm"
MODEL="${MODEL:-$HOME/Desktop/models/Qwen3-8B-Q4_K_M.gguf}"
PROMPT_TOKENS="${1:-4096}"
GEN_TOKENS="${2:-10}"

if [ ! -f "$MODEL" ]; then
    echo "❌ Model not found: $MODEL"
    echo "   Run benchmarks/phase3_8b_poc_baseline.md setup first"
    exit 1
fi

if [ ! -f "$FUSION_DIR/build/bin/test-fusion-kv-tier-integration" ]; then
    echo "❌ test-fusion-kv-tier-integration not built. Run build_fusion_tests.sh first"
    exit 1
fi

# Copy the test binary next to its libggml deps
TEST_BIN="$FUSION_DIR/build/bin/test-fusion-kv-tier-integration"
LLAMA_BIN="$LLAMA_DIR/build/bin"

# Run the integration test with a tunable prompt size
# Note: test-fusion-kv-tier-integration currently does a 1-token decode only.
# For real prefill testing we need a separate driver — see W1 D5 follow-up.
export DYLD_LIBRARY_PATH="$LLAMA_BIN"
export OMP_NUM_THREADS=8  # macOS auto 不可靠 (D0 finding)

echo "=================================================="
echo "Phase 3 W1 D5: 8B + FusionKVTierManager validation"
echo "=================================================="
echo "Model:        $MODEL"
echo "Prompt:       $PROMPT_TOKENS tokens (raw, will be tokenized)"
echo "Gen:          $GEN_TOKENS tokens"
echo "Test binary:  $TEST_BIN"
echo ""

# Check tier manager will be called even with 1 token (proves integration)
echo "=== Phase A: 1-token forward (integration smoke test) ==="
"$TEST_BIN" "$MODEL" 2>&1 | grep -E "PASS|FAIL|stats" | head -20

echo ""
echo "=== Phase B: multi-token batch (prefill simulation) ==="

# Write a small driver that:
#   1. Loads model with n_ctx=$PROMPT_TOKENS+256
#   2. Creates context
#   3. Attaches tier manager
#   4. Runs llama_decode with N tokens in one batch (prefill)
#   5. Runs N single-token decode (generation)
#   6. Reports stats + timing

cat > /tmp/fusion_w1_d5_driver.cpp <<'EOF'
#include "llama.h"
#include "fusion_kv_tier.h"
#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <unistd.h>

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model> <prompt_tokens> <gen_tokens>\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    int n_prompt = atoi(argv[2]);
    int n_gen    = atoi(argv[3]);

    fprintf(stderr, "[D5] model=%s n_prompt=%d n_gen=%d\n", model_path, n_prompt, n_gen);

    // 1. Load model
    auto t0 = std::chrono::steady_clock::now();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "FAIL: model load\n"); return 1; }
    int n_layers = llama_model_n_layer(model);
    int n_embd   = llama_model_n_embd(model);
    int n_head   = llama_model_n_head(model);
    int n_head_kv = llama_model_n_head_kv(model);
    int head_dim = n_embd / n_head;
    int n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    int kv_size  = n_prompt + 256;  // match n_ctx
    fprintf(stderr, "[D5] model: layers=%d n_embd=%d head_dim=%d (n_head=%d, n_head_kv=%d) vocab=%d\n",
            n_layers, n_embd, head_dim, n_head, n_head_kv, n_vocab);

    // 2. Context
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = kv_size;
    cp.n_threads = 8;
    cp.n_batch  = n_prompt;  // allow big batches
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "FAIL: ctx\n"); llama_model_free(model); return 1; }
    auto t_load = std::chrono::steady_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_load - t0).count();
    fprintf(stderr, "[D5] load+ctx: %.0f ms\n", load_ms);

    // 3. Tier manager
    size_t ggml_sz = 256 * 1024 * 1024;  // 256 MB for L0 staging (8B 36 layers × head_dim × kv_size)
    struct ggml_init_params gp = { ggml_sz, nullptr, false };
    ggml_context* ggml_ctx = ggml_init(gp);
    if (!ggml_ctx) { fprintf(stderr, "FAIL: ggml_init\n"); llama_free(ctx); llama_model_free(model); return 1; }
    // block_size: 512 tokens per block (8B; 70B would use 128)
    fusion::FusionKVTierManager mgr(ggml_ctx, n_layers, kv_size, head_dim,
                                     GGML_TYPE_F16, /*block_size=*/512);
    char ssd_path[256];
    snprintf(ssd_path, sizeof(ssd_path), "/tmp/fusion_w1_d5_%d", (int)getpid());
    mgr.set_ssd_path(ssd_path);

    // 4. Attach
    fusion_kv_tier_attach(ctx, reinterpret_cast<fusion_kv_tier_t*>(&mgr));
    fprintf(stderr, "[D5] tier manager attached (kv_size=%d, head_dim=%d, n_layers=%d)\n",
            kv_size, head_dim, n_layers);

    // 5. Prefill: tokenize prompt, run llama_decode once with full batch
    std::string prompt_text(n_prompt, 'a');  // 1 char per token (rough approx)
    std::vector<llama_token> tokens(n_prompt);
    for (int i = 0; i < n_prompt; ++i) tokens[i] = i % n_vocab;

    llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    batch.n_tokens = n_prompt;
    for (int i = 0; i < n_prompt; ++i) {
        batch.token[i]    = tokens[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = (i == n_prompt - 1);  // only need logits on last
    }
    auto s_before_prefill = mgr.get_stats();
    fprintf(stderr, "[D5] prefill: %d tokens in 1 batch\n", n_prompt);
    auto tp0 = std::chrono::steady_clock::now();
    int rc = llama_decode(ctx, batch);
    auto tp1 = std::chrono::steady_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();
    if (rc != 0) { fprintf(stderr, "FAIL: prefill rc=%d\n", rc); return 1; }
    fprintf(stderr, "[D5] prefill: %.0f ms (%.1f t/s)\n", prefill_ms,
            n_prompt / (prefill_ms / 1000.0));
    auto s_after_prefill = mgr.get_stats();
    fprintf(stderr, "[D5] prefill stats: gpu_copies=%zu (was %zu) promotions=%zu ssd_reads=%zu\n",
            s_after_prefill.n_gpu_copies, s_before_prefill.n_gpu_copies,
            s_after_prefill.n_promotions, s_after_prefill.n_ssd_reads);

    // 6. Generation: N single-token decodes
    auto s_before_gen = mgr.get_stats();
    auto tg0 = std::chrono::steady_clock::now();
    llama_token last_token = tokens.back();
    llama_batch single = llama_batch_init(1, 0, 1);
    for (int i = 0; i < n_gen; ++i) {
        single.n_tokens = 1;
        single.token[0]    = last_token;
        single.pos[0]      = n_prompt + i;
        single.n_seq_id[0] = 1;
        single.seq_id[0][0] = 0;
        single.logits[0]   = true;
        rc = llama_decode(ctx, single);
        if (rc != 0) { fprintf(stderr, "FAIL: gen step %d rc=%d\n", i, rc); return 1; }
        // Sample a random next token (no real sampler; we just want to advance)
        last_token = (last_token + 1) % n_vocab;
    }
    auto tg1 = std::chrono::steady_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(tg1 - tg0).count();
    fprintf(stderr, "[D5] gen: %d tokens in %.0f ms (%.2f t/s)\n",
            n_gen, gen_ms, n_gen / (gen_ms / 1000.0));
    auto s_after_gen = mgr.get_stats();
    fprintf(stderr, "[D5] gen stats delta: gpu_copies=+%zu promotions=+%zu ssd_reads=+%zu evictions=+%zu\n",
            s_after_gen.n_gpu_copies - s_before_gen.n_gpu_copies,
            s_after_gen.n_promotions - s_before_gen.n_promotions,
            s_after_gen.n_ssd_reads - s_before_gen.n_ssd_reads,
            s_after_gen.n_evictions - s_before_gen.n_evictions);

    auto s_final = mgr.get_stats();
    fprintf(stderr, "[D5] final: L0=%.2f MB L1=%.2f MB L2=%.2f MB\n",
            s_final.bytes_in_gpu / (1024.0 * 1024.0),
            s_final.bytes_in_cpu / (1024.0 * 1024.0),
            s_final.bytes_in_ssd / (1024.0 * 1024.0));

    // Cleanup
    fusion_kv_tier_attach(ctx, nullptr);
    llama_batch_free(batch);
    llama_batch_free(single);
    llama_free(ctx);
    llama_model_free(model);
    ggml_free(ggml_ctx);
    return 0;
}
EOF

# Build the driver
clang++ -std=c++17 -O2 \
    -I "$LLAMA_DIR/ggml/include" \
    -I "$LLAMA_DIR/include" \
    -I "$LLAMA_DIR/src" \
    -I "$FUSION_DIR/src" \
    /tmp/fusion_w1_d5_driver.cpp \
    "$FUSION_DIR/build/obj/fusion_kv_tier.o" \
    -L "$LLAMA_BIN" \
    -lggml -lggml-base -lggml-cpu -lggml-blas -lllama \
    -Wl,-rpath,"$LLAMA_BIN" \
    -o /tmp/fusion_w1_d5_driver 2>&1 | grep -E "error|warning" | head -5 || true
if [ ! -f /tmp/fusion_w1_d5_driver ]; then
    echo "❌ Failed to build driver"
    exit 1
fi

# Run it
/tmp/fusion_w1_d5_driver "$MODEL" "$PROMPT_TOKENS" "$GEN_TOKENS" 2>&1 | tail -30
