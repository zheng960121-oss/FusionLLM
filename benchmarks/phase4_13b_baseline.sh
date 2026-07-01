#!/bin/bash
# Phase 4 13B baseline benchmark (post-降级到13B decision 2026-07-01)
#
# Downloads Qwen2.5-14B-Instruct Q4_K_M and runs:
#   1. Vanilla llama-cli baseline (no mlock) — 4K + 32K context
#   2. With --mlock
#   3. FusionLLM W1 D5 driver (mlock + KV tier + sliding window)
#
# Compares against Phase 2 (7B) and Phase 3 W1 (8B 32K) baselines.
#
# Usage:
#   ./benchmarks/phase4_13b_baseline.sh
#
# Expected model size: ~8.5 GB (Q4_K_M), download 10-30 min depending on network.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FUSION_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_DIR="$HOME/Desktop/llama.cpp-fusionllm"
MODEL_DIR="$HOME/Models"
MODEL_FILE="$MODEL_DIR/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
# bartowski mirror (Qwen 官方 GGUF 仓库不存在)
MODEL_URL="https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf"

cd "$FUSION_DIR"

echo "=================================================="
echo "Phase 4 13B Baseline Benchmark"
echo "=================================================="
echo "Device: $(system_profiler SPHardwareDataType | grep -E 'Chip|Memory' | tr '\n' ' ')"
echo "Date:   $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 1. Download model if not present
if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading Qwen2.5-14B-Instruct Q4_K_M (~8.5GB)..."
    mkdir -p "$MODEL_DIR"
    curl -L -o "$MODEL_FILE" "$MODEL_URL" 2>&1 | tail -3
fi
ls -lh "$MODEL_FILE"
echo ""

# 2. Layer inspection
echo "=== Layer Inspection ==="
DYLD_LIBRARY_PATH="$LLAMA_DIR/build/bin" \
    ./build/bin/fusion_inspect "$MODEL_FILE" 2>&1 | tail -20
echo ""

# 3. Baseline runs (no mlock)
echo "=== Baseline (no mlock) - 4K + 32K context ==="
for n_ctx in 4096 32768; do
    for i in 1 2 3; do
        echo -n "  [no-mlock, n_ctx=$n_ctx] Run $i: "
        "$LLAMA_DIR/build/bin/llama-cli" \
            -m "$MODEL_FILE" -ngl 99 -t 8 -c "$n_ctx" --no-mlock \
            -p "$(python3 -c "print('The capital of France is ' * $((n_ctx / 8)))")" \
            -n 20 -st --no-display-prompt 2>&1 | \
            grep -oE "(Prompt|Generation): [0-9.]+ t/s" | tr '\n' ' '
        echo ""
    done
done
echo ""

# 4. With --mlock
echo "=== With --mlock (full model mlock) - 4K + 32K context ==="
for n_ctx in 4096 32768; do
    for i in 1 2 3; do
        echo -n "  [--mlock, n_ctx=$n_ctx] Run $i: "
        "$LLAMA_DIR/build/bin/llama-cli" \
            -m "$MODEL_FILE" -ngl 99 -t 8 -c "$n_ctx" --mlock \
            -p "$(python3 -c "print('The capital of France is ' * $((n_ctx / 8)))")" \
            -n 20 -st --no-display-prompt 2>&1 | \
            grep -oE "(Prompt|Generation): [0-9.]+ t/s" | tr '\n' ' '
        echo ""
    done
done
echo ""

# 5. FusionLLM W1 D5 driver (mlock + KV tier + sliding window)
echo "=== FusionLLM W1 D5 driver (mlock + KV tier + sliding window) ==="
if [ -f "$FUSION_DIR/build/bin/test-fusion-kv-tier-integration" ]; then
    for n_ctx in 4096 32768; do
        for i in 1 2 3; do
            echo -n "  [fusion, n_ctx=$n_ctx] Run $i: "
            DYLD_LIBRARY_PATH="$LLAMA_DIR/build/bin" \
            OMP_NUM_THREADS=8 \
                bash -c "/tmp/fusion_w1_d5_driver '$MODEL_FILE' $n_ctx 20 2>&1 | grep -oE '(prefill|gen): [0-9.]+ t/s' | tr '\n' ' '"
            echo ""
        done
    done
else
    echo "❌ test-fusion-kv-tier-integration not built, skipping FusionLLM run"
fi
echo ""

# 6. Memory state
echo "=== Memory state (during/after run) ==="
vm_stat | head -8
echo ""

echo "=================================================="
echo "Phase 4 13B Baseline Complete"
echo "=================================================="
echo ""
echo "Next: write phase4_13b_baseline_report.md with comparison to:"
echo "  - 7B (Phase 2): 175 t/s prefill, 6B physical"
echo "  - 8B 4K (W1 D5): 577 t/s prefill"
echo "  - 8B 32K (W1 D6): 153 t/s prefill, 11 t/s gen"
echo "  - 14B (Phase 4): <this run>"
