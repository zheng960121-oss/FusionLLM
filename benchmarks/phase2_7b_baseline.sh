#!/bin/bash
# Phase 2 7B baseline benchmark
# Downloads Qwen2.5-7B Q4_K_M, runs baseline + mlock + selective mlock
#
# Usage: ./phase2_7b_baseline.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$HOME/Models"
MODEL_FILE="$MODEL_DIR/qwen2.5-7b-instruct-q4_k_m.gguf"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf"

cd "$SCRIPT_DIR/.."

echo "=== Phase 2 7B Baseline Benchmark ==="
echo "Device: $(system_profiler SPHardwareDataType | grep -E 'Chip|Memory' | tr '\n' ' ')"
echo ""

# 1. Download model if not present
if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading Qwen 2.5-7B Q4_K_M (~4.5GB)..."
    mkdir -p "$MODEL_DIR"
    curl -L -o "$MODEL_FILE" "$MODEL_URL" 2>&1 | tail -3
fi
ls -lh "$MODEL_FILE"
echo ""

# 2. Run layer inspection
echo "=== Layer Inspection ==="
DYLD_LIBRARY_PATH="$HOME/Desktop/llama.cpp-fusionllm/build/bin" \
    ./build/bin/fusion_inspect "$MODEL_FILE" 2>&1 | tail -40
echo ""

# 3. Baseline run (no mlock)
echo "=== Baseline (no mlock) - 5 runs ==="
LLAMA_DIR="$HOME/Desktop/llama.cpp-fusionllm"
for i in 1 2 3 4 5; do
    echo -n "Run $i: "
    "$LLAMA_DIR/build/bin/llama-cli" \
        -m "$MODEL_FILE" -ngl 99 -t 4 \
        -p "The capital of France is" -n 20 -st --no-display-prompt 2>&1 | \
        grep -oE "(Prompt|Generation): [0-9.]+ t/s" | tr '\n' ' '
    echo ""
done
echo ""

# 4. Run with --mlock
echo "=== With --mlock (full model mlock) - 5 runs ==="
for i in 1 2 3 4 5; do
    echo -n "Run $i: "
    "$LLAMA_DIR/build/bin/llama-cli" \
        -m "$MODEL_FILE" -ngl 99 -t 4 --mlock \
        -p "The capital of France is" -n 20 -st --no-display-prompt 2>&1 | \
        grep -oE "(Prompt|Generation): [0-9.]+ t/s" | tr '\n' ' '
    echo ""
done
echo ""

# 5. Selective mlock via fusion_inspect
echo "=== Selective mlock (6 layers) ==="
DYLD_LIBRARY_PATH="$HOME/Desktop/llama.cpp-fusionllm/build/bin" \
    ./build/bin/fusion_inspect "$MODEL_FILE" --mlock-window 6 2>&1 | tail -15
echo ""

# 6. Check how much physical memory is used after mlock
echo "=== Memory state after 6-layer mlock ==="
ps aux | grep -E "fusion_inspect" | grep -v grep | head -3 || true
vmmap $(pgrep -f fusion_inspect 2>/dev/null) 2>&1 | grep -E "WIRED|MLOCK" | head -10 || true