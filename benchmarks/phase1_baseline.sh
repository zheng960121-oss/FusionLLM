#!/bin/bash
# FusionLLM Phase 1 Baseline Benchmark
# Run vanilla llama.cpp on M5 to establish baseline before adding fusion driver
#
# Usage: ./phase1_baseline.sh <model_path>
# Example: ./phase1_baseline.sh ~/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf

set -e
cd "$(dirname "$0")/../llama.cpp-fusionllm"

MODEL_PATH="${1:-models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    echo "Download one first, e.g.:"
    echo "  huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf"
    exit 1
fi

echo "=== FusionLLM Phase 1 Baseline Benchmark ==="
echo "Device: $(system_profiler SPHardwareDataType | grep -E 'Chip|Memory' | tr '\n' ' ')"
echo "Model: $MODEL_PATH"
echo "Binary: vanilla llama.cpp (no fusion driver yet)"
echo ""

# Run with pipe EOF to auto-exit
echo "Running prompt: 'The capital of France is'"
RESULT=$(./build/bin/llama-cli \
    -m "$MODEL_PATH" \
    -ngl 99 \
    -t 4 \
    -p "The capital of France is" \
    -n 20 \
    --no-display-prompt 2>&1 <<EOF
/exit
EOF
)

echo "$RESULT" | grep -E "Prompt:|Generation:|t/s" | head -10

echo ""
echo "=== Notes ==="
echo "- Baseline: vanilla llama.cpp with Metal 4 on Apple M5"
echo "- 'n -g l99': all layers on GPU (for small model like 0.5B, all fit)"
echo "- 'n -t 4': 4 CPU threads for I/O"
echo "- 'n -n 20': generate 20 tokens"
echo ""
echo "Next steps:"
echo "  1. Phase 1.1: Add -DFUSION_DRIVER compile flag"
echo "  2. Phase 1.2: Hook mlock into ggml weight loading"
echo "  3. Phase 1.3: Re-run this script with fusion driver, compare"