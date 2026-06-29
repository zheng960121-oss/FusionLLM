#!/bin/bash
# Build FusionLLM Phase 6 S8 Markov head test
# Requires: llama.cpp-fusionllm built in ~/Desktop/llama.cpp-fusionllm/
#
# Usage: ./build_fusion_tests.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$HOME/Desktop/llama.cpp-fusionllm}"

if [ ! -d "$LLAMA_DIR" ]; then
    echo "❌ llama.cpp-fusionllm not found at $LLAMA_DIR"
    exit 1
fi
if [ ! -d "$LLAMA_DIR/build/bin" ]; then
    echo "❌ llama.cpp-fusionllm not built. Build it first:"
    echo "  cd $LLAMA_DIR && cmake -B build -DGGML_METAL=ON && cmake --build build -j 8"
    exit 1
fi

mkdir -p "$SCRIPT_DIR/build/bin"

echo "=== Building fusion_markov_head.o ==="
clang++ -std=c++17 -O2 -c \
    -I "$LLAMA_DIR/ggml/include" \
    -I "$LLAMA_DIR/include" \
    "$SCRIPT_DIR/src/fusion_markov_head.cpp" \
    -o "$SCRIPT_DIR/build/fusion_markov_head.o"

echo "=== Building test-fusion-markov-head ==="
clang++ -std=c++17 -O2 \
    -I "$LLAMA_DIR/ggml/include" \
    -I "$LLAMA_DIR/include" \
    "$SCRIPT_DIR/tests/test-fusion-markov-head.cpp" \
    "$SCRIPT_DIR/build/fusion_markov_head.o" \
    -L "$LLAMA_DIR/build/bin" \
    -lggml -lggml-base -lggml-cpu -lggml-blas \
    -Wl,-rpath,"$LLAMA_DIR/build/bin" \
    -o "$SCRIPT_DIR/build/bin/test-fusion-markov-head"

if [ $? -eq 0 ]; then
    echo "✅ test-fusion-markov-head built: $SCRIPT_DIR/build/bin/test-fusion-markov-head"
    echo ""
    echo "Run: DYLD_LIBRARY_PATH=$LLAMA_DIR/build/bin $SCRIPT_DIR/build/bin/test-fusion-markov-head"
else
    echo "❌ Build failed"
    exit 1
fi
