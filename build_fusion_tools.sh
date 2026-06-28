#!/bin/bash
# Build FusionLLM tools (fusion_inspect + future fusion_driver tools)
# Requires: llama.cpp-fusionllm built in ~/Desktop/llama.cpp-fusionllm/
#
# Usage: ./build_fusion_tools.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$HOME/Desktop/llama.cpp-fusionllm}"

if [ ! -d "$LLAMA_DIR" ]; then
    echo "❌ llama.cpp-fusionllm not found at $LLAMA_DIR"
    echo "Run: git clone --depth=1 https://github.com/ggerganov/llama.cpp.git $LLAMA_DIR"
    exit 1
fi

if [ ! -d "$LLAMA_DIR/build/bin" ]; then
    echo "❌ llama.cpp-fusionllm not built. Build it first:"
    echo "  cd $LLAMA_DIR && cmake -B build -DGGML_METAL=ON && cmake --build build -j 8"
    exit 1
fi

mkdir -p "$SCRIPT_DIR/build/bin"

echo "=== Building fusion_inspect ==="
echo "LLAMA_DIR: $LLAMA_DIR"

clang++ -std=c++17 -O2 \
    -I "$LLAMA_DIR/ggml/include" \
    -I "$LLAMA_DIR/include" \
    "$SCRIPT_DIR/src/fusion_inspect.cpp" \
    -L "$LLAMA_DIR/build/bin" \
    -lggml-base -lggml-cpu \
    -Wl,-rpath,"$LLAMA_DIR/build/bin" \
    -o "$SCRIPT_DIR/build/bin/fusion_inspect"

if [ $? -eq 0 ]; then
    echo "✅ fusion_inspect built: $SCRIPT_DIR/build/bin/fusion_inspect"
    echo ""
    echo "Run: DYLD_LIBRARY_PATH=$LLAMA_DIR/build/bin $SCRIPT_DIR/build/bin/fusion_inspect <model.gguf>"
else
    echo "❌ Build failed"
    exit 1
fi