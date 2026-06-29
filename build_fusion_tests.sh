#!/bin/bash
# Build all FusionLLM Phase 6 tests (S2-S8)
#
# Strategy: Most fusion code (S2-S7) is already compiled into llama.cpp's
# libllama.dylib.  We just need to:
#   1. Compile our S8 fusion_markov_head.cpp separately (new code, not in libllama)
#   2. Link tests against libllama + libggml + our S8 object
#
# Tests:
#   S2 test-fusion-hs-extract       (needs llama lib - calls set_embeddings_layer_inp)
#   S3 test-fusion-draft-model      (needs llama lib - calls load_from_gguf)
#   S4 test-fusion-spec-decode      (needs llama lib - calls rejection_sample etc)
#   S5 test-fusion-window-spec-coord (needs llama lib - calls window_get_*)
#   S7 test-fusion-dspart-attention (standalone, no llama lib needed for skeleton)
#   S8 test-fusion-markov-head     (standalone, no llama lib needed)
#
# Usage: ./build_fusion_tests.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$HOME/Desktop/llama.cpp-fusionllm}"

if [ ! -d "$LLAMA_DIR" ]; then
    echo "❌ llama.cpp-fusionllm not found at $LLAMA_DIR"
    exit 1
fi
if [ ! -f "$LLAMA_DIR/build/bin/libllama.dylib" ]; then
    echo "❌ llama.cpp-fusionllm libllama.dylib not built. Build it first:"
    echo "  cd $LLAMA_DIR && cmake -B build -DGGML_METAL=ON && cmake --build build -j 8"
    exit 1
fi

mkdir -p "$SCRIPT_DIR/build/bin"
mkdir -p "$SCRIPT_DIR/build/obj"

CXXFLAGS="-std=c++17 -O2"

echo "=== Building S8 fusion_markov_head.o (new code, not in libllama) ==="
clang++ $CXXFLAGS \
    -I "$LLAMA_DIR/ggml/include" \
    -I "$LLAMA_DIR/include" \
    -I "$LLAMA_DIR/src" \
    -I "$SCRIPT_DIR/src" \
    -c "$SCRIPT_DIR/src/fusion_markov_head.cpp" \
    -o "$SCRIPT_DIR/build/obj/fusion_markov_head.o" 2>&1 | tail -3
echo ""

# Test: extras
test_window_spec_extras=""
test_dspart_extras=""
test_spec_decode_extras=""
test_draft_model_extras=""
test_hs_extract_extras=""

build_test() {
    local name="$1"
    local extras="$2"
    local need_llama="$3"  # "1" if needs libllama, "0" otherwise
    local objs="$extras $SCRIPT_DIR/build/obj/fusion_markov_head.o"
    local obj_args=""
    for o in $objs; do
        if [ -f "$o" ]; then
            obj_args="$obj_args $o"
        fi
    done
    local lib_args="-lggml -lggml-base -lggml-cpu -lggml-blas"
    if [ "$need_llama" = "1" ]; then
        lib_args="$lib_args -lllama"
    fi
    echo "  $name$(if [ "$need_llama" = "1" ]; then echo " [+llama]"; fi)"
    local defs=""
    if [ "$name" = "test-fusion-hs-extract" ]; then
        defs="-DFUSION_HS_INTEGRATION_TEST"
    fi
    clang++ $CXXFLAGS $defs \
        -I "$LLAMA_DIR/ggml/include" \
        -I "$LLAMA_DIR/include" \
        -I "$LLAMA_DIR/src" \
        -I "$SCRIPT_DIR/src" \
        "$SCRIPT_DIR/tests/${name}.cpp" \
        $obj_args \
        -L "$LLAMA_DIR/build/bin" \
        $lib_args \
        -Wl,-rpath,"$LLAMA_DIR/build/bin" \
        -o "$SCRIPT_DIR/build/bin/${name}" 2>&1 | tail -5
}

echo "=== Building tests ==="
# Standalone tests (no llama lib needed) - only test-fusion-markov-head
build_test "test-fusion-markov-head"   ""                       "0"

# Tests that need llama lib (S2-S5 + S7)
build_test "test-fusion-dspart-attention" ""                       "1"
build_test "test-fusion-window-spec-coord" "$test_window_spec_extras"  "1"
build_test "test-fusion-spec-decode"       "$test_spec_decode_extras"  "1"
build_test "test-fusion-draft-model"       "$test_draft_model_extras"  "1"
build_test "test-fusion-hs-extract"        "$test_hs_extract_extras"   "1"

echo ""
echo "=== Build complete ==="
echo "Binaries in $SCRIPT_DIR/build/bin/:"
ls -la "$SCRIPT_DIR/build/bin/test-fusion-"* 2>/dev/null | awk '{print "  " $NF}'
echo ""
echo "Run with: DYLD_LIBRARY_PATH=$LLAMA_DIR/build/bin $SCRIPT_DIR/build/bin/test-fusion-<name>"
