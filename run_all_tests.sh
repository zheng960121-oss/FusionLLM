#!/bin/bash
# Run all FusionLLM Phase 6 tests with one command
#
# Tests:
#   S2 FusionHSExtractor (unit + integration with Qwen 0.5B)
#   S3 FusionDSparkModel (loading - needs DSpark GGUF for full)
#   S4 FusionSpecDecoder (rejection sampling)
#   S5 FusionWindow ↔ Spec Decoding coordination
#   S7 DSparkAttention (skeleton mode)
#   S8 MarkovHead (vanilla ops, full PyTorch reference comparison)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$HOME/Desktop/llama.cpp-fusionllm}"
QWEN05B="$LLAMA_DIR/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
DSPARK_GGUF="${DSPARK_GGUF:-/tmp/dspark_qwen3_4b.gguf}"
DSPARK_MOCK_DIR="${DSPARK_MOCK_DIR:-/tmp/mock_dspark}"

export DYLD_LIBRARY_PATH="$LLAMA_DIR/build/bin"

# Build first
"$SCRIPT_DIR/build_fusion_tests.sh" > /dev/null 2>&1

# Generate DSpark GGUF if not present (uses mock weights from --from-mock)
if [ ! -f "$DSPARK_GGUF" ]; then
    echo "[setup] generating DSpark GGUF via tools/dspark_to_gguf.py --from-mock ..."
    python3 "$SCRIPT_DIR/tools/dspark_to_gguf.py" \
        --draft-config qwen3_4b \
        --target-gguf "$QWEN05B" \
        --output "$DSPARK_GGUF" \
        --from-mock \
        --mock-output "$DSPARK_MOCK_DIR" 2>&1 | tail -3
    echo ""
fi

pass=0
fail=0
results=()

run_test() {
    local name="$1"
    local args="${2:-}"
    local out
    out=$("$SCRIPT_DIR/build/bin/$name" $args 2>&1)
    local rc=$?
    if [ $rc -eq 0 ]; then
        pass=$((pass+1))
        results+=("✅ $name")
    else
        fail=$((fail+1))
        results+=("❌ $name (rc=$rc)")
    fi
    echo "$out" | grep -E "Summary|passed|failed|⚠" | tail -3
    echo "---"
}

echo "=================================================="
echo "FusionLLM Phase 6 Test Suite"
echo "=================================================="
echo ""

echo "=== S2 FusionHSExtractor (unit + integration) ==="
run_test "test-fusion-hs-extract" "$QWEN05B"
echo ""

echo "=== S3 FusionDSparkModel (loading) ==="
run_test "test-fusion-draft-model" "$DSPARK_GGUF"
echo ""

echo "=== S4 Rejection Sampling ==="
run_test "test-fusion-spec-decode"
echo ""

echo "=== S5 Window ↔ Spec Decoding Coordination ==="
run_test "test-fusion-window-spec-coord"
echo ""

echo "=== S7 DSpark Attention (skeleton) ==="
run_test "test-fusion-dspart-attention"
echo ""

echo "=== S8 Markov Head Vanilla ==="
run_test "test-fusion-markov-head"
echo ""

echo "=================================================="
echo "Summary: $pass passed, $fail failed"
echo "=================================================="
for r in "${results[@]}"; do
    echo "  $r"
done

exit $fail
