#!/bin/bash
# PoC 批量编译 + 运行脚本
# 用法: ./run_all_pocs.sh

set -e
cd "$(dirname "$0")"

POCS=(
    "poc1_happy_path"
    "poc1_page_fault_test"
    "poc1_test4_realistic"
    "poc4_kv_cache_ssd"
)

echo "=== FusionLLM PoC Batch Runner ==="
echo ""

PASSED=0
FAILED=0

for poc in "${POCS[@]}"; do
    echo "▶ $poc"
    
    if [ ! -f "$poc.swift" ]; then
        echo "  ❌ $poc.swift not found"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    echo "  Compiling..."
    if ! swiftc -O "$poc.swift" -o "$poc" 2>&1; then
        echo "  ❌ Compilation failed"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    echo "  Running..."
    if "./$poc"; then
        echo "  ✅ PASSED"
        PASSED=$((PASSED + 1))
    else
        echo "  ❌ FAILED"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "=== Summary ==="
echo "Passed: $PASSED / $((PASSED + FAILED))"
echo "Failed: $FAILED / $((PASSED + FAILED))"

if [ $FAILED -gt 0 ]; then
    exit 1
fi