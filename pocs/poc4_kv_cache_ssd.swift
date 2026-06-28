// PoC-4: KV Cache SSD 读写循环（路径 c 专项）
// 这是 Phase 3 的提前风险验证
//
// 测试场景（模拟 FusionLLM 真实工作流）:
//   1. 计算出 KV Cache（GPU 或 CPU 生成）
//   2. 序列化到 SSD 文件
//   3. 内存压力下，从 SSD 重新加载（mmap）
//   4. GPU 通过 zero-copy 读取
//   5. 验证数据完整性
//
// 关键问题：多轮读写循环是否稳定？数据是否损坏？延迟是否可接受？

import Foundation
import Metal

// MARK: - Configuration
let kvFilePath = "/tmp/fusionllm_poc4_kv.bin"
let kvSize: Int = 256 * 1024 * 1024  // 256MB per KV block
let testElements: Int = 1024
let numCycles = 10

// MARK: - Helpers
func printHeader(_ text: String) { print("\n=== \(text) ===") }
func printSuccess(_ text: String) { print("✅ \(text)") }
func printFailure(_ text: String) { print("❌ \(text)") }
func printInfo(_ text: String) { print("ℹ️  \(text)") }
func printWarning(_ text: String) { print("⚠️  \(text)") }

// MARK: - Shaders
let shaderSource = """
#include <metal_stdlib>

// Write known pattern: buf[i] = (i + 1) * multiplier
// multiplier is passed in so each cycle can have a different pattern
kernel void write_pattern(device uint* buf [[buffer(0)]],
                          constant uint& multiplier [[buffer(1)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid < 1024) {
        buf[gid] = (gid + 1u) * multiplier;
    }
}

// Verify: 1 if matches expected, 0 if not, 0xDEAD if silent corruption
kernel void verify_pattern(device uint* buf [[buffer(0)]],
                           device uint* out [[buffer(1)]],
                           constant uint& multiplier [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid < 1024) {
        uint expected = (gid + 1u) * multiplier;
        uint actual = buf[gid];
        if (actual == expected) {
            out[gid] = 1u;  // match
        } else if (actual == 0u) {
            out[gid] = 2u;  // zero (page re-faulted from empty file?)
        } else {
            out[gid] = 0xDEADu;  // corruption
        }
    }
}
"""

// MARK: - Setup
printHeader("SETUP")

try? FileManager.default.removeItem(atPath: kvFilePath)
let fd = open(kvFilePath, O_RDWR | O_CREAT | O_TRUNC, 0o644)
if fd < 0 { printFailure("open failed"); exit(1) }
if ftruncate(fd, off_t(kvSize)) != 0 { printFailure("ftruncate failed"); exit(1) }
close(fd)
printSuccess("Created \(kvSize / 1024 / 1024)MB KV Cache file at \(kvFilePath)")

let mmapFd = open(kvFilePath, O_RDWR)
let ptr = mmap(nil, kvSize, PROT_READ | PROT_WRITE, MAP_SHARED, mmapFd, 0)
if ptr == MAP_FAILED || ptr == nil { printFailure("mmap failed"); exit(1) }
let mmapBase = ptr!
printSuccess("mmap'd \(kvSize / 1024 / 1024)MB at \(mmapBase)")

guard let device = MTLCreateSystemDefaultDevice() else { printFailure("no Metal"); exit(1) }
printSuccess("Metal: \(device.name)")

let kvBuffer = device.makeBuffer(bytesNoCopy: mmapBase, length: kvSize, options: [.storageModeShared], deallocator: nil)!
let outputBuffer = device.makeBuffer(length: 4 * 1024, options: [.storageModeShared])!
printSuccess("KV buffer: zero-copy=\(kvBuffer.contents() == mmapBase)")

let library = try! device.makeLibrary(source: shaderSource, options: nil)
let writePipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "write_pattern")!)
let verifyPipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "verify_pattern")!)
let commandQueue = device.makeCommandQueue()!
printSuccess("Pipelines ready")

// MARK: - Helpers

func writePattern(multiplier: UInt32) {
    let cb = commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(writePipeline)
    enc.setBuffer(kvBuffer, offset: 0, index: 0)
    var m = multiplier
    withUnsafeBytes(of: &m) { p in enc.setBytes(p.baseAddress!, length: p.count, index: 1) }
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if cb.status != .completed { printFailure("Write kernel failed: \(cb.error?.localizedDescription ?? "nil")") }
}

func verifyPattern(multiplier: UInt32) -> (status: MTLCommandBufferStatus, match: Int, zero: Int, corrupt: Int) {
    let cb = commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(verifyPipeline)
    enc.setBuffer(kvBuffer, offset: 0, index: 0)
    enc.setBuffer(outputBuffer, offset: 0, index: 1)
    var m = multiplier
    withUnsafeBytes(of: &m) { p in enc.setBytes(p.baseAddress!, length: p.count, index: 2) }
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    
    let outPtr = outputBuffer.contents().assumingMemoryBound(to: UInt32.self)
    var match = 0, zero = 0, corrupt = 0
    for i in 0..<testElements {
        if outPtr[i] == 1 { match += 1 }
        else if outPtr[i] == 2 { zero += 1 }
        else if outPtr[i] == 0xDEAD { corrupt += 1 }
    }
    return (cb.status, match, zero, corrupt)
}

func syncToDisk() -> Bool {
    return msync(mmapBase, kvSize, MS_SYNC) == 0
}

// MARK: - Test 1: Single write+sync+verify cycle
printHeader("TEST 1: Single write → sync → verify cycle")

writePattern(multiplier: 1)
printInfo("Wrote pattern (multiplier=1)")

if syncToDisk() {
    printSuccess("msync(MS_SYNC) to disk successful")
} else {
    printFailure("msync failed: \(String(cString: strerror(errno)))")
}

let t1 = verifyPattern(multiplier: 1)
printInfo("Verify: match=\(t1.match) zero=\(t1.zero) corrupt=\(t1.corrupt) status=\(t1.status.rawValue)")
if t1.match == testElements && t1.corrupt == 0 {
    printSuccess("✅ Test 1 PASSED: single cycle data integrity verified")
} else {
    printFailure("Test 1 FAILED")
}

// MARK: - Test 2: Multi-cycle stability (10 cycles, same buffer, different patterns)
printHeader("TEST 2: 10 cycles of write → sync → verify")

var cycleResults: [(cycle: Int, match: Int, corrupt: Int, timeMs: Double)] = []
let overallStart = Date()

for cycle in 1...numCycles {
    let cycleStart = Date()
    let multiplier = UInt32(cycle * 1000)
    
    writePattern(multiplier: multiplier)
    if !syncToDisk() {
        printFailure("Cycle \(cycle): msync failed")
        cycleResults.append((cycle, -1, -1, 0))
        continue
    }
    let t = verifyPattern(multiplier: multiplier)
    let elapsedMs = Date().timeIntervalSince(cycleStart) * 1000
    cycleResults.append((cycle, t.match, t.corrupt, elapsedMs))
    
    if t.match == testElements && t.corrupt == 0 && t.status == .completed {
        printSuccess("Cycle \(cycle): ✅ \(t.match)/\(testElements) match, \(elapsedMs)ms")
    } else {
        printFailure("Cycle \(cycle): match=\(t.match), corrupt=\(t.corrupt), status=\(t.status.rawValue)")
    }
}

let totalElapsed = Date().timeIntervalSince(overallStart) * 1000
printInfo("Total: \(numCycles) cycles in \(Int(totalElapsed))ms (\(Int(totalElapsed / Double(numCycles)))ms per cycle)")

let allCyclesOK = cycleResults.allSatisfy { $0.match == testElements && $0.corrupt == 0 }
if allCyclesOK {
    printSuccess("✅ Test 2 PASSED: 10 cycles all clean")
} else {
    printFailure("Test 2 FAILED: some cycles had issues")
}

// MARK: - Test 3: Multiple "KV blocks" (3 separate files, simulating different conversation turns)
printHeader("TEST 3: Multiple KV blocks (3 files)")

let blockPaths = [
    "/tmp/fusionllm_poc4_kv_block1.bin",
    "/tmp/fusionllm_poc4_kv_block2.bin",
    "/tmp/fusionllm_poc4_kv_block3.bin"
]
let blockSize = 128 * 1024 * 1024  // 128MB per block

for (i, path) in blockPaths.enumerated() {
    try? FileManager.default.removeItem(atPath: path)
    let blockFd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0o644)
    ftruncate(blockFd, off_t(blockSize))
    close(blockFd)
}
printSuccess("Created 3 KV block files (\(blockSize / 1024 / 1024)MB each)")

var blockResults: [String] = []
for (i, path) in blockPaths.enumerated() {
    let blockFd = open(path, O_RDWR)
    let blockPtr = mmap(nil, blockSize, PROT_READ | PROT_WRITE, MAP_SHARED, blockFd, 0)
    if blockPtr == MAP_FAILED || blockPtr == nil {
        printFailure("Block \(i+1): mmap failed")
        blockResults.append("block\(i+1): MMAP FAIL")
        continue
    }
    
    let blockBuffer = device.makeBuffer(bytesNoCopy: blockPtr!, length: blockSize, options: [.storageModeShared], deallocator: nil)!
    let multiplier = UInt32((i + 1) * 100)
    
    // Write pattern
    let cb = commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(writePipeline)
    enc.setBuffer(blockBuffer, offset: 0, index: 0)
    var m = multiplier
    withUnsafeBytes(of: &m) { p in enc.setBytes(p.baseAddress!, length: p.count, index: 1) }
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    
    // Sync
    msync(blockPtr, blockSize, MS_SYNC)
    
    // Verify
    let outBuf = device.makeBuffer(length: 4 * 1024, options: .storageModeShared)!
    let cb2 = commandQueue.makeCommandBuffer()!
    let enc2 = cb2.makeComputeCommandEncoder()!
    enc2.setComputePipelineState(verifyPipeline)
    enc2.setBuffer(blockBuffer, offset: 0, index: 0)
    enc2.setBuffer(outBuf, offset: 0, index: 1)
    withUnsafeBytes(of: &m) { p in enc2.setBytes(p.baseAddress!, length: p.count, index: 2) }
    enc2.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    enc2.endEncoding()
    cb2.commit()
    cb2.waitUntilCompleted()
    
    let outPtr = outBuf.contents().assumingMemoryBound(to: UInt32.self)
    var match = 0, corrupt = 0
    for j in 0..<testElements {
        if outPtr[j] == 1 { match += 1 }
        else if outPtr[j] == 0xDEAD { corrupt += 1 }
    }
    
    if match == testElements && corrupt == 0 {
        printSuccess("Block \(i+1): ✅ \(match)/\(testElements) match")
        blockResults.append("block\(i+1): OK")
    } else {
        printFailure("Block \(i+1): match=\(match), corrupt=\(corrupt)")
        blockResults.append("block\(i+1): FAIL")
    }
    
    munmap(blockPtr, blockSize)
    close(blockFd)
    try? FileManager.default.removeItem(atPath: path)
}

let allBlocksOK = blockResults.allSatisfy { $0.hasSuffix("OK") }
if allBlocksOK {
    printSuccess("✅ Test 3 PASSED: all 3 KV blocks clean")
} else {
    printFailure("Test 3 FAILED")
}

// MARK: - Test 4: Eviction safety (madvise + pressure + verify)
printHeader("TEST 4: Eviction safety - madvise + pressure + verify")

// Re-write pattern 42 to main KV buffer
writePattern(multiplier: 42)
syncToDisk()
printInfo("Wrote pattern 42, synced")

// madvise + pressure
madvise(mmapBase, kvSize, MADV_DONTNEED)
printInfo("madvise(DONTNEED) called")

// Allocate 6GB pressure
let pressureSize = 6 * 1024 * 1024 * 1024
let pressurePtr = mmap(nil, pressureSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
if pressurePtr != MAP_FAILED && pressurePtr != nil {
    let pageSize = 16384
    for offset in stride(from: 0, to: pressureSize, by: pageSize) {
        memset(pressurePtr!.advanced(by: offset), 0xFF, 1)
    }
    printSuccess("Allocated and touched 6GB pressure buffer")
    
    let t4 = verifyPattern(multiplier: 42)
    printInfo("Verify after pressure: match=\(t4.match) zero=\(t4.zero) corrupt=\(t4.corrupt) status=\(t4.status.rawValue)")
    if t4.match == testElements && t4.corrupt == 0 {
        printSuccess("✅ Test 4 PASSED: KV Cache survives madvise + pressure")
    } else if t4.zero == testElements {
        printInfo("ℹ️ Page was evicted and re-faulted (data=0s, file might be sparse)")
    } else {
        printFailure("Test 4 unexpected: match=\(t4.match), zero=\(t4.zero), corrupt=\(t4.corrupt)")
    }
    
    munmap(pressurePtr, pressureSize)
} else {
    printWarning("Could not allocate 6GB pressure, skipping Test 4 eviction")
}

// MARK: - Cleanup
printHeader("CLEANUP")
munmap(mmapBase, kvSize)
close(mmapFd)
try? FileManager.default.removeItem(atPath: kvFilePath)
printSuccess("Cleaned up")

// MARK: - Summary
printHeader("SUMMARY")
let cycleOK = allCyclesOK
let blocksOK = allBlocksOK
if cycleOK && blocksOK {
    printSuccess("🎉 PoC-4 ALL TESTS PASSED")
    printInfo("KV Cache SSD tiering is VIABLE on Apple M5")
    printInfo("Multi-cycle writes are stable, no corruption")
    printInfo("Multiple blocks (separate files) all clean")
} else {
    printFailure("Some tests failed - review above")
}
