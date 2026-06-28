// PoC-1 Page Fault Test: Verify GPU access after page eviction
// Goal: Determine what happens when GPU accesses mmap'd region
//       after madvise(DONTNEED) / msync(MS_INVALIDATE) / memory pressure
//
// Critical question: Will Metal panic? Timeout? Silent error? Or work correctly?
//
// Target: Apple M5 MacBook Air 16GB

import Foundation
import Metal

// MARK: - Configuration
let filePath = "/tmp/fusionllm_poc1_pf_test.bin"
let fileSize: Int = 4 * 1024 * 1024 * 1024  // 4GB sparse file
let mlockSize: Int = 1 * 1024 * 1024 * 1024 // 1GB mlocked region
let testElements: Int = 1024
let fillValue: UInt32 = 0xAA

// MARK: - Helpers
func printHeader(_ text: String) { print("\n=== \(text) ===") }
func printSuccess(_ text: String) { print("✅ \(text)") }
func printFailure(_ text: String) { print("❌ \(text)") }
func printInfo(_ text: String) { print("ℹ️  \(text)") }
func printWarning(_ text: String) { print("⚠️  \(text)") }

// Shader: classify what GPU sees
let shaderSource = """
#include <metal_stdlib>

kernel void fill_buffer(device uint* buf [[buffer(0)]],
                        constant uint& value [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid < 1024) {
        buf[gid] = value;
    }
}

// Classify what's in the buffer:
//   1 = data preserved (still 0xAA, page wasn't actually freed)
//   2 = re-faulted from file (now 0, file was 0 originally)
//   0xDEAD = silent corruption (something else entirely)
kernel void classify_buffer(device uint* buf [[buffer(0)]],
                            device uint* out [[buffer(1)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid < 1024) {
        uint v = buf[gid];
        if (v == 0xAA) {
            out[gid] = 1u;
        } else if (v == 0u) {
            out[gid] = 2u;
        } else {
            out[gid] = 0xDEADu;
        }
    }
}
"""

// MARK: - Setup
printHeader("SETUP: Create file + mmap + mlock + fill 0xAA")

// Create file
try? FileManager.default.removeItem(atPath: filePath)
let fd = open(filePath, O_RDWR | O_CREAT | O_TRUNC, 0o644)
if fd < 0 { printFailure("open failed: \(String(cString: strerror(errno)))"); exit(1) }
if ftruncate(fd, off_t(fileSize)) != 0 { printFailure("ftruncate failed"); exit(1) }
close(fd)
printSuccess("Created 4GB file")

// mmap
let mmapFd = open(filePath, O_RDWR)
let ptr = mmap(nil, fileSize, PROT_READ | PROT_WRITE, MAP_SHARED, mmapFd, 0)
if ptr == MAP_FAILED || ptr == nil { printFailure("mmap failed"); exit(1) }
let mmapBase = ptr!
printSuccess("mmap'd 4GB at \(mmapBase)")

// mlock first 1GB
memset(mmapBase, 0, mlockSize)  // touch pages first
if mlock(mmapBase, mlockSize) != 0 {
    printWarning("mlock failed: \(String(cString: strerror(errno)))")
} else {
    printSuccess("mlock'd 1GB")
}

// Metal device
guard let device = MTLCreateSystemDefaultDevice() else {
    printFailure("No Metal device"); exit(1)
}
printSuccess("Metal device: \(device.name) (unified memory: \(device.hasUnifiedMemory))")

// Create Buffer A (in mlock region)
let bufferAPtr = mmapBase
guard let bufferA = device.makeBuffer(bytesNoCopy: bufferAPtr, length: mlockSize, options: [.storageModeShared], deallocator: nil) else {
    printFailure("makeBuffer A failed"); exit(1)
}
printSuccess("Buffer A: \(bufferA.length / 1024 / 1024)MB, zero-copy: \(bufferA.contents() == bufferAPtr)")

// Output buffer (separate, NOT from mmap)
guard let outputBuffer = device.makeBuffer(length: 4 * 1024, options: [.storageModeShared]) else {
    printFailure("makeBuffer output failed"); exit(1)
}
printSuccess("Output buffer: 4KB (separate, for classify results)")

// Compile shader
let library: MTLLibrary
do {
    library = try device.makeLibrary(source: shaderSource, options: nil)
} catch {
    printFailure("Shader compile failed: \(error)"); exit(1)
}
guard let fillFunction = library.makeFunction(name: "fill_buffer") else { printFailure("fill_buffer not found"); exit(1) }
guard let classifyFunction = library.makeFunction(name: "classify_buffer") else { printFailure("classify_buffer not found"); exit(1) }
printSuccess("Shader compiled")

let commandQueue = device.makeCommandQueue()!
let fillPipeline = try! device.makeComputePipelineState(function: fillFunction)
let classifyPipeline = try! device.makeComputePipelineState(function: classifyFunction)

// Fill Buffer A with 0xAA
do {
    let cb = commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(fillPipeline)
    enc.setBuffer(bufferA, offset: 0, index: 0)
    var v = fillValue
    withUnsafeBytes(of: &v) { p in enc.setBytes(p.baseAddress!, length: p.count, index: 1) }
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if cb.status == .completed {
        printSuccess("Buffer A filled with 0xAA")
    } else {
        printFailure("Fill failed: \(cb.error?.localizedDescription ?? "nil")"); exit(1)
    }
}

// Verify via CPU
let bufferAUInt32 = bufferAPtr.assumingMemoryBound(to: UInt32.self)
var initialCorrect = 0
for i in 0..<testElements {
    if bufferAUInt32[i] == 0xAA { initialCorrect += 1 }
}
printInfo("Initial CPU read: \(initialCorrect)/\(testElements) elements are 0xAA")
if initialCorrect != testElements { printFailure("Initial fill failed"); exit(1) }

// Helper: run classify kernel and report
func runClassifyKernel(label: String) -> (status: MTLCommandBufferStatus, preserved: Int, reFaulted: Int, corrupt: Int) {
    let cb = commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(classifyPipeline)
    enc.setBuffer(bufferA, offset: 0, index: 0)
    enc.setBuffer(outputBuffer, offset: 0, index: 1)
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    
    var preserved = 0, reFaulted = 0, corrupt = 0
    let outPtr = outputBuffer.contents().assumingMemoryBound(to: UInt32.self)
    for i in 0..<testElements {
        if outPtr[i] == 1 { preserved += 1 }
        else if outPtr[i] == 2 { reFaulted += 1 }
        else if outPtr[i] == 0xDEAD { corrupt += 1 }
    }
    
    printInfo("[\(label)] preserved=\(preserved), reFaulted=\(reFaulted), corrupt=\(corrupt)")
    if cb.status != .completed {
        printFailure("[\(label)] kernel status: \(cb.status.rawValue), error: \(cb.error?.localizedDescription ?? "nil")")
    }
    return (cb.status, preserved, reFaulted, corrupt)
}

// MARK: - Test 1: madvise(DONTNEED) + immediate GPU read
printHeader("TEST 1: madvise(DONTNEED) + immediate GPU read")

let r1 = madvise(bufferAPtr, mlockSize, MADV_DONTNEED)
if r1 == 0 {
    printSuccess("madvise(DONTNEED) returned success")
} else {
    printFailure("madvise(DONTNEED) failed: \(String(cString: strerror(errno)))")
}

let t1 = runClassifyKernel(label: "Test 1")
if t1.status == .completed {
    if t1.corrupt > 0 {
        printFailure("❌ SILENT CORRUPTION: \(t1.corrupt) elements have unexpected values")
    } else if t1.preserved == testElements {
        printSuccess("✅ Data preserved (page not actually freed by madvise - macOS lazy)")
    } else if t1.reFaulted == testElements {
        printSuccess("✅ Page re-faulted from file (data is 0s, as expected)")
    } else {
        printInfo("⚠️ Mixed result")
    }
} else {
    printFailure("❌ Kernel failed: status=\(t1.status.rawValue)")
}

// MARK: - Test 2: re-mlock + re-fill + force free with allocation
printHeader("TEST 2: Force free via 8GB allocation + GPU read")

// Re-mlock and re-fill
printInfo("Re-mlock and re-fill 0xAA...")
munlock(bufferAPtr, mlockSize)
mlock(bufferAPtr, mlockSize)
do {
    let cb = commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(fillPipeline)
    enc.setBuffer(bufferA, offset: 0, index: 0)
    var v = fillValue
    withUnsafeBytes(of: &v) { p in enc.setBytes(p.baseAddress!, length: p.count, index: 1) }
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if cb.status != .completed { printFailure("Re-fill failed"); exit(1) }
}
printSuccess("Re-mlock + re-fill done")

// madvise
madvise(bufferAPtr, mlockSize, MADV_DONTNEED)
printInfo("madvise(DONTNEED) done, now allocating 8GB to force actual free...")

// Allocate 8GB to create memory pressure
var pressureBuffer: UnsafeMutableRawPointer? = nil
do {
    pressureBuffer = mmap(nil, 8 * 1024 * 1024 * 1024, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
    if pressureBuffer == MAP_FAILED || pressureBuffer == nil {
        printWarning("Could not allocate 8GB, trying 4GB...")
        pressureBuffer = mmap(nil, 4 * 1024 * 1024 * 1024, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
    }
    if pressureBuffer != MAP_FAILED && pressureBuffer != nil {
        memset(pressureBuffer!, 0xFF, 4 * 1024 * 1024 * 1024)  // touch it
        printSuccess("Allocated pressure buffer, touched 4GB to force page reclaim")
    } else {
        printWarning("Could not allocate pressure buffer: \(String(cString: strerror(errno)))")
    }
}

// Now read with GPU
let t2 = runClassifyKernel(label: "Test 2")
if t2.status == .completed {
    if t2.corrupt > 0 {
        printFailure("❌ SILENT CORRUPTION under memory pressure: \(t2.corrupt) elements")
    } else if t2.preserved == testElements {
        printInfo("ℹ️ Data still preserved (madvise is very lazy on macOS)")
    } else if t2.reFaulted == testElements {
        printSuccess("✅ Page re-faulted from file (real page free happened)")
    } else {
        printInfo("⚠️ Mixed result")
    }
} else {
    printFailure("❌ Kernel failed under memory pressure: status=\(t2.status.rawValue)")
}

// Free pressure buffer
if pressureBuffer != nil && pressureBuffer != MAP_FAILED {
    munmap(pressureBuffer, 8 * 1024 * 1024 * 1024)
}

// MARK: - Test 3: msync(MS_INVALIDATE) + GPU read
printHeader("TEST 3: msync(MS_INVALIDATE) + GPU read")

// Re-fill
printInfo("Re-fill 0xAA...")
do {
    let cb = commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(fillPipeline)
    enc.setBuffer(bufferA, offset: 0, index: 0)
    var v = fillValue
    withUnsafeBytes(of: &v) { p in enc.setBytes(p.baseAddress!, length: p.count, index: 1) }
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
}

// msync to invalidate
let msyncResult = msync(bufferAPtr, mlockSize, MS_INVALIDATE)
if msyncResult == 0 {
    printSuccess("msync(MS_INVALIDATE) returned success")
} else {
    printFailure("msync(MS_INVALIDATE) failed: \(String(cString: strerror(errno)))")
}

let t3 = runClassifyKernel(label: "Test 3")
if t3.status == .completed {
    if t3.corrupt > 0 {
        printFailure("❌ SILENT CORRUPTION after msync: \(t3.corrupt) elements")
    } else if t3.preserved == testElements {
        printInfo("ℹ️ Data still preserved (msync didn't free)")
    } else if t3.reFaulted == testElements {
        printSuccess("✅ msync triggered re-fault (data is 0s from file)")
    } else {
        printInfo("⚠️ Mixed result")
    }
} else {
    printFailure("❌ Kernel failed after msync: status=\(t3.status.rawValue)")
}

// MARK: - Cleanup
printHeader("CLEANUP")
munlock(mmapBase, mlockSize)
munmap(mmapBase, fileSize)
close(mmapFd)
try? FileManager.default.removeItem(atPath: filePath)
printSuccess("Cleaned up")

// MARK: - Summary
printHeader("SUMMARY")
let allCompleted = t1.status == .completed && t2.status == .completed && t3.status == .completed
let noCorruption = t1.corrupt == 0 && t2.corrupt == 0 && t3.corrupt == 0
if allCompleted && noCorruption {
    printSuccess("🎉 ALL TESTS PASSED: No crashes, no timeouts, no silent corruption")
    printInfo("FusionLLM architecture is VIABLE on Apple M5")
} else {
    if !allCompleted {
        printFailure("Some kernels FAILED - check error output above")
    }
    if !noCorruption {
        printFailure("SILENT CORRUPTION detected - architecture needs revision")
    }
}
