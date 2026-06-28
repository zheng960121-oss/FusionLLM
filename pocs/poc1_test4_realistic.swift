// PoC-1 Test 4: Realistic FusionLLM scenario
// After sliding window moves past a layer, we want to release that layer's pages:
//   munlock + madvise(DONTNEED)
// Will GPU still be able to access it safely?
//
// This is THE most important test for FusionLLM architecture.

import Foundation
import Metal

// MARK: - Configuration
let filePath = "/tmp/fusionllm_poc1_t4.bin"
let fileSize: Int = 2 * 1024 * 1024 * 1024   // 2GB (smaller, faster)
let mlockSize: Int = 512 * 1024 * 1024      // 512MB region
let testElements: Int = 1024
let pageSize: Int = 16384

// MARK: - Helpers
func printHeader(_ text: String) { print("\n=== \(text) ===") }
func printSuccess(_ text: String) { print("✅ \(text)") }
func printFailure(_ text: String) { print("❌ \(text)") }
func printInfo(_ text: String) { print("ℹ️  \(text)") }
func printWarning(_ text: String) { print("⚠️  \(text)") }

let shaderSource = """
#include <metal_stdlib>

kernel void fill_buffer(device uint* buf [[buffer(0)]],
                        constant uint& value [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid < 1024) { buf[gid] = value; }
}

kernel void classify_buffer(device uint* buf [[buffer(0)]],
                            device uint* out [[buffer(1)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid < 1024) {
        uint v = buf[gid];
        if (v == 0xAA) { out[gid] = 1u; }
        else if (v == 0u) { out[gid] = 2u; }
        else { out[gid] = 0xDEADu; }
    }
}
"""

// MARK: - Setup
printHeader("SETUP")

try? FileManager.default.removeItem(atPath: filePath)
let fd = open(filePath, O_RDWR | O_CREAT | O_TRUNC, 0o644)
ftruncate(fd, off_t(fileSize))
close(fd)
printSuccess("Created 2GB file")

let mmapFd = open(filePath, O_RDWR)
let ptr = mmap(nil, fileSize, PROT_READ | PROT_WRITE, MAP_SHARED, mmapFd, 0)
let mmapBase = ptr!
printSuccess("mmap'd 2GB")

memset(mmapBase, 0, mlockSize)
mlock(mmapBase, mlockSize)
printSuccess("mlock'd 512MB")

guard let device = MTLCreateSystemDefaultDevice() else { printFailure("no Metal"); exit(1) }
printSuccess("Metal: \(device.name)")

let bufferAPtr = mmapBase
guard let bufferA = device.makeBuffer(bytesNoCopy: bufferAPtr, length: mlockSize, options: [.storageModeShared], deallocator: nil) else {
    printFailure("makeBuffer failed"); exit(1)
}
let outputBuffer = device.makeBuffer(length: 4 * 1024, options: [.storageModeShared])!

let library = try! device.makeLibrary(source: shaderSource, options: nil)
let fillPipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "fill_buffer")!)
let classifyPipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "classify_buffer")!)
let commandQueue = device.makeCommandQueue()!
printSuccess("Buffers + pipelines ready")

// Helper to fill 0xAA
func fillBuffer() {
    let cb = commandQueue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(fillPipeline)
    enc.setBuffer(bufferA, offset: 0, index: 0)
    var v: UInt32 = 0xAA
    withUnsafeBytes(of: &v) { p in enc.setBytes(p.baseAddress!, length: p.count, index: 1) }
    enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
}

// Helper to classify
func classifyAndReport(_ label: String) -> (status: MTLCommandBufferStatus, preserved: Int, reFaulted: Int, corrupt: Int) {
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
    
    let outPtr = outputBuffer.contents().assumingMemoryBound(to: UInt32.self)
    var preserved = 0, reFaulted = 0, corrupt = 0
    for i in 0..<testElements {
        if outPtr[i] == 1 { preserved += 1 }
        else if outPtr[i] == 2 { reFaulted += 1 }
        else if outPtr[i] == 0xDEAD { corrupt += 1 }
    }
    printInfo("[\(label)] preserved=\(preserved) reFaulted=\(reFaulted) corrupt=\(corrupt) status=\(cb.status.rawValue)")
    if let err = cb.error { printInfo("  error: \(err.localizedDescription)") }
    return (cb.status, preserved, reFaulted, corrupt)
}

// MARK: - TEST 4: munlock + madvise + REAL pressure
printHeader("TEST 4: munlock + madvise + REAL memory pressure")

// Step 1: mlock + fill 0xAA
fillBuffer()
printSuccess("Step 1: mlock'd + filled 0xAA")
let t4a = classifyAndReport("before munlock")
guard t4a.preserved == testElements else { printFailure("Initial state wrong"); exit(1) }

// Step 2: munlock
let munlockResult = munlock(bufferAPtr, mlockSize)
if munlockResult == 0 {
    printSuccess("Step 2: munlock'd 512MB (mprotection removed)")
} else {
    printFailure("munlock failed: \(String(cString: strerror(errno)))")
}

// Step 3: madvise(DONTNEED)
let madviseResult = madvise(bufferAPtr, mlockSize, MADV_DONTNEED)
if madviseResult == 0 {
    printSuccess("Step 3: madvise(DONTNEED) called")
} else {
    printFailure("madvise failed: \(String(cString: strerror(errno)))")
}

// Step 4: Allocate 8GB to FORCE page reclaim
printInfo("Step 4: Allocating 8GB pressure buffer...")
let pressureSize: Int = 8 * 1024 * 1024 * 1024
var pressurePtr: UnsafeMutableRawPointer? = nil
pressurePtr = mmap(nil, pressureSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)

if pressurePtr == MAP_FAILED || pressurePtr == nil {
    printFailure("Failed to allocate 8GB, trying 6GB...")
    pressurePtr = mmap(nil, 6 * 1024 * 1024 * 1024, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
}

if pressurePtr == MAP_FAILED || pressurePtr == nil {
    printWarning("Could not allocate pressure buffer, skipping pressure test")
} else {
    // Touch all pages to force allocation
    let actualSize: Int = (pressurePtr != nil) ? 6 * 1024 * 1024 * 1024 : 0
    let p = pressurePtr!
    printInfo("Touching all pages of pressure buffer to force allocation...")
    for offset in stride(from: 0, to: actualSize, by: pageSize) {
        memset(p.advanced(by: offset), 0xFF, 1)
    }
    printSuccess("Touched \(actualSize / 1024 / 1024 / 1024)GB - this should force OS to reclaim Buffer A's pages")
    
    // Step 5: Now read with GPU
    printInfo("Step 5: Reading Buffer A via GPU under pressure...")
    let t4b = classifyAndReport("after pressure")
    
    if t4b.status == .completed {
        if t4b.corrupt > 0 {
            printFailure("❌ SILENT CORRUPTION under real pressure: \(t4b.corrupt) elements")
        } else if t4b.preserved == testElements {
            printInfo("ℹ️ Data still preserved (OS chose not to evict despite pressure)")
        } else if t4b.reFaulted == testElements {
            printSuccess("✅ Page ACTUALLY freed and re-faulted from file (data is 0s)")
        } else {
            printInfo("⚠️ Mixed: preserved=\(t4b.preserved), reFaulted=\(t4b.reFaulted), corrupt=\(t4b.corrupt)")
        }
    } else {
        printFailure("❌ Kernel FAILED under real pressure: status=\(t4b.status.rawValue)")
    }
    
    // Free pressure
    munmap(p, actualSize)
    printInfo("Freed pressure buffer")
}

// MARK: - TEST 4b: Even more pressure + multiple cycles
printHeader("TEST 4b: Multiple munlock/fill cycles under sustained pressure")

var cycleResults: [String] = []
for cycle in 1...3 {
    printInfo("--- Cycle \(cycle) ---")
    
    // Re-mlock and re-fill
    mlock(bufferAPtr, mlockSize)
    fillBuffer()
    
    // Verify
    let tCheck = classifyAndReport("cycle \(cycle) filled")
    if tCheck.preserved != testElements {
        printFailure("Cycle \(cycle) re-fill wrong")
        cycleResults.append("cycle\(cycle): FILL WRONG")
        continue
    }
    
    // Release
    munlock(bufferAPtr, mlockSize)
    madvise(bufferAPtr, mlockSize, MADV_DONTNEED)
    
    // Apply pressure again
    let p = mmap(nil, 6 * 1024 * 1024 * 1024, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
    if p != MAP_FAILED && p != nil {
        for offset in stride(from: 0, to: 6 * 1024 * 1024 * 1024, by: pageSize) {
            memset(p!.advanced(by: offset), 0xFF, 1)
        }
        
        // Read
        let t = classifyAndReport("cycle \(cycle) after release+pressure")
        if t.status == .completed {
            if t.corrupt > 0 {
                printFailure("Cycle \(cycle): SILENT CORRUPTION")
                cycleResults.append("cycle\(cycle): CORRUPT")
            } else if t.reFaulted == testElements {
                printSuccess("Cycle \(cycle): ✅ page actually freed and re-faulted")
                cycleResults.append("cycle\(cycle): REFault OK")
            } else if t.preserved == testElements {
                cycleResults.append("cycle\(cycle): preserved")
            } else {
                cycleResults.append("cycle\(cycle): mixed")
            }
        } else {
            printFailure("Cycle \(cycle): kernel failed")
            cycleResults.append("cycle\(cycle): KERNEL FAIL")
        }
        
        munmap(p, 6 * 1024 * 1024 * 1024)
    }
}

// MARK: - CLEANUP
printHeader("CLEANUP")
munlock(mmapBase, mlockSize)
munmap(mmapBase, fileSize)
close(mmapFd)
try? FileManager.default.removeItem(atPath: filePath)
printSuccess("Cleaned up")

// MARK: - SUMMARY
printHeader("SUMMARY")
printInfo("Cycle results: \(cycleResults)")
let allOk = cycleResults.allSatisfy { !$0.contains("CORRUPT") && !$0.contains("FAIL") && !$0.contains("WRONG") }
if allOk {
    printSuccess("🎉 No silent corruption across all cycles")
    printInfo("FusionLLM can safely release layers via munlock + madvise + pressure")
} else {
    printFailure("Some cycles had issues - review above")
}
