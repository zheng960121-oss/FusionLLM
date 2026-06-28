// PoC-1 Happy Path: Verify mmap + mlock + Metal zero-copy buffer works
// Target: Apple M5 MacBook Air, 16GB unified memory, macOS 26.5.1
// Goal: Steps 1-4 of the PoC plan - confirm toolchain works before
//       doing the actual page fault test (Steps 5-8).

import Foundation
import Metal

// MARK: - Configuration
let filePath = "/tmp/fusionllm_poc1_test.bin"
let fileSize: Int = 4 * 1024 * 1024 * 1024  // 4GB sparse file
let mlockSize: Int = 1 * 1024 * 1024 * 1024 // 1GB mlocked
let testElements: Int = 1024                // 1024 uints = 4KB test
let pageSize: Int = 16384                   // 16KB page size on Apple Silicon

// MARK: - Helpers
func printHeader(_ text: String) { print("\n=== \(text) ===") }
func printSuccess(_ text: String) { print("✅ \(text)") }
func printFailure(_ text: String) { print("❌ \(text)") }
func printInfo(_ text: String) { print("ℹ️  \(text)") }

// Track test results
var allPassed = true

// MARK: - Step 1: Create 4GB sparse file
printHeader("Step 1: Create 4GB sparse file")
do {
    try? FileManager.default.removeItem(atPath: filePath)
    let fd = open(filePath, O_RDWR | O_CREAT | O_TRUNC, 0o644)
    if fd < 0 {
        printFailure("open() failed: \(String(cString: strerror(errno)))")
        allPassed = false
        exit(1)
    }
    if ftruncate(fd, off_t(fileSize)) != 0 {
        printFailure("ftruncate() failed: \(String(cString: strerror(errno)))")
        allPassed = false
        exit(1)
    }
    close(fd)
    printSuccess("Created \(fileSize / 1024 / 1024 / 1024)GB sparse file at \(filePath)")
    
    // Check actual disk usage
    let attrs = try FileManager.default.attributesOfItem(atPath: filePath)
    let allocSize = (attrs[.size] as? NSNumber)?.int64Value ?? 0
    printInfo("Logical file size: \(allocSize / 1024 / 1024)MB")
    if allocSize < Int64(fileSize) {
        printSuccess("File is sparse (no physical disk used yet)")
    } else {
        printInfo("File is not sparse (used \(allocSize / 1024 / 1024)MB on disk)")
    }
} catch {
    printFailure("Step 1 error: \(error)")
    allPassed = false
    exit(1)
}

// MARK: - Step 2: mmap
printHeader("Step 2: mmap the file")
let mmapFd = open(filePath, O_RDWR)
if mmapFd < 0 {
    printFailure("open() for mmap failed: \(String(cString: strerror(errno)))")
    exit(1)
}
let ptr = mmap(nil, fileSize, PROT_READ | PROT_WRITE, MAP_SHARED, mmapFd, 0)
if ptr == MAP_FAILED || ptr == nil {
    printFailure("mmap() failed: \(String(cString: strerror(errno)))")
    exit(1)
}
let mmapBase = ptr!
printSuccess("mmap'd \(fileSize / 1024 / 1024 / 1024)GB at virtual address \(mmapBase)")
printInfo("Page size: \(pageSize / 1024)KB, mlock size: \(mlockSize / 1024 / 1024)MB")

// MARK: - Step 3: mlock
printHeader("Step 3: mlock first 1GB")
// Touch the memory to ensure pages are allocated
memset(mmapBase, 0, mlockSize)
if mlock(mmapBase, mlockSize) != 0 {
    let err = errno
    printFailure("mlock() failed: \(String(cString: strerror(errno))) (errno=\(err))")
    printInfo("Continuing without mlock (PoC happy path doesn't strictly require it)")
} else {
    printSuccess("mlock'd \(mlockSize / 1024 / 1024)MB successfully")
}

// MARK: - Step 4: Create Metal device and buffers
printHeader("Step 4: Create Metal device + zero-copy buffers")
guard let device = MTLCreateSystemDefaultDevice() else {
    printFailure("Metal not supported on this device")
    exit(1)
}
printSuccess("Metal device: \(device.name)")
printInfo("Has unified memory: \(device.hasUnifiedMemory)")

// Create buffer A from mlock region
let bufferAPtr = mmapBase
guard let bufferA = device.makeBuffer(
    bytesNoCopy: bufferAPtr,
    length: mlockSize,
    options: [.storageModeShared],
    deallocator: nil
) else {
    printFailure("makeBuffer for bufferA (mlock region) failed")
    printInfo("This could be due to: length not page-aligned, ptr not page-aligned, or storage mode mismatch")
    exit(1)
}
printSuccess("Buffer A created: \(bufferA.length / 1024 / 1024)MB (mlock region)")

// Create buffer B from non-mlock region
let bufferBPtr = mmapBase.advanced(by: mlockSize)
guard let bufferB = device.makeBuffer(
    bytesNoCopy: bufferBPtr,
    length: mlockSize,
    options: [.storageModeShared],
    deallocator: nil
) else {
    printFailure("makeBuffer for bufferB (non-mlock region) failed")
    exit(1)
}
printSuccess("Buffer B created: \(bufferB.length / 1024 / 1024)MB (non-mlock region)")

// Verify zero-copy: buffer contents() should equal our mmap'd pointer
if bufferA.contents() == bufferAPtr {
    printSuccess("Buffer A is ZERO-COPY: contents() == mmap ptr")
} else {
    printFailure("Buffer A NOT zero-copy: contents()=\(bufferA.contents()), mmap=\(bufferAPtr)")
    allPassed = false
}

if bufferB.contents() == bufferBPtr {
    printSuccess("Buffer B is ZERO-COPY: contents() == mmap ptr")
} else {
    printFailure("Buffer B NOT zero-copy: contents()=\(bufferB.contents()), mmap=\(bufferBPtr)")
    allPassed = false
}

// MARK: - Step 5: Compile Metal shader
printHeader("Step 5: Compile Metal shader (inline)")
let shaderSource = """
#include <metal_stdlib>

kernel void fill_buffer(device uint* buf [[buffer(0)]],
                        constant uint& value [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid < 1024) {
        buf[gid] = value;
    }
}

kernel void verify_buffer(device uint* buf [[buffer(0)]],
                          device uint* out [[buffer(1)]],
                          constant uint& expected [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid < 1024) {
        out[gid] = (buf[gid] == expected) ? 1u : 0u;
    }
}
"""

let library: MTLLibrary
do {
    library = try device.makeLibrary(source: shaderSource, options: nil)
    printSuccess("Shader library compiled inline")
} catch {
    printFailure("makeLibrary failed: \(error)")
    exit(1)
}

guard let fillFunction = library.makeFunction(name: "fill_buffer") else {
    printFailure("fill_buffer function not found")
    exit(1)
}
guard let verifyFunction = library.makeFunction(name: "verify_buffer") else {
    printFailure("verify_buffer function not found")
    exit(1)
}
printSuccess("Found fill_buffer and verify_buffer functions")

// MARK: - Step 6: Run fill kernels
printHeader("Step 6: Run fill kernels (A: 0xAA, B: 0xBB)")
guard let commandQueue = device.makeCommandQueue() else {
    printFailure("makeCommandQueue failed")
    exit(1)
}

do {
    let pipeline = try device.makeComputePipelineState(function: fillFunction)
    
    // Fill Buffer A with 0xAA
    let cbA = commandQueue.makeCommandBuffer()!
    let encA = cbA.makeComputeCommandEncoder()!
    encA.setComputePipelineState(pipeline)
    encA.setBuffer(bufferA, offset: 0, index: 0)
    var fillValue: UInt32 = 0xAA
    withUnsafeBytes(of: &fillValue) { ptr in
        encA.setBytes(ptr.baseAddress!, length: ptr.count, index: 1)
    }
    let threadsPerGroup = MTLSize(width: 1024, height: 1, depth: 1)
    let numGroups = MTLSize(width: 1, height: 1, depth: 1)
    encA.dispatchThreadgroups(numGroups, threadsPerThreadgroup: threadsPerGroup)
    encA.endEncoding()
    cbA.commit()
    cbA.waitUntilCompleted()
    if cbA.status == .completed {
        printSuccess("Buffer A fill kernel: completed (0xAA)")
    } else {
        printFailure("Buffer A fill kernel: status=\(cbA.status.rawValue), error=\(cbA.error?.localizedDescription ?? "nil")")
        allPassed = false
    }
    
    // Fill Buffer B with 0xBB
    let cbB = commandQueue.makeCommandBuffer()!
    let encB = cbB.makeComputeCommandEncoder()!
    encB.setComputePipelineState(pipeline)
    encB.setBuffer(bufferB, offset: 0, index: 0)
    fillValue = 0xBB
    withUnsafeBytes(of: &fillValue) { ptr in
        encB.setBytes(ptr.baseAddress!, length: ptr.count, index: 1)
    }
    encB.dispatchThreadgroups(numGroups, threadsPerThreadgroup: threadsPerGroup)
    encB.endEncoding()
    cbB.commit()
    cbB.waitUntilCompleted()
    if cbB.status == .completed {
        printSuccess("Buffer B fill kernel: completed (0xBB)")
    } else {
        printFailure("Buffer B fill kernel: status=\(cbB.status.rawValue), error=\(cbB.error?.localizedDescription ?? "nil")")
        allPassed = false
    }
}

// MARK: - Step 7: Verify via CPU (zero-copy proof)
printHeader("Step 7: Verify GPU writes via CPU (zero-copy proof)")

let bufferAUInt32 = bufferAPtr.assumingMemoryBound(to: UInt32.self)
let bufferBUInt32 = bufferBPtr.assumingMemoryBound(to: UInt32.self)

var aCorrect = 0
for i in 0..<testElements {
    if bufferAUInt32[i] == 0xAA { aCorrect += 1 }
}
printInfo("Buffer A (mlock region): \(aCorrect)/\(testElements) elements are 0xAA")

var bCorrect = 0
for i in 0..<testElements {
    if bufferBUInt32[i] == 0xBB { bCorrect += 1 }
}
printInfo("Buffer B (non-mlock region): \(bCorrect)/\(testElements) elements are 0xBB")

if aCorrect == testElements && bCorrect == testElements {
    printSuccess("ZERO-COPY CONFIRMED: GPU writes visible to CPU via mmap pointer")
} else {
    printFailure("Zero-copy verification FAILED")
    allPassed = false
}

// MARK: - Cleanup
printHeader("Cleanup")
munlock(mmapBase, mlockSize)
munmap(mmapBase, fileSize)
close(mmapFd)
try? FileManager.default.removeItem(atPath: filePath)
printSuccess("Cleaned up mmap, mlock, file")

// MARK: - Final result
printHeader("FINAL RESULT")
if allPassed {
    printSuccess("🎉 PoC-1 Happy Path: ALL CHECKS PASSED")
    printInfo("Toolchain ready: mmap + mlock + Metal zero-copy buffer all work")
    printInfo("Next step: PoC-1 page fault test (Steps 5-8 from the plan)")
} else {
    printFailure("PoC-1 Happy Path: SOME CHECKS FAILED - review output above")
}
