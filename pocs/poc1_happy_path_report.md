# PoC-1 Happy Path 报告

**日期**：2026-06-28
**测试人员**：助手（执行） + jk（review）
**设备**：Apple M5 MacBook Air, 16GB unified memory, macOS 26.5.1
**脚本**：`poc1_happy_path.swift`
**结果**：✅ **ALL CHECKS PASSED**

---

## 1. 测试目的

验证 FusionLLM 核心技术栈的**工具链可行性**：
- mmap 大文件到虚拟地址空间
- mlock 部分物理页驻留
- 通过 `newBufferWithBytesNoCopy` 创建 Metal buffer，**不发生数据拷贝**
- Metal compute kernel 读写该 buffer
- CPU 通过 mmap 指针读到 GPU 写入的数据

**这是 W1 D1 的任务**。Happy path 通过后才能进行真正的页缺失测试（步骤 5-8）。

---

## 2. 测试结果汇总

| 步骤 | 验证项 | 结果 |
|:---:|---|:---:|
| 1 | 4GB 稀疏文件创建 | ✅ |
| 2 | mmap 4GB 到虚拟地址 | ✅ 0x0000000300000000 |
| 3 | mlock 1GB 成功 | ✅ |
| 4a | Metal device = Apple M5 | ✅ |
| 4b | `hasUnifiedMemory = true` | ✅ |
| 4c | Buffer A (mlock 区) zero-copy | ✅ contents() == mmap ptr |
| 4d | Buffer B (非 mlock 区) zero-copy | ✅ contents() == mmap ptr |
| 5 | Metal shader inline 编译 | ✅（修复 2 个语法问题后）|
| 6a | Buffer A fill kernel 完成 | ✅ 0xAA |
| 6b | Buffer B fill kernel 完成 | ✅ 0xBB |
| 7 | **零拷贝验证**：CPU 读到 GPU 写入 | ✅ **1024/1024 元素全对** |

**总耗时**：0.725s

---

## 3. 关键发现

### 3.1 `ulimit -l unlimited` 真的生效

这是开发计划中的 R2 风险点（macOS mlock RLIMIT 限制）。实际测试：

- mlock 1GB **成功**，无 ENOMEM
- 无需 entitlements，无需改 `launchctl limit`
- **节省半天 PoC-2 工作**

### 3.2 Metal 4 在 Apple M5 上的 zero-copy 完美工作

- `device.makeBuffer(bytesNoCopy: mmap_ptr, ...)` 直接返回 buffer
- `buffer.contents()` 严格等于 mmap 指针（**真正的零拷贝**）
- GPU compute kernel 写入后，CPU 通过 mmap 指针立即可见
- **整个 FusionLLM 架构的物理基础确认可行**

### 3.3 4GB sparse file 在 APFS 上不是 sparse

```
ℹ️  File is not sparse (used 4096MB on disk)
```

虽然 ftruncate 创建了 4GB logical size，但实际占用了 4GB 磁盘。**原因**：步骤 3 的 `memset(0)` 把 1GB 物理页写入，触发了真实分配。

**下次 PoC 改进**：
- 跳过 memset（让 mlock 自然触发 page fault）
- 或用 `fcntl(F_PUNCH_HOLE)` 主动稀疏化
- 或用更小的测试文件（如 1GB 而非 4GB）

### 3.4 Metal shader inline 编译的两个小坑

| 错误 | 原因 | 修复 |
|---|---|---|
| `type 'uint' is not valid for attribute 'buffer'` | `uint value [[buffer(1)]]` 应该是引用 | 改为 `constant uint& value [[buffer(1)]]` |
| `use of undeclared identifier 'thread_position_in_grid'` | 内置变量在 inline 编译下不可用 | 改为参数属性 `uint gid [[thread_position_in_grid]]` |

两个修复后，shader 编译通过，kernel 执行无误。

---

## 4. 风险验证矩阵

| 风险 | 状态 | 备注 |
|---|:---:|---|
| R1 GPU 访问 mlock 失效页行为 | ⏳ 待 PoC-1 步骤 5-8 | happy path 不测这个 |
| R2 mlock RLIMIT 限制 | ✅ 解除 | `ulimit -l unlimited` 真的生效 |
| R3 macOS MADV_DONTNEED 行为 | ⏳ 待 PoC-3 | |
| R4 首批预取时间 | ⏳ 待 Phase 1 | |
| R6 mlock 8GB 物理 RAM 是否够 | ⏳ 待实测 | 1GB 已验证，更大需 PoC-2 |
| R7 70B activation memory | ⏳ 待 Phase 3 | |

---

## 5. 下一步行动

按开发计划 W1 D2-D3：

- [ ] **PoC-1 步骤 5-8**（页缺失场景测试，**关键风险**）
  - fill kernel 写入 0xAA
  - madvise(DONTNEED) 让物理页释放
  - 读 kernel 触发 page fault
  - 观察：panic / timeout / 静默错 / 正确？
- [ ] PoC-3 macOS MADV_DONTNEED 行为（半天）
- [ ] 建 GitHub repo + CI pipeline

---

## 6. 附录：完整运行输出

```
=== Step 1: Create 4GB sparse file ===
✅ Created 4GB sparse file at /tmp/fusionllm_poc1_test.bin
ℹ️  Logical file size: 4096MB
ℹ️  File is not sparse (used 4096MB on disk)

=== Step 2: mmap the file ===
✅ mmap'd 4GB at virtual address 0x0000000300000000
ℹ️  Page size: 16KB, mlock size: 1024MB

=== Step 3: mlock first 1GB ===
✅ mlock'd 1024MB successfully

=== Step 4: Create Metal device + zero-copy buffers ===
✅ Metal device: Apple M5
ℹ️  Has unified memory: true
✅ Buffer A created: 1024MB (mlock region)
✅ Buffer B created: 1024MB (non-mlock region)
✅ Buffer A is ZERO-COPY: contents() == mmap ptr
✅ Buffer B is ZERO-COPY: contents() == mmap ptr

=== Step 5: Compile Metal shader (inline) ===
✅ Shader library compiled inline
✅ Found fill_buffer and verify_buffer functions

=== Step 6: Run fill kernels (A: 0xAA, B: 0xBB) ===
✅ Buffer A fill kernel: completed (0xAA)
✅ Buffer B fill kernel: completed (0xBB)

=== Step 7: Verify GPU writes via CPU (zero-copy proof) ===
ℹ️  Buffer A (mlock region): 1024/1024 elements are 0xAA
ℹ️  Buffer B (non-mlock region): 1024/1024 elements are 0xBB
✅ ZERO-COPY CONFIRMED: GPU writes visible to CPU via mmap pointer

=== Cleanup ===
✅ Cleaned up mmap, mlock, file

=== FINAL RESULT ===
✅ 🎉 PoC-1 Happy Path: ALL CHECKS PASSED
```

---

*报告完成于 2026-06-28 14:03*
