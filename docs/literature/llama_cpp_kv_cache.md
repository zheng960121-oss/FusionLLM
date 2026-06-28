# llama.cpp KV Cache 实现笔记

**项目**：llama.cpp (main branch, ggml 0.15.3)
**commit**：27c8bb4 (2026-06-28)
**链接**：https://github.com/ggml-org/llama.cpp
**相关讨论**：https://github.com/ggml-org/llama.cpp/discussions/21961 (Paged KV cache)
**阅读日期**：2026-06-28

---

## 1. 当前 KV Cache 状态

llama.cpp **目前没有完整的 PagedAttention 实现**（截至 commit 27c8bb4），但有：

1. **KV cache 量化**（`cache_type_k` / `cache_type_v` 参数）
2. **Discussion #21961** 中的 paged KV cache 设计讨论（未合并到主分支）
3. **连续的 KV cache 内存布局**（layer-block 形式）

### 1.1 已有的 KV cache 量化（最成熟）

```bash
llama-cli -m model.gguf \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    ...
```

支持的类型：f16、f32、q4_0、q4_1、q5_0、q5_1、q8_0

**对 FusionLLM 的启示**：
- ✅ **直接可用**：Phase 3 可以先用 KV cache 量化（4x 压缩）
- ✅ 不需要重写 paged attention
- 💡 70B 模型 KV cache 在 4K 上下文 = 10.5GB，q4 量化后 = 2.6GB

### 1.2 Discussion #21961 (未合并的 Paged KV cache)

讨论链接：https://github.com/ggml-org/llama.cpp/discussions/21961
- 标题："Paged KV cache and scheduler: Phase 1 (design feedback requested)"
- 状态：设计阶段，未合并
- 基于 vLLM PagedAttention（Kwon et al., 2023）
- 目标：消除 unified KV cache 内部碎片，支持高并发 serving

**对 FusionLLM 的启示**：
- ⚠️ 这个 PR 还没合并到 llama.cpp 主分支
- 💡 我们可以参考这个设计做自己的实现
- ⚠️ 单流推理场景下，paged attention 收益有限（没有并发序列共享）

---

## 2. llama.cpp 现有 KV cache 内存布局

基于 `common/arg.cpp` 和 `src/llama.cpp` 代码分析：

```
KV cache 内存布局（每层）：
+--------------------------------+
| K[0] [head 0, head 1, ...]   |  ← 所有 head 的 K 向量连续存储
| K[1]                          |
| ...                           |
| K[seq_len-1]                  |
+--------------------------------+
| V[0]                          |  ← V 向量
| V[1]                          |
| ...                           |
| V[seq_len-1]                  |
+--------------------------------+
```

**特点**：
- 每层独立分配
- 连续内存（不是 paged）
- 大小 = `2 × n_layer × n_head × head_dim × seq_len × sizeof(type)`

**70B 模型 4K 上下文（fp16）**：
- 2 × 80 × 64 × 128 × 4096 × 2 = 10.7 GB
- 量化到 q8_0 = 5.4 GB
- 量化到 q4_0 = 2.7 GB

---

## 3. llama.cpp GGML Backend Buffer 机制

llama.cpp 通过 `ggml_backend_buffer_t` 抽象管理 buffer：

```cpp
// 创建 buffer
ggml_backend_buffer_t buf = ggml_backend_alloc_ctx(ctx, size);

// 映射到 host memory
void * host_ptr = ggml_backend_buffer_get_base(buf);

// 用作 tensor
struct ggml_tensor * t = ggml_new_tensor(ctx, type, n_dims, dims);
ggml_set_name(t, "weight");
```

**Metal backend 特定**：
- `ggml-metal.m` 用 `newBufferWithBytesNoCopy` 包装 mmap'd 内存
- 这跟我们 PoC-1 验证的机制**完全一样**

**对 FusionLLM 的启示**：
- ✅ 我们的 mlock hook 应该在 `ggml_backend_buffer_get_base` 之后
- ✅ 权重已经被 mmap，mlock 只是把虚拟地址的物理页 pin 住
- ✅ 无需修改 ggml 内部结构，只需在合适时机插入 mlock 调用

---

## 4. 集成路径

### 4.1 mlock hook 位置（推荐）

在 llama.cpp 的模型加载流程中，最佳 mlock 时机是：
1. 读取 model file 到 memory
2. 解析 GGUF metadata
3. 分配每个 tensor 的 buffer
4. **👉 在这里插入 mlock（per-tensor 或 per-layer）**
5. 上传到 GPU

具体代码位置：`src/llama.cpp` 中的 `llama_model_load_internal()`，大约在 `t->data` 分配之后。

### 4.2 `--fusion-driver` 编译开关

```cpp
// 在 common/arg.cpp 添加
if (getenv("FUSION_DRIVER") || cmd_params.fusion_driver) {
    // 启用 mlock
    fprintf(stderr, "FusionLLM driver: mlocking weight buffers\n");
    ggml_backend_buffer_set_fusion_driver(buf, true);
}
```

### 4.3 内存压力监测

llama.cpp 有 `n_threads` 和 `n_threads_batch` 控制 CPU 线程数。

可以添加：
- `fusion_window_size` (默认 6) - 滑动窗口层数
- `fusion_max_mlock_gb` (默认 6) - 最大 mlock 内存
- `fusion_kv_cache_path` (默认 /tmp) - KV cache 落盘路径

---

## 5. 关键 API 列表

| API | 用途 | 备注 |
|---|---|---|
| `ggml_backend_buffer_t` | 抽象 buffer | 我们的 hook 点 |
| `ggml_backend_buffer_get_base()` | 获取 host 指针 | mlock 需要这个 |
| `ggml_new_tensor()` | 创建 tensor | llama.cpp 调用 |
| `ggml_set_name()` | 设置 tensor 名称 | 调试用 |
| `ggml_mlock_supported()` | 检查 mlock 支持 | 运行时检查 |
| `mlock(ptr, size)` | pin 物理页 | macOS 已验证有效 |

---

## 6. ThunderLLAMA 参考

**项目**：https://github.com/lisihao/ThunderLLAMA
- Apple Silicon Paged Attention for llama.cpp
- 专门为 M1/M2/M3/M4 优化

**对 FusionLLM 的启示**：
- 💡 有人已经在做 Apple Silicon + paged attention
- ⚠️ 但他们的优化方向跟我们不同（他们做并发 serving，我们做 SSD offload）
- 可以参考他们的 Metal kernel 优化

---

## 7. 路径 c Phase 3 集成路径

```
1. 集成 llama.cpp 最新版 (已做)
2. 添加 FUSION_DRIVER compile flag
3. 在 llama_model_load_internal() 加 mlock hook
4. 验证 7B baseline 性能不退化
5. 集成 KV cache 量化 (cache_type_k/v q4_0)
6. 添加 KV cache 落盘逻辑（基于 PoC-4 已验证机制）
7. 添加 LRU 决策
8. 实测 70B + 32K 上下文
```

**预计工作量**：
- Step 1-2: 1 天
- Step 3-4: 2-3 天
- Step 5: 1 天
- Step 6-7: 5-7 天
- Step 8: 1 周（含调优）

**总计 4-6 周**，跟 Phase 3 估算一致。

---

## 8. 推荐阅读顺序

给后续开发者：

1. 📄 看 `src/llama.cpp` 的 `llama_model_load_internal()`（理解 mlock hook 插入点）
2. 📄 看 `ggml/src/ggml-metal.m` 的 `ggml_metal_buffer_init`（理解 buffer 包装）
3. 📄 看 `common/arg.cpp` 的 `cache_type_k/v` 参数（参考如何加新 CLI 参数）
4. 📄 看 `examples/llama-bench`（参考如何做性能 benchmark）
5. 📄 看 LMCache 的 `cache_engine.py`（参考 KV block 管理设计）

---

*笔记完成于 2026-06-28*
