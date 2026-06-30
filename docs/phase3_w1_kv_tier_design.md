# Phase 3 W1: KV Cache 三层分级（设计文档）

**日期**: 2026-06-30 凌晨
**Phase**: Phase 3 / W1 (Selective KV Offload)
**状态**: 设计阶段 (起草), 实现待启动
**依赖**: 
- ✅ Phase 2 mlock (在 fusion_mmap_map 已验证)
- ✅ PoC-4 SSD offload (10 轮 write→mmap→GPU read 零错误)
- ✅ Phase 3 D0 (8B 4K prefill 455 t/s, path verified)

---

## 1. 业务目标

**Phase 3 W1 核心 KPI**: 在 16GB M5 Air 上跑 **8B-70B Q4 + 32K-128K 长 context**，通过 KV cache 三层分级：

| Tier | 存储 | 容量 | 用途 | 访问延迟 |
|:----:|:-----|:----:|:-----|:---------|
| **L0** | GPU Metal buffer | ~500MB | 当前 + sliding window 内 KV | <1μs |
| **L1** | CPU RAM (mlock'd) | ~2GB | 最近 1K-4K tokens KV | ~100ns |
| **L2** | SSD (mmap'd) | 6-24GB | 冷 KV blocks (按 layer × token-range 切分) | ~10ms (mmap read) |

**预算验证（已写进 `benchmarks/phase3_kv_cache_test_report.md`）**:
- 8B 32K q4_0 KV: ~50MB → 全在 L1
- 70B 32K q4_0 KV: ~2GB → 全在 L0/L1
- 70B 128K q4_0 + SSD: 6-12GB → L2 offload 必须

---

## 2. 设计架构

### 2.1 KV cache 切分单位

**Block 维度**: `[layer_id, start_token, end_token, kv_type]`（按 layer × token-range 切分）

```
Block granularity 决策:
- Per-layer × per-token-range (而非 per-cell 或 per-head)
- Block size: ~512 tokens (8B) 或 ~128 tokens (70B, 因为 70B KV dims 大)
- 切分原因:
  - L2 SSD mmap 按 page (4KB) 加载 → block 越大越 amortize mmap 代价
  - Attention 通常连续访问 token range → block 越大命中率高
  - Block 太小 → 元数据开销大（per-block index、SSD 路径）
  - Block 太大 → promote/demote 粒度粗，浪费 L1 内存
```

### 2.2 Tier 状态机

```
                    promote (cold→warm)
       L2 SSD  ────────────────────►  L1 CPU RAM
       (mmap)                       (mlock)
        ▲                                │
        │  demote (warm→cold, LRU)        │ promote (warm→hot, attention)
        │                                ▼
       demote                          L0 GPU
       (hot→cold, sliding)            (Metal buffer)
        ▲                                ▲
        │   promote                       │  promote (initial)
        │   (SSD→CPU)                    │  (model load)
        └────────────────────────────────┘

State transitions:
- 模型加载: 所有 KV block 在 L2 (SSD, mmap)
- prefill: 需要访问的 block promote 到 L0/L1
- decode: 当前 token 范围 promote 到 L0
- sliding window: 离开 window 的 block demote 到 L1/L2
```

### 2.3 与 llama.cpp KV cache 集成

llama.cpp KV cache API（已看 `src/llama-kv-cache.h`）:
- `kv_size`: KV cache 总容量
- `k_l`, `v_l`: per-layer K/V tensors (`cache_k_l%d`, `cache_v_l%d`)
- K shape: `[n_embd_k_gqa, kv_size]` (2D)
- V shape: `[n_embd_v_gqa, kv_size]` (2D)

**集成策略** (类似 Phase 6 DSpark forward 集成):
1. **不替换 llama_kv_cache**，而是 layer wrapper
2. 在 attention forward 时 hook: `kv_tier_manager.ensure_for_attention(layer_id, start_tok, end_tok)`
3. 确保 access 路径的 KV block 在正确 tier
4. L2 → L1 promotion 通过 mmap + mlock（已 PoC-4 验证）
5. L1 → L0 通过 host→device copy (`ggml_backend_tensor_copy`)

---

## 3. 关键 API 设计

### 3.1 数据结构（`src/fusion_kv_tier.h`）

```cpp
namespace fusion {

// Tier enum
enum class KVTier {
    L0_GPU = 0,    // Metal buffer (热)
    L1_CPU = 1,    // mlock'd RAM (warm)
    L2_SSD = 2,    // mmap'd SSD (cold)
};

// KV block location
struct KVBlockLocation {
    KVTier tier;
    void*  ptr;              // GPU ptr / CPU ptr / SSD file offset
    size_t size_bytes;
    int    layer_id;
    int    start_token;
    int    end_token;
    ggml_type kv_type;       // fp16 / q8_0 / q4_0
};

// Per-layer tier manager
class FusionKVTierManager {
public:
    // Constructor: bind to llama_kv_cache
    FusionKVTierManager(
        ggml_context* ctx,
        int n_layers,
        int kv_size,         // max tokens per layer
        int kv_head_dim,     // GQA: head_dim × n_head_kv
        ggml_type kv_type    // fp16 / q8_0 / q4_0
    );

    // === Promotion / Demotion ===
    bool promote_to_gpu(int layer_id, int start_tok, int end_tok);
    bool promote_to_cpu(int layer_id, int start_tok, int end_tok);
    bool demote_to_ssd(int layer_id, int start_tok, int end_tok);

    // === Queries ===
    KVBlockLocation get(int layer_id, int start_tok, int end_tok);
    bool is_in_gpu(int layer_id, int start_tok, int end_tok);

    // === Hook for llama.cpp attention forward ===
    // Call before each attention forward to ensure K/V is in correct tier.
    // Returns GPU pointer to KV block (may trigger L2→L0 promotion).
    ggml_tensor* ensure_for_attention(int layer_id, int start_tok, int end_tok);

    // === SSD backend ===
    // Set SSD base path (per-model KV cache file).
    void set_ssd_path(const std::string& ssd_dir);
    // Flush L1 → L2 (call after sliding window moves).
    bool flush_to_ssd(int layer_id, int start_tok, int end_tok);

    // === Stats ===
    struct Stats {
        size_t bytes_in_gpu;
        size_t bytes_in_cpu;
        size_t bytes_in_ssd;
        size_t n_promotions;     // L2→L1
        size_t n_evictions;     // L1→L2
        size_t n_gpu_copies;    // L1→L0
    };
    Stats get_stats() const;

private:
    // Internal block table (per layer × per block)
    // Indexed by [layer_id][block_id]
    std::vector<std::vector<KVBlockLocation>> blocks_;

    // GPU staging buffer (for L1→L0 copies)
    ggml_tensor* gpu_buffer_ = nullptr;
    int gpu_buffer_size_ = 0;

    // SSD file paths per layer
    std::vector<std::string> ssd_paths_;
};

}  // namespace fusion
```

### 3.2 集成 hooks（llama.cpp 改动点）

在 `src/llama-kv-cache.cpp` 加 3 个 hook:
1. `kv_tier_manager_->ensure_for_attention(layer, start_tok, end_tok)` 在 attention forward 前
2. `kv_tier_manager_->flush_to_ssd(layer, start_tok, end_tok)` 在 sliding window 移动后
3. `kv_tier_manager_->demote_to_ssd(layer, evicted_start, evicted_end)` 在 cell 被驱逐时

类似 Phase 6 DSpark forward hook（已经成功）。

---

## 4. 实现路径（3 个文件）

### 4.1 `src/fusion_kv_tier.h` (200 行)

**目的**: 公共 API + 数据结构定义

**关键内容**:
- `enum class KVTier`
- `struct KVBlockLocation`
- `class FusionKVTierManager`
- 命名空间 `fusion`

### 4.2 `src/fusion_kv_tier.cpp` (300 行)

**目的**: Tier manager 实现 + SSD offload + mlock

**关键内容**:
- Constructor: 初始化 block table + SSD paths
- `promote_to_*` / `demote_to_*`: 实现 tier 切换
- `flush_to_ssd`: L1 RAM → SSD 文件（写 + fsync）
- `ensure_for_attention`: L2→L1 promotion（mmap + mlock）
- 复用 `fusion_mmap_map` 已有 API（`fusion_mmap_mlock_layer` 等）

### 4.3 `tests/test-fusion-kv-tier.cpp` (150 行)

**目的**: 单元测试（mock llama_kv_cache）

**测试用例**:
1. **Constructor test**: 正确分配 block table
2. **promote/demote test**: L0↔L1↔L2 切换路径正确
3. **SSD roundtrip test**: L1→L2→L1 数据一致性（PoC-4 升级版）
4. **Concurrent access test**: 多线程 promote/demote 不冲突
5. **Large KV test**: 32K context 8B (50MB) 完整 offload
6. **Stats test**: get_stats() 数值正确

---

## 5. 关键技术决策

### 5.1 Block size 选择

| Decision | 选项 | 选择 | 理由 |
|:---------|:-----|:-----|:------|
| Block size (8B) | 128/256/512/1024 tokens | **512** | 平衡 SSD mmap 粒度 vs 元数据开销 |
| Block size (70B) | 64/128/256/512 tokens | **128** | 70B KV dims 大，block 大了 SSD offload 一次搬运多 |
| Block boundary | layer / kv_dim / token | **layer × token** | attention 按 layer × token range 访问 |

### 5.2 Promotion 策略

| Strategy | Description | 选择 |
|:---------|:------------|:----:|
| **Eager (prefill 后立刻)** | prefill 完所有访问的 block 都 promote | ✅ 默认 |
| **Lazy (按需)** | attention 时才 promote（要等 SSD mmap latency） | ❌ 太慢 |
| **Prefetch (异步)** | 后台线程预取下个 block | W2 优化 |

### 5.3 SSD offload 数据格式

**选择: raw bytes** (跟 llama.cpp KV cache layout 一致)
- 每个 layer 一个 SSD file: `kv_layer_<N>.bin`
- Block offset = `block_id × block_size_bytes`
- **不压缩**: 压缩收益小（KV 已经是 fp16/q8_0），CPU 解压开销抵消收益
- **不加密**: SSD 加密是 OS 层面（FileVault），不重复

### 5.4 SSD file 持久化 vs 临时

**选择: 临时 (`/tmp/fusion_kv_cache/`)**
- 每次 decode 重新生成 KV cache
- 模型切换时自动清理
- 优点: 不需要文件管理；缺点: 每次重启都要重做 prefill
- 未来 W4: 加持久化（同名 GGUF 自动 resume）

---

## 6. 性能预算

### 6.1 8B Q4_K_M 长 context 预估

| Context | L0 (GPU) | L1 (CPU) | L2 (SSD) | Total | Expected t/s |
|:-------:|:--------:|:--------:|:--------:|:-----:|:------------:|
| 4K | 50MB | 0 | 0 | 50MB | 455 prompt + 24 gen (D0 数据) |
| 8K | 100MB | 0 | 0 | 100MB | ~220 prompt + 23 gen |
| 32K | 400MB | 0 | 0 | 400MB | ~55 prompt + 20 gen |
| 128K | 1.6GB | 0 | 0 | 1.6GB | ~14 prompt + 15 gen (sliding window 优化) |

**8B 在 16GB Air 跑 128K context 估 11GB total**（4.7 GB 模型 + 1.6 GB KV + 5 GB OS/system）。需 sliding window 减负。

### 6.2 70B Q4_K_M 长 context 预估

| Context | L0 | L1 | L2 | SSD offload | Expected t/s |
|:-------:|:--:|:--:|:--:|:-----------:|:------------:|
| 4K | 200MB | 0 | 0 | 0 | 估算 30 prompt + 4 gen |
| 32K | 1.6GB | 0 | 0 | 0 | 估算 5 prompt + 2 gen |
| 128K | 6.4GB | 0 | 0 | ✓ (4 GB offload) | 估算 1.5 prompt + 1.5 gen (sliding) |

**70B 在 16GB Air 跑 128K 必须 KV SSD offload**（预算 15GB，留 1GB OS）。

---

## 7. 测试计划（W1 D5）

### 7.1 单元测试 (`tests/test-fusion-kv-tier.cpp`)

7 个 case, 30 个 EXPECT:

1. **Constructor + block table**
2. **promote L0→L1→L2 单调降级**
3. **demote L0→L1→L2 单调降级**
4. **L2→L1 promotion roundtrip (数据一致性)**
5. **Multi-layer 多 block 并发**
6. **Stats 统计正确**
7. **Large KV 32K 8B (50MB) 全 offload**

### 7.2 真机测试 (W3 D11-D15)

- 8B 32K prefill + generation (validate W1)
- 70B 4K baseline (Phase 3 D15 主 KPI)
- 70B 32K + SSD offload (Phase 3 终极 KPI)

---

## 8. 时间线

| Day | 任务 | 估算 |
|:----|:-----|:-----|
| D1 | `fusion_kv_tier.h` + 数据结构 | 4h |
| D2 | `fusion_kv_tier.cpp` skeleton + L0↔L1 | 4h |
| D3 | L2 SSD offload + mmap + mlock | 4h |
| D4 | llama.cpp fork 集成（3 个 hook）| 4h |
| D5 | `tests/test-fusion-kv-tier.cpp` (7 cases) | 4h |
| **D6** | **8B 32K 真机验证** | 4h |

**W1 总时长**: 1.5 周 (含真机验证)

---

## 9. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|:---|:---:|:---:|:---|
| mlock 失败（macOS 16GB 限制）| 高 | 中 | 用 `posix_madvise(MADV_SEQUENTIAL)` 不真 mlock |
| SSD latency p95 > 50ms | 中 | 高 | Block size 调大 + 异步 prefetch (W2) |
| Llama.cpp KV cache API 改变 | 中 | 高 | 接口抽象层 + version check |
| Block boundary 不对齐 attention | 中 | 中 | 自动对齐到 attention head boundary |

---

## 10. W1 完成后

- W2 (Sliding KV 调度): 同窗口 weights + KV cache
- W3 (70B 真机): 32K-128K 长 context benchmark
- W4 (持久化): KV cache resume across runs

---

*W1 设计完成 - 2026-06-30 02:40*
*核心 KPI: 70B Q4 + 32K-128K context 在 16GB M5 Air 跑通*
*实现路径: src/fusion_kv_tier.h/cpp + tests/ + llama.cpp fork 集成*