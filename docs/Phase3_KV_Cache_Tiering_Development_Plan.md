# Phase 3: KV Cache 分层（GPU/CPU/SSD）开发方案

**日期**：2026-06-29
**状态**：用户选定 A（Phase 3）作为 Phase 6 完成后的下一步
**目标**：在 16GB M5 Air 上跑 70B Q4 + 32K-128K 长上下文

---

## 1. 业务目标

**Path C 主线的核心 KPI**：跑 70B 模型 + 长上下文（32K-128K）在 16GB 设备上。

**唯一可行的内存预算**（来自 `benchmarks/phase3_kv_cache_test_report.md`）：

| Context | 70B + sliding window | 16GB 够吗？|
|:---:|:---:|:---:|
| 8K q4_0 | 9 GB | ✅ |
| 32K q4_0 + SSD | 15 GB | ✅ |
| 128K q4_0 + SSD | 15 GB | ✅ |

**关键洞察**：70B + 32K+ **必须** KV cache SSD offload，否则 OOM。

---

## 2. 当前状态

| 阶段 | 状态 |
|---|---|
| PoC-4 KV SSD 落盘机制 | ✅ 已验证（10 轮 write→mmap→GPU read 零错误） |
| 7B KV 实测（4K/8K × fp16/q4_0） | ✅ 已有数据 |
| 70B 内存预算推断 | ✅ 表已写 |
| **Selective offload 实现** | ❌ **缺失** |
| **Sliding KV cache（按访问模式）** | ❌ **缺失** |
| **32K-128K 真机测试** | ❌ **缺失** |

**当前可工作**：4K-8K 上下文（KV cache 全部在 RAM）
**目标可工作**：32K-128K 上下文（KV cache 分层：热 RAM + 冷 SSD）

---

## 3. 设计架构

### 3.1 三层 KV Cache 分层

```
┌──────────────────────────────────────────────────────┐
│              KV Cache Tiering                          │
├──────────────────────────────────────────────────────┤
│                                                       │
│  L0 (GPU Metal buffer) — ~500MB                       │
│  ├── Current decode position 的 K/V（最热）           │
│  └── Last N=128 tokens（sliding window）              │
│                                                       │
│  L1 (CPU RAM mlock'd) — ~2GB                         │
│  ├── Recent 1K-4K tokens 的 K/V                       │
│  └── Attention mask 大概率命中                        │
│                                                       │
│  L2 (SSD mmap'd) — 6-24GB on disk                    │
│  ├── 冷 K/V blocks（按 layer × token range 切分）      │
│  └── Lazy load：attention 需要时再 fetch              │
│                                                       │
│  L3 (compressed Q4_0) — 60% 内存节省                  │
│  └── 用于 L1/L2 中可量化的部分                        │
│                                                       │
└──────────────────────────────────────────────────────┘
```

### 3.2 与已有机制的协同

| 已有机制 | Phase 3 协同 |
|---|---|
| Phase 2 sliding window (weights mlock) | **同窗口**：weight layer k 和它的 KV cache 同时 mlock |
| Phase 6 DSpark spec decode | KV cache crop 时考虑 spec decoding 边界 |
| PoC-4 SSD mmap 机制 | **直接复用**：KV blocks 落盘到 SSD 后 mmap |

### 3.3 关键技术决策

| 决策点 | 选项 | 选择 |
|---|---|---|
| 分层单位 | Per-layer × Per-token-range | **Per-token-range**（更灵活） |
| 何时 offload | Prefill 后立刻 / Decode 时 lazy | **Prefill 后立刻**（避免 decode 时长延迟） |
| 何时 fetch L2 | 提前 / 按需 | **按需**（decode 时 attention mask 决定） |
| KV 量化 | fp16 / q4_0 | **q4_0 默认**（节省 60% 内存，~15-40% 慢可接受） |
| L2 压缩 | 无 / q4_0 | **q4_0**（SSD I/O 是瓶颈） |
| Sliding window 集成 | 独立 / 同窗口 | **同窗口**（weight + KV 共用 sliding） |

---

## 4. 任务分解（3 周 Sprint）

### 4.1 W1（5 天）— Selective KV Offload 实现

**核心模块**：`fusion_kv_tier.{h,cpp}`

#### D1-D2: KV Tier 管理器
- [ ] 数据结构：`KVTier` (gpu_ptr / cpu_ptr / ssd_path / size)
- [ ] API：`tier_promote(layer, token_range)` / `tier_demote(layer, token_range)`
- [ ] 集成 llama.cpp KV cache layout

#### D3-D4: SSD Offload
- [ ] 把 llama.cpp KV cache 写到 SSD 文件（per layer × per token range block）
- [ ] 用 PoC-4 已验证的 mmap 机制读取
- [ ] 处理 mmap 与 Metal buffer 的转换（host ↔ device copy）

#### D5: Tier Manager + 测试
- [ ] `KVTierManager` class：自动追踪哪些 K/V blocks 在哪一层
- [ ] 单元测试：promote/demote 路径正确性

### 4.2 W2（5 天）— Sliding KV 调度

**核心模块**：`fusion_kv_window.{h,cpp}`（对应 Phase 2 sliding window for weights）

#### D6-D7: Sliding Window for KV
- [ ] 复用 `fusion_window.h` 的 sliding 逻辑
- [ ] 同窗口：weight layer k mlock 时，对应的 KV token range 也 mlock
- [ ] 离开窗口的 KV 自动 demote 到 SSD

#### D8-D9: Attention 集成
- [ ] Hook llama.cpp attention forward，让它在 decode 前先 ensure K/V 在正确 tier
- [ ] L2 fetch 是异步（用 dispatch queue 不阻塞 compute）
- [ ] GPU buffer 不够时主动 offload

#### D10: Sliding KV 测试
- [ ] 单元测试：window 滑动时 K/V tier 正确变化
- [ ] 性能：滑动窗口切换不阻塞 decode

### 4.3 W3（5 天）— 长上下文真机测试 + 报告

#### D11-D12: 32K 上下文
- [ ] 加载 70B Q4_K_M target + 32K KV cache
- [ ] 验证：内存峰值 < 16GB
- [ ] 性能：decode t/s 在合理范围（vs 4K baseline）

#### D13-D14: 128K 上下文
- [ ] 同样测试 128K
- [ ] SSD offload 命中率统计
- [ ] Latency p50/p95

#### D15: 报告 + 文档
- [ ] `benchmarks/phase3_kv_tiering_report.md`
- [ ] 数据汇总：内存峰值、t/s、SSD 命中率、latency

---

## 5. 关键技术挑战

### 5.1 llama.cpp KV cache 与 Metal buffer 集成

llama.cpp 当前 KV cache 布局：
```cpp
// 每个 layer 有自己的 K/V tensor
model.layers[il].k_cache  // ggml_tensor, Metal 分配
model.layers[il].v_cache
```

**挑战**：Metal buffer 不能直接 mmap（设备内存）。需要在 host RAM 维护 shadow buffer。

**方案**：
1. Prefill 时：host 端维护 shadow K/V tensor
2. Layer forward 时：把 shadow K/V 复制到 Metal buffer
3. Decode 后：写回 shadow K/V
4. SSD offload：从 shadow 落盘

### 5.2 Sliding 窗口与 attention mask 协同

长上下文的 attention 是 O(n²)，但实际只关心：
- 当前位置（最新 token）
- Sliding window 内的 history（最近 N tokens）
- 系统 prompt（如果有）

**优化**：
- 计算 attention 时只 fetch 实际需要的 K/V blocks
- L2 SSD block 大小：~512KB（per layer × per token chunk）
- Attention scheduler 决定哪些 blocks 需要 load

### 5.3 与 Phase 2/6 协同

- **Phase 2 sliding window**：weights 和 KV cache 共享 sliding logic
- **Phase 6 DSpark**：spec decode 的 verify forward 一次访问多个 K/V block，KV tier manager 要一次 promote 多个
- **Prefill vs Decode**：prefill 时 KV cache 全部进 L2（SSD），decode 时 promote 当前访问的

---

## 6. 验收标准

### 6.1 必达指标

- [ ] 70B Q4 + 32K 上下文在 M5 Air 16GB 上跑通
- [ ] 内存峰值 < 16GB（预算 14GB，留 2GB 系统）
- [ ] Decode 速度 ≥ 1 t/s（vs 8K baseline 不低于 50%）
- [ ] SSD offload 命中率 > 90%（少访问 SSD 时才访问）
- [ ] SSD fetch latency p95 < 50ms（不阻塞 GPU compute）

### 6.2 加分指标

- [ ] 70B Q4 + 128K 上下文跑通（内存峰值 < 16GB）
- [ ] 与 Phase 6 DSpark 协同（spec decode + KV tiering）
- [ ] 量化精度验证：q4_0 KV 输出 perplexity < 5% 退化
- [ ] 跨模型泛化：同样代码跑 Qwen3-8B 也工作

---

## 7. 资源需求

### 7.1 硬件
- M5 MacBook Air 16GB（已有）
- 200GB+ 可用 SSD（70B Q4 + 长 KV cache + shadow buffer）
- 无需 GPU 服务器

### 7.2 模型
- `Qwen2.5-70B-Instruct-Q4_K_M` (~40GB) — 需下载
- 备用：`Qwen3-8B-Q4_K_M` (~5GB) — 快速验证用
- 备用：`meta-llama/Llama-3-70B-Instruct-Q4_K_M` — 跨架构测试

### 7.3 工具
- Python + gguf + safetensors（已有）
- llama.cpp kv cache 内部 API（已有 `llama_kv_cache`）
- PoC-4 落盘代码（已有）

---

## 8. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|:---:|:---:|:---|
| 70B 模型下载时间 > 4h | 高 | 低 | 先用 8B 验证架构，再下 70B |
| Metal ↔ host KV copy 是性能瓶颈 | 中 | 高 | profile + 用 `ggml_backend_buffer_copy` 异步 |
| SSD 访问 latency > 100ms | 中 | 中 | 大 block size（~2MB per layer）+ 异步 prefetch |
| KV q4_0 量化损失影响输出质量 | 中 | 中 | 默认 q4_0，perplexity < 5% 退化（已实测） |
| 32K 上下文 attention O(n²) 计算瓶颈 | 高 | 中 | Flash Attention + sliding mask 减少实际计算 |

---

## 9. 文档交付物

| 文件 | 内容 |
|---|---|
| `src/fusion_kv_tier.{h,cpp}` | KV 三层管理 |
| `src/fusion_kv_window.{h,cpp}` | KV sliding 调度 |
| `tests/test-fusion-kv-tier.cpp` | 单元测试 |
| `tests/test-fusion-kv-window-e2e.cpp` | E2E 长上下文测试 |
| `benchmarks/phase3_kv_tiering_report.md` | 真机数据报告 |
| `docs/phase3_architecture.md` | 架构设计（如果需要） |

---

## 10. 与其他 Phase 的关系

```
Phase 2 (✅ 完成)              Phase 6 (60% 完成)
sliding window weights         spec decode
        ↓                              ↓
        └──────────────┬───────────────┘
                       ↓
              Phase 3 (待启动)
              KV cache tiering
                       ↓
                 70B + 32K-128K
                       ↓
                  Path C 完成
                       ↓
                 Phase 5 (Ollama)
```

**关键依赖**：
- Phase 3 必须 Phase 2 完成（sliding 逻辑共享）
- Phase 3 与 Phase 6 并行（KV tiering 不影响 spec decode）
- Phase 3 完成后才能跑 Phase 4（长上下文测试）

---

## 11. 时间线汇总

| 周 | 任务 | 里程碑 |
|---|---|---|
| W1 | Selective KV offload 实现 | D5: tier manager 单元测试 PASS |
| W2 | Sliding KV 调度 | D10: window 切换测试 PASS |
| W3 | 长上下文真机测试 | D15: 70B + 32K 跑通 + 报告 |

**Sprint 总时长**：3 周
**前置依赖**：Phase 6 完成 100%（attention + Markov head 实跑）
**后续**：Phase 4（长上下文测试）+ Phase 5（Ollama 集成）

---

*Phase 3 开发方案 v1 - 2026-06-29 01:50*
*基于 benchmarks/phase3_kv_cache_test_report.md + Path C 主线规划*
*关键 PoC-4 机制已验证（SSD 落盘 + mmap 读取）*
