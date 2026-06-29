# Phase 3 PoC: 8B Q4_K_M Baseline (M5 Air 16GB)

**日期**: 2026-06-30 凌晨
**状态**: 部分完成 — 短 context 路径验证 OK, 长 prefill 路径卡死
**目的**: 验证 8B Q4_K_M 在 M5 Air 16GB 跑得动 + Metal backend 工作

---

## 1. 测试环境

| 项目 | 值 |
|:-----|:---|
| 设备 | M5 MacBook Air, 16GB unified memory |
| OS | macOS (M-series) |
| 模型 | Qwen3-8B-Q4_K_M.gguf (4.7 GB) |
| Backend | Metal 4 + Accelerate + CPU (3 backends) |
| 路径 | `~/Desktop/models/Qwen3-8B-Q4_K_M.gguf` |

### 关键命令参数

```bash
./llama-cli -m Qwen3-8B-Q4_K_M.gguf \
    -p "..." \
    -n 50 \
    -c 8192 \
    -ngl 999 \      # 全部 layer offload 到 Metal
    -t 8 \          # 显式 8 线程 (auto 在 macOS 不可靠)
    --no-display-prompt
```

---

## 2. 测试结果

### 2.1 短 context (512 tokens) — ✅ 路径 OK

```
[ Prompt: 55.0 t/s | Generation: 31.9 t/s ]
```

- **Prompt 55.0 t/s**: prefill 1 token 极快 (token 数量少)
- **Generation 31.9 t/s**: decode 单核
- 内存 RSS ~5.2 GB

### 2.2 32K context (1 token prefill) — ✅ 路径 OK

```
[ Prompt: 55.1 t/s | Generation: 31.9 t/s ]
```

- 32K context buffer 分配 OK
- 但 **实际 KV cache 只有 6 tokens** (1 prompt + 5 gen)
- 没填满 32K → 没真验证 long-context prefill 性能

### 2.3 8K context (8K token prompt prefill) — ❌ 卡死

- 跑了 11+ 分钟没出 t/s 数字
- 100% CPU 单核
- 估算: 8B 36 层 × 8K tokens prefill ≈ 8-15 分钟 (CPU 单核)
- 问题: **mmap 路径** (`mmap_base is NULL, skipping mlock`)
  - 模型 weights 在 mmap 区域
  - 每次 page fault 从 mmap load → CPU bound
  - 没 mlock 进 RAM → 频繁 page fault

### 2.4 4K context (4K token prompt prefill) — ❌ 也卡死

- 60+ 秒没出 t/s 数字
- 跟 8K 同样 mmap + CPU 单核问题

---

## 3. 根因分析

### 3.1 短 context work 的原因

- 1-token prefill 不触发频繁 page fault
- Generation 阶段 KV cache 很小 (≤6 tokens), 在 L0/L1 内存
- Metal backend 处理少量 token 极快 (55 t/s)

### 3.2 长 context 卡死的原因

**问题 1: mmap + 无 mlock**
- llama.cpp fork 编译时 mlock 失败 (`mmap_base is NULL`)
- 模型 weights 在 mmap 区域, 没强制锁定 RAM
- 每次访问触发 page fault → RAM 加载 → 严重 latency

**问题 2: 单核瓶颈**
- `ps` 报 100% CPU 但其他核心空闲
- macOS 上 OpenMP 多线程调度不可靠
- llama-cli 大量用 Apple Accelerate BLAS (单线程 AMX) + Metal

**问题 3: 短跑"OK"是错觉**
- 短 context 没填满 KV cache, 没真测 long-context prefill
- 32K context 1 token prefill 跟 512 context 1 token prefill 实际 workload 一样

---

## 4. 关键数据点

| Context | Prompt tokens | Status | Notes |
|:-------:|:-------------:|:------:|:------|
| 512 | 1 | ✅ 55.0 t/s | 路径 OK |
| 32K | 1 | ✅ 55.1 t/s | buffer 分配 OK, 但没填 KV |
| 4K | 4K | ❌ >60s | mmap + CPU 单核 |
| 8K | 8K | ❌ >11min | mmap + CPU 单核 |

### 4.1 内存占用

| Stage | RSS |
|:------|:----|
| 短跑 baseline | 5.2 GB |
| 32K context (1 token) | 5.2 GB |
| 8K context (8K tokens) | 6.2 GB → 5.8 GB (OS reclaims inactive) |

### 4.2 vs 7B baseline (Phase 1)

| Model | Context | Prompt | Generation |
|:------|:-------:|:------:|:-----------:|
| 7B Q4_K_M | 4K | 175 t/s | 28 t/s |
| **8B Q4_K_M** | 512 | **55 t/s** | **32 t/s** |

- 8B Generation **更快** (32 vs 28 t/s) — Metal + M5 比 7B 时跑的 M1 快
- 8B Prompt **慢** (55 vs 175 t/s) — 短 prompt prefill 不饱和, 不是真比较

---

## 5. Phase 3 入口确认

8B 8K+ 长 context prefill 慢 (单核 CPU 11+ 分钟) → **Phase 3 KV SSD offload 必须**.

**Phase 3 W1 目标**:
- Selective KV offload: 冷 KV block 落盘到 SSD
- Sliding window for KV: 长 context KV cache 分层
- 实测 8B 8K prefill **降到 1-2 分钟** (10x 加速)

**W1 实现前必须解决**:
- [ ] mlock 问题: 让模型 weights 在 RAM (不需要 SSD)
- [ ] Metal 真启用验证: 短跑时 55 t/s prompt 已经走 Metal, 但长 prefill 是不是走 Metal 不确定
- [ ] 8 线程真协同: 加 `OMP_NUM_THREADS=8` env var 验证

---

## 6. 明天继续路径

1. **D0-PoC 修** (30 min):
   - 跑 4K prefill 时设 `OMP_NUM_THREADS=8` + 改用 mmap disable
   - 看是不是真多核跑 (top 应该看到 8 核 80%+ 而不是 1 核 100%)
   - 拿到 4K/8K prefill baseline 数据

2. **W1 设计** (1 hour):
   - `src/fusion_kv_tier.h` 头文件 (3 层 KV cache 数据结构)
   - `docs/phase3_w1_kv_tier_design.md` (W1 详细设计)
   - `tests/test-fusion-kv-tier.cpp` (单元测试设计)

3. **W1 实现** (3-5 days):
   - Selective KV SSD offload (mmap 验证已 OK)
   - Sliding window for KV
   - llama.cpp fork 集成 (类似 Phase 6 DSpark forward 集成)

---

*PoC 路径验证: 8B 短 context OK, 长 context 待修*
*Phase 3 入口: KV cache 分层必须, 价值是 10x prefill 加速*
*8B 模型路径: `~/Desktop/models/Qwen3-8B-Q4_K_M.gguf`*
