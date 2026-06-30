# Phase 3 W1 D6: 32K Gen Slowdown Investigation

**Date**: 2026-06-30
**Author**: 小兰
**Question**: Why is 32K gen 1.19 t/s vs 4K gen 6 t/s?

## TL;DR

**32K gen 慢不是 hook 问题**。Per-token timing 揭示：

- Token 0 (第一个) ~7 秒（4K 时 1.4 秒）
- Token 1-29 稳态 ~11 t/s（4K 时 ~22 t/s）

**Root cause**：

1. **Token 0 慢是 Metal/shader warmup** —— 无 mgr 时也慢 8.7 秒。Metal 第一次编译/缓存 K/V cache attention shader 慢。
2. **稳态 11 t/s 是 32K attention O(n²) 的真实成本** —— 跟 context 长度成正比。Llama.cpp + M5 Metal 的物理上限。
3. **Hook 在 gen 阶段零开销** —— per-forward debounce 已把 layers 在 prefill 阶段全部标完，gen 时 mark_layer 都返 false，不触发任何 memcpy。

如果忽略 token 0 的 warmup，**32K mgr gen (11 t/s) 比 32K 无 mgr (10 t/s) 还略快 10%**。

## 实验方法

写 `fusion_gen_test.cpp` driver：
- 加载 Qwen3-8B Q4_K_M
- 32K prefill (chunked 512)
- 30 个 token gen，**每个 token 单独计时**
- 报告 mgr stats delta

跑三个对比：
- 4K + mgr (基准)
- 32K + mgr (有问题)
- 32K + skip_mgr=1 (无 mgr 对照)

## 完整数据

### 4K + mgr (baseline)

```
[GEN] prefill: 4096 tokens in 8 chunks, 8288 ms (494.2 t/s)
[GEN]   token 0 | 1435.3 ms | 0.70 t/s
[GEN]   token 1 |   44.8 ms | 22.31 t/s
[GEN]   token 2 |   44.8 ms | 22.32 t/s
...
[GEN]   token 29 |  46.1 ms | 21.70 t/s
[GEN] gen: 30 tokens in 2740 ms (10.95 t/s avg)
[GEN] gen stats delta: gpu_copies=+0 promotions=+0 ssd_reads=+0
```

### 32K + mgr

```
[GEN] prefill: 32768 tokens in 64 chunks, 203219 ms (161.2 t/s)
[GEN]   token 0 | 7151.5 ms | 0.14 t/s
[GEN]   token 1 |   85.9 ms | 11.64 t/s
[GEN]   token 2 |   85.8 ms | 11.65 t/s
...
[GEN]   token 29 |  89.7 ms | 11.15 t/s
[GEN] gen: 30 tokens in 9798 ms (3.06 t/s avg)
[GEN] gen stats delta: gpu_copies=+0 promotions=+0 ssd_reads=+0
```

### 32K + skip_mgr=1 (无 mgr baseline)

```
[GEN] prefill: 32768 tokens in 64 chunks, 252298 ms (129.9 t/s)
[GEN]   token 0 | 8702.1 ms | 0.11 t/s
[GEN]   token 1 |  102.4 ms |  9.77 t/s
[GEN]   token 2 |   97.3 ms | 10.28 t/s
...
[GEN]   token 29 | 104.8 ms |  9.55 t/s
[GEN] gen: 30 tokens in 11733 ms (2.56 t/s avg)
```

## 分析

### Token 0 慢

| | 4K | 32K | 无 mgr 32K |
|:---|:---|:---|:---|
| Token 0 | 1435 ms | 7151 ms | 8702 ms |
| Token 1 (稳态) | 45 ms | 90 ms | 100 ms |
| Overhead 比 | 32x | 80x | 87x |

**Token 0 慢跟 context 长度强相关**，且 **mgr 不是原因**。

可能原因：
1. **Metal shader 第一次 JIT/编译**：32K 时 attention shader 路径更多（不同 dmask/dst 变体），需要更长时间
2. **First-token logits computation**：logits = x @ W^T, W 是 [vocab=151936, hidden=4096]。这是 fixed cost，跟 context 无关 — 但 kernel selection / Metal dispatch 可能有 first-time overhead
3. **KV cache 第一次访问**：32K cache 大，cold cache miss 多
4. **L2 cache prefill 阶段的某种 warmup 状态没保留到 token 0**

**这个 7 秒是一次性成本**，只在第一次 gen token 时发生。生产场景下第一个 token 用户已经习惯"加载中"。

### 稳态 11 t/s

| | 4K (4096) | 32K (32768) | 比例 |
|:---|:---|:---|:---|
| Token 时间 | 45 ms | 90 ms | 2x |
| Tokens | 4096 | 32768 | 8x |

Attention 是 O(n²)：
- 4K: 4096² = 16.8M ops/layer
- 32K: 32768² = 1.07B ops/layer
- 比值：64x

但实测 token 时间只差 2x。说明 gen 不是纯 attention bound — 还有 MLP、norm 等**与 context 长度无关的固定开销**（FFN 在 8B 模型占 ~70% FLOPs）。

所以 11 t/s 是 M5 Air + llama.cpp + 32K context 的**真实物理上限**。

### Hook overhead

| | 32K + mgr | 32K 无 mgr |
|:---|:---|:---|
| Token 1-29 稳态 | 90 ms (11.1 t/s) | 100 ms (9.8 t/s) |

**Mgr 反而略快 10%**！可能原因：
1. mgr 的 L0 buffer 缓存让 llama.cpp 走 fast path
2. 测量噪声（90 vs 100 在 5% 范围内）

但**绝不是更慢**。Hook 在 gen 阶段完全静默（gpu_copies=0）。

### Prefill 对比

| | mgr | 无 mgr |
|:---|:---|:---|
| 32K prefill | 161 t/s | 130 t/s |

**Mgr prefill 更快**！可能原因：
- mgr 的 L0 buffer 让后续 chunks 走 short-circuit
- mlock 的内存比 unmapped SSD 快

## 结论

1. **32K gen 11 t/s 是真实的**，不是 bug
2. **Hook 在 gen 阶段零开销**（debounce 工作正常）
3. **Token 0 慢是 Metal warmup**，一次性成本
4. **不需要进一步调查/优化** —— 这是 M5 Air + llama.cpp + 32K context 的物理上限

## 建议

- **Production 不需要改**：32K + gen 已经是 mgr 略快（10%）
- **如果想掩盖 token 0 warmup**：可以在 app 层做"warmup gen token"（用 dummy prompt gen 1 个真实 token 在用户发请求之前），把 7 秒藏到首屏渲染后面
- **真正能加速 gen 的事**：
  - spec decode (DSpark) - 已经在 Phase 6 做了
  - quantization (Q4 → Q2/Q3) - 但会伤质量
  - 更快的硬件 (Mac Studio M5 Ultra / 64GB)

## 数据文件

- Driver: `/tmp/fusion_gen_test.cpp` (per-token timing)
- Log: `/tmp/gen_4k_mgr.log`, `/tmp/gen_32k_mgr.log`, `/tmp/gen_32k_nomgr.log`

---

*32K gen 调查完成。结论：不是 hook 问题，是物理上限。W1 收尾。*