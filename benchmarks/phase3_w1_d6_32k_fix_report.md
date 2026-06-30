# Phase 3 W1 D6: 32K Metal Buffer Hang Fix

**Date**: 2026-06-30
**Author**: 小兰
**Commits**: main `4468f25`, fork `62b90fc`

## 问题

老大说"修 D6 32K 的 Metal buffer 问题（需要深入 llama.cpp 内部）"。

实际**不是 Metal buffer 问题**（W1 D6 v7 已经修过，commit `13d9d21`）。
**真正的问题**：W2 改动（per-forward debounce 用的 `mark_layer_in_current_forward` / `begin_forward`）让 32K prefill 卡死——0% CPU 10GB RSS 4+ 小时。

## 根因

`FusionKVTierManager` 构造函数**没初始化**：
- `layer_seen_in_forward_` (std::vector<char>)
- `forward_id_` (uint64_t)
- `current_forward_id_` (uint64_t)

第一次 `llama_decode` 进入 hook：
1. `begin_forward()` 调用 `forward_id_++`，但 `forward_id_` 是 UB 初始值
2. `mark_layer_in_current_forward(0)` 访问 `layer_seen_in_forward_[0]`，但 vector 是**空的**
3. → vector out-of-bounds → 内存访问违例
4. → ggml_metal abort callback 触发，set_abort_callback 把 SIGABRT 转成 `exit(-1)` 或类似
5. → 测试进程退出，但 log 看起来像"在 mark_layer 之后什么都没发生"

W2 之前的版本不需要这些字段，所以没人发现。W2 加了 debounce 之后，第一次 forward 才暴露。

## 修复

7 行代码，在构造函数体里加：

```cpp
// 3b) Per-forward debounce state — MUST be initialized in constructor;
//     mark_layer_in_current_forward() reads layer_seen_in_forward_[il]
//     and accesses current_forward_id_; if either is uninitialized the
//     first llama_decode's hook will crash (D6 32K hang root cause).
layer_seen_in_forward_.assign(n_layers_, 0);
forward_id_         = 0;
current_forward_id_ = 0;
```

同步 main 和 fork 两个仓库。

## 验证

| 测试 | 修复前 | 修复后 |
|:---|:---|:---|
| **4K regression** | — | ✅ 577.5 t/s prefill, 6.08 t/s gen |
| **32K D6 (prefill)** | ❌ hang 4h+ 0% CPU 10GB RSS | ✅ **153.1 t/s, 214s = 3.5 分钟** |

### 32K D6 详细数据

```
[D5] prefill chunk 64/64 done (tokens 32768/32768)
[D5] prefill: 32768 tokens in 64 chunks, 214000 ms (153.1 t/s)
[D5] prefill stats: gpu_copies=288 promotions=612 ssd_reads=612
[D5] gen: 10 tokens in 8374 ms (1.19 t/s)
[D5] final: L0=36.00 MB L1=40.50 MB L2=216.00 MB
```

- **prefill**: 153.1 t/s（vs W1 D5 v7 之前 ~165 t/s 基本持平，**之前是 hang**）
- **gen**: 1.19 t/s（比 4K 的 6.08 t/s 慢 5x — 大 context gen 本身慢，**不在 D6 scope**）

### 4K 回归数据

```
[D5] prefill chunk 8/8 done (tokens 4096/4096)
[D5] prefill: 4096 tokens in 8 chunks, 7092 ms (577.5 t/s)
[D5] gen: 10 tokens in 1643 ms (6.08 t/s)
[D5] final: L0=13.50 MB L1=0.00 MB L2=27.00 MB
```

跟 W2 4K baseline (574 t/s prefill, 6.07 t/s gen) 完全一致 ✅

## W1 综合现状

| 里程碑 | 状态 |
|:---|:---|
| W1 D1-D4 (设计 + API + SSD + 集成) | ✅ |
| W1 D5 (8B 4K E2E) | ✅ |
| **W1 D6 (32K prefill)** | ✅ **本报告** |
| W2 sliding window | ✅ (4K +0% overhead) |

## 关键教训

### ⚠️ **任何新增成员字段必须在构造函数里初始化**

std::vector 默认空构造。如果 hook 或 method 第一次调用就访问，下标是越界 → UB → 看起来"hang"。

教训应用范围：
- `mark_layer_in_current_forward` 用的 `layer_seen_in_forward_`
- `begin_forward` 用的 `current_forward_id_`
- 任何后续加的 per-forward / per-batch state 必须走同样的 init 路径

### ⚠️ **双仓库 + 修改构造函数** 的纪律

- main repo 的 mgr .cpp 改了 → fork 的 libllama.dylib 必须 rebuild
- fork 的 mgr .cpp 改了 → main repo 的 driver binary 必须 rebuild
- 这次同时改了两个仓库的构造函数，然后两个都 rebuild，OK

### ⚠️ **"看起来 hang" 不一定是 hang**

这次问题表象：
- 进程不退（kill -0 仍然 YES）
- 0% CPU
- RSS 10GB+
- log 停在 "mark_layer: layer=0"

实际是 **vector OOB → ggml_metal abort callback → 进程被静默终止**，但主进程在 fork 阶段没退（因为还没等 Metal init 完成）→ 看起来像 hang。

排查方法：往构造函数里加 init 后跑一次就破案了。

## 下一步候选

- 修 gen 慢 (32K context gen 1.19 t/s) — 是不是 hook 在大 context 累积？或者 K/V cache 太大？
- Ollama 集成（Phase 5）
- 70B 真机（64GB Mac Studio）

---

*W1 D6 hang 修复完成。老大再次拍板 W1 收尾。*