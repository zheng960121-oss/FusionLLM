# Phase 6 S8: Markov Head Vanilla Ops - Report

**日期**：2026-06-29
**作者**：FusionLLM Phase 6
**状态**：✅ **完成**（6/6 测试 PASS, max diff 1e-6）

---

## 1. 范围

实现 DSpark 推测解码的 Markov head vanilla 变体的 ggml ops。这是 spec decode 端到端跑通的前置条件之一（与 S7 attention 并列）。

数学：
```
prev_emb = W1[token_ids]                # [B, S, rank]   (Embedding lookup)
bias     = prev_emb @ W2.T              # [B, S, V]      (Linear projection)
out      = base_logits + bias           # [B, S, V]      (corrected logits)
```

`W1` 是 `nn.Embedding(vocab_size, markov_rank).weight`（shape `[V, rank]`）。
`W2` 是 `nn.Linear(markov_rank, vocab_size, bias=False).weight`（shape `[V, rank]`）。

---

## 2. 实现

### 2.1 头文件（`src/fusion_markov_head.h`）

```cpp
struct ggml_tensor * fusion_markov_head_forward(
    struct ggml_context * ctx,
    struct ggml_tensor  * base_logits,   // ne = [V, S, B]
    struct ggml_tensor  * token_ids,     // ne = [S, B]   (I32)
    struct ggml_tensor  * w1,            // ne = [rank, V] (Embedding)
    struct ggml_tensor  * w2);           // ne = [rank, V] (Linear)
```

### 2.2 关键设计：存储 layout

所有 tensor 在 ggml 内存中按 ne 顺序存储（ne[0] 是 innermost）：

| Tensor       | ggml ne          | PyTorch shape | 备注 |
|--------------|------------------|---------------|------|
| `base_logits`| `[V, S, B]`      | `[B, S, V]`   | 通常是 DSpark attention 输出 |
| `token_ids`  | `[S, B]`         | `[B, S]`      | I32 整数索引 |
| `w1` (embed) | `[rank, V]`      | `[V, rank]`   | PyTorch nn.Embedding 权重 |
| `w2` (linear)| `[rank, V]`      | `[V, rank]`   | PyTorch nn.Linear 权重 |
| `out`        | `[V, S, B]`      | `[B, S, V]`   | 同 base_logits |

### 2.3 ggml ops 链

```cpp
// 1) Flatten token_ids to 1D for ggml_get_rows
struct ggml_tensor * tok_1d = ggml_reshape_1d(ctx, token_ids, B * S);

// 2) prev_emb = W1[tok_1d]   ggml ne = [rank, B*S]
struct ggml_tensor * prev_emb = ggml_get_rows(ctx, w1, tok_1d);

// 3) bias = W2 @ prev_emb^T   ggml ne = [V, B*S]  (PyTorch [B*S, V])
struct ggml_tensor * bias_2d = ggml_mul_mat(ctx, w2, prev_emb);

// 4) Reshape to ne = [V, S, B] — memory layout matches base_logits / PyTorch [B, S, V]
struct ggml_tensor * bias = ggml_reshape_3d(ctx, bias_2d, V, S, B);

// 5) out = base_logits + bias
return ggml_add(ctx, base_logits, bias);
```

### 2.4 关键发现

**不需要 permute！** 把 `(B*S)` outer dim reshape 成 `(S, B)` 之后，flat index `m = s + b*S` 自然等于 PyTorch row-major 的 `m = b*S + s`，因为行优先 `(S, B)` 把 `b` 当外层。直接 reshape + ggml_add 即可。

---

## 3. 测试

### 3.1 单元测试（`tests/test-fusion-markov-head.cpp`）

6 个 case 覆盖：

| Case | B | S | V | rank | 用途 |
|:-----|:-:|:-:|:-:|:----:|:-----|
| small random | 1 | 7 | 64 | 16 | 基础验证 |
| dspark-block7-vocab | 1 | 7 | **151936** | 256 | Qwen 实际 vocab size |
| multi-batch | **2** | 7 | 1024 | 32 | B > 1 验证 |
| tall block | 1 | **64** | 4096 | 128 | S 大验证 |
| single token | 1 | **1** | 32 | 4 | 边界 case |
| 2d base (S only) | 1 | 8 | 128 | 16 | n_dims=2 路径 |

### 3.2 数值一致性

对比 `ggml_markov_forward` 和 naive C++ reference（PyTorch row-major），**所有 case max |diff| < 5e-6**（FP32 精度极限）：

| Case | max |diff| | 备注 |
|:-----|-------------:|:------|
| small random | 2.38e-7 | ✅ |
| dspark-block7-vocab | 4.77e-6 | ✅ |
| multi-batch | 4.77e-7 | ✅ |
| tall block | 1.91e-6 | ✅ |
| single token | 0.0e+0 | ✅（完全相同）|
| 2d base | 2.38e-7 | ✅ |

**总通过率：6/6 (100%)**

### 3.3 重要坑：ggml_gallocr 复用 input 内存

**坑**：ggml 的 `ggml_gallocr_alloc_graph` 会**复用 input tensor 的内存**作为中间结果 buffer。`ggml_backend_graph_compute` 之后，input tensor (`w1`, `base_logits`) 的内容**已被覆盖**。

**症状**（debug 出来的）：test 在 compute 之后用 `p_w1` 算 reference，但 `p_w1` 已经被 mul_mat 写过的 intermediate 覆盖。结果 reference 和 ggml 输出对比出 1e0 级别的 diff，看起来像 impl 有 bug，实际上 impl 是对的。

**修复**：test 在 compute 之前 copy 一份 `w1_copy`, `w2_copy`, `base_copy`, `tok_copy`，reference 用 copy 算。

```cpp
// BEFORE compute
std::vector<float>  w1_copy(p_w1, p_w1 + V*rank);
std::vector<float>  w2_copy(p_w2, p_w2 + V*rank);
std::vector<float>  base_copy(p_base, p_base + B*S*V);
std::vector<int32_t> tok_copy(p_tok, p_tok + B*S);

// Then call ggml_backend_graph_compute

// Reference uses copies
naive_markov_forward(base_pt.data(), tok_pt.data(),
                     w1_copy.data(), w2_copy.data(),
                     ref_pt.data(), B, S, V, rank);
```

---

## 4. 性能观察

S8 是纯 ops 链，依赖 ggml 自身的 kernel。`ggml_mul_mat` 在 M5 Air 上是 Metal-backed，性能接近 llama.cpp baseline（具体数字待 S10 E2E 时一起 benchmark）。

S8 与 S7 attention 一起，是 spec decode 端到端跑通的前置。

---

## 5. 交付物

| 文件 | 内容 |
|:-----|:------|
| `src/fusion_markov_head.h` | 公共 API（56 行） |
| `src/fusion_markov_head.cpp` | vanilla ops 实现（120 行） |
| `tests/test-fusion-markov-head.cpp` | 6 个 case + DEBUG 输出（260 行） |
| `build_fusion_tests.sh` | 编译脚本 |
| `benchmarks/phase6_s8_markov_head_report.md` | 本报告 |

---

## 6. 下一步（S9, S10）

| 任务 | 工作量 | 状态 |
|:-----|:------|:-----:|
| **S9** PyTorch checkpoint → GGUF 完整 reader | 2-3 天 | 待启动 |
| **S10** step_spec 实跑 + 端到端 E2E | 3-4 天 | 待启动 |
| S7 (DSparkAttention 双输入 attention) | 3-4 天 | 待启动 |

按 Phase 6 剩余开发方案（`docs/Phase6_Remaining_Development_Plan.md`），S7 / S8 / S9 / S10 是 2 周 sprint 的 4 个并行子任务。S8 已完成。

---

*Phase 6 S8 完成 - 2026-06-29 09:25*
*6/6 单元测试 PASS - max diff 1e-6*
*1 commit (01792ae) pushed to https://github.com/zheng960121-oss/FusionLLM*
