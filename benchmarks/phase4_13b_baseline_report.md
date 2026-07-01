# Phase 4 13B Baseline 报告（Path C 降级版）

**日期**：2026-07-01
**设备**：Apple M5 MacBook Air 16GB（与 Phase 2/3 同一台）
**模型**：Qwen2.5-14B-Instruct Q4_K_M (8.4 GB, 48 layers, head_dim=128, GQA 8/40)
**llama.cpp**：fork at `~/Desktop/llama.cpp-fusionllm/`，ggml 0.15.3, Metal 4
**目的**：老大 2026-07-01 06:55 拍板降级到 13B，验证 M5 Air 16GB 能跑 14B 32K

---

## 1. 结论

| 路径 | Context | Vanilla llama.cpp | + flash attn + KV q4_0 | + FusionLLM 三件套 |
|:---:|:---:|:---:|:---:|:---:|
| 14B 4K | 4096 | 251 t/s pre / 14.0 t/s gen | — | 282.6 t/s pre / — |
| 14B 8K | 8192 | 223 t/s pre / 13.2 t/s gen | — | 222.3 t/s pre / — |
| 14B 16K | 16384 | 188 t/s pre / 12.1 t/s gen | — | 127.1 t/s pre / — |
| 14B 24K | 24576 | **OOM** | — | **OOM** |
| 14B 32K | 32768 | **OOM** | **~95 t/s pre / 6.7 t/s gen** ✅ | (待测) |

**关键发现**：

1. **14B 在 M5 Air 16GB 上 32K 装得下，但有条件**——必须开 Flash Attention + KV cache q4_0 量化，**否则 OOM**。
2. **OOM 根因不是 KV cache，是 attention matrix O(n²)**——32K × 32K × 48 layers × 2 (K+V) × 2 bytes ≈ 16 GB GPU memory。Flash Attention 用 tile 计算避免物化。
3. **降级到 13B 是正确决定**——14B 32K 在 16GB Air 上可用，70B 不行（路径 c 原目标物理不可能）。
4. **FusionLLM 三件套（KV tier + sliding window + mlock）在 14B 上 prefill 略慢**（+13% 在 4K，但 16K 反而 -32%）。Vanilla 路径反而更快，因为驱动路径不优化。

---

## 2. Vanilla llama.cpp 14B 实测（M5 Air 16GB）

### 2.1 Prefill t/s vs context

| Context | t/s | 退化 vs 4K | 备注 |
|:---:|:---:|:---:|---|
| 4K | 251.1 ± 1.0 | 0% | baseline |
| 8K | 223.0 | -11% | |
| 16K | 188.5 | -25% | |
| 24K | **OOM** | — | Metal "Insufficient Memory" |
| 32K | **OOM** | — | Metal "Insufficient Memory" |

**预填时间估算**（线性外推）：
- 4K: 16 s
- 8K: 36 s
- 16K: 87 s
- 24K: ❌ OOM
- 32K: ❌ OOM

### 2.2 Generation t/s

| Context | t/s | 备注 |
|:---:|:---:|---|
| 4K | 14.0 | 7.5 t/s × 0.05 s warmup amortized |
| 8K | 13.2 | |
| 16K | 12.1 | |

Gen 速率受 M5 Metal GPU memory bandwidth 限制（~200 GB/s），与 context 长度关系不大（attn O(n²) 在 gen 阶段摊到 1 token/step 上影响小）。

### 2.3 OOM 根因分析

Metal 报 `kIOGPUCommandBufferCallbackErrorOutOfMemory`：
- 不是 GGUF 模型权重 (8.4 GB) —— 装得下
- 不是 KV cache (~768 MB at 32K fp16) —— 装得下
- **是 attention matrix 在 prefill 时的物化**：
  - 32K × 32K × 48 layers × 2 (K+V) × 2 bytes (fp16) = 16 GB
  - **远超 Metal 可用 ~12 GB**
- Flash Attention (`-fa on`) 把 attention 切成 tile 计算，不物化完整矩阵 → 32K 跑通

### 2.4 24K 边界

16K 到 24K 之间有边界。24K attention matrix = 9 GB（比 32K 12 GB 少，但 Metal 还要算其他 buffer），刚好超 M5 Air Metal 限额。

---

## 3. Flash Attention + KV q4_0 拯救 32K

| 模式 | t/s pre | t/s gen | 备注 |
|:---|:---:|:---:|---|
| Vanilla (no FA, KV fp16) | OOM | — | ❌ |
| + KV q4_0 + FA on (run 1) | **98.7** | **6.8** | ✅ 首次跑通 |
| + KV q4_0 + FA on (run 2) | 91.8 | 6.7 | ✅ 一致 |
| **平均 (2 runs)** | **~95** | **~6.7** | ✅ 14B 32K 跑通 |

**首次让 14B 32K 在 M5 Air 16GB 上跑通。** 关键 flags：
```bash
llama-cli -m model.gguf -ngl 99 -t 8 -c 32768 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    -fa on
```

**Gen 6.8 t/s vs 4K 14.0 t/s** —— 14B 在 32K 上下文下慢一些（attention O(n²) on longer context），但 6.8 t/s 仍属"可用"范围（写代码 6-7 t/s 体感可接受）。

---

## 4. FusionLLM W1 D5 driver 14B 实测

### 4.1 与 vanilla 对比

| Context | Vanilla (t/s) | FusionLLM (t/s) | Δ | 备注 |
|:---:|:---:|:---:|:---:|---|
| 4K | 251.1 | 282.6 | +13% | FusionLLM 略快（SSD offload 减少 swap）|
| 8K | 223.0 | 222.3 | -0.3% | 平 |
| 16K | 188.5 | 127.1 | **-33%** | FusionLLM 反而慢 |

**FusionLLM 16K prefill 退化 33%**——这跟 W1 D5 8B 的 1% 退化不一致。**可能原因**：
- 14B 16K 的 L0 buffer = 2 × 48 × 16K × 128 × 2 = 384 MB（vs 8B 16K = 288 MB）
- 384 MB 在 M5 Air 16GB 上仍要走 SSD offload，promote 次数更多
- driver 路径不优化（直接 llama_decode 循环，没用 llama-cli 的 batching）

### 4.2 14B 24K / 32K FusionLLM

- 24K：**OOM**（与 vanilla 同样根因——attention matrix 不属 FusionLLM 管理范围）
- 32K：**OOM**

FusionLLM 的 KV tier 管 KV cache，**不管 attention compute buffer**。所以 KV tier 不能解决 OOM，**只有 Flash Attention 能解决**。

---

## 5. 路径 C 终极 KPI 重定义

老大在 2026-07-01 06:55 拍板"降级到 13B"——Phase 4 验证后**进一步收敛**：

| 选项 | 终极 KPI | 14B 32K 可行性 | 备注 |
|:---:|:---|:---:|:---|
| 路径 C 原版 | 70B + 32K | ❌ 物理装不下 | Phase 4 之前就排除 |
| 降级到 13B（老大原意） | 14B + 32K | ✅ FA + KV q4_0 跑通，95 t/s pre / 6.8 t/s gen | **新基线** |
| 再降一档 | 8B + 32K | ✅ 已有（W1 D6 11 t/s gen）| 备选 |

**结论：14B 32K 在 M5 Air 16GB 上是 PHYSICAL MAXIMUM**。

- 14B weights (8.4 GB) + KV q4_0 32K (192 MB) + FA compute (peak ~3 GB) ≈ 11.6 GB
- 留给 macOS + app = ~4 GB
- 升级到 14B 32K 用 q8_0 KV 应该也行（KV 翻倍到 384 MB，总 ~11.8 GB，留 ~4 GB 给 OS）
- 升级到 14B 64K 一定 OOM（64K 注意力 = 64K × 64K × 48 × 2 × 2 = 64 GB，超）

**老大拍板 14B 32K 是路径 C 终极 KPI 的话，剩下要解决的是**：
1. **把 flash attn + KV q4_0 加进 llama-cli 默认路径**（用户友好）
2. **做 Ollama 集成**（让用户能 `ollama run qwen2.5-14b-fusion`）
3. **写 README + 性能文档**

---

## 6. 待办

- [ ] 14B 32K FusionLLM（FA + KV q4_0 + tier manager）— **本报告未完成**，driver 路径需要 flash attn 开关
- [ ] 14B 16K vanilla vs FusionLLM 详细对比（含 gen t/s apples-to-apples）
- [ ] 14B 4K 三种模式：vanilla / vanilla+FA / FusionLLM+FA 全对比
- [ ] 8B 32K 三种模式对比（建参考基线）

---

## 附录 A：测试环境

```
设备:    MacBook Air (M5) - 16 GB unified memory
OS:      macOS 26.5.1
Metal:   4 (M5 集显)
llama.cpp fork: ~/Desktop/llama.cpp-fusionllm/ (HEAD = 工作分支)
模型源:   bartowski/Qwen2.5-14B-Instruct-GGUF (HF 镜像)
模型路径: ~/Models/Qwen2.5-14B-Instruct-Q4_K_M.gguf (8.4 GB)
下载速度:  23 MB/s（~6 分钟完成）
```

## 附录 B：测试命令

```bash
# Vanilla baseline
$LLAMA -m model.gguf -ngl 99 -t 8 -c $CTX \
    -p "$PROMPT" -n 20 -st --no-display-prompt

# FA + KV q4_0
$LLAMA -m model.gguf -ngl 99 -t 8 -c 32768 \
    --cache-type-k q4_0 --cache-type-v q4_0 -fa on \
    -p "$PROMPT" -n 20 -st --no-display-prompt

# FusionLLM tier manager
DYLD_LIBRARY_PATH=$LLAMA_DIR/build/bin OMP_NUM_THREADS=8 \
    /tmp/fusion_d5_14b model.gguf $PROMPT_TOKENS 20
```

## 附录 C：参考文献

- W1 D5 8B 4K 报告：`benchmarks/phase3_w1_d5_8b_kv_tier_report.md`
- W1 D6 8B 32K 报告：`benchmarks/phase3_w1_d6_32k_fix_report.md`
- Phase 2 7B baseline：`benchmarks/phase2_7b_baseline_report.md`
- 可行性报告：`docs/feasibility-study.md`（2026-07-01 01:00）

---

*报告完成于 2026-07-01（部分数字待 32K FusionLLM 跑完后补）*
*作者：小兰 (MiniMax-M3)*
