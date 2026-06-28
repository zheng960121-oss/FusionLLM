# DSpark × FusionLLM 集成设计方案

**日期**：2026-06-28
**状态**：设计阶段（PoC 未跑）
**作者**：FusionLLM 设计组

---

## 1. 背景

**DSpark**（DeepSeek × 北京大学 联合开源，2026-06-27）：
- 推测解码（speculative decoding）框架，单用户提速 60-85%
- 配套 **DeepSpec** 工具链（MIT license），含 DSpark / DFlash / Eagle3 三种算法
- 已发布 checkpoint：`dspark_qwen3_4b_block7` / `dspark_qwen3_8b_block7` / `dspark_qwen3_14b_block7` 等

**FusionLLM**（本项目）：
- 内存层调度器（sliding window + selective mlock + SSD 分层）
- 已完成 Phase 2：真实 mlock/munlock 验证通过（0.5B 100% mlock 成功，7B 414 次 window_advance）

**核心洞察**：DSpark（compute 层加速） 与 FusionLLM（memory 层加速）**正交可叠**，组合可实现 "memory + compute 双优化"。

---

## 2. DSpark 核心机制（DeepSpec 源码分析）

### 2.1 Draft Model 架构

DSpark draft 不是完整 transformer，而是 **target hidden state → 少量 transformer → Markov head**：

```
target_hidden_states [B, S, d]  (从 target model 多个中间层抽出并拼接)
   ↓
embed_tokens(draft_input_ids)  [B, block_size, d]
   ↓
5× DecoderLayer  (num_draft_layers=5)
   ↓
hidden_states  [B, block_size, d]
   ↓
LM head → base_logits
   +
Markov head(prev_token_id, hidden) → bias
   ↓
corrected_logits
   ↓
sample block_size 个 draft tokens
```

### 2.2 关键参数（Qwen3-4B config）

```python
block_size = 7           # 每次 propose 7 个 draft tokens
num_draft_layers = 5     # draft backbone 5 层（vs target 36 层）
target_layer_ids = [1, 9, 17, 25, 33]  # 抽 target 5 个中间层 hidden state
markov_rank = 256         # Markov head 维度
markov_head_type = 'vanilla'
```

### 2.3 推测解码循环

```
prefill:  target(input_ids) → output_logits, hidden_states
          extract hidden_states[target_layer_ids] 作为 draft 输入

while start < max_length:
    propose:
        draft_input = [current_token, MASK×6]  (block_size-1 个 mask)
        draft_hidden = draft_model(draft_input, target_hidden)  # 一次 forward
        draft_tokens, draft_probs = markov_sample(draft_hidden, base_logits)

    verify:
        target(input_ids=[current_token] + draft_tokens)  # 一次 forward 验证
        for each draft position i:
            accept_prob = min(1, target_prob[draft_tokens[i]] / draft_prob[draft_tokens[i]])
            if rand() < accept_prob: accept
            else: replace with resample(target_probs[i], draft_probs[i]); break

    commit:
        accepted_draft_tokens + bonus_token 全部进 KV cache
        crop KV cache to start + accepted + 1

    update:
        target_hidden_states = extract(hidden_states[verified positions])
```

### 2.4 Draft Model 大小估算

Qwen3-4B Q4 ≈ 2.5GB
DSpark draft (Qwen3-4B target):
- 5 层 transformer ≈ 555M 参数
- Embedding share with target (可以不存)
- Markov head: 256 × vocab ≈ 100M
- **Draft 总量**: ~600M 参数（bf16 ~1.2GB，Q4 ~350MB）

---

## 3. FusionLLM 现状回顾

### 3.1 已完成能力（Phase 0-2）

| Phase | 能力 | 状态 |
|:---:|:---|:---:|
| 0 | 设备扫描（PoC-1~4） | ✅ |
| 1 | baseline benchmark（0.5B / 7B） | ✅ |
| 2 | **滑动窗口 + selective mlock** | ✅ |
| 3 | KV Cache 分层 | 🔄 设计完成 |
| 5 | Ollama 集成（合并 Phase 3） | ⏳ |

### 3.2 关键集成点

| 位置 | 现有能力 | DSpark 需要 |
|:---|:---|:---|
| 模型加载 | llama.cpp fork，self-mmap + populate layer→tensor ranges | 同样机制可用，加载 **draft model** 单独 GGUF |
| Forward | `llm_graph_context::cb()` 每个 tensor 带 `il` | 需要在指定 `target_layer_ids` 处**保存 hidden state** |
| KV Cache | llama.cpp 自带 KV cache | spec decoding 需要**精确 crop** + **commit accepted** |
| Sliding window | `cb_func` 触发 `fusion_window_advance(il)` | 不变；window advance 现在由 verify 触发 |

### 3.3 缺口分析

| DSpark 需要 | 当前 FusionLLM | 缺口 |
|:---|:---|:---|
| Target hidden state 抽取 | ❌ 没有保存中间层 hidden state | **新增**：`fusion_hs_extract.cpp/h` |
| Draft model loader | ❌ 只有 target model loader | **新增**：`fusion_draft_model.cpp/h`（加载 draft GGUF） |
| Markov head 实现 | ❌ | **移植**：`fusion_dspark_markov.cpp`（从 PyTorch → ggml） |
| Spec decoding loop | ❌ 标准 autoregressive | **新增**：`fusion_speculative_decode.cpp` |
| 动态 KV cache crop | ✅ llama.cpp 自带 | 需调整调用时机 |
| 与 sliding window 协同 | ✅ Phase 2 已完成 | verify 阶段触发 window_advance |

---

## 4. 集成架构设计

### 4.1 顶层架构

```
┌─────────────────────────────────────────────────────────────┐
│                  FusionLLM + DSpark System                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────┐    ┌─────────────────────────┐    │
│  │  Target Model (70B)  │    │  Draft Model (~600M)    │    │
│  │  - Sliding window    │    │  - Always mlock         │    │
│  │  - SSD ↔ RAM ↔ GPU   │    │  - Markov head          │    │
│  │  - 5/36 layers hot   │    │  - 5 layers backbone    │    │
│  └──────────────────────┘    └─────────────────────────┘    │
│           ▲                            ▲                     │
│           │                            │                     │
│  ┌────────┴────────────────────────────┴─────────┐           │
│  │       Spec Decoding Loop Controller          │           │
│  │  - propose: draft → block_size tokens        │           │
│  │  - verify: target → accept/reject             │           │
│  │  - commit: KV cache crop + advance window    │           │
│  └───────────────────────────────────────────────┘           │
│           ▲                                                   │
│           │                                                   │
│  ┌────────┴──────────────────────────────────────┐           │
│  │       Hidden State Extraction Layer            │           │
│  │  - Hook target_layer_ids 处的 forward          │           │
│  │  - Save to draft input buffer                  │           │
│  └───────────────────────────────────────────────┘           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 数据流（一次完整 spec decoding step）

```
输入: 当前 token (accepted_prev_token) + position start

1. PROPOSE 阶段
   ├─ target_layer_hidden_states (已在上次 verify 后保存)
   ├─ draft_input = [accepted_prev_token, MASK×(block_size-1)]
   ├─ draft_forward(target_layer_hidden, draft_input)
   ├─ markov_sample → draft_tokens [block_size]
   └─ 输出: draft_tokens, draft_probs

2. VERIFY 阶段
   ├─ target_verify_input = [accepted_prev_token] + draft_tokens
   ├─ target_forward(target_verify_input)
   │    ├─ 触发 sliding window advance（每访问一层 layer 就 mlock 一次）
   │    ├─ 保存 target_layer_hidden_states（用于下次 propose）
   │    └─ 输出 target_probs [block_size, vocab]
   ├─ rejection sampling:
   │    for i in 0..block_size-1:
   │        accept_prob = min(1, target_probs[i, draft_tokens[i]] / draft_probs[i, draft_tokens[i]])
   │        if rand() < accept_prob: accept draft_tokens[i]
   │        else: resample(target_probs[i], draft_probs[i]) → bonus_token; break
   └─ 输出: accepted_count + bonus_token

3. COMMIT 阶段
   ├─ KV cache crop to (start + accepted_count + 1)
   ├─ sliding window: mlock 包含 accepted tokens 的 layers, munlock 已滚出的 layers
   └─ 移动 start += accepted_count + 1

性能影响：
- Verify 阶段 forward 1 次（每次 batch_size = block_size）
- Propose 阶段 forward 1 次（draft model 很小）
- 单个 token 平均 forward 次数: 1 + 1/block_size = ~1.14 (block_size=7)
- Target layer 访问频率: 降低 ~7 倍 → sliding window 切换开销降低 ~7 倍
```

### 4.3 内存布局（16GB M5 Air 70B 场景）

| 组件 | 大小 | 位置 | mlock 策略 |
|:---|:---:|:---|:---:|
| Target model weights | ~40GB Q4 | SSD + RAM 分层 | Sliding window（6 层 ~3GB mlocked） |
| Draft model weights | ~350MB Q4 | RAM | Always mlock（太小） |
| Target KV cache (4K ctx) | ~250MB | GPU/RAM | llama.cpp 默认 |
| Draft KV cache | ~30MB | GPU/RAM | llama.cpp 默认 |
| Activations | 临时 | GPU/RAM | 无持久化 |
| **总驻留** | ~4GB | RAM+GPU | mlock 保护 |

**结论**：即使在 16GB Air 上，70B target + 350MB draft + KV cache 也放得下，DSpark + FusionLLM 协同后甚至能跑。

---

## 5. 实现路线图

### 5.1 阶段划分（8 周）

| 阶段 | 时间 | 交付物 | 验证标准 |
|:---|:---:|:---|:---|
| **S1. 深度读 + 详细 spec** | 1 周 | `docs/DSpark_FusionLLM_detailed_spec.md` | 团队 review 通过 |
| **S2. Hidden state 抽取** | 2 周 | `src/fusion_hs_extract.{h,cpp}` | 7B target 抽 [1,9,17,25,33] hidden state 验证数值一致 |
| **S3. Draft model 移植** | 2 周 | `src/fusion_draft_model.{h,cpp}` + Q4 量化脚本 | Qwen3-4B draft model 加载 + 单 token 生成正确 |
| **S4. Spec decoding loop** | 1 周 | `src/fusion_speculative_decode.cpp` | Qwen3-4B + DSpark 端到端 generate 文本质量一致 |
| **S5. 与 sliding window 协同** | 1 周 | `src/fusion_window_speculative.cpp` | 70B target + sliding + DSpark 跑通 |
| **S6. Benchmark + 报告** | 1 周 | `benchmarks/fusion_dspark_70b_report.md` | 加速比 + 内存峰值数据齐全 |

### 5.2 S1 详细任务（本周内）

- [x] 读 DSpark paper（已读 PDF 摘要）
- [x] 读 DeepSpec 核心源码（evaluator + draft_ops + markov_head + common.py）
- [ ] 读完整 `deepspec/modeling/dspark/qwen3/modeling.py`（533 行）
- [ ] 读 `deepspec/eval/dspark/confidence_head.py`（605 行）
- [ ] 对比 `transformers.Qwen3Model` 实现细节
- [ ] 写 `DSpark_FusionLLM_detailed_spec.md`：
  - 详细数据流图
  - API 接口设计（C++ 类层次）
  - ggml ops 映射表
  - 测试用例

### 5.3 S2-S5 关键工程决策

| 决策点 | 选项 | 建议 |
|:---|:---|:---|
| Hidden state 存储位置 | GPU buffer / CPU pinned mem / 普通 RAM | **GPU buffer**（最快，DSpark draft 直接读） |
| Markov head 实现 | 重写 ggml ops / 调用 ggml_mul_mat | **ggml_mul_mat**（已是 ggml 原语） |
| Draft model 格式 | 单独 GGUF / 合并到 target GGUF | **单独 GGUF**（独立加载、量化、版本管理） |
| Confidence head | 实现完整 / 第一版省略 | **第一版省略**（用全部 block_size tokens，简化） |
| Sliding window 触发时机 | verify 阶段 / propose 阶段 | **verify 阶段**（draft 阶段只跑 draft model） |
| Spec decoding 失败兜底 | 全拒绝回退 autoregressive | **必须实现**（首次 draft 质量不稳定时） |

---

## 6. 与原 Path C 的关系

**Path C 原始定义**（FusionLLM 启动时）：
- Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4
- Phase 5 = Ollama 集成（合并入 Phase 3）
- 总目标：70B + 32K 长上下文 + Ollama 友好

**DSpark 集成不影响 Path C 主干**：
- Path C 是 memory 层路线
- DSpark 是 compute 层增强（独立维度）
- 两者在同一模型上叠加，**DSpark 不替代任何 Phase**

**新 Phase 建议（融入 Path C）**：
- 在 Phase 3（KV cache 分层）**之后**插入 **Phase 6：DSpark Speculative Decoding**
- Phase 6 工作量 ~8 周
- 完成 Phase 6 后，整体能力 = 70B + 32K + Ollama + DSpark 60-85% 加速

---

## 7. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|:---|:---:|:---:|:---|
| DSpark checkpoint 转 GGUF 数值漂移 | 中 | 高 | 同一组输入对比 PyTorch vs llama.cpp 输出，逐 token diff < 1e-3 |
| Hidden state 抽取时序错位 | 中 | 高 | 强 unit test + 端到端 numerical check |
| Rejection sampling 边界条件（首个 token 等） | 高 | 中 | 复用 DeepSpec 的 unit test 思路 |
| Sliding window 与 spec decoding 资源竞争 | 中 | 中 | spec decode 优先级高于 window advance（verify 必须等） |
| M5 Air 性能不达预期 | 低 | 中 | 70B 在 Air 上跑不动是预期的，target 设备 = Mac Studio 64GB |
| DSpark 模型 Q4 量化损失 | 中 | 中 | 跑 mt-bench + gsm8k 对比 bf16 vs Q4 |

---

## 8. 今晚已完成（A2 任务收尾）

- [x] 验证 DSpark 真实存在（2026-06-27 开源）
- [x] clone DeepSpec repo (`~/Desktop/DeepSpec`)
- [x] 读 README：确认支持 Qwen3-4B/8B/14B + Eagle3 + DFlash + DSpark
- [x] 读 `deepspec/eval/dspark/{evaluator,draft_ops}.py`：完整 spec decoding loop
- [x] 读 `deepspec/modeling/dspark/{common,markov_head}.py`：Markov head + 数据流
- [x] 读 `deepspec/modeling/dspark/qwen3/{config,modeling}.py`：Qwen3-4B DSpark 模型
- [x] 读 `config/dspark/dspark_qwen3_4b.py`：关键参数（block_size=7, num_draft_layers=5）
- [x] 写出本设计文档

---

## 9. 下一步建议

| 选项 | 工作量 | 价值 |
|:---|:---:|:---|
| **继续 A2 深入**：读完整 533 行 modeling.py + 写更详细的 spec（3-4h） | 中 | 高 |
| **A1 PoC**：装环境 + 跑 Qwen3-4B + DSpark GSM8K（2-3h） | 高 | 高（拿真实加速比） |
| **A2 收尾 + 立项**：把 S1-S6 拆成 issue，预约下周 S2 开始 | 低 | 中 |
| **暂停**：今晚到这里，明晚再决定 | 0 | - |

**今晚建议**：A2 收尾 + 立项。明天或下周开始 S2。

---

*设计文档 v1 - 2026-06-28 23:30*
*基于 DeepSpec HEAD commit（2026-06-28 clone）*
