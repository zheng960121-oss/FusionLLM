# Phase 6 剩余工作开发方案（Next 2-Week Sprint）

**日期**：2026-06-29
**目标**：让 DSpark 推测解码端到端在 M5 Air 上跑起来
**当前状态**：Phase 6 完成度 60%（设计 + 骨架 + 算法 + 协同 17/17 测试 PASS）
**目标完成度**：Phase 6 → **100% 可工作**，拿到真实加速比数字

---

## 1. 范围与目标

### 1.1 业务目标

完成 **DSpark × FusionLLM 端到端集成**：
- Qwen3-4B target + DSpark draft 在 M5 Air 上跑通
- 真实加速比数字（vs autoregressive baseline）
- 数值一致性验证（PyTorch reference < 1e-3）

### 1.2 技术目标

1. **Qwen3DSparkAttention 双输入 attention 完整实现**（核心 blocker）
2. **Markov head vanilla 实际 ops**（spec decode 必需）
3. **PyTorch checkpoint → GGUF 完整 reader**（从 HF 拿真实权重）
4. **step_spec 完整 forward 集成**（spec decode 端到端跑）
5. **E2E 测试 + 数值一致性验证**

### 1.3 非目标（明确不做）

- ❌ Confidence head 完整实现（第一版省略）
- ❌ Markov head gated / rnn 变体
- ❌ 70B target 真机验证（需要 64GB 硬件）
- ❌ madvise 优化（独立后续 phase）
- ❌ Ollama 集成（Path C 主线后续 phase）

---

## 2. 任务分解（4 个子任务，2 周时间）

### 2.1 S7 — Qwen3DSparkAttention 双输入 attention 实现（3-4 天）

**技术挑战**：DSparkAttention 是 DSpark 的核心创新——K/V 不是来自单一输入，而是 `concat(target_context, draft_noise)`。

**对应 PyTorch 实现**（`deepspec/modeling/dspark/qwen3/modeling.py:60-130`）：

```python
def forward(self, hidden_states, target_hidden_states, position_embeddings, ...):
    bsz, q_len = hidden_states.shape[:-1]      # draft tokens 数 = block_size
    ctx_len = target_hidden_states.shape[1]     # target context 长度
    
    # Q from draft input only
    q = self.q_proj(hidden_states)
    
    # K/V from BOTH target context AND draft noise
    k_ctx = self.k_proj(target_hidden_states)
    k_noise = self.k_proj(hidden_states)
    v_ctx = self.v_proj(target_hidden_states)
    v_noise = self.v_proj(hidden_states)
    
    # Concat K/V: [target_ctx, draft_noise]
    k = torch.cat([k_ctx, k_noise], dim=1)
    v = torch.cat([v_ctx, v_noise], dim=1)
    
    # Apply RoPE on full concatenated sequence
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    
    # Standard attention
    attn_output = sdpa(q, k, v, ...)
    
    return self.o_proj(attn_output)
```

**ggml 实现**（对应 ops 映射）：

```cpp
// 1. Q projection from draft input
q = ggml_mul_mat(self.q_proj_w, hidden_states);    // [n_embd, block_size*hidden]
q = ggml_reshape_4d(q, bsz, q_len, n_head, head_dim);
q = ggml_rms_norm(q, self.q_norm_w);                // per-head RMSNorm
q = ggml_cont(q);                                    // make contiguous

// 2. K from target context
k_ctx = ggml_mul_mat(self.k_proj_w, target_hs);     // [n_embd_kv, ctx_len*hidden]

// 3. K from draft noise
k_noise = ggml_mul_mat(self.k_proj_w, hidden_states);

// 4. Concat K (along sequence dim)
k = ggml_concat(ctx, k_ctx, k_noise, 1);              // [n_embd_kv, (ctx_len+q_len)*hidden]

// 5. Same for V
v_ctx = ggml_mul_mat(self.v_proj_w, target_hs);
v_noise = ggml_mul_mat(self.v_proj_w, hidden_states);
v = ggml_concat(ctx, v_ctx, v_noise, 1);

// 6. Reshape to multi-head
k = ggml_reshape_4d(k, bsz, ctx_len+q_len, n_head_kv, head_dim);
v = ggml_reshape_4d(v, bsz, ctx_len+q_len, n_head_kv, head_dim);

// 7. Apply RoPE (must be on FULL concatenated sequence)
position_ids = ...  // [start, start+ctx_len+q_len)
cos = ggml_compute_cos(position_ids);
sin = ggml_compute_sin(position_ids);
q = ggml_rope(ctx, q, cos, sin);
k = ggml_rope(ctx, k, cos, sin);

// 8. SDPA / Flash Attention
attn_out = ggml_flash_attn_ext(ctx, q, k, v, mask, scale);

// 9. Output projection
attn_out = ggml_mul_mat(self.o_proj_w, attn_out);
```

**关键决策**：
- ✅ 用 `ggml_concat` 拼接 K/V（已支持）
- ✅ 用 `ggml_flash_attn_ext` SDPA（已支持）
- ⚠️ 整段 RoPE 需要 `position_ids` 是连续的（target 端 + draft 端无缝隙）
- ⚠️ GQA（grouped query attention）需要 `repeat_interleave`，ggml 用 `ggml_repeat` 实现

**验收标准**：
- [ ] `FusionDSparkAttention::forward()` 完整实现
- [ ] 单元测试：与 PyTorch reference 对比，diff < 1e-3
- [ ] 性能：单次 attention forward < 5ms (M5 Air)

### 2.2 S8 — Markov head vanilla 实际 ops（2-3 天）

**对应 PyTorch**（`deepspec/modeling/dspark/markov_head.py:7-50`）：

```python
class VanillaMarkov(nn.Module):
    def __init__(self, vocab_size, markov_rank):
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, vocab_size, bias=False)
    
    def get_prev_embeddings(self, token_ids):
        return self.markov_w1(token_ids)  # [B, S, rank]
    
    def project_bias(self, latent):
        return self.markov_w2(latent)  # [B, S, vocab]
    
    def apply_block_logits(self, base_logits, token_ids, hidden_states):
        # base_logits: [B, num_blocks, block_size, vocab]
        # token_ids:   [B, num_blocks, block_size]
        prev_emb = self.get_prev_embeddings(token_ids)
        bias = self.project_bias(prev_emb)
        return base_logits + bias
```

**ggml 实现**：

```cpp
// prev_emb: [B, num_blocks*block_size, markov_rank]
prev_emb = ggml_get_rows(markov_w1_w, token_ids);

// bias: [B, num_blocks*block_size, vocab]
// 注意: ggml Linear weight shape = [vocab, rank]
bias = ggml_mul_mat(markov_w2_w, prev_emb);

// base_logits + bias: 需要 broadcast bias 到 base_logits shape
// base_logits: [B, num_blocks, block_size, vocab]
// bias: [B, num_blocks*block_size, vocab]
// Reshape bias to [B, num_blocks, block_size, vocab]
bias = ggml_reshape_4d(bias, B, num_blocks, block_size, vocab);

corrected_logits = ggml_add(ctx, base_logits, bias);
```

**采样循环**（per draft position）：
```python
# sample_block_tokens: 每个 step 调 markov head
for step_idx in range(proposal_len):
    step_hidden = hidden_states[:, step_idx, ...]
    step_logits = base_logits[:, step_idx, :] + markov.compute_step_bias(...)
    next_token = sample_tokens(step_logits)
    prev_token_ids = next_token
```

**验收标准**：
- [ ] `markov_head_forward()` 完整实现（vanilla）
- [ ] 单元测试：与 PyTorch reference 对比，diff < 1e-3
- [ ] 集成：与 S7 attention 组合，forward 端到端跑通

### 2.3 S9 — PyTorch checkpoint → GGUF 完整 reader（2-3 天）

**当前状态**：`tools/dspark_to_gguf.py` 只写 metadata，不读 tensor。

**需要做的**：
1. 用 `safetensors` 或 `pytorch` 库读 HF checkpoint
2. 按 DSpark tensor naming convention 提取权重：
   ```
   -layers.{il}.self_attn.q_proj.weight
   -layers.{il}.self_attn.k_proj.weight
   -...
   -fc.weight
   -hidden_norm.weight
   -norm.weight
   -markov_head.markov_w1.weight
   -markov_head.markov_w2.weight
   ```
3. 量化（可选）：bf16 → Q4_K_M 用 `gguf.PyTorchConverter` 或 llama.cpp `quantize` 命令
4. 写到 GGUF 文件，用 `GGUFWriter.add_tensor()`

**对应脚本**：

```python
# tools/dspark_to_gguf.py 扩展
def add_draft_tensors(writer, dspark_cfg, pytorch_state_dict):
    """读 PyTorch checkpoint 并写到 GGUF"""
    
    n_layers = dspark_cfg['num_draft_layers']
    
    # Decoder layers
    for il in range(n_layers):
        # Attention
        writer.add_tensor(f"layers.{il}.attn_q.weight", 
                          pytorch_state_dict[f'layers.{il}.self_attn.q_proj.weight'])
        writer.add_tensor(f"layers.{il}.attn_k.weight", 
                          pytorch_state_dict[f'layers.{il}.self_attn.k_proj.weight'])
        # ... 其他 tensors
    
    # FC + norms
    writer.add_tensor("fc.weight", pytorch_state_dict['fc.weight'])
    writer.add_tensor("hidden_norm.weight", pytorch_state_dict['hidden_norm.weight'])
    writer.add_tensor("norm.weight", pytorch_state_dict['norm.weight'])
    
    # Markov head
    writer.add_tensor("markov_head.markov_w1.weight",
                      pytorch_state_dict['markov_head.markov_w1.weight'])
    writer.add_tensor("markov_head.markov_w2.weight",
                      pytorch_state_dict['markov_head.markov_w2.weight'])
```

**验收标准**：
- [ ] 下载 `deepseek-ai/dspark_qwen3_4b_block7` (50MB)
- [ ] 转换生成完整 GGUF (~300MB with Q4 量化)
- [ ] C++ 端 `load_from_gguf` 能加载实际权重

### 2.4 S10 — step_spec 完整 forward + E2E 测试（3-4 天）

**当前状态**：`step_spec` 是骨架，verify forward 是 TODO。

**需要做的**：
1. 实现 `step_spec` 完整 forward：
   ```cpp
   // 1. prepare draft input
   draft_input = [accepted_token, MASK×(block_size-1)]
   
   // 2. extract target hidden states（用 FusionHSExtractor）
   target_hs = hs_extractor->extract(ctx);  // [1, ctx_len, concat_dim]
   
   // 3. draft forward
   draft_logits = draft_model->forward(ctx, target_hs, draft_input, position_ids, draft_kv);
   
   // 4. sample draft tokens（用 llama_sample_token）
   draft_tokens = llama_sample_token(draft_logits, temperature);
   draft_probs = llama_sample_probs(draft_logits, temperature);
   
   // 5. verify forward (target model on [accepted, draft_tokens])
   verify_input = concat(accepted_token, draft_tokens);
   target_output = target_model_forward(ctx, verify_input);
   target_probs = softmax(target_output.logits, temperature);
   
   // 6. rejection sampling
   (accepted_count, bonus_token) = rejection_sample(target_probs, draft_probs, draft_tokens, ...);
   
   // 7. commit + crop KV cache
   output_ids[start:start+accepted_count+1] = [draft_tokens[:accepted_count], bonus_token];
   llama_kv_cache_crop(target_kv, start + accepted_count + 1);
   llama_kv_cache_crop(draft_kv, start + accepted_count + 1);
   
   // 8. extract new target hidden states
   ```

2. 端到端测试：
   - 加载 Qwen3-4B target + DSpark draft GGUF
   - 跑 100 tokens spec decoding
   - 输出文本质量（perplexity）与 reference 对比 < 5% 退化
   - 加速比 vs autoregressive baseline ≥ 1.3×

**验收标准**：
- [ ] step_spec 实跑 Qwen3-4B + DSpark 100 tokens
- [ ] 文本质量与 HuggingFace reference 一致
- [ ] 加速比 vs autoregressive ≥ 1.3×
- [ ] 内存峰值 < 8GB (在 M5 Air 16GB 内)

---

## 3. 时间线（2 周 Sprint）

| Day | S7 Attention | S8 Markov | S9 Reader | S10 E2E |
|---|---|---|---|---|
| D1 (Mon) | ✅ 设计 + 框架 | — | — | — |
| D2 (Tue) | ✅ q_proj + k/v_proj + concat | — | — | — |
| D3 (Wed) | ✅ RoPE + SDPA + o_proj | — | — | — |
| D4 (Thu) | ✅ 数值一致性测试 | ✅ VanillaMarkov 框架 | — | — |
| D5 (Fri) | — | ✅ get_rows + mul_mat + add | ✅ safetensors 读 | — |
| D6 (Sat) | — | ✅ apply_block_logits 完整 | ✅ tensor 写 + 量化 | — |
| D7 (Sun) | — | ✅ 数值一致性测试 | ✅ E2E load GGUF | ✅ step_spec 框架 |
| D8 (Mon) | — | — | — | ✅ 实跑 100 tokens |
| D9 (Tue) | — | — | — | ✅ 文本质量 vs HF ref |
| D10 (Wed) | — | — | — | ✅ 加速比 benchmark + 报告 |

**关键里程碑**：
- **D3**：Attention 单元测试 PASS（与 PyTorch diff < 1e-3）
- **D6**：Markov head 单元测试 PASS
- **D7**：能加载真实 DSpark GGUF
- **D8**：spec decode 端到端跑通
- **D10**：拿到真实加速比数字 + Phase 6 完成度 100%

---

## 4. 资源需求

### 4.1 硬件
- M5 MacBook Air 16GB（已有）
- ~2GB 磁盘 for models（Qwen3-4B target + DSpark draft GGUF）
- 无需 GPU 服务器

### 4.2 数据
- `deepseek-ai/dspark_qwen3_4b_block7` from HuggingFace (~50MB checkpoint)
- 不需要训练数据（用预训练 checkpoint）

### 4.3 工具
- Python 3.14 + `gguf` package（已有）
- `safetensors` package（需要安装：`pip3 install safetensors`）
- `transformers` package（用于 tokenizer，不需安装大包）

### 4.4 知识
- DeepSpec 源码（已 clone 在 `~/Desktop/DeepSpec/`）
- llama.cpp ggml ops 文档（已有，参考 `ggml.h`）
- 本地已有 MEMORY.md + daily memory

---

## 5. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|:---:|:---:|:---|
| ggml 双输入 attention concat+rope 性能瓶颈 | 中 | 中 | profile + 用 fused kernel 替代 |
| PyTorch checkpoint tensor name 与 DSpark 不完全匹配 | 高 | 中 | 加 fuzzy matching + 详细 log |
| Qwen3-4B target 加载 + draft + KV cache > 8GB（OOM） | 中 | 高 | 用 `-ngl 20`（部分层 offload CPU） |
| spec decode 加速比 < 1.3×（达不到 DSpark paper 的 1.6×） | 低 | 低 | 调整 block_size（5-10）+ accept 阈值 |
| M5 Air Metal backend 与 spec decode KV cache 交互 bug | 中 | 高 | 准备 fallback：CPU only 跑 |

---

## 6. 验收标准（Phase 6 完成度 → 100%）

### 6.1 必达指标

- [ ] 所有 4 个 S 子任务完成
- [ ] `tests/test-fusion-spec-decode` E2E 测试 PASS
- [ ] 17 个单元测试 + 1 个 E2E 测试全 PASS
- [ ] 数值一致性：spec decode logits vs PyTorch reference diff < 1e-3
- [ ] 文本质量：spec decode 输出 perplexity < 5% 退化
- [ ] 加速比：≥ 1.3× vs autoregressive
- [ ] 内存峰值：< 8GB (M5 Air 16GB 限制)

### 6.2 加分指标

- [ ] 加速比 ≥ 1.5×（接近 DSpark paper 实测 1.6-1.85×）
- [ ] 跨模型泛化：同样代码跑 Qwen3-8B 也工作
- [ ] 文档完整：更新 `benchmarks/phase6_dspark_e2e_report.md`

---

## 7. 文档交付物

| 文件 | 内容 |
|---|---|
| `src/fusion_draft_model.cpp` | 完整 attention + markov head 实现（替换骨架） |
| `tools/dspark_to_gguf.py` | 扩展为完整 reader + writer |
| `tests/test-fusion-spec-decode-e2e.cpp` | E2E 测试（加载真实 GGUF + 100 tokens） |
| `benchmarks/phase6_dspark_e2e_report.md` | 真实加速比数字 + 内存数据 |

---

## 8. 与 Path C 主线的关系

**Phase 6 完成后的 Path C 主线工作**：
- Phase 3 (KV cache 分层) — 设计完成，待实现
- Phase 4 (长上下文 32K-128K 测试) — 待启动
- Phase 5 (Ollama 集成) — 待启动

**Phase 6 完成度 → 100% 后的下一步候选**：

| 选项 | 工作量 | 价值 |
|---|---|---|
| A. Phase 3 KV cache 分层实现 | 2-3 周 | Path C 主线推进 |
| B. Phase 6 优化（madvise + 70B） | 2-3 周 | 性能调优 |
| C. Ollama 集成（Path C Phase 5） | 2 周 | 用户可用性 |
| D. 招聘 Metal 工程师 + 团队对齐 | 1-2 周 | 团队扩张 |

---

## 9. 一句话总结

**2 周 sprint 目标**：让 DSpark × FusionLLM 端到端跑通 + 真实加速比数字。

**关键技术风险**：Qwen3DSparkAttention 双输入 attention 实现（ggml concat + RoPE 拼装）。

**最大 ROI**：S7（attention 实现）完成后，其余 S8/S9/S10 是相对直接的串联工作。

**资源**：M5 Air 16GB + HuggingFace 预训练 checkpoint（无需训练）。

---

*开发方案 v1 - 2026-06-29 01:40*
*基于 Phase 6 S1-S6 60% 完成状态*
*基于 DeepSpec HEAD (2026-06-28 clone) 完整源码分析*
