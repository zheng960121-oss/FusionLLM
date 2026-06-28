# DSpark × FusionLLM 详细技术规范 (S1 Deliverable)

**日期**：2026-06-28
**阶段**：S1 完成
**基于**：DeepSpec HEAD (2026-06-28 clone) 完整源码分析
**目标读者**：S2-S6 实现工程师

---

## 1. 范围

本文档定义 DSpark × FusionLLM 集成的：
1. 详细数据流图（含每个张量的 shape）
2. C++ API 接口（类层次 + 公共方法）
3. ggml ops 映射表（每个 PyTorch op 对应 ggml op）
4. 测试用例（端到端数值一致性）

参考实现：[DeepSpec HEAD](https://github.com/deepseek-ai/DeepSpec)，重点文件：
- `deepspec/modeling/dspark/qwen3/modeling.py` (533 行)
- `deepspec/eval/dspark/draft_ops.py` (153 行)
- `deepspec/modeling/dspark/markov_head.py` (319 行)
- `deepspec/modeling/dspark/common.py` (309 行)
- `deepspec/eval/base_evaluator.py` (`generate_decoding_sample` 是 spec decoding 主循环)

---

## 2. 详细数据流图

### 2.1 Spec Decoding 单步数据流

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入:
  - input_ids:        [B=1, L_ctx]          (已接受的 context)
  - past_kv_target:   DynamicCache           (target KV cache)
  - past_kv_draft:    DynamicCache           (draft KV cache)
  - target_hs:        [1, ctx_len, d]        (从 target 中间层抽出)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【PROPOSE 阶段】
  draft_input = [accepted_token, MASK × (block_size-1)]   shape: [1, block_size]
  draft_emb   = embed_tokens(draft_input)                shape: [1, block_size, d]
  
  ── fusion_draft_model.forward(draft_emb, target_hs)
  │   ┌─────────────────────────────────────────────────────┐
  │   │ target_hs_proj = RMSNorm(Linear(target_hs))         │ [1, ctx_len, d]
  │   │   # Linear weight: [len(target_layer_ids)*d, d]
  │   │   # 这里是关键 fusion 点：把多个 target layer 的 hs 拼成一个
  │   ├─────────────────────────────────────────────────────┤
  │   │ for layer in 5× Qwen3DSparkDecoderLayer:            │
  │   │   # Standard Pre-Norm block, 但 attention 是双输入:  │
  │   │   residual = hidden_states                          │ [1, block_size, d]
  │   │   hidden_states = RMSNorm(hidden_states)            │
  │   │   attn_out = DSparkAttention(                        │
  │   │       hidden_states,       # Q from draft emb        │
  │   │       target_hidden_states # K/V from context + noise│
  │   │   )                                                  │
  │   │   hidden_states = residual + attn_out                │
  │   │   residual = hidden_states                            │
  │   │   hidden_states = RMSNorm(hidden_states) + MLP(hidden_states)
  │   └─────────────────────────────────────────────────────┘
  draft_hidden = RMSNorm(final_hidden)                    shape: [1, block_size, d]
  draft_logits = Linear(draft_hidden)                     shape: [1, block_size, V]
  
  # Markov head 调整 logits (基于 prev_token + hidden_state)
  draft_logits = markov_head.apply_block_logits(...)
  
  # Sample block_size 个 draft tokens
  draft_tokens, draft_probs = markov_head.sample_block_tokens(...)
                                                       shape: [1, block_size], [1, block_size, V]

【VERIFY 阶段】
  verify_input = [accepted_token] + draft_tokens          shape: [1, block_size + 1]
  
  target_output = target_model.forward(verify_input, past_kv_target)
  target_logits = target_output.logits                   shape: [1, block_size + 1, V]
  target_hs_new = target_output.hidden_states[k]         # k ∈ target_layer_ids
                                                       shape: [1, block_size + 1, d] × len(layer_ids)
  
  # Rejection sampling (per draft position i):
  accept_prob[i] = min(1, target_probs[i, draft_tokens[i]] / draft_probs[i, draft_tokens[i]])
  accept_mask[i] = (rand() < accept_prob[i]).cumprod()  # 一遇 reject 全停
  
  accepted_count = sum(accept_mask)
  bonus_token    = resample(target_probs[accepted_count], draft_probs[accepted_count])
  
【COMMIT 阶段】
  committed = [draft_tokens[:accepted_count], bonus_token]   shape: [1, accepted_count + 1]
  output_ids[:, start:start+accepted_count+1] = committed
  
  # KV cache crop
  past_kv_target.crop(start + accepted_count + 1)
  past_kv_draft.crop(start + accepted_count + 1)
  
  # Sliding window (FusionLLM 协同)
  for il in target_layer_ids:
      fusion_window_advance(il, accepted_count + 1)  # 触发 mlock
  
  start += accepted_count + 1

【UPDATE 阶段】
  target_hs = extract_context_feature(target_hs_new, target_layer_ids)
  # extract_context_feature: 从 hidden_states[layer_id + 1] 抽出并 concat
  # 注意: hidden_states[0] 是 embedding output, hidden_states[k+1] 是 layer k output
  # 然后 RMSNorm + Linear 投影回去（已在 _forward_backbone 头部）
```

### 2.2 Hidden State 抽取协议

**关键不变量**：`extract_context_feature` 必须返回 `[1, seq_len, len(target_layer_ids) * d]`，按 layer_ids 升序拼接。

```python
def extract_context_feature(hidden_states, layer_ids):
    return torch.cat(
        [hidden_states[0 if layer_id == -1 else layer_id + 1] for layer_id in layer_ids],
        dim=-1,
    )
```

注意：
- `hidden_states` 是 tuple，长度 = num_target_layers + 1（layer 0 ~ num_target_layers + final norm）
- `hidden_states[0]` = embedding output
- `hidden_states[k]` = layer k-1 output (k ≥ 1)
- 所以 `hidden_states[layer_id + 1]` = layer `layer_id` output
- `layer_id == -1` 时取 embedding output（即 layer 0 input）

**对应 C++ 接口**：`fusion_extract_context_hs(ggml_context* ctx, ggml_tensor* hidden_states_tuple, int* layer_ids, int n_layer_ids)`

---

## 3. C++ API 接口设计

### 3.1 类层次总览

```
fusion/
├── fusion_draft_model.h/cpp      # DSpark draft model
│   ├── class FusionDSparkConfig  # 配置（block_size, num_layers, target_layer_ids, markov_rank）
│   ├── class FusionDSparkModel   # 主模型类
│   │   ├── bool load_from_gguf(path)
│   │   ├── ggml_tensor* forward(ggml_context*, target_hs, draft_input_emb, position_ids)
│   │   └── void sample_block_tokens(...)  // 复用 llama.cpp sampling
│   └── class FusionMarkovHead    # Markov head
│       ├── VanillaMarkov
│       ├── GatedMarkovHead
│       └── RNNHead
│
├── fusion_hs_extract.h/cpp        # Hidden state 抽取
│   ├── class FusionHSExtractor
│   │   ├── ggml_tensor* extract_target_layers(target_hidden_states, layer_ids)
│   │   └── void hook_into_llama_forward(...)  // 注册 hook 到 target model
│
├── fusion_speculative_decode.h/cpp # Spec decoding loop
│   ├── class FusionSpecDecoder
│   │   ├── int step(...)           # 单步 propose + verify + commit
│   │   ├── int generate(...)       // 完整 generate(token_ids, max_new_tokens)
│   │   └── AcceptanceStats stats  # accepted_count, draft_tokens_per_proposal, etc.
│
└── fusion_window_speculative.h/cpp # 与 FusionLLM 滑动窗口协同
    └── void on_verify_step(layer_ids, accessed_positions)
```

### 3.2 FusionDSparkConfig

```cpp
struct FusionDSparkConfig {
    // 从 GGUF metadata 读取（自定义 keys）
    int32_t block_size = 7;            // fusion.draft.block_size
    int32_t num_draft_layers = 5;       // fusion.draft.num_layers
    std::vector<int32_t> target_layer_ids = {1, 9, 17, 25, 33};  // fusion.draft.target_layer_ids
    int32_t markov_rank = 256;          // fusion.draft.markov_rank
    std::string markov_head_type = "vanilla";  // fusion.draft.markov_head_type
    int32_t mask_token_id = 151669;     // fusion.draft.mask_token_id
    bool enable_confidence_head = false;  // 第一版省略
};
```

GGUF metadata 扩展（添加到 GGUF 写入）：
```python
# Python training side:
metadata = {
    "fusion.draft.block_size": 7,
    "fusion.draft.num_layers": 5,
    "fusion.draft.target_layer_ids": [1, 9, 17, 25, 33],
    "fusion.draft.markov_rank": 256,
    "fusion.draft.markov_head_type": "vanilla",
    "fusion.draft.mask_token_id": 151669,
}
```

### 3.3 FusionDSparkModel

```cpp
class FusionDSparkModel {
public:
    // 加载 draft GGUF（独立文件，例如 draft_model.gguf）
    bool load_from_gguf(const std::string& path);
    
    // 初始化时复用 target 的 embed_tokens 和 lm_head（共享权重）
    void share_target_embeddings(ggml_tensor* target_embed, ggml_tensor* target_lm_head);
    
    // 主 forward（inference）
    // target_hs:        [1, ctx_len, len(layer_ids) * d]
    // draft_input_ids:  [1, block_size]
    // position_ids:     [1, block_size]
    // past_kv_draft:    之前的 draft KV cache
    // 返回: [1, block_size, V] logits
    ggml_tensor* forward(
        ggml_context* ctx,
        ggml_tensor* target_hs,
        ggml_tensor* draft_input_ids,
        ggml_tensor* position_ids,
        llama_kv_cache* past_kv_draft
    );
    
    // 采样 block_size 个 draft tokens + Markov head 调整
    // 返回: (sampled_token_ids [1, block_size], probs [1, block_size, V])
    std::pair<ggml_tensor*, ggml_tensor*> sample_block_tokens(
        ggml_context* ctx,
        ggml_tensor* logits,
        ggml_tensor* first_prev_token_id,
        float temperature = 0.0f
    );
    
    // 配置访问器
    const FusionDSparkConfig& config() const { return cfg_; }

private:
    FusionDSparkConfig cfg_;
    // 权重 tensors（从 GGUF 加载）
    ggml_tensor* fc_weight_ = nullptr;           // [d, len(layer_ids) * d]
    ggml_tensor* hidden_norm_weight_ = nullptr; // [d]
    ggml_tensor* embed_tokens_weight_ = nullptr; // [V, d]（可能与 target 共享）
    ggml_tensor* lm_head_weight_ = nullptr;      // [V, d]（可能与 target 共享）
    std::vector<ggml_tensor*> attn_q_, attn_k_, attn_v_, attn_o_;  // 5 层
    std::vector<ggml_tensor*> attn_q_norm_, attn_k_norm_;
    std::vector<ggml_tensor*> mlp_gate_, mlp_up_, mlp_down_;
    std::vector<ggml_tensor*> input_layernorm_, post_attention_layernorm_;
    ggml_tensor* final_norm_weight_ = nullptr;
    
    // Markov head
    std::unique_ptr<FusionMarkovHead> markov_head_;
};
```

### 3.4 FusionHSExtractor

```cpp
class FusionHSExtractor {
public:
    // 注册 hook 到 llama_model forward：在指定 layer_ids 处复制 hidden state
    void hook_layers(llama_model* model, const std::vector<int32_t>& layer_ids);
    
    // 提取最近一次 forward 的 target hidden states
    // 返回: [1, seq_len, len(layer_ids) * d]（已拼接）
    ggml_tensor* extract(ggml_context* ctx);
    
private:
    std::vector<int32_t> hooked_layer_ids_;
    std::vector<ggml_tensor*> saved_hs_;  // 每个 hooked layer 一个
    llama_model* hooked_model_ = nullptr;
};
```

**Hook 实现思路**：修改 `llama-model.cpp::forward()`，在指定 layer 输出后调用回调保存 hidden state。

### 3.5 FusionSpecDecoder

```cpp
class FusionSpecDecoder {
public:
    FusionSpecDecoder(
        llama_model* target_model,
        FusionDSparkModel* draft_model,
        FusionHSExtractor* hs_extractor,
        llama_context* target_ctx,
        FusionWindowManager* window_mgr  // FusionLLM
    );
    
    // 完整 generate
    struct GenerateResult {
        std::vector<int32_t> output_ids;
        int num_accepted_draft;       // 总接受的 draft tokens
        int num_proposed;             // 总 propose 的 draft tokens
        int num_verify_calls;         // verify 调用次数（= num_proposed / block_size）
        double acceptance_length;     // 平均每个 verify step 接受的 tokens
    };
    
    GenerateResult generate(
        const std::vector<int32_t>& input_ids,
        int max_new_tokens,
        float temperature = 0.0f,
        const std::vector<int32_t>& stop_token_ids = {}
    );

private:
    // 单步 propose + verify + commit
    int step_spec(
        std::vector<int32_t>& output_ids,  // 当前已接受的所有 token
        int& start,                          // 当前生成位置
        llama_kv_cache& target_kv,
        llama_kv_cache& draft_kv,
        ggml_tensor*& target_hs_out,        // 下次 propose 用的 hidden states
        std::mt19937& rng
    );
    
    // Fallback autoregressive（draft model 拒绝时）
    int step_autoregressive(...);
};
```

### 3.6 与 FusionLLM 滑动窗口的协同

```cpp
class FusionWindowManager {
public:
    // 原 Phase 2 接口
    void on_layer_access(int il);
    
    // 新增：spec decode 协调
    void on_verify_step(const std::vector<int32_t>& accessed_layer_ids, int accepted_count);
    // 在 verify step 结束后调用，参数是这次 verify 实际访问的所有 layer（= target_layer_ids）
    // 内部：调用 fusion_window_advance 多次，触发 mlock
};
```

**协同规则**：
- **propose 阶段**：只跑 draft model，不触发 target 的 window advance（因为 target model 不被访问）
- **verify 阶段**：跑 target model 一次 forward，触发 `cb_func(il)`，所有访问的 layer 都会 mlock
- **commit 阶段**：根据 accepted_count 计算窗口滑动

**效果**：在 70B target + 6 层滑动窗口下：
- autoregressive: 每 token 一次 target forward → 6 层滑动（每 6 个 token 切换一次）
- spec decoding (block_size=7, accept=5): 每 verify 一次 target forward → 6 层滑动，但平均 6 tokens 才一次 verify（accept 5 + bonus 1）
- **窗口切换频率降低 ~6 倍**

---

## 4. ggml Ops 映射表

| PyTorch op | ggml op | 备注 |
|---|---|---|
| `nn.Embedding.forward(id)` | `ggml_get_rows(weight, id)` | weight: `[V, d]`, id: `[B, S]` |
| `nn.Linear.forward(x)` | `ggml_mul_mat(weight, x)` | weight: `[out, in]`, x: `[B, S, in]` → `[B, S, out]` (注意 ggml 是 `[in, out]` 存储) |
| `Qwen3RMSNorm(x)` | `ggml_rms_norm(x, weight)` | weight: `[d]` |
| `apply_rotary_pos_emb(q, k, cos, sin)` | `ggml_rope(q, cos, sin)` + 类似 k | ggml 已原生支持 |
| Attention forward | `ggml_flash_attn_ext` 或自定义 | **关键挑战**: DSparkAttention 的 K/V 是拼接的，需要专门处理 |
| `torch.cat([k_ctx, k_noise], dim=1)` | `ggml_concat(k_ctx, k_noise, 1)` | ggml 原生支持 |
| Softmax | `ggml_soft_max` | 标准 |
| Sample categorical | llama.cpp `llama_sample_token` | 复用 |
| Top-K / Top-P | llama.cpp `llama_sample_top_k/top_p` | 复用 |
| Markov head `Embedding` | `ggml_get_rows` | vocab-level bias embedding |
| Markov head `Linear` | `ggml_mul_mat` | bias 投影 |

### 4.1 DSparkAttention 的 ggml 实现

DSparkAttention 是最大挑战，原始 PyTorch 实现：

```python
q = q_proj(hidden_states)              # [B, q_len, H*head_dim]
k_ctx = k_proj(target_hidden_states)   # [B, ctx_len, H_kv*head_dim]
k_noise = k_proj(hidden_states)        # [B, q_len, H_kv*head_dim]
k = cat([k_ctx, k_noise], dim=1)       # [B, ctx_len + q_len, H_kv*head_dim]
v = cat([v_proj(target), v_proj(noise)], dim=1)  # 同上
```

**ggml 实现思路**：
```cpp
// 1. Q from draft input
q = ggml_mul_mat(self.q_proj_w, hidden_states);   // [d, q_len*hidden]
q = ggml_reshape(q, ...);
q = ggml_rms_norm(q, self.q_norm_w);              // per-head rmsnorm

// 2. K from target context
k_ctx = ggml_mul_mat(self.k_proj_w, target_hs);    // [d, ctx_len*hidden]
v_ctx = ggml_mul_mat(self.v_proj_w, target_hs);    // [d, ctx_len*hidden]

// 3. K from draft input (noise)
k_noise = ggml_mul_mat(self.k_proj_w, hidden_states);  // [d, q_len*hidden]
v_noise = ggml_mul_mat(self.v_proj_w, hidden_states);  // [d, q_len*hidden]

// 4. Concat K/V
k = ggml_concat(ctx, k_ctx, k_noise, 1);   // [B, ctx_len + q_len, ...]
v = ggml_concat(ctx, v_ctx, v_noise, 1);

// 5. Apply RoPE (必须整段一起做，因为 position_ids 是连续的)
//    target 的 position 是 [start, start+ctx_len)
//    draft 的 position 是 [start+ctx_len, start+ctx_len+q_len)
cos = ggml_compute_cos(position_ids);
sin = ggml_compute_sin(position_ids);
q = ggml_rope(q, cos, sin);
k = ggml_rope(k, cos, sin);

// 6. SDPA / Flash Attention
attn_out = ggml_flash_attn_ext(ctx, q, k, v, mask, scale);  // [B, q_len, H*head_dim]

// 7. Output projection
attn_out = ggml_mul_mat(self.o_proj_w, attn_out);
```

**关键点**：RoPE 必须知道每个位置的 position_id。target context 和 draft input 在序列上是连续的，所以 `position_ids` 是一个完整范围 [start, start + ctx_len + q_len)。

### 4.2 Markov Head Vanilla 实现

```python
# VanillaMarkov.forward:
prev_emb = markov_w1(prev_token_id)   # [B, markov_rank]  (Embedding lookup)
bias = markov_w2(prev_emb)            # [B, vocab]         (Linear)
corrected_logits = base_logits + bias
```

ggml 等价：
```cpp
prev_emb = ggml_get_rows(markov_w1_w, prev_token_id);  // [B, markov_rank]
bias = ggml_mul_mat(markov_w2_w, prev_emb);              // [B, vocab] (注意: ggml 是 [vocab, markov_rank])
corrected_logits = ggml_add(base_logits, bias);
```

### 4.3 Hidden State 抽取的 ggml 实现

```cpp
// extract_context_feature:
// 输入: hidden_states 是 tuple[num_target_layers + 1]
// 输出: [B, S, len(target_layer_ids) * d]

std::vector<ggml_tensor*> parts;
for (int layer_id : target_layer_ids) {
    ggml_tensor* hs = (layer_id == -1) 
        ? hidden_states[0] 
        : hidden_states[layer_id + 1];
    parts.push_back(hs);
}
ggml_tensor* concat = ggml_concat(ctx, parts[0], parts[1], 2);  // dim=-1
// ...
```

但问题：现有 llama.cpp forward 中 `hidden_states` 不直接对外暴露。需要修改 forward 流程让它在指定 layer 保存中间结果。

**实现方案**：
- 在 `llama_model.forward()` 中，遍历每层后检查 `if layer_idx in target_layer_ids`，复制 `hidden_states` 到 `saved_hs[layer_idx]`
- 或者更优雅：用一个全局 `FusionHSExtractor` 实例，forward 时回调

---

## 5. 测试用例

### 5.1 数值一致性测试（最重要）

**目标**：C++ 实现与 PyTorch 参考实现对同一输入产出 bit-exact（或 < 1e-3 tolerance）的 logits。

```cpp
// tests/test_dspark_numerical_match.cpp
TEST(DSpark, Qwen3_4B_Block7_ForwardMatchesPyTorch) {
    // 1. 加载 PyTorch 参考输出（从 training 时 dump 的 .pt 文件）
    auto ref_logits = load_tensor("tests/ref_data/dspark_qwen3_4b_block7_logits.pt");
    
    // 2. 准备 C++ 端输入（同样 prompt + 同样 draft input）
    std::vector<int32_t> prompt = {151644, 872, 198, ...};  // 真实 prompt tokens
    std::vector<int32_t> draft_input = {prompt.back(), MASK, MASK, MASK, MASK, MASK, MASK};
    auto target_hs = load_tensor("tests/ref_data/dspark_qwen3_4b_target_hs.pt");
    
    // 3. C++ forward
    FusionDSparkModel draft_model;
    draft_model.load_from_gguf("models/qwen3-4b-dspark-block7-q4.gguf");
    auto cpp_logits = draft_model.forward(ctx, target_hs, draft_input, position_ids, nullptr);
    
    // 4. 对比
    ASSERT_LT(max_abs_diff(cpp_logits, ref_logits), 1e-3);
}
```

**数据准备**：用 DeepSpec training pipeline dump 一组 (target_hs, draft_input, expected_logits) 三元组，作为 C++ 实现的 reference。

### 5.2 单元测试

```cpp
TEST(MarkovHead, VanillaForward) {
    // 测试 vanilla Markov head 单步 forward
    FusionMarkovHead head(MarkovType::VANILLA, vocab_size=1000, markov_rank=64);
    auto logits = random_tensor({1, 7, 1000});
    auto prev_id = random_int_tensor({1, 7});
    auto corrected = head.apply_block_logits(logits, prev_id, hidden_states);
    // 验证: corrected - logits = markov_w2(markov_w1(prev_id))
}

TEST(MarkovHead, GatedForward) {
    // 测试 gated Markov head
    FusionMarkovHead head(MarkovType::GATED, vocab_size=1000, markov_rank=64, hidden_size=512);
    // ... 类似
}

TEST(HSExtract, ConcatCorrectOrder) {
    // 验证 extract 返回的 hidden state 是按 layer_ids 升序拼接
    auto hs_tuple = make_dummy_hidden_states(36, 1, 10, 128);  // 36 层, batch=1, seq=10, d=128
    auto extracted = extract_target_layers(hs_tuple, {1, 9, 17});
    // 验证 shape = [1, 10, 3 * 128] = [1, 10, 384]
    // 验证 extracted[0, :, :128] == hs_tuple[2]  (layer_id + 1 = 2)
    // 验证 extracted[0, :, 128:256] == hs_tuple[10]
    // 验证 extracted[0, :, 256:384] == hs_tuple[18]
}

TEST(SpecDecode, FullStep) {
    // 测试单步 spec decode（不带 sliding window）
    auto target = load_model("qwen3-4b-q4.gguf");
    auto draft = load_draft_model("qwen3-4b-dspark-q4.gguf");
    auto decoder = FusionSpecDecoder(target.get(), draft.get(), ...);
    
    std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    auto result = decoder.generate(prompt, max_new_tokens=20);
    
    // 验证：output 应该与 PyTorch reference 输出一致
    auto ref_output = load_tensor("tests/ref_data/dspark_qwen3_4b_output.pt");
    ASSERT_EQ(result.output_ids, ref_output);
}

TEST(WindowSpecCoord, SlidingWindowCorrectness) {
    // 测试 spec decode 与 sliding window 的协同
    FusionWindowManager wm(window_size=6);
    FusionSpecDecoder decoder(target, draft, hs_extractor, ctx, &wm);
    
    // mock verify 阶段访问 layer 10
    wm.on_verify_step({10}, accepted_count=5);
    
    // 验证 layer 10 的 mlock 被触发
    ASSERT_TRUE(wm.is_layer_mlocked(10));
    // 验证 [10-6+1, 10-1] = [5, 9] 的 mlock 被维持
    for (int il = 5; il <= 9; il++) {
        ASSERT_TRUE(wm.is_layer_mlocked(il));
    }
}
```

### 5.3 集成测试

```cpp
TEST(EndToEnd, Qwen3_4B_WithDSparkOnM5Air) {
    // 端到端：Qwen3-4B target + DSpark draft + FusionLLM sliding window
    auto target = load_model("models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf");
    auto draft = load_draft_model("models/qwen3-4b-dspark-q4.gguf");
    FusionSpecDecoder decoder(target, draft, hs_extractor, ctx, &window);
    
    // 生成 100 tokens
    auto result = decoder.generate(prompt_tokens, 100);
    
    // 验收:
    // 1. 输出文本质量与 reference 一致（perplexity < 5% 退化）
    // 2. 加速比 ≥ 1.3x（vs autoregressive）
    // 3. 内存峰值 < 16GB
    // 4. Sliding window 切换次数 < autoregressive 的 1/3
}
```

---

## 6. 实施清单 (S2-S6)

### 6.1 S2 (Hidden State 抽取, 2 周)

- [ ] 修改 `src/llama-model.cpp::forward()` 暴露中间层 hidden state
- [ ] 创建 `src/fusion_hs_extract.cpp/h`
- [ ] 单元测试：`extract_target_layers` 输出 shape / 顺序正确
- [ ] 集成测试：从 7B model 抽 target_layer_ids=[1,9,17,25,33] hidden state，与 PyTorch 对比 < 1e-3

### 6.2 S3 (Draft Model 移植, 2 周)

- [ ] 创建 `src/fusion_draft_model.cpp/h`
- [ ] GGUF writer 脚本（Python 端）：把 PyTorch DSpark checkpoint → GGUF，带自定义 metadata
- [ ] 实现 Qwen3DSparkAttention（双输入 attention）
- [ ] 实现 Qwen3DSparkDecoderLayer（Pre-Norm block）
- [ ] 实现 Markov head (vanilla 优先，gated/rnn 后续)
- [ ] Confidence head 第一版省略
- [ ] 数值一致性测试（与 PyTorch 对比）

### 6.3 S4 (Spec Decoding Loop, 1 周)

- [ ] 创建 `src/fusion_speculative_decode.cpp/h`
- [ ] `step_spec`: propose + verify + commit
- [ ] Fallback autoregressive（draft 拒绝时）
- [ ] Rejection sampling（精确匹配 PyTorch 数值）
- [ ] 端到端测试：Qwen3-4B + DSpark 跑 100 tokens，输出与 reference 一致

### 6.4 S5 (与 Sliding Window 协同, 1 周)

- [ ] 扩展 `src/fusion_window.cpp` 加 `on_verify_step` 接口
- [ ] 修改 `src/fusion_speculative_decode.cpp` 在 verify 结束后调用
- [ ] 单元测试：window 切换次数 / 时序正确

### 6.5 S6 (Benchmark + 报告, 1 周)

- [ ] 在 Qwen3-4B + Qwen2.5-7B 上跑 benchmark
- [ ] 测加速比（target only vs target + draft）
- [ ] 测内存峰值（无 window vs sliding window）
- [ ] 写 `benchmarks/fusion_dspark_70b_report.md`

---

## 7. 风险与缓解（更新）

| 风险 | 概率 | 影响 | 缓解 |
|:---|:---:|:---:|:---|
| DSpark checkpoint → GGUF 数值漂移 | 中 | 高 | **S2 必做**：同 prompt 同权重对比 PyTorch vs C++ 输出 diff < 1e-3 |
| DSparkAttention 的 ggml concat+rope 性能 | 中 | 中 | **S3 早期**：profile 这个 op，如果瓶颈考虑 cuBLAS 风格 fused kernel |
| Hidden state 保存时机错位（异步 GPU 计算） | 高 | 高 | **S2 必做**：在 forward 后立即同步 + 强 unit test |
| Sliding window 与 spec decode 资源竞争 | 中 | 中 | **S5**：spec decode verify 完成后才触发 window_advance |
| 70B 在 16GB Air 上 OOM | 中 | 中 | **S6**：验证 70B target + 350MB draft + KV cache 是否超出 16GB；如超出转 64GB Mac Studio |

---

## 8. 后续文档

S1 完成。下一步：
1. S1 团队 review（如果团队存在）
2. 立项 S2-S6（拆 issue / 任务卡）
3. 准备数值参考数据集（dump PyTorch 输出作为 C++ 实现的 reference）

---

*详细技术规范 v1 - 2026-06-28 23:50*
*基于 DeepSpec HEAD commit (2026-06-28 clone) + DSpark paper PDF*
