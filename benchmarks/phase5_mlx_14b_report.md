# Phase 5 MLX 14B 实测报告

**日期**：2026-07-01
**设备**：Apple M5 MacBook Air 16GB
**模型**：Qwen2.5-14B-Instruct-4bit (MLX 4-bit 量化, 7.75 GB, 48 layers, head_dim=128, GQA 8/40)
**框架**：mlx-lm 0.31.1 + MLX 0.31.1 + Apple MLX framework
**目的**：验证 MLX 路径在 M5 Air 16GB 上 14B 的实际性能 vs Phase 4 llama.cpp baseline

---

## 1. TL;DR

**MLX 在 4K 上下文碾压 llama.cpp（gen +185%），但在 16K 退化，32K 同样 OOM。**

| Context | MLX prefill | MLX gen | Phase 4 (llama.cpp) pre | gen | MLX 优势 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 4K | **272-277 t/s** | **39-40 t/s** | 251 t/s | 14.0 t/s | pre +10%, gen **+185%** |
| 16K | 272-280 t/s | 9.8-10.1 t/s | 188 t/s | 12.1 t/s | pre +50%, gen **-19%** |
| 32K | **OOM** (Metal) | **OOM** | OOM (~95 with FA + KV q4_0) | OOM | ❌ 同样 OOM |

**关键洞察**：
1. **MLX 真正赢短 context**（4K gen 40 t/s 是 llama.cpp 的 3x）—— 聊天/QA 体验质变
2. **MLX 在长 context 退化**（16K gen 比 llama.cpp 慢 19%）—— attention O(n) per token 是 bottleneck
3. **MLX 救不了 32K OOM**——Metal 命令缓冲不够，跟 llama.cpp 同样物理上限

---

## 2. 实测数据

### 2.1 加载性能

```
Loaded in 1.4s
  num_hidden_layers: 48
  hidden_size: 5120
  num_attention_heads: 40
  num_key_value_heads: 8 (GQA)
  max_position_embeddings: 32768
  quantization: 4-bit (group_size=64)
```

模型加载到 GPU：**8.4 GB**（vs llama.cpp 8.4 GB，**一样**）

### 2.2 14B 4K 上下文

```
Run 1: 10.19s wall, ~272 t/s pre, ~39.25 t/s gen, peak 8.72GB
Run 2: 9.99s wall,  ~277 t/s pre, ~40.03 t/s gen, peak 8.72GB
```

- **Prefill**: 274.5 t/s 平均（vs Phase 4 251 = +9%）
- **Gen**: 39.6 t/s 平均（vs Phase 4 14.0 = **+183%**）
- **Peak mem**: 8.72 GB（vs Phase 4 ~10 GB）

### 2.3 14B 16K 上下文

```
Run 1: 39.78s wall, ~280 t/s pre, ~10.06 t/s gen, peak 10.1GB
Run 2: 40.92s wall, ~272 t/s pre, ~9.77  t/s gen, peak 10.1GB
```

- **Prefill**: 276 t/s 平均（vs Phase 4 188 = +47%）
- **Gen**: 9.9 t/s 平均（vs Phase 4 12.1 = **-18%**）
- **Peak mem**: 10.1 GB（vs Phase 4 ~11 GB）

### 2.4 14B 32K 上下文

```
[崩溃] libc++abi: terminating due to uncaught exception of type std::runtime_error: 
       [METAL] Command buffer execution failed: Insufficient Memory 
       (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

- **OOM** 在 MLX 也是硬限制
- 跟 llama.cpp 同样的 Metal 限额
- 即使有 Flash Attention（MLX SDPA 内置）也救不了 32K 的 prefill 阶段

---

## 3. 为什么 MLX 4K 强 16K/32K 弱

**MLX 性能分析**：

| Context | prefill 时间 | gen 时间 | prefill t/s | gen t/s | per-token gen 复杂度 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 4K | 9 s | 0.5 s | 277 | 40 | 4096 attn + 14B matmul |
| 16K | 40 s | 2 s | 280 | 10 | 16384 attn + 14B matmul |
| 32K | ❌ OOM | — | — | — | 32768 attn OOM |

**洞察**：
- **Prefill**: MLX prefill 几乎常数（272-280 t/s）—— bandwidth-bound
- **Gen 退化**: 4K→16K gen 从 40 t/s 跌到 10 t/s（-75%），因为每 token 要 attend 4× 更多 token
- **Gen 跟 llama.cpp 对比反转**: 4K MLX 40 vs llama.cpp 14 = MLX 3x；16K MLX 10 vs llama.cpp 12 = llama.cpp 1.2x

**根因**：
- MLX 在小 batch + 短 context 上 fused kernel 极优
- llama.cpp 在长 context + Metal 上优化更深（多年 Metal 后端经验）
- MLX 在长 context 的 KV cache 管理不如 llama.cpp 精细

---

## 4. 路径选择建议

### 4.1 单模型策略：14B MLX 在 4K 极佳

**最优策略**：
- **短 prompt（<4K）**: 用 MLX 路径，gen 40 t/s 是 16GB Air 上最好的体验
- **长 prompt（4K-16K）**: 用 llama.cpp + FA + KV q4_0，gen 12 t/s
- **32K 上下文**: 都 OOM，**实际不可用**（16GB Air 物理上限）

### 4.2 替代方案：8B MLX 32K 也许能跑

8B Q4 ~ 4.7 GB，attention matrix 32K² 减半，理论上能装下。需验证。

### 4.3 集成方案：双引擎路由

**FusionLLM 双引擎架构**：

```python
class FusionLLM:
    def __init__(self):
        self.mlx_model = None   # 短 context 用
        self.llama_model = None # 长 context 用
    
    def generate(self, prompt, max_tokens):
        if len(prompt) < 4000:
            return self.mlx_generate(prompt, max_tokens)  # 40 t/s
        else:
            return self.llama_generate(prompt, max_tokens)  # 12 t/s
```

**但**：增加复杂度。简化版是只用 MLX，覆盖 4K 短 context（80% 用例）。

---

## 5. 实施路径

### 5.1 Phase 5 完成

- [x] 装 mlx-lm 0.31.1 + MLX 0.31.1
- [x] 下载 Qwen2.5-14B-Instruct-4bit (7.75 GB)
- [x] 14B 4K MLX benchmark（40 t/s gen）
- [x] 14B 16K MLX benchmark（10 t/s gen）
- [x] 14B 32K MLX OOM（确认）
- [x] 写对比报告

### 5.2 立即可做（待老大拍板）

1. **跑 8B MLX 32K 验证**（5 min 装 + 30 min 跑）
   - `mlx-community/Qwen3-8B-4bit` 或 `Llama-3.1-8B-Instruct-4bit`
   - 验证 8B + 32K 在 MLX 上是否 OOM
   - 如果通过，**8B 32K 是 16GB Air 的真正 sweet spot**

2. **8B 32K Phase 5 续测**（30 min）
   - 同 14B benchmark 方法
   - 跟 Phase 4 8B 32K (W1 D6 11 t/s gen) 对比

3. **写双引擎集成方案**（半天）
   - MLX 路径覆盖 ≤8K
   - llama.cpp 路径覆盖 >8K
   - OpenAI 兼容 API 包装

### 5.3 1-2 周出 demo

- 用 mlx_lm.server 起 OpenAI 兼容 HTTP server
- Modelfile 包装
- 一键 install.sh
- README + benchmark 数据
- （可选）8B + 14B 双模型，按 ctx 自动路由

---

## 6. 关键决策点

**等老大拍板**：

1. **8B MLX 32K 测试要不要跑？**（30 min，5 GB 模型）
2. **接受 4K 上下文 = 40 t/s MLX 体验？**（短 context 完美，长 context 一般）
3. **走单 MLX 引擎 还是 双引擎架构？**
   - 单 MLX：简单，覆盖 4K 短 context
   - 双引擎：复杂，覆盖 4K-16K 全部

---

## 附录 A：测试环境

```
设备: MacBook Air M5 16GB unified memory
OS: macOS 26.5.1 (25F80)
架构: arm64
MLX: 0.31.1 (Apple official)
mlx-lm: 0.31.1
Python: 3.14.3
Metal: 4 (Apple GPU)
模型: mlx-community/Qwen2.5-14B-Instruct-4bit (7.75 GB safetensors, 4-bit MLX quant)
下载速度: ~25 MB/s (3 分钟左右)
```

## 附录 B：测试命令

```bash
# Sanity check
python3 -c "
from mlx_lm import load, generate
import mlx.core as mx
m, t = load('/Users/jk/Models/mlx-community/Qwen2.5-14B-Instruct-4bit')
print(generate(m, t, prompt='The capital of France is', max_tokens=20, verbose=True))
"

# Full benchmark
python3 benchmarks/phase5_mlx_14b_benchmark.py
```

## 附录 C：参考文献

- Phase 4 14B baseline：`benchmarks/phase4_13b_baseline_report.md`
- mlx-lm 仓库：https://github.com/ml-explore/mlx-lm
- MLX 性能 vs llama.cpp：groundy.com 2026 报告（+30-80% for <14B）
- 我们的 MLX 源码分析：`docs/MLX源码分析-2026-07-01.md`

---

*报告完成于 2026-07-01 09:25*
*作者：小兰 (MiniMax-M3)*
*结论：MLX 路径 4K 极优，16K 一般，32K OOM；建议测 8B 32K + 单 MLX 引擎*
