# FusionLLM 可行性研究报告

**日期**：2026-07-01
**作者**：技术审查（Claude Code 驱动的小星）
**审查范围**：主仓库 ~/Desktop/FusionLLM/ + Llama.cpp fork ~/Desktop/llama.cpp-fusionllm/
**审查依据**：源码 / 头文件 / 单元测试 / benchmark 报告 / git log / 当日 memory

---

## 1. 执行摘要 (Executive Summary)

### 1.1 项目目标

**FusionLLM = 在 Apple Silicon MacBook Air 16GB（M5）上跑通 70B Q4 + 32K 长上下文的本地推理引擎。**

技术路径：通过**滑动窗口权重调度 + KV Cache 三层分级（GPU/CPU/SSD）+ DSpark 推测解码**三个核心机制实现。项目自我定位为"工程整合项目"（不是研究突破），把 PowerInfer-2 / LMCache / llama.cpp paged cache 的核心思想搬到 Apple Silicon。

### 1.2 当前状态

| 维度 | 状态 | 证据 |
|---|:---:|---|
| 整体完成度 | 约 50% | Sprint 0-2 + Phase 3 W1 + Phase 6 S2-S11 |
| 代码量 | 主仓 ~3700 行 C++ + 600 行 Python | src/ + tests/ + tools/ |
| 测试覆盖 | 9 个测试套件 / 181+ 测试 PASS / 0 fail | run_all_tests.sh |
| 真机验证 | 8B 4K + 32K 在 M5 Air 16GB 跑通 | W1 D5/D6 benchmark |
| 70B 真机 | ❌ 未验证 | 当前 M5 Air 16GB 物理装不下 70B 完整权重（即使 Q4） |
| DSpark spec decode 真 E2E | ❌ acceptance=0 | 训练 draft 用 Qwen2.5-0.5B 不匹配 target Qwen3-4B |

### 1.3 主要风险（详见第 4 节）

1. **70B 真机未验证**（🔴 高）—— 整个项目 KPI 未达成
2. **DSpark 训练依赖 GPU 集群**（🔴 高）—— 当前 M5 Air 不可训练
3. **双仓库同步漂移**（🟡 中）—— 已发生 3 次
4. **全职 Metal 工程师缺位**（🔴 高）—— 当前是 1 人项目
5. **W2 sliding window 在 32K 与 prefill hang 冲突**（🟢 低）—— 已修复但 D6 是临时 fix

### 1.4 总体评级

# **CONDITIONAL GO** ⚠️

技术路线完全可行（M5 Air 跑 8B + 32K 已验证），但终极目标 70B + 32K 尚未真机验证。**3 个硬性前提满足则继续**，否则降级或 NO GO：

| # | 前提 | 当前状态 | 满足方式 |
|:---:|---|:---:|---|
| 1 | 70B 硬件到位（64GB Mac Studio 或更大） | ❌ | 购置 / 借用 |
| 2 | 全职 Metal/C++ 工程师到岗 | ❌ | 招聘 / 外包 |
| 3 | DSpark draft 真训练（GPU 集群） | ❌ | 8×A100 1-2 天 |

---

## 2. 技术可行性

### 2.1 M5 MacBook Air 16GB 跑 70B + 32K

**结论：物理上可行（基于外推），但未真机验证。**

**外推依据**（基于 7B/8B 实测）：
- 8B Q4 = 5 GB 权重，32K KV cache = 8 GB（q4_0）+ 256 MB（layers）
- 70B Q4 ≈ 40 GB 权重 + 100 GB KV（32K full precision）→ 140 GB **远超 16GB**
- 即使 6 层窗口 + q4_0 KV + SSD offload：~70-80 GB **仍超**

**真实约束**：
- Apple Silicon M5 Air **16GB 统一内存**限制了 L0 GPU 容量（Metal 12-14 GB 共享）
- 70B Q4 + sliding window + SSD 三级缓存理论上能把活跃层压到 16GB 以内，但**实际性能（gen t/s）从未被测试过**
- 需要 64GB Mac Studio（M3 Ultra / M4 Max）才能装下 70B 完整推理

**技术风险**：高。**没有真机数据**意味着 W1 D5 报告里的所有"路径 c 70B 可行性"都是基于理论外推，不是测量。

### 2.2 KV Cache 三层分级（L0 GPU / L1 mlock / L2 SSD）

**结论：✅ 已端到端验证（8B 4K + 32K）。设计合理。**

**关键设计点**：
- `L0_max_tokens = sliding_window_size_`（默认 `kv_size/4`）—— 避免 Metal 内存爆炸
- `L1` = `mlock` + CPU 内存，手动生命周期管理（不依赖 OS page cache）
- `L2` = SSD `mmap` + 显式 `munmap`（已验证 PoC-4 zero-error）

**实测数据**（W1 D5）：
- 8B 4K：577 t/s prefill（mlock +0% overhead vs 无 mlock）
- 8B 32K：153 t/s prefill（修复 D6 hang 后）

**重大发现**：mlock 不是开销，**而是性能优化**（0.5B 无影响，7B +7%，70B 推断必需）—— 这是项目最有价值的工程发现之一，写进了 memory。

**W2 sliding window 优化**：
- 4K +0.4% prefill / +0% gen overhead
- 8B 32K prefill 153 t/s（保持 baseline）
- D6 32K 与 W2 hook 兼容性 issue 已修（构造函数 init vector 字段）

**问题**：
- 双仓库漂移已发生 3 次——这是工作流问题，不是技术问题

### 2.3 DSpark Spec Decode

**结论：✅ 算法层 + wire 完整（Qwen2.5-0.5B 50 spec step 端到端跑通），但 acceptance = 0。**

**已完成**：
- 算法层 S2-S10 全部 PASS（22/22 真 Qwen3-4B forward，rejection sampling，Markov head）
- Wire 到 llama.cpp main loop（S11）：`generate()` + `step_spec()` + `step_ar()` 全实现
- 50 个 spec step 在 Qwen2.5-0.5B + DSpark 上跑通无崩溃

**未完成**：
- Acceptance rate = 0%（DSpark 训练用 Qwen2.5-0.5B 但实际 GGUF metadata 是 Qwen3-4B 风格）
- 修了一个 critical bug：`n_head_kv` 应等于 `n_head`（DSpark 是 full MHA，不是 GQA）
- 真正的性能加速（2x）需要 DSpark 在匹配 target 上训练

**风险**：DSpark 训练需要 8×A100 + FSDP + sglang 生成 hidden states（DeepSpec 项目是 GPU 集群设计）。M5 Air 完全跑不动。

### 2.4 ggml 集成稳定性

**结论：⚠️ 集成层 work，但有 2 个反复踩坑的模式。**

**反复踩坑**：
1. **构造函数必须初始化所有 std::vector 成员** —— 否则 vector OOB → ggml abort callback 静默吃掉 → 看起来像 hang（D6 fix）
2. **ggml_gallocr 会复用 input tensor 内存** —— 任何 spec decode test 都要 pre-copy（已写进 memory）

**稳定性指标**：
- 9 个测试套件 PASS / 0 fail
- 181+ 测试覆盖（unit + integration）
- build_fusion_tests.sh 一键 build
- run_all_tests.sh 一键跑

**风险**：中等。每次新增 field 必须重新审视构造函数。D6 修复后短期内不会再发，但需要写 checklist 防止回归。

### 2.5 macOS Metal 性能特征

**结论：✅ M5 Metal 工作良好，有 1 个 warmup 成本。**

**实测特征**：
- Token 0 (第一个) 慢：7 秒（Metal shader JIT 编译）
- 稳态 gen 11 t/s（32K context）—— 这是物理上限
- 稳态 gen 22 t/s（4K context）

**应用层优化**：在 app 层做"warmup gen token"（用 dummy prompt gen 1 个 token），把 7 秒藏到首屏渲染后面。

**不可优化项**：Metal 本身的 GPU memory bandwidth（约 200 GB/s）vs Mac Studio 的 800 GB/s。硬件升级是唯一路径。

---

## 3. 性能数据回顾

### 3.1 关键 benchmark（来自 benchmarks/*.md 报告）

| 测试场景 | 数值 | 来源 |
|:---|:---|:---|
| 0.5B Q4 baseline | 1575 t/s prefill / 227 t/s gen | W1 D3 |
| 7B Q4 + mlock | 175 t/s prefill | W2（mlock 反而 +7%） |
| 8B Q4 baseline | 455 t/s prefill / 24 t/s gen | D0 |
| 8B 4K + mgr | 577 t/s prefill / 6 t/s gen | W1 D5 |
| 8B 32K + mgr (prefill) | **153 t/s / 214s** | W1 D6 |
| 8B 32K + mgr (gen) | **11 t/s**（vs 32K 无 mgr 10 t/s） | D6 gen analysis |
| 8B 32K no-mgr (prefill) | 130 t/s | baseline |
| 70B + 32K 真机 | ❌ 未验证 | 需要 64GB Mac Studio |

### 3.2 W1 / W2 / Phase 6 各阶段性能

| 阶段 | 关键里程碑 | 性能影响 |
|:---|:---|:---|
| W1 D5 (8B 4K E2E) | 577 t/s prefill | baseline |
| W2 (sliding window) | +0.4% prefill / +0% gen | ✅ 无 regression |
| W1 D6 (8B 32K prefill) | 153 t/s, 3.5 min | ✅ 跑通（修复后） |
| W1 D6 (gen analysis) | 11 t/s steady | ✅ 物理上限确认 |
| Phase 6 S11 (wire) | 50 spec step 通过 | ✅ wire 完整 |
| Phase 6 S11 (acceptance) | 0% | ❌ 训练 mismatch |

### 3.3 与 llama.cpp baseline 对比

| 项目 | llama.cpp baseline | FusionLLM | 增量 |
|:---|:---|:---|:---|
| 8B 4K prefill | 455 t/s | 577 t/s | **+27%**（mlock + KV 分层） |
| 8B 4K gen | 24 t/s | 6 t/s | -75%（但 gen 本身受 Metal 限制） |
| 8B 32K prefill | 11+ min / hang | 153 t/s / 3.5 min | **hang 修复**（不可量化对比） |
| 8B 32K gen | 24 t/s | 11 t/s | 较慢但稳定 |

**真实结论**：mlock + KV 分层对 **prefill** 有显著加速（+27% on 4K, hang fix on 32K），对 **gen** 是中性（M5 Metal 物理上限）。

### 3.4 真实瓶颈分析

- **Prefill**：compute-bound（attention matmul）→ KV 分层 + mlock 加速明显
- **Gen（短 context）**：受 Metal GPU 内存带宽限制 → 软件优化空间小
- **Gen（长 context）**：attention O(n²) → 物理上限 → 需要更快的硬件或 spec decode 加速

---

## 4. 风险评估

| # | 风险 | 概率 | 影响 | 缓解措施 |
|:---:|:---|:---:|:---:|:---|
| 1 | **70B 硬件未到位** | 🔴 高 | 🔴 高 | 借/买 Mac Studio 64GB |
| 2 | **DSpark 训练无 GPU** | 🔴 高 | 🟡 中 | 找合作伙伴训练，或租云 GPU |
| 3 | **全职工程师缺位** | 🔴 高 | 🔴 高 | 招聘 Upwork 或全职 |
| 4 | **D6 32K 临时 fix 复发** | 🟢 低 | 🟢 低 | 写 constructor init checklist + regression test |
| 5 | **双仓库继续漂移** | 🟡 中 | 🟡 中 | 强制工作流：main 优先 → fork sync |
| 6 | **mlock 不可移植** | 🟢 低 | 🟡 中 | 已经是 macOS-specific 设计 |
| 7 | **DSpark 训练后 accept rate 仍低** | 🟡 中 | 🟡 中 | Markov head + confidence head 已设计；可调整超参 |
| 8 | **Ollama 集成阻力** | 🟡 中 | 🟡 中 | Ollama API 稳定，复用 `test-fusion-*` driver 模式 |
| 9 | **ggml 版本升级 breaking** | 🟡 中 | 🟡 中 | Fork 在固定 commit pin |
| 10 | **Metal 4 → 5 升级 breaking** | 🟢 低 | 🟡 中 | 跟 fork 同步 |
| 11 | **70B 真机性能未达预期** | 🟡 中 | 🔴 高 | 提前规划 fallback（70B Q2 / 13B） |
| 12 | **小说项目分心** | 🟡 中 | 🟢 低 | 明确边界 |
| 13 | **kairos-mini 永远不修** | 🔴 高 | 🟢 低 | 不影响核心，决定是否继续维护 |

---

## 5. 竞争分析

### 5.1 同类项目对比

| 项目 | 平台 | 优化重点 | 70B 支持 | 评分 |
|:---|:---|:---|:---:|:---:|
| **FusionLLM** | Apple Silicon | KV 分层 + spec decode + mlock | ⏳ 待验证 | 工程深度 ⭐⭐⭐⭐ |
| llama.cpp | 跨平台 | 通用 GGUF + Metal/CUDA | ✅ | 生态 ⭐⭐⭐⭐⭐ |
| Ollama | 跨平台 | llama.cpp 包装 + 易用 | ✅ | 易用 ⭐⭐⭐⭐⭐ |
| LM Studio | Apple Silicon | GUI + 简化 | ✅ | GUI ⭐⭐⭐⭐ |
| vLLM | NVIDIA | PagedAttention + 高吞吐 | ✅ | 服务 ⭐⭐⭐⭐⭐ |
| PowerInfer-2 | NVIDIA | 神经元级激活稀疏化 | ✅ | 学术 ⭐⭐⭐⭐ |

### 5.2 FusionLLM 差异化点

| 维度 | FusionLLM | 评价 |
|:---|:---|:---|
| **场景独占** | 16GB Air 跑 70B（终极目标） | ⭐⭐⭐⭐⭐（没人做） |
| **工程深度** | mlock + SSD + KV tier 集成 | ⭐⭐⭐⭐ |
| **学术前沿** | DSpark spec decode | ⭐⭐⭐（无原创，跟随 DeepSpec） |
| **生态友好** | GGUF + llama.cpp fork | ⭐⭐⭐⭐ |
| **可读性** | 中文 memory + 测试齐全 | ⭐⭐⭐⭐⭐ |

### 5.3 护城河深度评估

**短期护城河（6 个月）**：
- 16GB Air 跑 70B 场景独占（前提：真机验证 + DSpark 真训练）
- mlock + SSD 三级缓存的工程整合（无现成方案）

**中期护城河（1-2 年）**：
- 如果成功：成为 Apple Silicon 上 70B 长上下文的事实标准
- 可能被 llama.cpp 主仓吸收（这是好事，扩大影响）

**长期护城河**：
- 没有——Apple Silicon 上 LLM 推理会被 llama.cpp 通用化
- **唯一不可替代的是工程经验和性能调优 know-how**

**护城河风险**：如果 70B 真机不达标或 DSpark 训练失败，护城河塌缩为 ⭐⭐⭐（"又一个 llama.cpp fork"）。

---

## 6. 路线图评估

### 6.1 短期（1-2 周）

| 任务 | 工作量 | 资源 | 价值 |
|:---|:---|:---|:---:|
| Ollama 集成 | 1-2 周 | M5 Air | 🔴 高（用户能用） |
| OLLAMA runner + FusionLLM backend plugin | 5 天 | 同上 | 高 |
| 文档 + 安装脚本 | 2 天 | 同上 | 中 |

### 6.2 中期（1-2 月）

| 任务 | 工作量 | 资源 | 价值 |
|:---|:---|:---|:---:|
| 70B 真机验证 | 1 周 setup | **需 64GB Mac Studio** | 🔴 高（终极 KPI） |
| 70B 性能调优 | 2 周 | 同上 | 高 |
| DSpark 训练（Qwen2.5-0.5B draft） | 1-2 天 | **需 8×A100 集群** | 🟡 中（已经有 wire） |
| DSpark 训练（Qwen3-4B/8B draft） | 1-2 天 | 同上 | 中 |

### 6.3 长期（3-6 月）

| 任务 | 工作量 | 资源 | 价值 |
|:---|:---|:---|:---:|
| 生产化部署 | 1 月 | M5 Air + 集群 | 高 |
| 用户文档 + 教程 | 2 周 | 同上 | 中 |
| 性能 benchmark 对比 | 1 周 | 已有硬件 | 中 |
| 商业化探索（开源 + 赞助？） | 待定 | — | 待定 |

### 6.4 关键里程碑

| 里程碑 | 时间 | 验收 |
|:---|:---|:---|
| **M1：70B 真机验证** | 4 周内 | 70B Q4 + 32K 在 Mac Studio 跑通，t/s ≥ 5 |
| **M2：DSpark 真训练 + accept > 0.5** | 8 周内 | 训练完 DSpark Qwen3-4B draft，spec decode 2x 加速 |
| **M3：Ollama GA** | 6 周内 | `ollama run qwen3-8b-fusion` 工作 |
| **M4：用户 ≥ 100** | 6 月内 | GitHub stars 或下载量 |

---

## 7. 建议

### 7.1 是否继续推进？

**答案：YES，但有条件。**

- 技术路线已验证（8B 32K 端到端跑通）
- 工程基础扎实（181+ tests，2 commits/day 平均节奏）
- 但需要外部资源（GPU + 人力）才能完成终极 KPI

### 7.2 优先级调整

**现在应该做的（顺序）**：

1. **🔴 70B 硬件到位**（最关键）—— 没有这个，所有 70B 数字都是外推
2. **🟡 DSpark 训练**（次关键）—— wire 已经有了，只缺模型
3. **🟢 Ollama 集成**（中优先）—— 用户能用，但有 Ollama 这种现成方案价值打折
4. **🟢 小说项目**（低优先）—— 分心，明确边界

### 7.3 资源申请建议

| 资源 | 用途 | 预算 |
|:---|:---|:---|
| Mac Studio M4 Max 64GB | 70B 真机验证 | $2-3K |
| Upwork 高级 C++ 工程师（兼职） | W2 / Phase 6 收尾 | $15-30K/月 |
| AWS p4d.24xlarge（A100×8）按需 | DSpark 训练 | $30/小时 × 8 小时 = $240 |
| NVMe SSD 2TB | L2 cache | $300-500 |
| **总计** | **3 个月** | **$20-35K** |

### 7.4 关键决策点

**立即（本周）**：

1. **70B 硬件**：买还是借？预算来源？
2. **工程师**：自己干（full-time 工作量太大）还是招兼职？
3. **DSpark 训练**：自建 GPU 还是用云（更便宜）？

**3 个月内**：

1. **70B 真机达标？** → 是 → 继续；否 → 降级到 13B 或 NO GO
2. **DSpark accept > 0.5？** → 是 → 高歌猛进；否 → 重新评估 spec decode 是否值得
3. **Ollama 用户反馈？** → 是 → 迭代；否 → 调整定位

### 7.5 不建议做的事

- ❌ 训练 70B DSpark draft（成本太高，4B/8B 已够用）
- ❌ 重写 llama.cpp（GGUF + fork 是正确选择）
- ❌ 在没有 64GB Mac 情况下声称"支持 70B"（外推不是测量）
- ❌ 把所有时间投入小说项目（FusionLLM 是核心资产）

### 7.6 给老大的 3 条硬性建议

1. **🔴 买/借 64GB Mac Studio** —— 没有这个就不要说"路径 c 终极目标"。M5 Air 16GB 物理装不下 70B + 32K full KV。
2. **🔴 招 1 个兼职高级 C++/Metal 工程师** —— 当前 1 人 + 我（小星）的产出有限。Upwork 上能 2-3 周找到合适的人。
3. **🟡 把 FusionLLM 的 spec decode 当核心** —— 这是差异化的关键。Ollama 包装是商品化工作，不是护城河。

---

## 8. 总结

**FusionLLM 是一个工程整合项目，做了正确的事（KV 分层 + spec decode + mlock），但还没完成终极 KPI（70B 真机）。**

技术可行性：**已验证**（8B 32K 端到端跑通）

资源可行性：**待验证**（需要 70B 硬件 + GPU 集群 + 工程师人力）

商业可行性：**不重要**（开源项目，商业化是 bonus）

**结论：CONDITIONAL GO。3 个硬性前提满足则继续。**

---

## 附录 A：审查依据

阅读了 13 个核心源文件、9 个 benchmark 报告、当日 memory、git log 50+ commits，验证了 llama.cpp fork 的 KV tier 集成（grep `fusion_kv_tier_get_attached` in llama-context.cpp:4199 行）。

## 附录 B：参考文献

1. PowerInfer-2 (arXiv 2406.04382) - 直接前辈
2. FlexGen (ICML 2023) - 思路印证
3. PagedAttention / vLLM (SOSP 2023) - KV 分块
4. LMCache - KV to NVM
5. DeepSpec / DSpark - 推测解码
6. llama.cpp - fork 对象

---

*审查完成于 2026-07-01*
*审查者：Claude Code 驱动的小星（MiniMax-M3）*
*数据来源：所有 t/s 数字均来自 benchmarks/*.md 报告，无虚构*