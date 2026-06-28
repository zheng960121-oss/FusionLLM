# FusionLLM

**Apple Silicon 上跑 70B+ 大模型本地推理引擎**

FusionLLM 通过**滑动窗口权重调度 + KV Cache SSD 分层**两个核心机制，让 16GB 统一内存的 MacBook Air（M5）也能流畅运行 70B Q4 模型 + 32K-128K 长上下文。

---

## 当前状态：Sprint 0 Gate ✅

**4 个 PoC 全部通过**（2026-06-28，1.5 小时完成 2.5 天计划工作量）：

| PoC | 状态 | 验证内容 |
|:---:|:---:|---|
| PoC-1 happy path | ✅ | mmap + mlock + Metal 零拷贝 buffer |
| PoC-1 缺页测试 | ✅ | GPU 访问释放页不崩、不静默错 |
| PoC-1 Test 4 真实场景 | ✅ | munlock + madvise + 压力下稳定 |
| PoC-4 KV Cache SSD | ✅ | 10 轮 write-msync-verify + 3 block + 压力 |

**关键风险已解除**：
- ✅ R1（GPU 缺页 panic/timeout/静默错）—— 5 场景全过
- ✅ R2（KV Cache SSD 一致性）—— 路径 c 核心风险

**重大设计发现**：
- macOS `madvise(DONTNEED)` 几乎从不真正释放 mmap 物理页
- 修正设计：**只用 mlock/munlock**，不依赖 madvise
- 活跃滑动窗口 mlock（保证驻留），其余 mmap 不 mlock（OS file cache + 压力下自动换出）

---

## 仓库结构

```
FusionLLM/
├── README.md                              # 本文件
├── LICENSE                                # MIT
├── docs/
│   ├── 技术路线方案.md                    # 架构、决策、风险
│   └── 开发计划.md                        # Sprint / Gate / 任务
├── pocs/                                  # 4 个 PoC 源码 + 报告
│   ├── poc1_happy_path.swift
│   ├── poc1_happy_path_report.md
│   ├── poc1_page_fault_test.swift
│   ├── poc1_page_fault_report.md
│   ├── poc1_test4_realistic.swift
│   ├── poc1_test4_realistic_report.md
│   ├── poc4_kv_cache_ssd.swift
│   └── poc4_kv_cache_ssd_report.md
├── benchmarks/                            # 性能数据（Phase 1+）
├── decisions/                             # ADR 决策记录
└── .github/workflows/
    └── poc-ci.yml                         # 自动化跑 PoC
```

---

## 如何运行 PoC

```bash
# 进入项目目录
cd FusionLLM/pocs/

# 编译 + 运行某个 PoC
swiftc -O poc1_happy_path.swift -o poc1_happy_path && ./poc1_happy_path
swiftc -O poc1_page_fault_test.swift -o poc1_page_fault_test && ./poc1_page_fault_test
swiftc -O poc1_test4_realistic.swift -o poc1_test4_realistic && ./poc1_test4_realistic
swiftc -O poc4_kv_cache_ssd.swift -o poc4_kv_cache_ssd && ./poc4_kv_cache_ssd
```

**要求**：
- Apple Silicon Mac（M1+ 推荐 M5）
- macOS 14+
- Xcode + Metal 4
- `ulimit -l unlimited`（用于 mlock 1GB+ 测试）

---

## 路线图

| Phase | 状态 | 目标 | 预计 |
|---|:---:|---|---|
| 0 风险关门 | ✅ | 4 PoC 全过 | 已完成 |
| 1 最小推理 | ⏳ | M5 跑 7B，对比 llama.cpp | 2-3 周 |
| 2 滑动窗口 | ⏳ | 30B 跑通，预取 stall < 10% | 3-4 周 |
| 3 KV Cache 分层 ⭐ | ⏳ | **70B + 32K 上下文跑通** | 4-6 周 |
| 4 长上下文 | ⏳ | 70B + 64K-128K | 4-8 周 |
| 5 生态对接 | ⏳ | Ollama 兼容 | 2-3 周（推迟）|

**总时间**：4-5 个月到路径 c 最低目标（70B + 32K），6-7 个月含 Ollama。

详细规划见 `docs/开发计划.md`。

---

## 核心参考文献

- [PowerInfer-2](https://arxiv.org/abs/2406.04382) —— 消费级 GPU + SSD 卸载的直接前辈
- [FlexGen](https://arxiv.org/abs/2303.06865) (ICML 2023) —— SSD 当扩展内存的思路印证
- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) (SOSP 2023) —— KV Cache 分块管理
- [LMCache](https://github.com/LMCache/LMCache) —— KV Cache 落 NVM 的工程实现

---

## License

MIT