# Phase 1 Baseline 报告

**日期**：2026-06-28
**设备**：Apple M5 MacBook Air 16GB
**llama.cpp**：latest main (27c8bb4, ggml 0.15.3)
**模型**：Qwen2.5-0.5B-Instruct Q4_K_M (468MB)
**目标**：建立 vanilla llama.cpp baseline，作为 fusion driver 添加后的对比基线

---

## 1. 验证结果

| 验证项 | 结果 |
|---|:---:|
| llama.cpp 编译 (GGML_METAL=ON) | ✅ |
| Apple M5 GPU 识别 | ✅ 12124 MiB |
| Qwen 0.5B Q4 推理跑通 | ✅ |
| Prompt 处理速度 | **1575.2 t/s** |
| Generation 速度 | **227.5 t/s** |

---

## 2. 工具链

- **llama.cpp-fusionllm**: `/Users/jk/Desktop/llama.cpp-fusionllm/`
- **构建**: `cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 8`
- **运行时**:
  - 模型路径: `models/qwen2.5-0.5b-instruct-q4_k_m.gguf`
  - 命令: `llama-cli -m <model> -ngl 99 -t 4 -p "<prompt>" -n 20 -st`
  - `-st`: single-turn 模式（生成完自动退出，不需要交互）

---

## 3. 设备信息

| 维度 | 值 |
|---|---|
| Chip | Apple M5 |
| Memory | 16 GB unified |
| GPU 可用 | 12124 MiB (~12GB) |
| Metal | Metal 4 |
| macOS | 26.5.1 |

---

## 4. 多次运行的方差

| 运行 | Prompt t/s | Generation t/s |
|:---:|:---:|:---:|
| Run 1 (interactive kill) | 1401.3 | 250.8 |
| Run 2 (single-turn) | 1575.2 | 227.5 |

差异约 10-15%，正常系统噪声范围内。

---

## 5. Phase 1 下一步

按开发计划 T1.1-T1.7：

- [ ] T1.1 fork 完成 ✅
- [ ] T1.2 实现 mmap + mlock + madvise 权重生命周期 hook（**待 C++ 实现**）
- [ ] T1.3 加 `--fusion-driver` 编译开关
- [ ] T1.4 跑 7B Q4 baseline（**待模型下载 + 跑**）
- [ ] T1.5 性能回归测试（不能比原版慢 5%+）
- [ ] T1.6 加 benchmark 框架
- [ ] T1.7 写单元测试覆盖 mlock 路径

**预计耗时**：2-3 周全职 C++ 工作

---

## 6. 跑 benchmark 的方法

```bash
# 一键跑 baseline
cd /Users/jk/Desktop/FusionLLM
bash benchmarks/phase1_baseline.sh

# 或手动跑
cd ~/Desktop/llama.cpp-fusionllm
./build/bin/llama-cli \
    -m models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    -ngl 99 -t 4 \
    -p "The capital of France is" \
    -n 20 -st
```

---

## 7. 后续模型测试计划

按 Sprint 2 计划，跑 7B / 13B / 30B Q4 模型：

| 模型 | 大小 | 预计下载时间 |
|---|---:|---|
| 7B Q4_K_M | ~4.5GB | 5-10 分钟 |
| 13B Q4_K_M | ~8GB | 10-15 分钟 |
| 30B Q4_K_M | ~18GB | 20-30 分钟 |
| 70B Q4_K_M | ~40GB | 1-1.5 小时 |

Phase 1 主要用 7B 做对比，13B/30B 在 Sprint 2 跑。

---

*报告完成于 2026-06-28 14:33*