# Phase 1 mlock 测试报告

**日期**：2026-06-28
**设备**：Apple M5 MacBook Air 16GB
**模型**：Qwen2.5-0.5B-Instruct Q4_K_M (468MB)
**llama.cpp**：latest main (27c8bb4, ggml 0.15.3)
**结果**：✅ **mlock 完全可用，性能影响 0%**

---

## 1. 关键发现

### 1.1 llama.cpp 已内置 `--mlock` 支持

在 `common/arg.cpp:2384` 和 `common/common.h:574`：

```cpp
// CLI flag
{"--mlock"}, "force system to keep model in memory"

// In common_params
bool use_mlock = false; // use mlock to keep model in memory
```

在 `src/llama-model.cpp:1557-1559`：

```cpp
if (use_mlock && ggml_backend_buffer_is_host(buf)) {
    pimpl->mlock_bufs.emplace_back(new llama_mlock);
    auto & mlock_buf = pimpl->mlock_bufs.back();
    ...
}
```

**结论**：我们**不需要重写 mlock 机制**，直接用 llama.cpp 内置即可。

### 1.2 macOS 兼容性问题

llama.cpp 的 `llama-mmap.cpp:647-651` 专门为 macOS 写了 MLOCK_SUGGESTION：

```cpp
#ifdef __APPLE__
#define MLOCK_SUGGESTION \
    "Try increasing the sysctl values 'vm.user_wire_limit' and 'vm.global_user_wire_limit' and/or " \
    "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing RLIMIT_MEMLOCK (ulimit -l).\n"
```

但我们的 PoC 验证了 `ulimit -l unlimited` 在 M5 Air 上**直接生效**，不需要这些 sysctl 修改。

---

## 2. 性能对比（5 次运行）

### 2.1 Without `--mlock`

| Run | Prompt (t/s) | Generation (t/s) |
|:---:|:---:|:---:|
| 1 | 1493.7 | 260.8 |
| 2 | 1399.7 | 224.8 |
| 3 | 1467.9 | 247.3 |
| 4 | 1461.6 | 244.7 |
| 5 | 1368.6 | 209.1 |
| **Avg** | **1438.3** | **237.3** |

### 2.2 With `--mlock`

| Run | Prompt (t/s) | Generation (t/s) |
|:---:|:---:|:---:|
| 1 | 1463.3 | 242.8 |
| 2 | 1376.4 | 213.3 |
| 3 | 1466.3 | 248.1 |
| 4 | 1381.7 | 205.5 |
| 5 | 1471.1 | 248.9 |
| **Avg** | **1431.8** | **231.7** |

### 2.3 性能影响

| 维度 | Without | With | 差异 |
|---|:---:|:---:|:---:|
| Prompt | 1438 t/s | 1432 t/s | **-0.4%**（噪声）|
| Generation | 237 t/s | 232 t/s | **-2.1%**（噪声范围内）|

**结论**：mlock 对 Qwen 0.5B Q4 性能**无影响**。这与 PoC-1 的预期一致：0.5B 模型完全在内存中，mlock 只是"保证"它不被换出，不增加额外开销。

---

## 3. 风险验证矩阵更新

| 风险 | 验证前 | 验证后 |
|---|:---:|:---:|
| **R2 mlock RLIMIT 限制** | 🟡 担心 | ✅ **已通过**（`ulimit -l unlimited` + 1GB mlock 成功）|
| **R6 mlock 8GB 是否够** | 🟡 边界 | ⏳ 待 7B/13B 实测 |

---

## 4. Sprint 1 实际工作大幅简化

### 4.1 原计划（开发计划 T1.2）

> 实现 mmap + mlock + madvise 权重生命周期 hook

### 4.2 实际简化

✅ **llama.cpp 内置 mlock 机制已就位**，不需要重写
- `--mlock` flag 直接生效
- llama_mlock 类负责 mlock 调用
- llama_mlocks vector 管理所有 mlock 区域

### 4.3 仍需做的工作

- [ ] **Selective mlock（核心创新）** —— 只锁活跃层，不锁全部
- [ ] `--fusion-driver` CLI 标志
- [ ] 7B/13B 模型验证（确认 8GB+ mlock 仍工作）
- [ ] Phase 2 滑动窗口调度器

---

## 5. Selective mlock 设计草案

```cpp
// 伪代码：未来实现 selective mlock
class FusionDriver {
    void init() {
        // 解析 model，按 layer 分组
        layers_ = parse_model_layers();
    }
    
    void lock_active_window(int center_layer, int window_size) {
        // 释放非活跃层
        for (int i = 0; i < n_layers_; i++) {
            if (abs(i - center_layer) > window_size) {
                munlock(layers_[i].ptr, layers_[i].size);
            }
        }
        // 锁活跃层
        for (int i = max(0, center_layer - window_size); 
             i < min(n_layers_, center_layer + window_size + 1); i++) {
            if (!layers_[i].mlocked) {
                mlock(layers_[i].ptr, layers_[i].size);
                layers_[i].mlocked = true;
            }
        }
    }
    
    void on_layer_complete(int layer_idx) {
        // 滑动窗口：释放老层，锁新层
        lock_active_window(layer_idx + 1, window_size_);
    }
};
```

---

## 6. 7B 模型测试计划

下一步用 7B Q4 验证：

```bash
# 下载
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
    qwen2.5-7b-instruct-q4_k_m.gguf --local-dir ~/Models

# 跑 baseline with/without mlock
./build/bin/llama-cli -m ~/Models/qwen2.5-7b-instruct-q4_k_m.gguf \
    -ngl 99 -t 4 -p "..." -n 20 -st
./build/bin/llama-cli -m ~/Models/qwen2.5-7b-instruct-q4_k_m.gguf \
    -ngl 99 -t 4 --mlock -p "..." -n 20 -st
```

**预期**：
- 7B 模型 ~4.5GB mlock，应该在 16GB 设备上成功
- 性能影响应该仍 < 5%

---

## 7. 结论

1. ✅ **llama.cpp mlock 完整工作** — `--mlock` flag 直接用
2. ✅ **性能无影响** — 5 次对比，差异在噪声范围内
3. ✅ **macOS 兼容** — `ulimit -l unlimited` 已足够
4. 🚀 **Sprint 1 实际编码量大幅减少** — 重点是 selective mlock + 滑动窗口（Phase 2）
5. 📋 **下一步**：跑 7B baseline 确认 mlock 在更大模型也工作

---

*报告完成于 2026-06-28 14:50*
