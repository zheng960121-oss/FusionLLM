# LMCache 文献笔记

**项目**：LMCache - LMCache: KV Cache Layer for LLM Serving
**链接**：https://github.com/LMCache/LMCache
**作者**：Yuhan Liu et al. (University of Chicago)
**发布日期**：2024 (论文 + 开源)
**阅读日期**：2026-06-28

---

## 1. 核心定位

LMCache 是**给 vLLM 加的 KV Cache 缓存层**，解决两个问题：
1. **跨 vLLM 实例共享 prefix KV cache**（多副本场景）
2. **用 NVM (SSD) 扩展 KV cache 容量**（单实例场景）

**注**：项目用 docker + Python 部署，与我们 macOS 上的 C/C++ 路径差异大。

---

## 2. 架构概览（基于 README）

```
+---------------------+
|   vLLM Engine(s)    |  ← 多个 vLLM 实例
+----------+----------+
           |
           v
+---------------------+
|     LMCache        |  ← 统一 KV cache 层
|  - GPU/CPU/SSD     |
+----------+----------+
           |
           v
+---------------------+
|  Backend Storage   |  ← Redis / Local Disk / S3
+---------------------+
```

**核心组件**：
- **LMCache Engine**：在 vLLM 内部拦截 KV cache 操作
- **Storage Backends**：Local disk、Redis、S3 等多种后端
- **Lookup/Store 接口**：基于 prefix matching 命中缓存

---

## 3. 关键设计点

### 3.1 Token-based KV Cache 切分

把 KV cache 切成 token-level 块（不是 layer-level）：
- 每块对应一定数量的 token（如 16 tokens）
- 可以基于 prefix 匹配命中（如"前 1024 token 完全相同"）

**对 FusionLLM 的启示**：
- 我们的设计是 layer-level 切分（更粗）
- LMCache 是 token-level 切分（更细）
- 路径 c 目标 70B + 32K-128K 上下文，**token-level 可能更优**
- 但实现复杂度也更高

### 3.2 Hot/Cold 数据识别

LMCache 用 LRU + access frequency 识别 hot/cold KV blocks。

**对 FusionLLM 的启示**：
- 我们的 PoC-4 已经验证了 KV block 落 SSD + 重新读取
- 加上 LRU 决策就是完整的 KV cache 分层
- 实现简单：FIFO + counter

### 3.3 Prefetch Policy

LMCache 用"基于 prefix 预测"做预取：
- 知道 prompt 前缀
- 预测接下来会用哪部分 KV
- 提前从 SSD 加载

**对 FusionLLM 的启示**：
- 在长对话场景很有用
- 但对单流推理，可以简化为"按需加载 + LRU 缓存"

---

## 4. 与 FusionLLM 的相关性

### 4.1 适用的部分

| LMCache 设计 | FusionLLM 借鉴价值 |
|---|---|
| KV block 落 SSD | ⭐⭐⭐ **直接借鉴**（已 PoC-4 验证）|
| LRU 决策 | ⭐⭐⭐ 简单可靠 |
| Prefetch 策略 | ⭐⭐ 可选优化 |
| Backend 抽象 | ⭐ 我们直接用 mmap，不需要后端抽象 |

### 4.2 不适用的部分

| LMCache 设计 | 不适用原因 |
|---|---|
| 跨 vLLM 实例共享 | FusionLLM 是单实例 |
| Python + Docker 部署 | 我们是 C++ 嵌入式 |
| Prefix matching 优化 | 单流场景下收益有限 |
| Redis / S3 后端 | 我们只用本地 SSD + mmap |

### 4.3 路径 c 核心借鉴

**LMCache 给路径 c 的最重要启示是：KV Cache 落 SSD 是已经被验证可行的方向**。

我们 PoC-4 已经在 m5 上验证了基本机制（write → mmap → GPU read），LMCache 的工程实现给了我们以下确认：
- ✅ KV block 切分（可以是 layer-level 也可以是 token-level）
- ✅ LRU 决策
- ✅ 落盘 + 重新加载的性能可接受
- ⚠️ 他们的延迟数据是基于 A100 + NVMe，我们是基于 M5 + Apple SSD，需要实测

---

## 5. 实际参考价值评估

**直接复用代码价值：低**（Python + vLLM 集成，C++ 路径不同）

**设计思想价值：中**：
- KV block 切分思想
- LRU 决策
- Hot/cold 分层

**路径 c 进度影响**：
- PoC-4 已经验证了基本机制 ✅
- Phase 3 实现时参考 LMCache 的工程设计
- 不必复现 LMCache 整体架构（场景不同）

---

## 6. 推荐阅读优先级

**给路径 c 的 Phase 3 实现**：

| 优先级 | 资源 | 用途 |
|:---:|---|---|
| 🔴 高 | LMCache KV store 源码 | 看 block 切分、LRU 实现 |
| 🟡 中 | LMCache 论文 | 理解系统设计取舍 |
| 🟢 低 | LMCache benchmark | 参考他们的性能数据 |

**具体看什么**：
- `lmcache/v1/storage_backend/local_cpu.py`（local backend）
- `lmcache/v1/storage_backend/serde.py`（序列化）
- `lmcache/v1/cache_engine.py`（cache engine 主体）

---

## 7. 与 PoC-4 的对应

| PoC-4 测试 | LMCache 对应实现 |
|---|---|
| 写 256MB KV block 到 SSD | `LocalCPUStorageBackend.put()` |
| mmap + GPU zero-copy read | `LocalCPUBackend.get()` + mmap 包装 |
| 10 轮 write-sync-verify | LMCache 的 eviction + reload 循环 |
| madvise + 压力 | LMCache 的 memory limit + 自动换出 |

**结论**：PoC-4 验证了 LMCache 核心机制，Phase 3 可以直接借鉴其 API 设计。

---

*笔记完成于 2026-06-28*
