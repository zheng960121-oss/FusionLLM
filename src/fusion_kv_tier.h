// FusionLLM Phase 3 W1: KV Cache 三层分级（GPU/CPU/SSD）
// 设计文档: docs/phase3_w1_kv_tier_design.md
//
// 提供 3 层 KV cache 分级管理:
//   L0 GPU Metal buffer — 热 KV (<500MB, <1us 访问)
//   L1 CPU mlock'd RAM  — warm KV (~2GB, ~100ns 访问)
//   L2 SSD mmap'd       — 冷 KV (6-24GB, ~10ms mmap)
//
// 集成到 llama.cpp KV cache 通过 3 个 hook:
//   - ensure_for_attention(layer, start_tok, end_tok)  // pre-attention
//   - flush_to_ssd(layer, start_tok, end_tok)         // sliding window 移动
//   - demote_to_ssd(layer, start, end)                // cell 被驱逐
//
// 用法示例:
//   auto* mgr = new fusion::FusionKVTierManager(ctx, n_layers, kv_size, kv_head_dim, type);
//   mgr->set_ssd_path("/tmp/fusion_kv_cache");
//   // 在 attention forward 前:
//   auto* k_gpu = mgr->ensure_for_attention(layer_id, start_tok, end_tok);
//   auto* v_gpu = mgr->ensure_for_attention(layer_id, start_tok, end_tok);

#ifndef FUSION_KV_TIER_H
#define FUSION_KV_TIER_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Pull in ggml types directly — kv tier holds ggml_context / ggml_tensor /
// uses ggml_type in API.  Forward-declaring enum ggml_type with an explicit
// underlying type conflicts with the real ggml.h definition (changes between
// llama.cpp revisions), so just include the header.
extern "C" {
#include "ggml.h"
}

// C-compatible opaque type for the tier manager (C API uses this name)
typedef struct fusion_kv_tier fusion_kv_tier_t;

namespace fusion {

// ============================================================
// Tier 枚举
// ============================================================
enum class KVTier : uint8_t {
    L0_GPU = 0,    // Metal buffer (热 KV)
    L1_CPU = 1,    // mlock'd RAM (warm KV)
    L2_SSD = 2,    // mmap'd SSD (cold KV)
};

// ============================================================
// KV block location 描述符
// ============================================================
// 一个 KV block = 一个 layer 的 [start_token, end_token) 范围
// block 是 W1 tier 管理的最小单位
struct KVBlockLocation {
    KVTier tier;           // 当前在哪一层
    void*  ptr;            // tier 指针 (GPU ptr / CPU ptr / SSD file offset)
    size_t size_bytes;      // block 大小 (bytes)
    int    layer_id;        // 0..n_layers-1
    int    start_token;     // 包含
    int    end_token;       // 不包含
    ggml_type kv_type;      // F16 / Q8_0 / Q4_0
    bool   is_loaded;       // 是否已从 L2 load 到 L1/L0
};

// ============================================================
// Block metadata 持久化 (L2 SSD file header)
// ============================================================
// 每个 L2 SSD file 的开头写 KVBlockMeta:
//   - magic, version, kv_size, n_layers, block_size, kv_type
//   - 验证文件完整性
struct KVBlockMeta {
    uint32_t magic;         // 'F' 'K' 'V' 'B' = 0x42564B46
    uint32_t version;       // = 1
    uint32_t n_layers;
    uint32_t kv_size;
    uint32_t block_size;    // tokens per block
    uint32_t kv_type;       // ggml_type cast to uint32_t
    uint32_t reserved[2];
};

// ============================================================
// FusionKVTierManager: 3 层 KV cache 统一管理
// ============================================================
class FusionKVTierManager {
public:
    // === 统计信息 ===
    struct Stats {
        size_t bytes_in_gpu;     // L0 字节数
        size_t bytes_in_cpu;     // L1 字节数
        size_t bytes_in_ssd;     // L2 字节数
        size_t n_promotions;     // L2→L1 总次数
        size_t n_evictions;      // L1→L2 总次数
        size_t n_gpu_copies;     // L1→L0 总次数
        size_t n_ssd_reads;      // L2 读取次数
        size_t n_ssd_writes;     // L2 写入次数
    };

    // ============================================================
    // 构造 / 析构
    // ============================================================
    FusionKVTierManager(
        ggml_context* ctx,
        int n_layers,
        int kv_size,           // max tokens per layer
        int kv_head_dim,       // GQA: head_dim × n_head_kv
        ggml_type kv_type,     // F16 / Q8_0 / Q4_0
        int block_size = 512   // 默认 512 tokens per block (8B), 128 for 70B
    );
    ~FusionKVTierManager();

    // 禁止拷贝（持有 ggml resources）
    FusionKVTierManager(const FusionKVTierManager& other) = delete;
    FusionKVTierManager& operator=(const FusionKVTierManager& other) = delete;

    // ============================================================
    // SSD backend
    // ============================================================
    // 设置 SSD 缓存根目录 (per-model)
    // 目录结构: <ssd_dir>/<model_hash>/layer_<N>.bin
    // 模型 hash: model_path + key params (避免不同模型冲突)
    void set_ssd_path(const std::string& ssd_dir);

    // 模型加载完成时调用: 将所有 KV block 写入 L2 (SSD)
    // 这是初始状态 — block 在 SSD, 需要时才 promote
    bool flush_all_to_ssd();

    // ============================================================
    // Promotion / Demotion API
    // ============================================================
    // Promote 一个 token range 到指定 tier
    bool promote_to_gpu(int layer_id, int start_tok, int end_tok);
    bool promote_to_cpu(int layer_id, int start_tok, int end_tok);
    bool demote_to_ssd(int layer_id, int start_tok, int end_tok);

    // ============================================================
    // Queries
    // ============================================================
    // 查询一个 block 的当前位置
    KVBlockLocation get(int layer_id, int start_tok, int end_tok);

    // 快速检查 (不需要 full location lookup)
    bool is_in_gpu(int layer_id, int start_tok, int end_tok);
    bool is_in_cpu(int layer_id, int start_tok, int end_tok);
    bool is_in_ssd(int layer_id, int start_tok, int end_tok);

    // ============================================================
    // Hook for llama.cpp attention forward (W1 D4 集成)
    // ============================================================
    // 在 attention forward 前调用, 确保 K/V 在 L0 (GPU)
    // 返回: ggml_tensor* (GPU buffer 中的 K 或 V block)
    // 失败: nullptr
    ggml_tensor* ensure_for_attention(int layer_id, int start_tok, int end_tok, bool is_value = false);
    // ↑ is_value=false → K (key), is_value=true → V (value)

    // ============================================================
    // SSD persistence
    // ============================================================
    // Flush 一个 block 从 L1 CPU → L2 SSD
    bool flush_to_ssd(int layer_id, int start_tok, int end_tok);

    // ============================================================
    // Sliding window integration (W2)
    // ============================================================
    // 当 sliding window 移动时调用: 离开 window 的 block demote
    // 默认 sliding window size = kv_size / 4 (i.e., 8B 32K → window 8K)
    void set_sliding_window(int window_size);
    // 推进 window: [old_start, old_end) → [new_start, new_end)
    // 自动 demote 离开的 block, promote 进入的 block
    bool advance_window(int layer_id, int old_start, int old_end, int new_start, int new_end);

    // ============================================================
    // Stats
    // ============================================================
    Stats get_stats() const;
    void reset_stats();

    // ============================================================
    // Accessors for llama.cpp integration
    // ============================================================
    // The bridge needs to know the per-layer kv_size to pre-warm the cache.
    // All layers share the same kv_size in our model, so a single getter
    // suffices.
    int kv_size_for_layer() const { return kv_size_; }

    // ============================================================
    // Debug / Logging
    // ============================================================
    void print_summary() const;

private:
    // === 内部状态 ===
    ggml_context* ctx_ = nullptr;
    int n_layers_ = 0;
    int kv_size_ = 0;
    int kv_head_dim_ = 0;
    ggml_type kv_type_ = {};
    int block_size_ = 512;

    // Block table: blocks_[layer_id][block_id]
    // 每个 layer 的 block 数 = (kv_size + block_size - 1) / block_size
    std::vector<std::vector<KVBlockLocation>> blocks_;

    // L1 CPU RAM staging buffer (per layer)
    // size = kv_size × kv_head_dim × type_size
    std::vector<void*> l1_buffers_;
    std::vector<size_t> l1_buffer_sizes_;

    // L0 GPU staging buffer (per layer)
    // ggml tensors in ctx_, kept alive via context
    std::vector<ggml_tensor*> l0_k_tensors_;
    std::vector<ggml_tensor*> l0_v_tensors_;
    int l0_max_tokens_ = 0;

    // L2 SSD files
    std::string ssd_dir_;
    std::vector<std::string> ssd_file_paths_;  // per layer
    std::vector<void*> ssd_mmapped_;           // mmap'd regions
    std::vector<int>   ssd_fds_;               // open fds per layer (-1 if not open)

    // Sliding window
    int sliding_window_size_ = 0;

    // Stats
    mutable Stats stats_{};

    // === 内部 helpers ===
    int block_id(int token) const { return token / block_size_; }
    int block_start(int block_id) const { return block_id * block_size_; }
    int block_end(int block_id) const {
        int end = (block_id + 1) * block_size_;
        return end > kv_size_ ? kv_size_ : end;
    }
    int n_blocks_per_layer() const {
        return (kv_size_ + block_size_ - 1) / block_size_;
    }
    size_t block_byte_size() const {
        // K 和 V 各一份 (ggml_type 决定字节/元素)
        // 但我们以 token × head_dim 为单位
        return (size_t) block_size_ * kv_head_dim_ * ggml_type_size(kv_type_);
    }

    // Promote/demote 实现
    bool do_promote_to_cpu(int layer_id, int block_id);
    bool do_promote_to_gpu(int layer_id, int block_id);
    bool do_demote_to_ssd(int layer_id, int block_id);

    // SSD file I/O
    bool write_ssd_block(int layer_id, int block_id);
    bool read_ssd_block(int layer_id, int block_id);

    // GPU staging buffer ensure
    bool ensure_l0_buffers(int max_tokens);
};

}  // namespace fusion

// ============================================================
// C-linkage bridge for llama.cpp integration
// ============================================================
// llama.cpp links against this C API to attach / detach a tier manager from
// a llama_context.  The actual hook lives in graph_get_cb() and dispatches
// to FusionKVTierManager::ensure_for_attention() per layer per forward.
//
// Forward declaration of llama_context so we don't need to include
// llama.h here (keeps fusion_kv_tier.h standalone-buildable).
struct llama_context;

#ifdef __cplusplus
extern "C" {
#endif

// Attach a tier manager to a llama_context.  Pass mgr=nullptr to detach.
// After this call, the context's per-layer graph callback will call
// ensure_for_attention() for every layer in every forward.
// Note: mgr must outlive ctx (the context stores a raw pointer; we don't
// take ownership).
void fusion_kv_tier_attach(struct llama_context* ctx, fusion_kv_tier_t* mgr);

// Check if a tier manager is attached (returns 1 if yes, 0 if no)
int fusion_kv_tier_is_attached(struct llama_context* ctx);

// C++ accessor: returns the attached manager for ctx, or nullptr if none.
// Used by llama_context's graph_get_cb() to dispatch per-layer calls.
namespace fusion {
class FusionKVTierManager;
FusionKVTierManager* fusion_kv_tier_get_attached(struct llama_context* ctx);
}  // namespace fusion

#ifdef __cplusplus
}
#endif

#endif  // FUSION_KV_TIER_H