// FusionLLM Phase 3 W1: KV Tier Manager Implementation Skeleton
// 设计文档: docs/phase3_w1_kv_tier_design.md
//
// 本文件是 W1 D1-D2 骨架 (skeleton), 提供完整 API + stub 实现
// 后续 W1 D2-D5 渐进式实装: L0↔L1↔L2 切换 + llama.cpp 集成 + 测试

#include "fusion_kv_tier.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <sys/stat.h>

// ggml type size helper
extern "C" {
#include "ggml.h"
}

namespace fusion {

// ============================================================
// Constructor / Destructor
// ============================================================
FusionKVTierManager::FusionKVTierManager(
    ggml_context* ctx,
    int n_layers,
    int kv_size,
    int kv_head_dim,
    ggml_type kv_type,
    int block_size
) : ctx_(ctx),
    n_layers_(n_layers),
    kv_size_(kv_size),
    kv_head_dim_(kv_head_dim),
    kv_type_(kv_type),
    block_size_(block_size > 0 ? block_size : 512) {
    fprintf(stderr, "[KVTier] initializing: n_layers=%d, kv_size=%d, head_dim=%d, block_size=%d\n",
            n_layers_, kv_size_, kv_head_dim_, block_size_);

    // 1) 分配 block table
    int n_blocks = n_blocks_per_layer();
    blocks_.resize(n_layers_);
    for (int il = 0; il < n_layers_; ++il) {
        blocks_[il].resize(n_blocks);
        for (int b = 0; b < n_blocks; ++b) {
            blocks_[il][b] = KVBlockLocation{
                .tier        = KVTier::L2_SSD,
                .ptr         = nullptr,
                .size_bytes  = 0,
                .layer_id    = il,
                .start_token = block_start(b),
                .end_token   = block_end(b),
                .kv_type     = kv_type_,
                .is_loaded   = false,
            };
        }
    }

    // 2) L1 CPU staging buffer (per layer, kv_size 全量)
    l1_buffers_.resize(n_layers_, nullptr);
    l1_buffer_sizes_.resize(n_layers_, 0);
    size_t layer_byte_size = (size_t)kv_size_ * kv_head_dim_ * ggml_type_size(kv_type_);
    for (int il = 0; il < n_layers_; ++il) {
        l1_buffers_[il] = calloc(layer_byte_size, 1);
        l1_buffer_sizes_[il] = layer_byte_size;
        fprintf(stderr, "[KVTier]   layer %d: L1 buffer = %.2f MB\n",
                il, layer_byte_size / (1024.0 * 1024.0));
    }

    // 3) L0 GPU staging buffer placeholder (lazy alloc)
    l0_k_tensors_.resize(n_layers_, nullptr);
    l0_v_tensors_.resize(n_layers_, nullptr);
    l0_max_tokens_ = 0;

    // 4) SSD paths
    ssd_file_paths_.resize(n_layers_);
    ssd_mmapped_.resize(n_layers_, nullptr);
}

FusionKVTierManager::~FusionKVTierManager() {
    fprintf(stderr, "[KVTier] destroying\n");
    // Free L1 buffers
    for (auto* p : l1_buffers_) {
        if (p) free(p);
    }
    // Munmap SSD
    for (size_t i = 0; i < ssd_mmapped_.size(); ++i) {
        if (ssd_mmapped_[i]) {
            // TODO: munmap + close fd
            ssd_mmapped_[i] = nullptr;
        }
    }
    // ggml tensors in L0 are freed by ggml_context (caller's responsibility)
}

// ============================================================
// SSD backend
// ============================================================
void FusionKVTierManager::set_ssd_path(const std::string& ssd_dir) {
    ssd_dir_ = ssd_dir;
    fprintf(stderr, "[KVTier] SSD path set to: %s\n", ssd_dir_.c_str());

    // Create directory if not exists
    struct stat st;
    if (stat(ssd_dir_.c_str(), &st) != 0) {
        if (mkdir(ssd_dir_.c_str(), 0755) != 0) {
            fprintf(stderr, "[KVTier]   failed to create SSD dir: %s\n", ssd_dir_.c_str());
            return;
        }
    }

    // Set per-layer file paths
    for (int il = 0; il < n_layers_; ++il) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/layer_%03d.bin", ssd_dir_.c_str(), il);
        ssd_file_paths_[il] = path;
    }
}

bool FusionKVTierManager::flush_all_to_ssd() {
    if (ssd_dir_.empty()) {
        fprintf(stderr, "[KVTier] SSD path not set, skipping flush_all_to_ssd\n");
        return false;
    }
    fprintf(stderr, "[KVTier] flushing %d layers to SSD...\n", n_layers_);
    int n_blocks = n_blocks_per_layer();
    for (int il = 0; il < n_layers_; ++il) {
        for (int b = 0; b < n_blocks; ++b) {
            if (!write_ssd_block(il, b)) {
                fprintf(stderr, "[KVTier]   failed to write layer %d block %d\n", il, b);
                return false;
            }
            blocks_[il][b].tier = KVTier::L2_SSD;
            blocks_[il][b].is_loaded = false;
        }
    }
    fprintf(stderr, "[KVTier] flush complete\n");
    return true;
}

// ============================================================
// Promotion / Demotion
// ============================================================
bool FusionKVTierManager::promote_to_gpu(int layer_id, int start_tok, int end_tok) {
    if (layer_id < 0 || layer_id >= n_layers_) return false;
    int b_start = block_id(start_tok);
    int b_end   = block_id(end_tok - 1);  // inclusive
    bool ok = true;
    for (int b = b_start; b <= b_end; ++b) {
        if (blocks_[layer_id][b].tier == KVTier::L0_GPU) continue;
        if (blocks_[layer_id][b].tier == KVTier::L1_CPU) {
            if (!do_promote_to_gpu(layer_id, b)) ok = false;
        } else if (blocks_[layer_id][b].tier == KVTier::L2_SSD) {
            // L2 → L1 (load + mlock)
            if (!do_promote_to_cpu(layer_id, b)) { ok = false; continue; }
            stats_.n_ssd_reads++;
            // L1 → L0 (copy to GPU)
            if (!do_promote_to_gpu(layer_id, b)) { ok = false; continue; }
            stats_.n_promotions++;
        }
    }
    return ok;
}

bool FusionKVTierManager::promote_to_cpu(int layer_id, int start_tok, int end_tok) {
    if (layer_id < 0 || layer_id >= n_layers_) return false;
    int b_start = block_id(start_tok);
    int b_end   = block_id(end_tok - 1);
    bool ok = true;
    for (int b = b_start; b <= b_end; ++b) {
        if (blocks_[layer_id][b].tier == KVTier::L1_CPU) continue;
        if (blocks_[layer_id][b].tier == KVTier::L2_SSD) {
            if (!do_promote_to_cpu(layer_id, b)) { ok = false; continue; }
            stats_.n_promotions++;
            stats_.n_ssd_reads++;
        }
        // Already in GPU: also need to copy to CPU for sliding window updates
        if (blocks_[layer_id][b].tier == KVTier::L0_GPU) {
            // TODO: copy GPU → CPU
            blocks_[layer_id][b].tier = KVTier::L1_CPU;
        }
    }
    return ok;
}

bool FusionKVTierManager::demote_to_ssd(int layer_id, int start_tok, int end_tok) {
    if (layer_id < 0 || layer_id >= n_layers_) return false;
    int b_start = block_id(start_tok);
    int b_end   = block_id(end_tok - 1);
    bool ok = true;
    for (int b = b_start; b <= b_end; ++b) {
        if (blocks_[layer_id][b].tier == KVTier::L2_SSD) continue;
        // 先把 GPU/CPU 数据写回 SSD
        if (!flush_to_ssd(layer_id, blocks_[layer_id][b].start_token, blocks_[layer_id][b].end_token)) {
            ok = false; continue;
        }
        blocks_[layer_id][b].tier = KVTier::L2_SSD;
        blocks_[layer_id][b].is_loaded = false;
        stats_.n_evictions++;
        stats_.n_ssd_writes++;
    }
    return ok;
}

// ============================================================
// Queries
// ============================================================
KVBlockLocation FusionKVTierManager::get(int layer_id, int start_tok, int end_tok) {
    if (layer_id < 0 || layer_id >= n_layers_) {
        return KVBlockLocation{};
    }
    int b = block_id(start_tok);
    if (b < 0 || b >= (int)blocks_[layer_id].size()) {
        return KVBlockLocation{};
    }
    return blocks_[layer_id][b];
}

bool FusionKVTierManager::is_in_gpu(int layer_id, int start_tok, int end_tok) {
    return get(layer_id, start_tok, end_tok).tier == KVTier::L0_GPU;
}

bool FusionKVTierManager::is_in_cpu(int layer_id, int start_tok, int end_tok) {
    return get(layer_id, start_tok, end_tok).tier == KVTier::L1_CPU;
}

bool FusionKVTierManager::is_in_ssd(int layer_id, int start_tok, int end_tok) {
    return get(layer_id, start_tok, end_tok).tier == KVTier::L2_SSD;
}

// ============================================================
// Hook for llama.cpp attention forward (W1 D4 stub)
// ============================================================
ggml_tensor* FusionKVTierManager::ensure_for_attention(int layer_id, int start_tok, int end_tok, bool is_value) {
    // TODO W1 D4: 集成到 llama.cpp attention forward
    // 当前 stub: promote 到 L0, 返回 GPU tensor
    if (!promote_to_gpu(layer_id, start_tok, end_tok)) {
        return nullptr;
    }
    // 返回 L0 buffer 中的对应 K/V tensor
    // (W1 D4: 实际集成时, 这里返回 llama_kv_cache 中对应位置的 view)
    return is_value ? l0_v_tensors_[layer_id] : l0_k_tensors_[layer_id];
}

// ============================================================
// SSD persistence
// ============================================================
bool FusionKVTierManager::flush_to_ssd(int layer_id, int start_tok, int end_tok) {
    int b = block_id(start_tok);
    if (b < 0 || b >= (int)blocks_[layer_id].size()) return false;
    return write_ssd_block(layer_id, b);
}

// ============================================================
// Sliding window integration
// ============================================================
void FusionKVTierManager::set_sliding_window(int window_size) {
    sliding_window_size_ = window_size > 0 ? window_size : kv_size_ / 4;
    fprintf(stderr, "[KVTier] sliding window size = %d\n", sliding_window_size_);
}

bool FusionKVTierManager::advance_window(int layer_id, int old_start, int old_end, int new_start, int new_end) {
    // Demote 离开的 block, promote 进入的 block
    bool ok = true;
    // Demote [old_start, new_start) if new_start > old_start
    if (new_start > old_start) {
        if (!demote_to_ssd(layer_id, old_start, new_start)) ok = false;
    }
    // Promote [old_end, new_end) if new_end > old_end (newly in window)
    if (new_end > old_end) {
        if (!promote_to_gpu(layer_id, old_end, new_end)) ok = false;
    }
    return ok;
}

// ============================================================
// Stats
// ============================================================
FusionKVTierManager::Stats FusionKVTierManager::get_stats() const {
    Stats s = stats_;
    s.bytes_in_gpu = 0;
    s.bytes_in_cpu = 0;
    s.bytes_in_ssd = 0;
    for (int il = 0; il < n_layers_; ++il) {
        for (const auto& b : blocks_[il]) {
            size_t sz = b.size_bytes > 0 ? b.size_bytes : block_byte_size();
            switch (b.tier) {
                case KVTier::L0_GPU: s.bytes_in_gpu += sz; break;
                case KVTier::L1_CPU: s.bytes_in_cpu += sz; break;
                case KVTier::L2_SSD: s.bytes_in_ssd += sz; break;
            }
        }
    }
    return s;
}

void FusionKVTierManager::reset_stats() {
    stats_ = Stats{};
}

void FusionKVTierManager::print_summary() const {
    Stats s = get_stats();
    fprintf(stderr, "[KVTier] === Summary ===\n");
    fprintf(stderr, "[KVTier]   L0 GPU: %.2f MB\n", s.bytes_in_gpu / (1024.0 * 1024.0));
    fprintf(stderr, "[KVTier]   L1 CPU: %.2f MB\n", s.bytes_in_cpu / (1024.0 * 1024.0));
    fprintf(stderr, "[KVTier]   L2 SSD: %.2f MB\n", s.bytes_in_ssd / (1024.0 * 1024.0));
    fprintf(stderr, "[KVTier]   Promotions (L2→L1): %zu\n", s.n_promotions);
    fprintf(stderr, "[KVTier]   Evictions (L1→L2): %zu\n", s.n_evictions);
    fprintf(stderr, "[KVTier]   GPU copies (L1→L0): %zu\n", s.n_gpu_copies);
    fprintf(stderr, "[KVTier]   SSD reads: %zu\n", s.n_ssd_reads);
    fprintf(stderr, "[KVTier]   SSD writes: %zu\n", s.n_ssd_writes);
}

// ============================================================
// Internal helpers (stubs for W1 D2-D5)
// ============================================================
bool FusionKVTierManager::do_promote_to_cpu(int layer_id, int block_id) {
    // TODO W1 D3: 实现 L2 → L1
    //  1) mmap SSD file (PoC-4 已验证)
    //  2) copy 到 l1_buffers_[layer_id]
    //  3) mlock (用 fusion_mmap_mlock_layer 模式)
    //  4) 更新 blocks_[layer_id][block_id].tier = L1_CPU
    //  5) blocks_[layer_id][block_id].ptr = l1_buffers_[layer_id] + offset
    fprintf(stderr, "[KVTier] STUB: do_promote_to_cpu(layer=%d, block=%d)\n", layer_id, block_id);
    if (!read_ssd_block(layer_id, block_id)) return false;
    blocks_[layer_id][block_id].tier = KVTier::L1_CPU;
    blocks_[layer_id][block_id].is_loaded = true;
    blocks_[layer_id][block_id].ptr = (char*)l1_buffers_[layer_id] + block_id * block_byte_size();
    return true;
}

bool FusionKVTierManager::do_promote_to_gpu(int layer_id, int block_id) {
    // TODO W1 D4: 实现 L1 → L0
    //  1) ensure L0 buffer 大小够 (ensure_l0_buffers)
    //  2) copy l1_buffers_[layer_id] → l0_k_tensors_[layer_id] via ggml_backend_tensor_copy
    //  3) 更新 blocks_[layer_id][block_id].tier = L0_GPU
    fprintf(stderr, "[KVTier] STUB: do_promote_to_gpu(layer=%d, block=%d)\n", layer_id, block_id);
    if (!ensure_l0_buffers(block_end(block_id))) return false;
    // TODO: actual host→device copy
    blocks_[layer_id][block_id].tier = KVTier::L0_GPU;
    stats_.n_gpu_copies++;
    return true;
}

bool FusionKVTierManager::do_demote_to_ssd(int layer_id, int block_id) {
    // demote 在 demote_to_ssd() 中通过 flush_to_ssd 实现
    // Skeleton: keep L1 buffer ptr valid even after tier becomes L2 (caller can
    // still touch data while we incrementally flush); real impl will track SSD
    // offset in ptr once mmap is set up.
    return true;
}

bool FusionKVTierManager::write_ssd_block(int layer_id, int block_id) {
    // TODO W1 D3: 写 SSD file (real impl: open + lseek + write + fsync + close)
    //  1) open(ssd_file_paths_[il], O_WRONLY|O_CREAT)
    //  2) lseek 到 block offset
    //  3) write l1_buffers_[il] + block offset
    //  4) fsync
    //  5) close
    if (ssd_file_paths_[layer_id].empty()) {
        fprintf(stderr, "[KVTier] STUB: write_ssd_block(layer=%d, block=%d) - no SSD path set, no-op\n",
                layer_id, block_id);
        return true;  // skeleton: don't fail when SSD path not set
    }
    fprintf(stderr, "[KVTier] STUB: write_ssd_block(layer=%d, block=%d, file=%s)\n",
            layer_id, block_id, ssd_file_paths_[layer_id].c_str());
    return true;
}

bool FusionKVTierManager::read_ssd_block(int layer_id, int block_id) {
    // TODO W1 D3: 读 SSD file (real impl: open + mmap + memcpy + mlock)
    //  1) open ssd_file_paths_[il]
    //  2) mmap (PoC-4 已验证)
    //  3) memcpy to l1_buffers_[il] + offset
    //  4) mlock (or posix_madvise SEQUENTIAL)
    if (ssd_file_paths_[layer_id].empty()) {
        fprintf(stderr, "[KVTier] STUB: read_ssd_block(layer=%d, block=%d) - no SSD path set, no-op (zero buffer)\n",
                layer_id, block_id);
        // Skeleton: zero-fill destination so caller gets deterministic data
        size_t offset = (size_t)block_id * block_byte_size();
        memset((char*)l1_buffers_[layer_id] + offset, 0, block_byte_size());
        return true;
    }
    fprintf(stderr, "[KVTier] STUB: read_ssd_block(layer=%d, block=%d)\n", layer_id, block_id);
    return true;
}

bool FusionKVTierManager::ensure_l0_buffers(int max_tokens) {
    if (max_tokens <= l0_max_tokens_ && l0_k_tensors_[0] != nullptr) {
        return true;  // already have enough
    }
    // TODO W1 D4: 分配 ggml tensors in ctx_
    //  for (int il = 0; il < n_layers_; ++il) {
    //      l0_k_tensors_[il] = ggml_new_tensor_2d(ctx_, kv_type_, kv_head_dim_, max_tokens);
    //      l0_v_tensors_[il] = ggml_new_tensor_2d(ctx_, kv_type_, kv_head_dim_, max_tokens);
    //      ggml_format_name(l0_k_tensors_[il], "kv_l0_k_l%d", il);
    //  }
    l0_max_tokens_ = max_tokens;
    fprintf(stderr, "[KVTier] STUB: ensure_l0_buffers(%d tokens)\n", max_tokens);
    return true;
}

}  // namespace fusion