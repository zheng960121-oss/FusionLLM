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
#include <cerrno>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

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

    // 4) SSD paths + file descriptors
    ssd_file_paths_.resize(n_layers_);
    ssd_mmapped_.resize(n_layers_, nullptr);
    ssd_fds_.assign(n_layers_, -1);  // -1 = not open
}

FusionKVTierManager::~FusionKVTierManager() {
    fprintf(stderr, "[KVTier] destroying\n");
    // Free L1 buffers (also munlock: calloc memory is not mlock'd by default,
    // but mlock we did in read_ssd_block needs undoing).
    for (size_t i = 0; i < l1_buffers_.size(); ++i) {
        if (l1_buffers_[i]) {
            munlock(l1_buffers_[i], l1_buffer_sizes_[i]);
            free(l1_buffers_[i]);
        }
    }
    // Munmap SSD
    for (size_t i = 0; i < ssd_mmapped_.size(); ++i) {
        if (ssd_mmapped_[i]) {
            // TODO: munmap + close fd
            ssd_mmapped_[i] = nullptr;
        }
    }
    // Close SSD fds
    for (int fd : ssd_fds_) {
        if (fd >= 0) close(fd);
    }
    ssd_fds_.clear();
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

    // Per-layer file size: 4 KB header + n_blocks * block_byte_size
    // (4 KB reserved for future KVBlockMeta magic + version + checksum)
    const size_t header_size = 4 * 1024;
    const size_t per_block = block_byte_size();
    const int n_blocks = n_blocks_per_layer();
    const size_t file_size = header_size + (size_t)n_blocks * per_block;

    // Set per-layer file paths + open fds
    int opened = 0;
    for (int il = 0; il < n_layers_; ++il) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/layer_%03d.bin", ssd_dir_.c_str(), il);
        ssd_file_paths_[il] = path;

        // Open with O_RDWR | O_CREAT so we can both read (promote) and write
        // (demote).  Mode 0644 = owner RW, group/other R.
        int fd = open(path, O_RDWR | O_CREAT, 0644);
        if (fd < 0) {
            fprintf(stderr, "[KVTier]   open(%s) failed: errno=%d %s\n", path, errno, strerror(errno));
            ssd_fds_[il] = -1;
            continue;
        }
        // Resize to expected size if smaller
        struct stat fst;
        if (fstat(fd, &fst) == 0 && (size_t)fst.st_size < file_size) {
            if (ftruncate(fd, (off_t)file_size) != 0) {
                fprintf(stderr, "[KVTier]   ftruncate(%s, %zu) failed: errno=%d %s\n",
                        path, file_size, errno, strerror(errno));
                close(fd);
                ssd_fds_[il] = -1;
                continue;
            }
        }
        ssd_fds_[il] = fd;
        opened++;
    }
    fprintf(stderr, "[KVTier]   opened %d/%d SSD files in %s (per-file=%.2f MB)\n",
            opened, n_layers_, ssd_dir_.c_str(),
            (double)file_size / (1024.0 * 1024.0));
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
// Hook for llama.cpp attention forward (W1 D4)
// ============================================================
// Called by llama.cpp's per-layer graph callback (or by a user wrapper) to
// ensure the K (or V) block for [start_tok, end_tok) is in L0 (GPU).  This
// triggers L2->L1->L0 promotion if needed, then returns a ggml_tensor view
// of the L0 staging buffer sliced to the requested token range.  The view
// shares storage with the full L0 tensor (no copy); it's safe to use as
// long as the manager outlives the compute graph.
ggml_tensor* FusionKVTierManager::ensure_for_attention(int layer_id, int start_tok, int end_tok, bool is_value) {
    if (layer_id < 0 || layer_id >= n_layers_) return nullptr;
    if (start_tok < 0 || end_tok <= start_tok) return nullptr;

    // 1) Promote all blocks in the range from L2/L1 → L0
    if (!promote_to_gpu(layer_id, start_tok, end_tok)) {
        return nullptr;
    }

    // 2) Pick the right L0 tensor (K or V) and return a view of [start, end)
    ggml_tensor* full = is_value ? l0_v_tensors_[layer_id] : l0_k_tensors_[layer_id];
    if (full == nullptr) return nullptr;

    int width = end_tok - start_tok;
    // ggml_view_2d(ctx, tensor, ne0, ne1, nb1, offset) - take a [ne0, width] view
    // starting at column start_tok.
    size_t byte_offset = (size_t)start_tok * (size_t)full->nb[1];
    ggml_tensor* view = ggml_view_2d(ctx_, full, (int64_t)full->ne[0], (int64_t)width,
                                       full->nb[1], byte_offset);
    if (view) {
        char vname[64];
        snprintf(vname, sizeof(vname), "fusion_kv_%s_l%d_t%d_%d",
                 is_value ? "v" : "k", layer_id, start_tok, end_tok);
        ggml_format_name(view, vname);
    }
    return view;
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
    // W1 D4: 实现 L1 → L0
    //  1) ensure L0 buffer 大小够 (ensure_l0_buffers)
    //  2) copy l1_buffers_[layer_id] → l0_k_tensors_[layer_id]
    //     - 在 CPU 后端: plain memcpy（ctx_ 分配的 buffer 就是 CPU memory）
    //     - 在 GPU 后端 (W2): ggml_backend_tensor_copy 跨设备
    //  3) 更新 blocks_[layer_id][block_id].tier = L0_GPU
    int end_tok = block_end(block_id);
    if (!ensure_l0_buffers(end_tok)) return false;

    // Copy L1 -> L0 for both K and V slices.
    // The L0 tensor is sized [n_embd_k_gqa, max_tokens]; we copy into the
    // [start, end) range using the block's L1 offset as the source row.
    int start_tok = block_start(block_id);
    size_t block_bytes = block_byte_size();
    size_t src_offset = (size_t)block_id * block_bytes;
    size_t dst_offset = (size_t)start_tok * (size_t)kv_head_dim_ * ggml_type_size(kv_type_);

    // K tensor
    if (l0_k_tensors_[layer_id] && l0_k_tensors_[layer_id]->data) {
        char* dst_k = (char*)l0_k_tensors_[layer_id]->data + dst_offset;
        const char* src_k = (const char*)l1_buffers_[layer_id] + src_offset;
        memcpy(dst_k, src_k, block_bytes);
    }
    // V tensor (same shape as K; copy independently for symmetry — in
    // practice, llama.cpp's K and V caches share the L1 staging layout)
    if (l0_v_tensors_[layer_id] && l0_v_tensors_[layer_id]->data) {
        char* dst_v = (char*)l0_v_tensors_[layer_id]->data + dst_offset;
        const char* src_v = (const char*)l1_buffers_[layer_id] + src_offset;
        memcpy(dst_v, src_v, block_bytes);
    }

    blocks_[layer_id][block_id].tier = KVTier::L0_GPU;
    // ptr stays pointing at the L1 buffer (canonical staging ptr); the L0
    // tensor is what callers should use for actual compute, accessed via
    // ensure_for_attention().
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

// SSD file layout (per layer):
//   [0 .. 4KB)                    : reserved header (KVBlockMeta magic + version + checksum)
//   [4KB + b*block_byte_size, 4KB + (b+1)*block_byte_size) : block b raw data
static constexpr size_t SSD_HEADER_SIZE = 4 * 1024;
static inline off_t ssd_block_offset(int block_id, size_t block_byte_size) {
    return (off_t)SSD_HEADER_SIZE + (off_t)block_id * (off_t)block_byte_size;
}

bool FusionKVTierManager::write_ssd_block(int layer_id, int block_id) {
    if (ssd_fds_.empty() || layer_id >= (int)ssd_fds_.size() || ssd_fds_[layer_id] < 0) {
        // No SSD path set or open failed — skeleton fallback (no-op)
        if (!ssd_file_paths_.empty() && !ssd_file_paths_[layer_id].empty()) {
            fprintf(stderr, "[KVTier] write_ssd_block(layer=%d, block=%d) - fd invalid, no-op\n",
                    layer_id, block_id);
        }
        return true;
    }
    int fd = ssd_fds_[layer_id];
    size_t bsz = block_byte_size();
    size_t buffer_offset = (size_t)block_id * bsz;
    off_t file_offset = ssd_block_offset(block_id, bsz);

    // pwrite is thread-safe; we don't need a lock for concurrent writers to
    // different blocks because each block has a unique file offset.
    ssize_t written = pwrite(fd, (const char*)l1_buffers_[layer_id] + buffer_offset,
                              bsz, file_offset);
    if (written < 0 || (size_t)written != bsz) {
        fprintf(stderr, "[KVTier] write_ssd_block(layer=%d, block=%d, fd=%d) failed: "
                "wrote %zd/%zu bytes, errno=%d %s\n",
                layer_id, block_id, fd, written, bsz, errno, strerror(errno));
        return false;
    }
    // Best-effort: don't fsync per block (10x+ slower on macOS).  flush_all_to_ssd()
    // can call fsync() at the end of a batch if the caller wants durability.
    return true;
}

bool FusionKVTierManager::read_ssd_block(int layer_id, int block_id) {
    if (ssd_fds_.empty() || layer_id >= (int)ssd_fds_.size() || ssd_fds_[layer_id] < 0) {
        // No SSD path set — zero-fill destination so caller gets deterministic data
        size_t bsz = block_byte_size();
        size_t buffer_offset = (size_t)block_id * bsz;
        if (!l1_buffers_.empty() && l1_buffers_[layer_id]) {
            memset((char*)l1_buffers_[layer_id] + buffer_offset, 0, bsz);
        }
        return true;
    }
    int fd = ssd_fds_[layer_id];
    size_t bsz = block_byte_size();
    size_t buffer_offset = (size_t)block_id * bsz;
    off_t file_offset = ssd_block_offset(block_id, bsz);

    // Read into the L1 buffer at the block's offset
    ssize_t got = pread(fd, (char*)l1_buffers_[layer_id] + buffer_offset,
                         bsz, file_offset);
    if (got < 0) {
        fprintf(stderr, "[KVTier] read_ssd_block(layer=%d, block=%d, fd=%d) failed: "
                "errno=%d %s\n",
                layer_id, block_id, fd, errno, strerror(errno));
        return false;
    }
    if ((size_t)got < bsz) {
        // Short read: file is shorter than expected (just-created or partial).
        // Zero the unread tail and treat as success.  This is the documented
        // behavior for blocks that haven't been written yet.
        memset((char*)l1_buffers_[layer_id] + buffer_offset + got, 0, bsz - (size_t)got);
    }

    // mlock the L1 buffer pages for this block.  mlock requires page-aligned
    // addresses, so expand the range to cover the whole page(s) that contain
    // the block.  The L1 buffer comes from calloc() which returns 16-byte
    // aligned memory; for the typical block_byte_size (>= 4 KB) the block
    // itself is page-aligned.  For smaller blocks we round down/up.
    long pg = sysconf(_SC_PAGESIZE);
    if (pg > 0) {
        uintptr_t base_addr = (uintptr_t)l1_buffers_[layer_id];
        uintptr_t block_start_addr = base_addr + buffer_offset;
        uintptr_t page_start = block_start_addr & ~((uintptr_t)pg - 1);
        uintptr_t block_end_addr = block_start_addr + bsz;
        uintptr_t page_end = (block_end_addr + (uintptr_t)pg - 1) & ~((uintptr_t)pg - 1);
        size_t lock_size = (size_t)(page_end - page_start);
        if (mlock((void*)page_start, lock_size) != 0) {
            // mlock failure is non-fatal (it'll just fall back to swap on
            // pressure).  Log at low verbosity.
            fprintf(stderr, "[KVTier] mlock(layer=%d, block=%d, %zu bytes) failed: "
                    "errno=%d %s (block still works, but may swap)\n",
                    layer_id, block_id, lock_size, errno, strerror(errno));
        }
    }
    return true;
}

bool FusionKVTierManager::ensure_l0_buffers(int max_tokens) {
    if (max_tokens <= l0_max_tokens_ && !l0_k_tensors_.empty() && l0_k_tensors_[0] != nullptr) {
        return true;  // already have enough
    }
    if (ctx_ == nullptr) {
        fprintf(stderr, "[KVTier] ensure_l0_buffers: no ggml context, L0 path disabled\n");
        return false;
    }
    // Allocate (or grow) the L0 staging tensors in ctx_.  Layout mirrors
    // llama.cpp's K/V cache: [n_embd_k_gqa, kv_size].
    //   ne[0] = n_embd_k_gqa (= kv_head_dim_ in our model)
    //   ne[1] = max_tokens (= kv_size for full context)
    // When max_tokens is less than kv_size, we allocate the smaller view and
    // let later calls grow it (ggml tensors in a context are immutable in
    // shape; we always allocate the full kv_size at first call).
    int alloc_tokens = max_tokens;
    if (alloc_tokens < kv_size_) alloc_tokens = kv_size_;  // never grow down

    // If tensors are already allocated but smaller, we need a fresh context
    // to realloc — but our ctx_ is caller-owned.  Strategy: allocate the full
    // kv_size once, and only realloc if a bigger request comes in.
    if (l0_k_tensors_.empty()) {
        l0_k_tensors_.assign(n_layers_, nullptr);
        l0_v_tensors_.assign(n_layers_, nullptr);
    }
    if (l0_k_tensors_[0] != nullptr && (int)l0_k_tensors_[0]->ne[1] >= alloc_tokens) {
        l0_max_tokens_ = (int)l0_k_tensors_[0]->ne[1];
        return true;
    }
    // (Re)allocate.  If the previous tensors are smaller, we cannot resize
    // in-place — caller must provide a bigger ctx_.  We assume the first
    // call has the right size.
    for (int il = 0; il < n_layers_; ++il) {
        char name_k[64], name_v[64];
        snprintf(name_k, sizeof(name_k), "fusion_kv_l0_k_l%d", il);
        snprintf(name_v, sizeof(name_v), "fusion_kv_l0_v_l%d", il);
        l0_k_tensors_[il] = ggml_new_tensor_2d(ctx_, kv_type_, (int64_t)kv_head_dim_, (int64_t)alloc_tokens);
        l0_v_tensors_[il] = ggml_new_tensor_2d(ctx_, kv_type_, (int64_t)kv_head_dim_, (int64_t)alloc_tokens);
        if (l0_k_tensors_[il]) ggml_format_name(l0_k_tensors_[il], "%s", name_k);
        if (l0_v_tensors_[il]) ggml_format_name(l0_v_tensors_[il], "%s", name_v);
    }
    l0_max_tokens_ = alloc_tokens;
    fprintf(stderr, "[KVTier] L0 buffers allocated: %d layers × [%d, %d] = %.2f MB/layer\n",
            n_layers_, kv_head_dim_, alloc_tokens,
            (double)kv_head_dim_ * alloc_tokens * ggml_type_size(kv_type_) / (1024.0 * 1024.0));
    return true;
}

}  // namespace fusion