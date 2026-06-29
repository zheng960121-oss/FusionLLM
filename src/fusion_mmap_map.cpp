// FusionLLM Phase 2: Layer-to-memory mapping implementation
//
// 关键设计：每个 layer 的 tensors 在文件里可能不连续
// （embedding / output / 其他 layers 的 tensors 散布在文件各位置）
// 所以 mlock 必须按 tensor 分别做，不能 mlock 整块（会失败）

#include "fusion_mmap_map.h"
#include "llama-model.h"
#include "ggml.h"
#include "gguf.h"

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <climits>

namespace fusion {

// 单个 tensor 在 mmap'd 文件中的精确位置（不含 gap）
struct TensorRange {
    void*  ptr;
    size_t size;
};

// 一个 layer = 多个 tensor ranges + 总数据量
struct LayerMapping {
    int                        layer_idx;
    std::vector<TensorRange>   tensors;
    size_t                     total_bytes = 0;
};

static std::vector<LayerMapping> g_mappings;
static std::unordered_map<int, LayerMapping*> g_lookup;

void mmap_map_build(int n_layers, void* mmap_base_ptr, size_t mmap_total_size) {
    g_mappings.clear();
    g_lookup.clear();
    (void)n_layers;
    (void)mmap_base_ptr;
    (void)mmap_total_size;
}

const LayerMapping* mmap_map_get(int layer_idx) {
    auto it = g_lookup.find(layer_idx);
    if (it == g_lookup.end()) return nullptr;
    return it->second;
}

int mmap_map_layer_count() {
    return (int)g_mappings.size();
}

size_t mmap_mlock_layer(int layer_idx) {
    LayerMapping* m = const_cast<LayerMapping*>(mmap_map_get(layer_idx));
    if (!m || m->tensors.empty()) return 0;

    size_t total_mlocked = 0;
    int success_count = 0;
    int fail_count = 0;
    int first_err = 0;
    for (size_t ti = 0; ti < m->tensors.size(); ti++) {
        const auto& t = m->tensors[ti];
        // 先 touch 每个 page 让 page-in 完成，避免 mlock 大块时 EAGAIN
        const size_t page_size = 4096;
        volatile char* p = (volatile char*)t.ptr;
        for (size_t off = 0; off < t.size; off += page_size) {
            volatile char touch = p[off];
            (void)touch;
        }
        errno = 0;
        if (mlock(t.ptr, t.size) == 0) {
            total_mlocked += t.size;
            success_count++;
        } else {
            fail_count++;
            if (first_err == 0) first_err = errno;
        }
    }
    if (success_count == (int)m->tensors.size()) {
        fprintf(stderr, "[FusionLLM] mlock layer %d: %d/%zu OK, %zu bytes\n",
                layer_idx, success_count, m->tensors.size(), total_mlocked);
    } else {
        fprintf(stderr, "[FusionLLM] mlock layer %d: %d/%zu OK, %d failed (err=%d %s)\n",
                layer_idx, success_count, m->tensors.size(), fail_count,
                first_err, first_err ? strerror(first_err) : "none");
    }
    return total_mlocked;
}

void mmap_munlock_layer(int layer_idx) {
    LayerMapping* m = const_cast<LayerMapping*>(mmap_map_get(layer_idx));
    if (!m) return;

    int success_count = 0;
    for (const auto& t : m->tensors) {
        if (munlock(t.ptr, t.size) == 0) {
            success_count++;
        }
    }
    if (success_count > 0) {
        fprintf(stderr, "[FusionLLM] munlocked layer %d: %d tensors\n",
                layer_idx, success_count);
    }
}

void mmap_map_cleanup() {
    for (auto& m : g_mappings) {
        for (const auto& t : m.tensors) {
            munlock(t.ptr, t.size);
        }
    }
    g_mappings.clear();
    g_lookup.clear();
}

void populate_from_gguf(gguf_context* gguf_ctx, void* mmap_base) {
    if (!gguf_ctx) return;
    if (mmap_base == nullptr) {
        fprintf(stderr, "[FusionLLM] populate_from_gguf: mmap_base is NULL, skipping mlock\n");
        return;
    }
    // 保留备用但当前未使用
    (void)mmap_base;
}

void mmap_map_populate(llama_model* model, const char* gguf_path) {
    if (!model || !gguf_path) return;

    g_mappings.clear();
    g_lookup.clear();

    // FusionLLM 自管 mmap：不依赖 llama_model->fusion_mmap_base()
    // llama.cpp 在 Metal backend + fit_params 场景会禁用 mmap（use_mmap=false）
    static void*  g_fusion_mmap_base = nullptr;
    static size_t g_fusion_mmap_size = 0;
    static int    g_fusion_mmap_fd   = -1;
    static std::string g_fusion_mmap_path;

    if (g_fusion_mmap_path != gguf_path) {
        if (g_fusion_mmap_base) {
            munmap(g_fusion_mmap_base, g_fusion_mmap_size);
            close(g_fusion_mmap_fd);
        }
        g_fusion_mmap_fd = open(gguf_path, O_RDONLY);
        if (g_fusion_mmap_fd < 0) {
            fprintf(stderr, "[FusionLLM] populate: open(%s) failed: %s\n", gguf_path, strerror(errno));
            return;
        }
        struct stat st;
        if (fstat(g_fusion_mmap_fd, &st) != 0) {
            fprintf(stderr, "[FusionLLM] populate: fstat failed: %s\n", strerror(errno));
            close(g_fusion_mmap_fd);
            return;
        }
        g_fusion_mmap_size = st.st_size;
        g_fusion_mmap_base = mmap(nullptr, g_fusion_mmap_size, PROT_READ, MAP_SHARED, g_fusion_mmap_fd, 0);
        if (g_fusion_mmap_base == MAP_FAILED) {
            fprintf(stderr, "[FusionLLM] populate: mmap failed: %s\n", strerror(errno));
            close(g_fusion_mmap_fd);
            g_fusion_mmap_base = nullptr;
            g_fusion_mmap_fd = -1;
            return;
        }
        // 预读首 page 让实际分配物理内存
        volatile char touch = *((char*)g_fusion_mmap_base);
        (void)touch;
        g_fusion_mmap_path = gguf_path;
        fprintf(stderr, "[FusionLLM] self-mmap: %s size=%.1f MB base=%p\n",
                gguf_path, g_fusion_mmap_size/1024.0/1024.0, g_fusion_mmap_base);
    }

    void* mmap_base = g_fusion_mmap_base;
    size_t mmap_size = g_fusion_mmap_size;
    (void)model;

    if (!mmap_base || mmap_size == 0) {
        return;
    }

    gguf_context* gctx = gguf_init_from_file(gguf_path, (gguf_init_params){ true, NULL });
    if (!gctx) {
        return;
    }

    // 按 layer index 分组，保留每个 tensor 的精确范围
    std::unordered_map<int, LayerMapping> layer_maps;

    int64_t n_tensors = gguf_get_n_tensors(gctx);
    for (int64_t i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gctx, i);
        if (!name) continue;

        if (strncmp(name, "blk.", 4) != 0) continue;
        int layer_idx = atoi(name + 4);
        if (layer_idx < 0) continue;

        int64_t offset = gguf_get_tensor_offset(gctx, i);
        size_t size = gguf_get_tensor_size(gctx, i);
        if (size == 0) continue;

        TensorRange tr;
        tr.ptr = (char*)mmap_base + offset;
        tr.size = size;

        auto& lm = layer_maps[layer_idx];
        lm.layer_idx = layer_idx;
        lm.tensors.push_back(tr);
        lm.total_bytes += size;
    }

    gguf_free(gctx);

    for (auto& [idx, lm] : layer_maps) {
        std::sort(lm.tensors.begin(), lm.tensors.end(),
                  [](const TensorRange& a, const TensorRange& b) {
                      return a.ptr < b.ptr;
                  });
        g_mappings.push_back(lm);
    }

    for (auto& m : g_mappings) {
        g_lookup[m.layer_idx] = &m;
    }

    size_t total_tensors = std::accumulate(g_mappings.begin(), g_mappings.end(), (size_t)0,
        [](size_t sum, const LayerMapping& lm) { return sum + lm.tensors.size(); });

    fprintf(stderr, "[FusionLLM] populated %zu layer mappings (%zu total tensor ranges, mmap_base=%p size=%zu)\n",
            g_mappings.size(), total_tensors, mmap_base, mmap_size);
}

} // namespace fusion

extern "C" void fusion_populate_from_gguf(void* gguf_ctx, void* mmap_base) {
    fusion::populate_from_gguf((gguf_context*)gguf_ctx, mmap_base);
}

extern "C" void fusion_mmap_map_build(int n_layers, void* base, size_t total) {
    fusion::mmap_map_build(n_layers, base, total);
}

extern "C" void fusion_mmap_populate(void* model, const char* gguf_path) {
    fusion::mmap_map_populate((llama_model*)model, gguf_path);
}

extern "C" int fusion_mmap_map_get_layer_count() {
    return fusion::mmap_map_layer_count();
}

extern "C" size_t fusion_mmap_mlock_layer(int layer_idx) {
    return fusion::mmap_mlock_layer(layer_idx);
}

extern "C" void fusion_mmap_munlock_layer(int layer_idx) {
    fusion::mmap_munlock_layer(layer_idx);
}

extern "C" void fusion_mmap_cleanup() {
    fusion::mmap_map_cleanup();
}