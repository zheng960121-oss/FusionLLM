// FusionLLM Layer Inspector
// Reads a GGUF model, identifies per-layer weight blocks, optionally mlock's
// the first N layers (selective mlock for sliding window).
//
// Build: see build_fusion_tools.sh
// Usage: ./fusion_inspect <model.gguf> [--mlock-window N]

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <numeric>

#include "ggml.h"
#include "gguf.h"

struct TensorInfo {
    std::string name;
    ggml_type   type;
    int64_t     offset;
    int64_t     size;
};

struct LayerInfo {
    int                        layer_idx;
    std::vector<TensorInfo>    tensors;
    int64_t                    total_bytes = 0;
    int64_t                    first_offset = INT64_MAX;
    int64_t                    last_offset_end = 0;
};

// Extract layer index from tensor name
// Patterns:
//   blk.N.xxx  -> N
//   token_embd.weight, output.*, rope_* -> -1 (non-layer)
static int extract_layer_idx(const char* name) {
    if (strncmp(name, "blk.", 4) == 0) {
        return atoi(name + 4);
    }
    return -1;  // non-layer tensor
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [--mlock-window N]\n", argv[0]);
        fprintf(stderr, "  --mlock-window N  mlock the first N layers (selective mlock)\n");
        return 1;
    }

    const char* model_path = argv[1];
    int window_size = 0;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--mlock-window") == 0 && i + 1 < argc) {
            window_size = atoi(argv[++i]);
        }
    }

    // 1. mmap the model file
    int fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "open failed: %s\n", strerror(errno));
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "fstat failed: %s\n", strerror(errno));
        return 1;
    }
    size_t file_size = st.st_size;
    void* mmap_ptr = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mmap_ptr == MAP_FAILED) {
        fprintf(stderr, "mmap failed: %s\n", strerror(errno));
        return 1;
    }

    // 2. Parse GGUF
    struct gguf_init_params params = { true, NULL };
    struct gguf_context* ctx = gguf_init_from_file(model_path, params);
    if (!ctx) {
        fprintf(stderr, "Failed to parse GGUF: %s\n", model_path);
        return 1;
    }

    int64_t n_tensors = gguf_get_n_tensors(ctx);
    size_t  alignment = gguf_get_alignment(ctx);

    fprintf(stderr, "\n=== FusionLLM Layer Inspector ===\n");
    fprintf(stderr, "Model:     %s\n", model_path);
    fprintf(stderr, "File size: %.1f MB\n", file_size / 1024.0 / 1024.0);
    fprintf(stderr, "Tensors:   %lld\n", n_tensors);
    fprintf(stderr, "Alignment: %zu bytes\n", alignment);
    fprintf(stderr, "\n");

    // 3. Iterate tensors, group by layer
    std::map<int, LayerInfo> layers;
    std::vector<TensorInfo> non_layer_tensors;
    int64_t total_weight_bytes = 0;

    for (int64_t i = 0; i < n_tensors; i++) {
        const char*  name   = gguf_get_tensor_name(ctx, i);
        int64_t      offset = gguf_get_tensor_offset(ctx, i);
        size_t       size   = gguf_get_tensor_size(ctx, i);
        ggml_type    type   = gguf_get_tensor_type(ctx, i);

        int layer_idx = extract_layer_idx(name);
        TensorInfo t{name, type, offset, (int64_t)size};

        if (layer_idx >= 0) {
            auto& layer = layers[layer_idx];
            layer.layer_idx = layer_idx;
            layer.tensors.push_back(t);
            layer.total_bytes += size;
            layer.first_offset = std::min<int64_t>(layer.first_offset, offset);
            layer.last_offset_end = std::max<int64_t>(layer.last_offset_end, offset + (int64_t)size);
        } else {
            non_layer_tensors.push_back(t);
        }

        total_weight_bytes += size;
    }

    // 4. Show layer info
    fprintf(stderr, "=== Layer Summary ===\n");
    fprintf(stderr, "Layers:           %lu\n", layers.size());
    fprintf(stderr, "Layer weight MB:  %.1f\n", total_weight_bytes / 1024.0 / 1024.0);
    fprintf(stderr, "Non-layer MB:     %.1f\n",
            std::accumulate(non_layer_tensors.begin(), non_layer_tensors.end(), (int64_t)0,
                [](int64_t s, const TensorInfo& t) { return s + t.size; }) / 1024.0 / 1024.0);
    fprintf(stderr, "\n");

    fprintf(stderr, "Per-layer weight size:\n");
    fprintf(stderr, "  Layer | Tensors | Size (MB) | mlock'd at 6-layer window\n");
    fprintf(stderr, "  ------+---------+-----------+------------------------\n");

    // Compute cumulative MB for visualization
    double cum_mb = 0;
    int max_layer = layers.empty() ? 0 : layers.rbegin()->first;
    int window_sum = 0;
    for (auto& [idx, layer] : layers) {
        double layer_mb = layer.total_bytes / 1024.0 / 1024.0;
        cum_mb += layer_mb;

        // Mark layers in the first 6-layer window
        const char* in_window = (idx < 6) ? "<-- in window" : "";

        fprintf(stderr, "  %5d | %7zu | %9.1f | %s\n",
                layer.layer_idx, layer.tensors.size(), layer_mb, in_window);
        if (idx < 6) window_sum += 1;
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "Cumulative MB: %.1f\n", cum_mb);

    // 5. Show what mlock of N layers would do
    if (!layers.empty() && window_size == 0) {
        // Auto-suggest: show 6-layer window stats
        int suggested = 6;
        int64_t window_bytes = 0;
        for (int i = 0; i < std::min(suggested, max_layer + 1); i++) {
            auto it = layers.find(i);
            if (it != layers.end()) {
                window_bytes += it->second.total_bytes;
            }
        }
        fprintf(stderr, "\n=== Sliding Window Analysis (6 layers) ===\n");
        fprintf(stderr, "First 6 layers: %.1f MB\n", window_bytes / 1024.0 / 1024.0);
        fprintf(stderr, "Rest of model:  %.1f MB (mmap'd, OS file cache manages)\n",
                (total_weight_bytes - window_bytes) / 1024.0 / 1024.0);
        fprintf(stderr, "\nUse --mlock-window N to actually mlock N layers\n");
    }

    // 6. If --mlock-window N, mlock the first N layers
    if (window_size > 0) {
        fprintf(stderr, "\n=== Selective mlock: first %d layers ===\n", window_size);

        int mlocked = 0;
        size_t total_mlocked = 0;
        for (int i = 0; i < std::min(window_size, max_layer + 1); i++) {
            auto it = layers.find(i);
            if (it == layers.end()) continue;
            auto& layer = it->second;

            // mlock the range from first_offset to last_offset_end
            void* ptr = (char*)mmap_ptr + layer.first_offset;
            size_t size = layer.last_offset_end - layer.first_offset;

            if (mlock(ptr, size) == 0) {
                mlocked++;
                total_mlocked += size;
                fprintf(stderr, "  ✅ Layer %2d: mlocked %.1f MB (offset %lld-%lld)\n",
                        i, size / 1024.0 / 1024.0, layer.first_offset, layer.last_offset_end);
            } else {
                fprintf(stderr, "  ❌ Layer %2d: mlock failed: %s\n", i, strerror(errno));
            }
        }
        fprintf(stderr, "\nMlocked %d layers, total %.1f MB (in physical RAM, guaranteed)\n",
                mlocked, total_mlocked / 1024.0 / 1024.0);
    }

    // Cleanup
    gguf_free(ctx);
    munmap(mmap_ptr, file_size);
    close(fd);

    return 0;
}
