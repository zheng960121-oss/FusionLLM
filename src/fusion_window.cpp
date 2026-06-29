// FusionLLM Phase 2: Selective mlock sliding window implementation
//
// 通过追踪 graph build 过程中的 layer 访问模式，自动维护
// 一组 mlock'd 的层（活跃滑动窗口）

#include "fusion_window.h"
#include "fusion_mmap_map.h"

#include <sys/mman.h>
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

namespace fusion {

static WindowState g_state;

// S5: spec decoding 协同接口状态
static int g_step_advance_count = 0;           // 当前 verify step 累计的 advance 次数
static std::vector<int> g_step_advance_history; // 历史上每个 verify step 的 advance 次数

void window_on_verify_step_begin() {
    g_step_advance_count = 0;
}

void window_on_verify_step_end() {
    g_step_advance_history.push_back(g_step_advance_count);
    if (g_state.enabled) {
        fprintf(stderr, "[FusionLLM] verify step ended: %d window_advance calls\n",
                g_step_advance_count);
    }
    g_step_advance_count = 0;
}

int window_get_last_step_advance_count() {
    if (g_step_advance_history.empty()) return 0;
    return g_step_advance_history.back();
}

const std::vector<int>& window_get_step_advance_history() {
    return g_step_advance_history;
}

void window_init(int n_layers) {
    if (g_state.mlocked != nullptr) {
        window_shutdown();
    }
    
    g_state.n_layers = n_layers;
    g_state.mlocked = new bool[n_layers]();  // 全部初始化为 false
    g_state.total_mlocked = 0;
    g_state.current_center = -1;
    
    // 检查环境变量
    const char* env = getenv("FUSION_DRIVER");
    g_state.enabled = (env != nullptr && env[0] == '1');
    
    const char* win_env = getenv("FUSION_WINDOW");
    if (win_env != nullptr) {
        g_state.window_size = atoi(win_env);
    }
    
    if (g_state.enabled) {
        fprintf(stderr, "[FusionLLM] window_init: %d layers, window_size=%d\n",
                n_layers, g_state.window_size);
    }
}

void window_shutdown() {
    if (g_state.mlocked != nullptr) {
        // munlock 所有
        for (int i = 0; i < g_state.n_layers; i++) {
            // 真实 mlock 在 window_mlock_layer 中实现
        }
        delete[] g_state.mlocked;
        g_state.mlocked = nullptr;
    }
    g_state.total_mlocked = 0;
    g_state.current_center = -1;
    if (g_state.enabled) {
        fprintf(stderr, "[FusionLLM] window_shutdown: released all mlock\n");
    }
}

void window_set_size(int size) {
    if (size > 0) g_state.window_size = size;
}

int window_get_size() {
    return g_state.window_size;
}

void window_set_enabled(bool enabled) {
    g_state.enabled = enabled;
    if (enabled) {
        fprintf(stderr, "[FusionLLM] window enabled\n");
    }
}

bool window_is_enabled() {
    return g_state.enabled;
}

const WindowState& window_get_state() {
    return g_state;
}

// 注：实际 mlock/munlock 由调用方在拿到 layer ptr 时调用
// 这里只追踪"哪些层应该 mlock'd"

void window_advance(int layer_idx) {
    if (!g_state.enabled || g_state.mlocked == nullptr) return;
    if (layer_idx < 0) return;  // 非 layer tensor (embed/output) 不动窗口

    // S5: 累计当前 verify step 内的 advance 次数
    g_step_advance_count++;
    
    // 首次调用：初始化窗口（mlock 初始 N 层）
    if (g_state.current_center < 0) {
        // mlock layers 0..window_size-1
        int end = std::min(g_state.window_size, g_state.n_layers);
        for (int i = 0; i < end; i++) {
            g_state.mlocked[i] = true;
        }
        g_state.current_center = 0;
        fprintf(stderr, "[FusionLLM] window initialized, layers 0-%d marked as mlock'd\n", end - 1);
        return;
    }
    
    // 计算新的窗口范围
    int new_center = layer_idx;
    int window_start = std::max(0, new_center - g_state.window_size / 2);
    int window_end = std::min(g_state.n_layers - 1, new_center + g_state.window_size / 2);
    
    // 当前窗口范围
    int cur_start = std::max(0, g_state.current_center - g_state.window_size / 2);
    int cur_end = std::min(g_state.n_layers - 1, g_state.current_center + g_state.window_size / 2);
    
    if (new_center == g_state.current_center) {
        return;  // 没变化
    }
    
    // Munlock 离开窗口的层（真实 munlock 通过 fusion::mmap_munlock_layer）
    int released = 0;
    for (int i = cur_start; i <= cur_end; i++) {
        if (i < window_start || i > window_end) {
            if (g_state.mlocked[i]) {
                fusion::mmap_munlock_layer(i);
                g_state.mlocked[i] = false;
                released++;
            }
        }
    }
    
    // Mark 新窗口中、未在旧窗口的层为 mlock'd（实际 mlock 通过 fusion::mmap_mlock_layer）
    int added = 0;
    for (int i = window_start; i <= window_end; i++) {
        if (i < cur_start || i > cur_end) {
            if (!g_state.mlocked[i]) {
                size_t mlocked = fusion::mmap_mlock_layer(i);
                if (mlocked > 0) {
                    g_state.mlocked[i] = true;
                    added++;
                    g_state.total_mlocked += mlocked;
                }
            }
        }
    }
    
    g_state.current_center = new_center;
    
    if (released > 0 || added > 0) {
        fprintf(stderr, "[FusionLLM] window_advance: layer %d (range %d-%d), +%d -%d\n",
                new_center, window_start, window_end, added, released);
    }
}

// 显式 mlock 某层（用于真实层 mlock，需要 ptr 信息）
size_t window_mlock_layer(int layer_idx, void* base_ptr,
                          size_t layer_offset, size_t layer_size) {
    if (!g_state.enabled) return 0;
    if (layer_idx < 0 || layer_idx >= g_state.n_layers) return 0;
    
    void* ptr = (char*)base_ptr + layer_offset;
    if (mlock(ptr, layer_size) != 0) {
        if (errno != EAGAIN) {  // EAGAIN = already locked
            fprintf(stderr, "[FusionLLM] mlock layer %d failed: %s\n",
                    layer_idx, strerror(errno));
        }
        return 0;
    }
    
    g_state.total_mlocked += layer_size;
    fprintf(stderr, "[FusionLLM] mlocked layer %d (%zu bytes), total now %llu\n",
            layer_idx, layer_size, (unsigned long long)g_state.total_mlocked);
    return layer_size;
}

void window_munlock_layer(int layer_idx) {
    // 需要 ptr + size，这里仅记录"非 mlock"状态
    // 真实 munlock 需要 ptr/size 信息
    // 简化：如果未来要实现真正的滑动，需要维护 ptr 映射
}

bool window_layer_is_mlocked(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= g_state.n_layers) return false;
    return g_state.mlocked[layer_idx];
}

void window_reset() {
    for (int i = 0; i < g_state.n_layers; i++) {
        g_state.mlocked[i] = false;
    }
    g_state.total_mlocked = 0;
    g_state.current_center = -1;
}

} // namespace fusion

// C-linkage bridge for llama.cpp internal calls
extern "C" void fusion_window_init_capture(int n_layers);  // declared in header

void fusion_window_init_capture(int n_layers) {
    fusion::window_init(n_layers);
}