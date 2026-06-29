// FusionLLM driver implementation (Phase 1 prototype)
//
// 当前阶段：仅占位实现，使用 llama.cpp 的 --mlock
// 后续阶段：实现真正的 selective mlock 和滑动窗口

#include "fusion_driver.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <cerrno>

namespace fusion {

static bool g_enabled = false;
static size_t g_total_mlocked = 0;
static int g_locked_layers = 0;

void driver_init() {
    const char* env = getenv("FUSION_DRIVER");
    g_enabled = (env != nullptr && env[0] == '1');
    
    if (g_enabled) {
        fprintf(stderr, "[FusionLLM] driver initialized (Phase 1 prototype)\n");
        fprintf(stderr, "[FusionLLM] selective mlock not yet implemented\n");
        fprintf(stderr, "[FusionLLM] use llama.cpp's --mlock flag for full-model mlock\n");
    }
}

void driver_shutdown() {
    if (g_enabled) {
        fprintf(stderr, "[FusionLLM] driver shutdown, mlocked %zu bytes total\n", g_total_mlocked);
    }
    g_enabled = false;
    g_total_mlocked = 0;
    g_locked_layers = 0;
}

bool driver_enabled() {
    return g_enabled;
}

size_t driver_lock_window(int center_layer, int window_size) {
    // TODO Phase 2: 实现真正的 selective mlock
    // 当前：只报告意图，不做实际 mlock
    if (g_enabled) {
        fprintf(stderr, "[FusionLLM] lock_window(center=%d, size=%d) - not yet implemented\n",
                center_layer, window_size);
    }
    return 0;
}

size_t driver_advance_to_layer(int next_layer) {
    // TODO Phase 2: 实现滑动窗口
    return driver_lock_window(next_layer, 3);  // 默认窗口大小 3
}

size_t driver_total_mlocked() {
    return g_total_mlocked;
}

int driver_locked_layer_count() {
    return g_locked_layers;
}

} // namespace fusion
