// FusionLLM Phase 2: Selective mlock sliding window
// 通过 llama.cpp 的 llm_graph_cb 回调实现真正的滑动窗口
//
// 工作原理：
// 1. 在 graph_get_cb() 注册 fusion_callback
// 2. 每个 tensor 创建时回调，传入 layer index (il)
// 3. 当 il 变化时，滑动窗口自动调整：
//    - mlock 新进入窗口的层
//    - munlock 离开窗口的层
//
// 这意味着：每次 graph build = 一次 layer 计算序列 = 滑动窗口完美工作
// 不需要修改 llama.cpp 的 compute 路径本身！

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace fusion {

// 窗口状态（每层一个）
struct WindowState {
    int      n_layers = 0;           // 模型总层数
    int      window_size = 6;        // 滑动窗口大小（前后各多少）
    bool*    mlocked = nullptr;      // 每层 mlock 状态
    uint64_t total_mlocked = 0;      // 总 mlock 字节数
    int      current_center = -1;    // 当前窗口中心 layer
    bool     enabled = false;        // 是否启用
};

// 初始化窗口（从环境变量 FUSION_DRIVER / FUSION_WINDOW 等读取配置）
void window_init(int n_layers);

// 关闭窗口（释放所有 mlock）
void window_shutdown();

// 设置窗口大小（默认 6）
void window_set_size(int size);

// 获取窗口大小
int window_get_size();

// 启用/禁用
void window_set_enabled(bool enabled);
bool window_is_enabled();

// 滑动窗口：当 layer N 被访问时调用
// 自动调整：mlock 进入窗口的层，munlock 离开窗口的层
// n_layer 在这里传递（首次调用时）
void window_advance(int layer_idx);

// ===== S5: Spec Decoding 协同接口 =====
// spec decode 的 verify 步骤会跑完整一次 target forward，访问所有 layers
// 这个接口让 spec decoder 能标记 "一次 verify step 开始/结束"，便于统计
// window advance 频率 vs autoregressive

// 标记一次 verify step 开始（清零 step counter）
void window_on_verify_step_begin();

// 标记一次 verify step 结束（记录 step advance 次数到 history）
void window_on_verify_step_end();

// 获取最近一次 verify step 触发的 window_advance 次数
int window_get_last_step_advance_count();

// 获取所有 verify step 的 advance 次数历史
const std::vector<int>& window_get_step_advance_history();

// 状态查询
const WindowState& window_get_state();

// 显式 mlock 某层（带 layer offset/size 范围，用于 Phase 3 真实层 mlock）
// 返回：mlock 成功的字节数
size_t window_mlock_layer(int layer_idx, void* base_ptr,
                          size_t layer_offset, size_t layer_size);

// 显式 munlock 某层
void window_munlock_layer(int layer_idx);

// 检查某层是否 mlocked
bool window_layer_is_mlocked(int layer_idx);

// 重置（清除所有 mlock）
void window_reset();

} // namespace fusion

// C-linkage bridge for llama.cpp internal calls
#ifdef __cplusplus
extern "C" {
#endif
void fusion_window_init_capture(int n_layers);
#ifdef __cplusplus
}
#endif