// FusionLLM driver header (Phase 1 prototype)
// 目的：扩展 llama.cpp 内置 --mlock，添加 selective mlock 和路径 c 集成
//
// 当前状态：Phase 1 最小实现
//   - 包装 llama.cpp 的 mlock 机制
//   - 提供 selective mlock API（layer-level 锁/释放）
//   - 跟踪活跃窗口
//
// 后续阶段：
//   - Phase 2: 滑动窗口调度器
//   - Phase 3: KV Cache SSD 分层

#pragma once

#include <cstddef>
#include <cstdint>

namespace fusion {

// 初始化驱动（在 main 开始时调用一次）
// 检查环境变量 FUSION_DRIVER / FUSION_MLOCK_WINDOW 等
void driver_init();

// 关闭驱动（释放所有资源）
void driver_shutdown();

// 报告驱动是否启用
bool driver_enabled();

// Selective mlock: 只锁指定 layer 范围
// center_layer: 当前计算的 layer 索引
// window_size: 前后各锁多少层
// 返回：实际锁定的字节数
size_t driver_lock_window(int center_layer, int window_size);

// 滑动窗口：当 layer N 计算完成，调用此函数
// 自动调整锁范围：释放已完成的旧层，锁新的下一批
size_t driver_advance_to_layer(int next_layer);

// 统计：总 mlock 字节数
size_t driver_total_mlocked();

// 统计：当前锁定的层数
int driver_locked_layer_count();

} // namespace fusion
