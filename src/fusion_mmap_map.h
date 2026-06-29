// FusionLLM Phase 2: Layer-to-memory mapping
//
// 建立 layer_idx → (mmap_base_ptr, offset, size) 映射
// 让 window_advance 可以做真实的 mlock/munlock

#pragma once

#include <cstdint>
#include <cstddef>

namespace fusion {

// LayerMapping 定义在 .cpp（避免重复定义）

// 从 llama.cpp 内部调用
// 在模型加载完成后调用，建立 layer → mmap 位置映射
// 必须在所有 mlock 操作之前调用
void mmap_map_build(int n_layers, void* mmap_base_ptr, size_t mmap_total_size);

// 从 llama_model + file path 直接填充（推荐使用）
void mmap_map_populate(void* model, const char* gguf_path);

// 查询
struct LayerMapping;
const LayerMapping* mmap_map_get(int layer_idx);
int  mmap_map_layer_count();

// 真实 mlock 一层（per-tensor ranges，避免 gap 失败）
size_t mmap_mlock_layer(int layer_idx);

// 真实 munlock 一层
void mmap_munlock_layer(int layer_idx);

// 清理所有 mlock 和映射
void mmap_map_cleanup();

} // namespace fusion

// C-linkage bridges
extern "C" {
void fusion_populate_from_gguf(void* gguf_ctx, void* mmap_base);
void fusion_mmap_map_build(int n_layers, void* base, size_t total);
void fusion_mmap_populate(void* model, const char* gguf_path);
int  fusion_mmap_map_get_layer_count();
size_t fusion_mmap_mlock_layer(int layer_idx);
void  fusion_mmap_munlock_layer(int layer_idx);
void  fusion_mmap_cleanup();
}