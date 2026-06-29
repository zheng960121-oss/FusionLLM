// FusionLLM S7 测试：DSpark Attention Forward
// 验证 forward() 能:
//  1. 编译并执行（不崩溃）
//  2. 在骨架模式下返回合理的 placeholder
//  3. 输入 shape 验证（target_hs + draft_input_ids + position_ids）

#include "fusion_draft_model.h"

#include <cstdio>
#include <vector>
#include <random>

static int passed = 0;
static int failed = 0;
#define EXPECT(cond, msg) do { \
    if (cond) { passed++; printf("  PASS: %s\n", msg); } \
    else { failed++; fprintf(stderr, "  FAIL: %s (line %d)\n", msg, __LINE__); } \
} while (0)

int main() {
    printf("=== FusionDSparkModel::forward() Test (S7) ===\n");

    // 1. 加载 metadata-only GGUF（之前生成的）
    fusion::FusionDSparkModel draft;
    if (!draft.load_from_gguf("/tmp/dspark_qwen3_4b.gguf")) {
        fprintf(stderr, "FAIL: load metadata-only GGUF\n");
        return 1;
    }
    EXPECT(draft.is_loaded(), "draft model loaded");
    EXPECT(draft.config().block_size == 7, "block_size = 7");
    EXPECT(draft.config().num_draft_layers == 5, "num_layers = 5");

    // 2. 验证 forward 在骨架模式（无实际权重）下不崩溃
    // 创建 dummy ggml context + tensors
    size_t ctx_size = 64 * 1024 * 1024;  // 64MB
    struct ggml_init_params iparams = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context* ctx = ggml_init(iparams);
    EXPECT(ctx != nullptr, "ggml_init OK");

    int ctx_len = 32;
    int n_embd = draft.config().n_embd;
    int concat_dim = draft.config().concat_dim();
    int block_size = draft.config().block_size;

    // target_hs: [concat_dim, ctx_len, 1] (placeholder)
    struct ggml_tensor* target_hs = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, concat_dim, ctx_len, 1);
    // draft_input_ids: [block_size, 1]
    struct ggml_tensor* draft_input_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, block_size, 1);
    // position_ids: [ctx_len + block_size, 1]
    struct ggml_tensor* position_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ctx_len + block_size);

    EXPECT(target_hs != nullptr, "target_hs allocated");
    EXPECT(draft_input_ids != nullptr, "draft_input_ids allocated");
    EXPECT(position_ids != nullptr, "position_ids allocated");

    EXPECT(target_hs->ne[0] == concat_dim, "target_hs ne[0] = concat_dim");
    EXPECT(target_hs->ne[1] == ctx_len, "target_hs ne[1] = ctx_len");
    EXPECT(draft_input_ids->ne[0] == block_size, "draft_input_ids ne[0] = block_size");

    // 3. forward 应该返回 nullptr（因为 fc weight 没加载 + embeddings 没共享）
    fprintf(stderr, "Calling forward() (expected to fail with placeholder message)...\n");
    struct ggml_tensor* out = draft.forward(ctx, target_hs, draft_input_ids, position_ids, nullptr);
    // 在骨架模式下（无 fc weight），forward 应该返回 nullptr
    EXPECT(out == nullptr, "forward returns nullptr in skeleton mode (no weights)");

    // 4. 测试：embeddings 共享但 fc 没设（仍然骨架）
    fprintf(stderr, "Setting dummy embeddings...\n");
    // 用 target_hs 同时作为 embed_tokens 和 lm_head 占位
    draft.share_target_embeddings(target_hs, target_hs);
    out = draft.forward(ctx, target_hs, draft_input_ids, position_ids, nullptr);
    // 仍然应该 nullptr（因为 fc = null）
    EXPECT(out == nullptr, "forward still nullptr (no fc weight)");

    ggml_free(ctx);

    printf("\n=== Summary: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
