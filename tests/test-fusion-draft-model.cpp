// FusionLLM S3 测试：Draft Model Loading
// 1. 加载 dspark metadata-only GGUF
// 2. 验证 config 解析正确（block_size, target_layer_ids, markov_rank 等）

#include "fusion_draft_model.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <dspark.gguf>\n", argv[0]);
        return 1;
    }

    printf("=== FusionDSparkModel Skeleton Test ===\n");

    fusion::FusionDSparkModel draft;
    if (!draft.load_from_gguf(argv[1])) {
        fprintf(stderr, "FAIL: load_from_gguf\n");
        return 1;
    }

    const auto& cfg = draft.config();
    printf("\nParsed config:\n");
    printf("  block_size         = %d\n", cfg.block_size);
    printf("  num_draft_layers   = %d\n", cfg.num_draft_layers);
    printf("  markov_rank        = %d\n", cfg.markov_rank);
    printf("  markov_head_type   = %s\n", cfg.markov_head_type.c_str());
    printf("  mask_token_id      = %d\n", cfg.mask_token_id);
    printf("  target_layer_ids   = [");
    for (size_t i = 0; i < cfg.target_layer_ids.size(); ++i) {
        printf("%s%d", i ? "," : "", cfg.target_layer_ids[i]);
    }
    printf("] (%zu layers)\n", cfg.target_layer_ids.size());
    printf("  target.n_embd      = %d\n", cfg.n_embd);
    printf("  target.n_head      = %d\n", cfg.n_head);
    printf("  target.n_head_kv   = %d\n", cfg.n_head_kv);
    printf("  target.n_vocab     = %d\n", cfg.n_vocab);
    printf("  target.head_dim    = %d\n", cfg.head_dim);
    printf("  concat_dim         = %d\n", cfg.concat_dim());

    // 验证关键字段
    int failed = 0;
    if (cfg.block_size != 7) {
        fprintf(stderr, "FAIL: block_size != 7\n"); failed++;
    }
    if (cfg.num_draft_layers != 5) {
        fprintf(stderr, "FAIL: num_draft_layers != 5\n"); failed++;
    }
    if (cfg.markov_rank != 256) {
        fprintf(stderr, "FAIL: markov_rank != 256\n"); failed++;
    }
    if (cfg.target_layer_ids.size() != 5 ||
        cfg.target_layer_ids[0] != 1 ||
        cfg.target_layer_ids[4] != 33) {
        fprintf(stderr, "FAIL: target_layer_ids mismatch\n"); failed++;
    }
    if (cfg.mask_token_id != 151669) {
        fprintf(stderr, "FAIL: mask_token_id != 151669\n"); failed++;
    }

    // 验证 forward 还能调（占位）
    printf("\nTesting forward() (skeleton, will return nullptr):\n");
    ggml_tensor* fake_target_hs = (ggml_tensor*)0x1234;
    ggml_tensor* fake_input = (ggml_tensor*)0x5678;
    ggml_tensor* fake_pos = (ggml_tensor*)0x9abc;
    auto* out = draft.forward(nullptr, fake_target_hs, fake_input, fake_pos, nullptr);
    if (out != nullptr) {
        fprintf(stderr, "FAIL: skeleton forward should return nullptr\n"); failed++;
    } else {
        printf("  PASS: skeleton forward returns nullptr as expected\n");
    }

    printf("\n=== Summary: %d failed ===\n", failed);
    return failed > 0 ? 1 : 0;
}
