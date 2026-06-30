// FusionLLM Phase 3 W1 测试: KV Tier Manager
// 设计文档: docs/phase3_w1_kv_tier_design.md
//
// 测试 7 个 case / ~30 个 EXPECT:
//   1. Constructor + block table
//   2. promote L0/L1/L2 路径
//   3. demote L0→L1→L2 路径
//   4. L2→L1 roundtrip 数据一致性 (PoC-4 升级版)
//   5. Multi-layer 多 block 并发
//   6. Stats 统计
//   7. Large KV (32K 8B, 50MB) 完整 offload

#include "fusion_kv_tier.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "ggml.h"
#include "ggml-backend.h"

static int g_passed = 0;
static int g_failed = 0;
#define EXPECT(cond, msg) do { \
    if (cond) { g_passed++; printf("  PASS: %s\n", msg); } \
    else      { g_failed++; fprintf(stderr, "  FAIL: %s (line %d)\n", msg, __LINE__); } \
} while (0)

// 写一个简单的 KV pattern 到 buffer (便于验证 roundtrip)
static void write_pattern(float* buf, int n_elements, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 0.1f);
    for (int i = 0; i < n_elements; ++i) {
        buf[i] = nd(rng);
    }
}

static bool check_pattern(const float* buf, int n_elements, int seed, float tol = 1e-4f) {
    // 重新生成期望 pattern 比较
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 0.1f);
    for (int i = 0; i < n_elements; ++i) {
        float expected = nd(rng);
        if (std::fabs(buf[i] - expected) > tol) {
            fprintf(stderr, "    pattern mismatch at %d: got %.4f, expected %.4f\n",
                    i, buf[i], expected);
            return false;
        }
    }
    return true;
}

// ============================================================
// Test 1: Constructor + block table
// ============================================================
static void test_constructor() {
    printf("\n=== Test 1: Constructor + block table ===\n");
    const int n_layers = 4;
    const int kv_size = 1024;
    const int head_dim = 64;
    const int block_size = 256;

    size_t ctx_size = 64 * 1024 * 1024;
    ggml_init_params params = { ctx_size, nullptr, true };
    ggml_context* ctx = ggml_init(params);
    EXPECT(ctx != nullptr, "ggml_init OK");

    fusion::FusionKVTierManager mgr(ctx, n_layers, kv_size, head_dim, GGML_TYPE_F32, block_size);

    // 验证 block table
    auto s = mgr.get_stats();
    int n_blocks_expected = (kv_size + block_size - 1) / block_size;  // = 4
    printf("    expected %d blocks per layer × %d layers = %d blocks total\n",
           n_blocks_expected, n_layers, n_blocks_expected * n_layers);
    EXPECT(s.bytes_in_ssd > 0, "all blocks initially in L2 SSD");
    EXPECT(s.bytes_in_gpu == 0, "L0 GPU initially empty");
    EXPECT(s.bytes_in_cpu == 0, "L1 CPU initially empty");

    ggml_free(ctx);
}

// ============================================================
// Test 2: promote L0/L1/L2 路径
// ============================================================
static void test_promote_paths() {
    printf("\n=== Test 2: promote path ===\n");
    const int n_layers = 2;
    const int kv_size = 512;
    const int head_dim = 32;
    const int block_size = 128;

    ggml_init_params params = { 32 * 1024 * 1024, nullptr, true };
    ggml_context* ctx = ggml_init(params);

    fusion::FusionKVTierManager mgr(ctx, n_layers, kv_size, head_dim, GGML_TYPE_F32, block_size);

    // Initial state
    auto block = mgr.get(0, 0, 128);
    EXPECT(block.tier == fusion::KVTier::L2_SSD, "block initially in L2 SSD");

    // Promote 0-128 (1 block, layer 0) to L1
    EXPECT(mgr.promote_to_cpu(0, 0, 128), "promote L2→L1 success");
    block = mgr.get(0, 0, 128);
    EXPECT(block.tier == fusion::KVTier::L1_CPU, "block now in L1");

    // Promote 0-128 (same block) to L0
    EXPECT(mgr.promote_to_gpu(0, 0, 128), "promote L1→L0 success");
    block = mgr.get(0, 0, 128);
    EXPECT(block.tier == fusion::KVTier::L0_GPU, "block now in L0 GPU");

    // Promote multi-block range (0-256 = 2 blocks)
    EXPECT(mgr.promote_to_cpu(0, 0, 256), "promote multi-block L2→L1");
    EXPECT(mgr.get(0, 0, 128).tier == fusion::KVTier::L1_CPU, "block 0 in L1");
    EXPECT(mgr.get(0, 128, 256).tier == fusion::KVTier::L1_CPU, "block 1 in L1");

    ggml_free(ctx);
}

// ============================================================
// Test 3: demote 路径
// ============================================================
static void test_demote_paths() {
    printf("\n=== Test 3: demote path ===\n");
    const int n_layers = 1;
    const int kv_size = 256;
    const int head_dim = 16;
    const int block_size = 64;

    ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context* ctx = ggml_init(params);
    fusion::FusionKVTierManager mgr(ctx, n_layers, kv_size, head_dim, GGML_TYPE_F32, block_size);

    // 先 promote 到 L0
    EXPECT(mgr.promote_to_gpu(0, 0, 256), "promote all to L0");
    EXPECT(mgr.get(0, 0, 64).tier == fusion::KVTier::L0_GPU, "block 0 in L0");

    // Demote 到 L2
    EXPECT(mgr.demote_to_ssd(0, 0, 256), "demote all to L2");
    EXPECT(mgr.get(0, 0, 64).tier == fusion::KVTier::L2_SSD, "block 0 back to L2");

    // Demote partial range
    EXPECT(mgr.promote_to_gpu(0, 0, 128), "promote block 0 to L0");
    EXPECT(mgr.promote_to_gpu(0, 128, 256), "promote block 1 to L0");
    EXPECT(mgr.demote_to_ssd(0, 0, 128), "demote block 0 only");
    EXPECT(mgr.get(0, 0, 64).tier == fusion::KVTier::L2_SSD, "block 0 in L2");
    EXPECT(mgr.get(0, 128, 192).tier == fusion::KVTier::L0_GPU, "block 1 still in L0");

    ggml_free(ctx);
}

// ============================================================
// Test 4: L2→L1 roundtrip 数据一致性 (PoC-4 升级版)
// ============================================================
static void test_ssd_roundtrip() {
    printf("\n=== Test 4: L2→L1 SSD roundtrip ===\n");
    const int n_layers = 1;
    const int kv_size = 256;
    const int head_dim = 16;
    const int block_size = 64;
    const int seed = 42;

    ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context* ctx = ggml_init(params);
    fusion::FusionKVTierManager mgr(ctx, n_layers, kv_size, head_dim, GGML_TYPE_F32, block_size);

    // Setup SSD path (use /tmp)
    char ssd_path[256];
    snprintf(ssd_path, sizeof(ssd_path), "/tmp/fusion_kv_tier_test_%d", getpid());
    mgr.set_ssd_path(ssd_path);

    // Promote to L1 first so .ptr is non-null (initial state is L2 with nullptr)
    EXPECT(mgr.promote_to_cpu(0, 0, 64), "promote L2→L1 first");

    // 写一个 pattern 到 L1 buffer
    int n_elements = block_size * head_dim;  // 64 * 16 = 1024 floats per block
    write_pattern((float*)mgr.get(0, 0, 64).ptr, n_elements, seed);

    // 模拟 flush 到 SSD
    EXPECT(mgr.flush_to_ssd(0, 0, 64), "flush to SSD");
    EXPECT(mgr.demote_to_ssd(0, 0, 64), "mark as L2");

    // 覆盖 L1 buffer (清零)
    memset(mgr.get(0, 0, 64).ptr, 0, n_elements * sizeof(float));

    // Promote 回 L1
    EXPECT(mgr.promote_to_cpu(0, 0, 64), "promote L2→L1");
    // W1 D3 real impl: read_ssd_block does pread() from the SSD file we
    // pwrite'd in flush_to_ssd.  Roundtrip should preserve the pattern.
    EXPECT(mgr.get(0, 0, 64).ptr != nullptr, "L1 ptr valid after promote");
    EXPECT(check_pattern((float*)mgr.get(0, 0, 64).ptr, n_elements, seed),
           "roundtrip data preserved (real SSD read)");

    // Clean up
    char rm_cmd[300];
    snprintf(rm_cmd, sizeof(rm_cmd), "rm -rf %s", ssd_path);
    (void)!system(rm_cmd);

    ggml_free(ctx);
}

// ============================================================
// Test 4b: Real SSD file layout (W1 D3 — verify on-disk file)
// ============================================================
// Exercises the actual on-disk SSD files (created by set_ssd_path) and
// verifies: (1) file is created with the correct size, (2) data written via
// the manager can be read back via plain pread, (3) mlock path doesn't crash
// even when called many times.  This is the bridge to W1 D4 llama.cpp
// integration: the file format we use here will be what attention hooks read.
static void test_real_ssd_files() {
    printf("\n=== Test 4b: Real SSD file layout (W1 D3) ===\n");
    const int n_layers = 2;
    const int kv_size = 256;
    const int head_dim = 32;
    const int block_size = 128;  // 128 * 32 * 4 (F32) = 16 KB per block

    ggml_init_params params = { 32 * 1024 * 1024, nullptr, true };
    ggml_context* ctx = ggml_init(params);
    fusion::FusionKVTierManager mgr(ctx, n_layers, kv_size, head_dim, GGML_TYPE_F32, block_size);

    char ssd_path[256];
    snprintf(ssd_path, sizeof(ssd_path), "/tmp/fusion_kv_tier_disk_%d", getpid());
    // Clean any prior state so ftruncate path is exercised
    char rm_cmd[300];
    snprintf(rm_cmd, sizeof(rm_cmd), "rm -rf %s", ssd_path);
    (void)!system(rm_cmd);

    mgr.set_ssd_path(ssd_path);

    // (1) Verify SSD files exist with correct size on disk
    // 2 layers × 2 blocks (256/128) × 16 KB = 32 KB data per layer
    // + 4 KB header each = 36 KB per file
    int n_blocks = (kv_size + block_size - 1) / block_size;
    size_t expected_data = (size_t)n_blocks * block_size * head_dim * 4;  // F32
    size_t expected_file = 4 * 1024 + expected_data;

    for (int il = 0; il < n_layers; ++il) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/layer_%03d.bin", ssd_path, il);
        struct stat st;
        EXPECT(stat(path, &st) == 0, "SSD file exists on disk");
        EXPECT((size_t)st.st_size == expected_file, "SSD file has expected size");
        if (st.st_size != (off_t)expected_file) {
            fprintf(stderr, "    expected=%zu, got=%lld\n", expected_file, (long long)st.st_size);
        }
    }

    // (2) Write a pattern via the manager, then read raw from disk
    EXPECT(mgr.promote_to_cpu(0, 0, kv_size), "promote layer 0 to L1");
    int n_elements = block_size * head_dim;
    std::mt19937 rng(123);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    float* l1_ptr = (float*)mgr.get(0, 0, block_size).ptr;
    for (int i = 0; i < n_elements; ++i) l1_ptr[i] = nd(rng);
    EXPECT(mgr.flush_to_ssd(0, 0, block_size), "flush to SSD");

    // Open the SSD file externally and pread — verify the data we wrote
    char path[1024];
    snprintf(path, sizeof(path), "%s/layer_%03d.bin", ssd_path, 0);
    int fd = open(path, O_RDONLY);
    EXPECT(fd >= 0, "external open of SSD file");
    if (fd >= 0) {
        std::vector<float> external(n_elements);
        // File offset: 4 KB header + block 0
        ssize_t got = pread(fd, external.data(), n_elements * sizeof(float), 4 * 1024);
        EXPECT((size_t)got == n_elements * sizeof(float), "external pread got full block");
        // Re-generate expected and compare
        std::mt19937 rng2(123);
        std::normal_distribution<float> nd2(0.0f, 1.0f);
        bool matches = true;
        for (int i = 0; i < n_elements; ++i) {
            float expected = nd2(rng2);
            if (std::fabs(external[i] - expected) > 1e-6f) { matches = false; break; }
        }
        EXPECT(matches, "external pread data matches what manager wrote");
        close(fd);
    }

    // (3) Multiple promote/demote cycle exercises mlock path
    for (int cycle = 0; cycle < 5; ++cycle) {
        EXPECT(mgr.demote_to_ssd(0, 0, block_size), "cycle: demote");
        // Overwrite L1 to verify next read actually pulls from SSD
        memset(mgr.get(0, 0, block_size).ptr, 0xAB, n_elements * sizeof(float));
        EXPECT(mgr.promote_to_cpu(0, 0, block_size), "cycle: promote");
        // After promote, the data we wrote should be back (not 0xAB)
        std::mt19937 rng3(123);
        std::normal_distribution<float> nd3(0.0f, 1.0f);
        bool cycle_ok = true;
        float* rb = (float*)mgr.get(0, 0, block_size).ptr;
        for (int i = 0; i < n_elements; ++i) {
            float expected = nd3(rng3);
            if (std::fabs(rb[i] - expected) > 1e-6f) { cycle_ok = false; break; }
        }
        EXPECT(cycle_ok, "5x promote/demote preserves data");
    }

    // Clean up
    snprintf(rm_cmd, sizeof(rm_cmd), "rm -rf %s", ssd_path);
    (void)!system(rm_cmd);
    ggml_free(ctx);
}

// ============================================================
// Test 5: Multi-layer 多 block
// ============================================================
static void test_multi_layer() {
    printf("\n=== Test 5: Multi-layer multi-block ===\n");
    const int n_layers = 8;
    const int kv_size = 1024;
    const int head_dim = 32;
    const int block_size = 256;
    const int n_blocks = (kv_size + block_size - 1) / block_size;  // = 4

    ggml_init_params params = { 64 * 1024 * 1024, nullptr, true };
    ggml_context* ctx = ggml_init(params);
    fusion::FusionKVTierManager mgr(ctx, n_layers, kv_size, head_dim, GGML_TYPE_F32, block_size);

    // 每个 layer 的每个 block 都 promote 到 L1
    for (int il = 0; il < n_layers; ++il) {
        for (int b = 0; b < n_blocks; ++b) {
            int start = b * block_size;
            int end = std::min(start + block_size, kv_size);
            EXPECT(mgr.promote_to_cpu(il, start, end), "promote block to L1");
        }
    }

    // 验证全部在 L1
    for (int il = 0; il < n_layers; ++il) {
        for (int b = 0; b < n_blocks; ++b) {
            EXPECT(mgr.is_in_cpu(il, b * block_size, std::min((b + 1) * block_size, kv_size)),
                   "all blocks in L1");
        }
    }

    // 只 demote layer 4 的 block 2
    int il_target = 4;
    int b_target = 2;
    int start = b_target * block_size;
    int end = std::min(start + block_size, kv_size);
    mgr.demote_to_ssd(il_target, start, end);
    EXPECT(mgr.is_in_ssd(il_target, start, end), "target block in L2");
    EXPECT(mgr.is_in_cpu((il_target + 1) % n_layers, start, end),
           "other layer same block still in L1");

    ggml_free(ctx);
}

// ============================================================
// ============================================================
// Test 5b: L0 path + ensure_for_attention (W1 D4)
// ============================================================
// Verifies the W1 D4 GPU staging path: ensure_l0_buffers actually allocates
// ggml tensors, do_promote_to_gpu copies L1 data into the L0 tensor buffer,
// and ensure_for_attention returns a correctly-shaped 2D view that the
// llama.cpp integration can use as the K (or V) input to attention.
static void test_l0_ensure_for_attention() {
    printf("\n=== Test 5b: L0 path + ensure_for_attention (W1 D4) ===\n");
    const int n_layers = 2;
    const int kv_size = 256;
    const int head_dim = 16;
    const int block_size = 64;

    // Use a real (alloc'ing) ggml context so the L0 tensors actually have
    // a data buffer for memcpy.  The kv_tier manager writes to it.
    size_t ctx_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 8
                    + 64 * 1024 * 1024;  // 64 MB for L0 staging
    ggml_init_params params = { ctx_size, nullptr, false /* no_alloc=false */ };
    ggml_context* ctx = ggml_init(params);
    EXPECT(ctx != nullptr, "ggml_init OK (alloc mode)");

    fusion::FusionKVTierManager mgr(ctx, n_layers, kv_size, head_dim, GGML_TYPE_F32, block_size);

    // (1) L0 buffers will be allocated lazily by promote_to_gpu
    auto s0 = mgr.get_stats();
    EXPECT(s0.bytes_in_gpu == 0, "L0 starts empty");

    // (2) Write a known pattern to L1, promote to L0, verify the L0 tensor
    // data matches what we wrote.
    int n_elements_per_block = block_size * head_dim;
    std::mt19937 rng(7777);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    // Promote block 0 of layer 0 first (L2→L1), then we can write into L1
    EXPECT(mgr.promote_to_cpu(0, 0, block_size), "promote L2→L1 first");
    float* l1_ptr = (float*)mgr.get(0, 0, block_size).ptr;
    EXPECT(l1_ptr != nullptr, "L1 ptr valid");
    for (int i = 0; i < n_elements_per_block; ++i) l1_ptr[i] = nd(rng);

    // Now promote to L0 (L1→L0 should copy L1 data into the L0 tensor)
    EXPECT(mgr.promote_to_gpu(0, 0, block_size), "promote L1→L0");

    // (3) Call ensure_for_attention and check the returned view
    ggml_tensor* k_view = mgr.ensure_for_attention(0, 0, block_size, false /* K */);
    EXPECT(k_view != nullptr, "ensure_for_attention returned non-null K view");
    if (k_view) {
        EXPECT(k_view->ne[0] == head_dim, "K view ne[0] = head_dim");
        EXPECT(k_view->ne[1] == block_size, "K view ne[1] = block_size");
        EXPECT(k_view->type == GGML_TYPE_F32, "K view type = F32");
        // Read back data and compare
        const float* view_data = (const float*)k_view->data;
        EXPECT(view_data != nullptr, "K view has data");
        if (view_data) {
            std::mt19937 rng2(7777);
            std::normal_distribution<float> nd2(0.0f, 2.0f);
            bool data_matches = true;
            for (int i = 0; i < n_elements_per_block; ++i) {
                float expected = nd2(rng2);
                if (std::fabs(view_data[i] - expected) > 1e-5f) {
                    data_matches = false; break;
                }
            }
            EXPECT(data_matches, "K view data matches L1 source (L1→L0 copy works)");
        }
    }

    // (4) V view test
    ggml_tensor* v_view = mgr.ensure_for_attention(0, 0, block_size, true /* V */);
    EXPECT(v_view != nullptr, "ensure_for_attention returned non-null V view");
    if (v_view) {
        EXPECT(v_view->ne[0] == head_dim, "V view ne[0] = head_dim");
        EXPECT(v_view->ne[1] == block_size, "V view ne[1] = block_size");
    }

    // (5) Ensure for attention with different range returns correct slice
    ggml_tensor* k_view2 = mgr.ensure_for_attention(1, 0, block_size, false);
    EXPECT(k_view2 != nullptr, "K view for layer 1");
    if (k_view2) {
        EXPECT(k_view2->ne[1] == block_size, "layer-1 K view ne[1] = block_size");
        // L1 was zero (never written for layer 1), so L0 should also be zero
        if (k_view2->data) {
            const float* d = (const float*)k_view2->data;
            int zero_count = 0;
            for (int i = 0; i < n_elements_per_block; ++i) if (d[i] == 0.0f) zero_count++;
            EXPECT(zero_count == n_elements_per_block, "layer-1 K view zero (no source data)");
        }
    }

    // (6) Invalid range should return nullptr
    ggml_tensor* bad = mgr.ensure_for_attention(0, -1, 0, false);
    EXPECT(bad == nullptr, "invalid range returns nullptr");
    bad = mgr.ensure_for_attention(-1, 0, 1, false);
    EXPECT(bad == nullptr, "invalid layer returns nullptr");

    ggml_free(ctx);
}

// Test 6: Stats 统计
// ============================================================
static void test_stats() {
    printf("\n=== Test 6: Stats 统计 ===\n");
    const int n_layers = 1;
    const int kv_size = 512;
    const int head_dim = 16;
    const int block_size = 128;
    const int n_blocks = (kv_size + block_size - 1) / block_size;  // = 4

    ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context* ctx = ggml_init(params);
    fusion::FusionKVTierManager mgr(ctx, n_layers, kv_size, head_dim, GGML_TYPE_F32, block_size);

    auto s = mgr.get_stats();
    printf("    initial: L0=%zu L1=%zu L2=%zu promotions=%zu evictions=%zu\n",
           s.bytes_in_gpu, s.bytes_in_cpu, s.bytes_in_ssd, s.n_promotions, s.n_evictions);
    EXPECT(s.bytes_in_ssd > 0, "initial L2 SSD bytes > 0");
    EXPECT(s.bytes_in_gpu == 0 && s.bytes_in_cpu == 0, "initial L0/L1 empty");
    EXPECT(s.n_promotions == 0 && s.n_evictions == 0, "initial counts zero");

    // 一些 promotion
    mgr.promote_to_cpu(0, 0, 256);  // 2 blocks L2→L1
    mgr.promote_to_gpu(0, 0, 128);   // 1 block L1→L0
    s = mgr.get_stats();
    printf("    after promotes: L0=%zu L1=%zu L2=%zu promotions=%zu\n",
           s.bytes_in_gpu, s.bytes_in_cpu, s.bytes_in_ssd, s.n_promotions);
    EXPECT(s.n_promotions == 2, "n_promotions == 2 (2 from L2 to L1)");
    EXPECT(s.n_gpu_copies == 1, "n_gpu_copies == 1");
    EXPECT(s.bytes_in_gpu > 0, "L0 has data");
    EXPECT(s.bytes_in_cpu > 0, "L1 has data");

    // demote
    mgr.demote_to_ssd(0, 0, 256);
    s = mgr.get_stats();
    EXPECT(s.n_evictions >= 1, "eviction count incremented");
    EXPECT(mgr.is_in_ssd(0, 0, 128), "demoted block 0 in L2");

    ggml_free(ctx);
}

// ============================================================
// Test 7: Large KV 完整 offload (8B 32K 模拟)
// ============================================================
static void test_large_kv() {
    printf("\n=== Test 7: Large KV 32K context (8B 模拟) ===\n");
    // 8B Q4_K_M 32K context KV size:
    //   36 layers × 32768 tokens × 8 KV heads × 128 head_dim × 2 (K+V) × 0.5 bytes (Q4) = 1.2 GB
    // 简化: 用 36 layers × 1024 tokens × 8 heads × 128 head_dim × 2 bytes (F16) = 144 MB
    const int n_layers = 36;
    const int kv_size = 1024;
    const int head_dim = 8 * 128;  // GQA: 8 heads × 128 head_dim
    const int block_size = 512;

    ggml_init_params params = { 256 * 1024 * 1024, nullptr, true };
    ggml_context* ctx = ggml_init(params);
    fusion::FusionKVTierManager mgr(ctx, n_layers, kv_size, head_dim, GGML_TYPE_F16, block_size);

    char ssd_path[256];
    snprintf(ssd_path, sizeof(ssd_path), "/tmp/fusion_kv_tier_large_%d", getpid());
    mgr.set_ssd_path(ssd_path);

    // Flush all to SSD
    EXPECT(mgr.flush_all_to_ssd(), "flush all 36 layers to SSD");

    auto s = mgr.get_stats();
    printf("    total L2 SSD: %.2f MB\n", s.bytes_in_ssd / (1024.0 * 1024.0));
    // Skeleton test: 36 × 1024 × 1024 × 2 (F16) ≈ 72 MB.  Real impl with Q4_0
    // for 32K context 8B will be ~600 MB; threshold here is loosened for skeleton.
    EXPECT(s.bytes_in_ssd > 50 * 1024 * 1024, "L2 SSD > 50 MB (8B 模拟)");

    // Promote first 2 blocks (1024 tokens) to L0 (attention needs them)
    EXPECT(mgr.promote_to_gpu(0, 0, 1024), "promote first 2 blocks to L0");
    s = mgr.get_stats();
    printf("    after promote: L0=%.2f MB L2=%.2f MB\n",
           s.bytes_in_gpu / (1024.0 * 1024.0), s.bytes_in_ssd / (1024.0 * 1024.0));
    EXPECT(s.bytes_in_gpu > 0, "L0 has data after promote");

    // Demote the first block (sliding window)
    mgr.demote_to_ssd(0, 0, 512);
    s = mgr.get_stats();
    EXPECT(mgr.is_in_ssd(0, 0, 512), "demoted block in L2");

    // Clean up
    char rm_cmd[300];
    snprintf(rm_cmd, sizeof(rm_cmd), "rm -rf %s", ssd_path);
    (void)!system(rm_cmd);

    ggml_free(ctx);
}

// ============================================================
// Test 8: Sliding window integration
// ============================================================
static void test_sliding_window() {
    printf("\n=== Test 8: Sliding window integration ===\n");
    const int n_layers = 1;
    const int kv_size = 1024;
    const int head_dim = 32;
    const int block_size = 128;
    const int window_size = 512;

    ggml_init_params params = { 32 * 1024 * 1024, nullptr, true };
    ggml_context* ctx = ggml_init(params);
    fusion::FusionKVTierManager mgr(ctx, n_layers, kv_size, head_dim, GGML_TYPE_F32, block_size);
    mgr.set_sliding_window(window_size);

    // 初始: window = [0, 512)
    // 推进: window = [128, 640) → demote [0, 128), promote [512, 640)
    EXPECT(mgr.advance_window(0, 0, 512, 128, 640), "advance window");
    EXPECT(mgr.is_in_ssd(0, 0, 128), "evicted [0, 128) in L2");
    EXPECT(mgr.get(0, 512, 640).tier == fusion::KVTier::L0_GPU, "promoted [512, 640) in L0");

    ggml_free(ctx);
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    (void)argc; (void)argv;
    // Force unbuffered stdout so PASS lines show before any potential crash
    setvbuf(stdout, nullptr, _IONBF, 0);
    printf("=== FusionKVTierManager Tests (Phase 3 W1) ===\n");

    test_constructor();
    test_promote_paths();
    test_demote_paths();
    test_ssd_roundtrip();
    test_real_ssd_files();
    test_multi_layer();
    test_l0_ensure_for_attention();
    test_stats();
    test_large_kv();
    test_sliding_window();

    printf("\n=== Summary: %d passed, %d failed ===\n", g_passed, g_failed);
    return g_failed > 0 ? 1 : 0;
}