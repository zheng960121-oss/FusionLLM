// PoC-5: Direct mlock on mmap'd GGUF file
// 目标：确认 llama.cpp 的 mmap + mlock 路径是否能真 mlock
// 不依赖 llama.cpp，只用 GGUF header + raw syscall

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "ggml.h"
#include "gguf.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];

    // 1. mmap 文件
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    fstat(fd, &st);
    size_t file_size = st.st_size;
    fprintf(stderr, "[PoC-5] file: %s (%zu bytes = %.1f MB)\n", path, file_size, file_size/1024.0/1024.0);

    void* base = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    fprintf(stderr, "[PoC-5] mmap base: %p, size: %zu\n", base, file_size);

    // 2. 解析 GGUF header
    gguf_init_params p = { true, NULL };
    gguf_context* ctx = gguf_init_from_file(path, p);
    if (!ctx) { fprintf(stderr, "gguf_init_from_file failed\n"); return 1; }

    int64_t n_tensors = gguf_get_n_tensors(ctx);
    fprintf(stderr, "[PoC-5] GGUF tensors: %lld\n", n_tensors);

    // 3. 找前几个 blk.0.xxx tensor，尝试 mlock
    int tried = 0;
    int success = 0;
    int failed = 0;
    int64_t total_mlocked = 0;
    for (int64_t i = 0; i < n_tensors && tried < 12; i++) {
        const char* name = gguf_get_tensor_name(ctx, i);
        if (!name) continue;
        // 只测 blk.0 (第一层)
        if (strncmp(name, "blk.0.", 6) != 0) continue;

        int64_t offset = gguf_get_tensor_offset(ctx, i);
        size_t   size   = gguf_get_tensor_size(ctx, i);
        if (size == 0) continue;

        void* ptr = (char*)base + offset;
        fprintf(stderr, "[PoC-5] mlock try: %s offset=%lld size=%zu ptr=%p\n",
                name, offset, size, ptr);

        // 触发缺页（先 touch 一次让 page-in）
        volatile char touch = *((char*)ptr);
        (void)touch;

        errno = 0;
        int ret = mlock(ptr, size);
        int e = errno;
        if (ret == 0) {
            fprintf(stderr, "  ✅ mlock OK (%zu bytes)\n", size);
            success++;
            total_mlocked += size;
        } else {
            fprintf(stderr, "  ❌ mlock FAILED: errno=%d (%s)\n", e, strerror(e));
            failed++;
        }
        tried++;
    }

    // 4. 统计
    fprintf(stderr, "\n[PoC-5] Summary: tried=%d success=%d failed=%d total_mlocked=%.1f MB\n",
            tried, success, failed, total_mlocked/1024.0/1024.0);

    // 5. 检查 RLIMIT_MEMLOCK
    struct rlimit rl;
    if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0) {
        fprintf(stderr, "[PoC-5] RLIMIT_MEMLOCK: soft=%llu hard=%llu (%.1f MB / %.1f MB)\n",
                (unsigned long long)rl.rlim_cur, (unsigned long long)rl.rlim_max,
                (double)rl.rlim_cur/1024/1024, (double)rl.rlim_max/1024/1024);
    } else {
        perror("getrlimit");
    }

    // 6. 检查 resident size 变化
    FILE* status = fopen("/proc/self/status", "r");
    if (status) {
        char line[256];
        while (fgets(line, sizeof(line), status)) {
            if (strncmp(line, "Locked:", 7) == 0) {
                fprintf(stderr, "[PoC-5] /proc/self/status: %s", line);
                break;
            }
        }
        fclose(status);
    } else {
        fprintf(stderr, "[PoC-5] /proc/self/status not available (likely macOS)\n");
    }

    // 7. 清理
    gguf_free(ctx);
    munmap(base, file_size);
    close(fd);
    return 0;
}
