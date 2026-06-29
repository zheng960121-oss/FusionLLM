// FusionLLM Markov Head — vanilla implementation (Phase 6 S8)
//
// Implements: corrected_logits = base_logits + W2 @ W1[token_ids]
//
// Shape contracts (all in ggml ne[] order, i.e. ne[0] is innermost/fastest):
//   base_logits:  ne = [V, S, B]            (PyTorch shape [B, S, V])
//   token_ids:    ne = [S, B]   (I32)       (PyTorch shape [B, S])
//   w1 (embed):   ne = [rank, V]            (PyTorch shape [V, rank])
//   w2 (linear):  ne = [rank, V]            (PyTorch shape [V, rank])
//   out:          ne = [V, S, B]            (PyTorch shape [B, S, V])
//
// Reference (PyTorch):
//   prev_emb = W1[token_ids]                # [B, S, rank]
//   bias     = prev_emb @ W2.T              # [B, S, V]
//   logits   = base_logits + bias
//
// Notes on ggml ops used:
//   ggml_get_rows(a, b):   a.ne[0] x b.ne[0] x b.ne[1] x b.ne[2] in ggml
//                          For our case: a=w1 ne=[rank,V], b=token_ids 1D ne=[B*S]
//                          -> result ne=[rank, B*S] in ggml = PyTorch [B*S, rank]
//   ggml_mul_mat(a, b):    result ne = a.ne[1] x b.ne[1] x b.ne[2] x b.ne[3]
//                          a=w2 ne=[rank,V], b=prev_emb ne=[rank, B*S]
//                          -> result ne=[V, B*S] in ggml = PyTorch [B*S, V]
//   ggml_reshape_3d       (re)interpret memory: from [V, B*S] to [V, B, S]
//                          (no data copy, just shape re-label)
//   ggml_permute(0,2,1,0) [V, B, S] -> [V, S, B]  (PyTorch [S,B,V] -> [B,S,V])
//   ggml_cont             make tensor contiguous in memory (data copy)
//   ggml_add              broadcast (matches base_logits ne)

#include "fusion_markov_head.h"

#include <cstdio>

struct ggml_tensor * fusion_markov_head_forward(
    struct ggml_context * ctx,
    struct ggml_tensor  * base_logits,
    struct ggml_tensor  * token_ids,
    struct ggml_tensor  * w1,
    struct ggml_tensor  * w2) {
    if (!ctx || !base_logits || !token_ids || !w1 || !w2) {
        fprintf(stderr, "[fusion_markov_head] null input\n");
        return nullptr;
    }

    const int n_dims = ggml_n_dims(base_logits);
    if (n_dims < 1 || n_dims > 3) {
        fprintf(stderr,
                "[fusion_markov_head] base_logits must be 1D, 2D or 3D; got n_dims=%d\n",
                n_dims);
        return nullptr;
    }

    // Vocab dim is the innermost (ne[0]).
    const int64_t V   = base_logits->ne[0];

    // Sequence dim(s).
    int64_t S = 0;
    int64_t B = 1;
    if (n_dims == 1) {
        // base_logits is effectively 1D ne=[V].  Treat as S=1, B=1.
        S = 1;
    } else if (n_dims == 2) {
        S = base_logits->ne[1];
    } else {
        S = base_logits->ne[1];
        B = base_logits->ne[2];
    }

    // w1 / w2 shape checks.  Both must be [rank, V] in ggml.
    if (w1->ne[1] != V) {
        fprintf(stderr, "[fusion_markov_head] w1 ne[1]=%lld != V=%lld\n",
                (long long) w1->ne[1], (long long) V);
        return nullptr;
    }
    if (w2->ne[1] != V) {
        fprintf(stderr, "[fusion_markov_head] w2 ne[1]=%lld != V=%lld\n",
                (long long) w2->ne[1], (long long) V);
        return nullptr;
    }
    const int64_t rank = w1->ne[0];
    if (w2->ne[0] != rank) {
        fprintf(stderr, "[fusion_markov_head] rank mismatch: w1.ne[0]=%lld, w2.ne[0]=%lld\n",
                (long long) w1->ne[0], (long long) w2->ne[0]);
        return nullptr;
    }

    // Token-id layout check.  Should be [S, B] (or [S] if B=1).
    if (token_ids->ne[0] != S) {
        fprintf(stderr,
                "[fusion_markov_head] token_ids ne[0]=%lld != S=%lld (must match base_logits S dim)\n",
                (long long) token_ids->ne[0], (long long) S);
        return nullptr;
    }
    if (B > 1 && token_ids->ne[1] != B) {
        fprintf(stderr,
                "[fusion_markov_head] token_ids ne[1]=%lld != B=%lld (must match base_logits B dim)\n",
                (long long) token_ids->ne[1], (long long) B);
        return nullptr;
    }

    // --- Step 1: prev_emb = W1[token_ids]   ggml ne = [rank, B*S] = PyTorch [B*S, rank] ---
    // ggml_get_rows requires token_ids 1D and a->ne[2] == b->ne[1] (both 1).
    // Easiest: reshape token_ids to 1D ne=[B*S].
    struct ggml_tensor * tok_1d = ggml_reshape_1d(ctx, token_ids, B * S);
    struct ggml_tensor * prev_emb = ggml_get_rows(ctx, w1, tok_1d);

    // --- Step 2: bias = W2 @ prev_emb^T     ggml ne = [V, B*S] = PyTorch [B*S, V] ---
    //   The (B*S) dim here is the "column" index of the result matrix.
    //   bias_2d[v, n] = sum_r w2[v, r] * prev_emb[r, n]
    //                 = sum_r w2[v, r] * w1[r, token_ids_PT[n/S, n%S]]
    //   where token_ids_PT[b, s] is the PyTorch-shaped token id (b = n/S, s = n%S).
    struct ggml_tensor * bias_2d = ggml_mul_mat(ctx, w2, prev_emb);

    // --- Step 3: reshape to 3D ne=[V, S, B] (PyTorch shape [B, S, V]) ---
    //   Reshape treats the (B*S) outer dim as a 2D grid of (S, B) in row-major
    //   (S inner, B outer), so flat index n = s + b*S.  This matches the PyTorch
    //   row-major order m = b*S + s of the (B, S) token grid.  Therefore:
    //     bias_3d[v, s, b] = bias_2d[v, n = s + b*S] = bias_PT[b, s, v]
    //   No data copy or permute needed; the reshape + ggml_add with base_logits
    //   (also ne=[V, S, B]) gives the corrected logits directly.
    struct ggml_tensor * bias = ggml_reshape_3d(ctx, bias_2d, V, S, B);

    // --- Step 4: add to base_logits (matching ne=[V, S, B]) ---
    if (n_dims == 3) {
        return ggml_add(ctx, base_logits, bias);
    } else if (n_dims == 2) {
        // base_logits ne=[V, S]
        struct ggml_tensor * bias_2d_out = ggml_reshape_2d(ctx, bias, V, S);
        return ggml_add(ctx, base_logits, bias_2d_out);
    } else {  // n_dims == 1; base_logits ne=[V]
        struct ggml_tensor * bias_1d_out = ggml_reshape_1d(ctx, bias, V);
        return ggml_add(ctx, base_logits, bias_1d_out);
    }
}
