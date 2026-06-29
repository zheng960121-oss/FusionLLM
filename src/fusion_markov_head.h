// FusionLLM Markov Head (Phase 6 S8)
//
// Vanilla Markov head for DSpark speculative decoding.
// Mathematically equivalent to:
//   prev_emb = embed(token_ids)            // [B, S, rank]
//   bias     = proj(prev_emb)              // [B, S, V]
//   logits   = base_logits + bias
//
// "Vanilla" = the bias depends only on the previous token IDs (memoryless).
// Gated/RNN variants are out of scope for S8.
//
// API:
//   ggml_tensor * fusion_markov_head_forward(
//       ggml_context * ctx,         // graph builder context
//       ggml_tensor  * base_logits, // [B, num_blocks, block_size, V] (or any [B, S, V])
//       ggml_tensor  * token_ids,   // [B, S] (int32)
//       ggml_tensor  * w1,          // [V, rank] embedding table
//       ggml_tensor  * w2,          // [V, rank] linear weight (out=V, in=rank)
//       int32_t       vocab_size,
//       int32_t       markov_rank);
//
// The returned tensor has the same shape as `base_logits`.
//
// Notes on shape contract:
//   * base_logits: any shape [..., V] (last dim must be vocab_size)
//   * token_ids:   [B, S] where S = product of all dims in base_logits except last
//   * w1, w2:      both [V, rank] in ggml storage (Linear weight is [out, in] in ggml)
//
// This file is header-only declarations. See fusion_markov_head.cpp for impl.

#ifndef FUSION_MARKOV_HEAD_H
#define FUSION_MARKOV_HEAD_H

#include <cstdint>
#include <ggml.h>

#ifdef __cplusplus
extern "C" {
#endif

// Compute the corrected logits: base_logits + markov_bias.
// `base_logits` is treated as a [B, S, V] tensor (any leading dims allowed).
// `token_ids`   is [B, S] int32.
// `w1`          is [V, rank] (embedding table).
// `w2`          is [V, rank] (ggml Linear weight; effective operation: out = w2 @ prev_emb).
// Returns a tensor with the same shape & type as `base_logits`.
struct ggml_tensor * fusion_markov_head_forward(
    struct ggml_context * ctx,
    struct ggml_tensor  * base_logits,
    struct ggml_tensor  * token_ids,
    struct ggml_tensor  * w1,
    struct ggml_tensor  * w2);

#ifdef __cplusplus
}
#endif

#endif // FUSION_MARKOV_HEAD_H
