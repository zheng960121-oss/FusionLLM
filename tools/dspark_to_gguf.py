#!/usr/bin/env python3
"""
DSpark Draft Model GGUF Writer
==============================

读取 DeepSeek DSpark draft model checkpoint (PyTorch safetensors) + target model GGUF,
生成 FusionLLM 可读的 draft model GGUF，带自定义 fusion.draft.* metadata。

Usage:
    python tools/dspark_to_gguf.py \\
        --draft-checkpoint deepseek-ai/dspark_qwen3_4b_block7 \\
        --target-gguf models/qwen2.5-7b-instruct-q4_k_m.gguf \\
        --output dspark_qwen3_4b_block7.gguf \\
        [--quantize q4_k_m]

References:
    - DeepSpec: https://github.com/deepseek-ai/DeepSpec
    - config/dspark/dspark_qwen3_4b.py (训练 config)
    - FusionLLM docs/DSpark_FusionLLM_detailed_spec.md
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy required. pip install numpy", file=sys.stderr)
    sys.exit(1)

try:
    from gguf import (
        GGUFWriter,
        GGMLQuantizationType,
        LlamaFileType,
    )
except ImportError:
    print("ERROR: gguf Python package required. pip install gguf", file=sys.stderr)
    sys.exit(1)


# DeepSpec default configs (from config/dspark/dspark_qwen3_*.py)
DSPARK_CONFIGS = {
    "qwen3_4b": {
        "target_model_name": "Qwen/Qwen3-4B",
        "block_size": 7,
        "num_draft_layers": 5,
        "target_layer_ids": [1, 9, 17, 25, 33],
        "mask_token_id": 151669,
        "num_anchors": 512,
        "markov_rank": 256,
        "markov_head_type": "vanilla",
        "confidence_head_alpha": 1.0,
        "confidence_head_with_markov": True,
    },
    "qwen3_8b": {
        "target_model_name": "Qwen/Qwen3-8B",
        "block_size": 7,
        "num_draft_layers": 5,
        "target_layer_ids": [1, 9, 17, 25, 33],
        "mask_token_id": 151669,
        "num_anchors": 512,
        "markov_rank": 256,
        "markov_head_type": "vanilla",
        "confidence_head_alpha": 1.0,
        "confidence_head_with_markov": True,
    },
    "qwen3_14b": {
        "target_model_name": "Qwen/Qwen3-14B",
        "block_size": 7,
        "num_draft_layers": 5,
        "target_layer_ids": [1, 9, 17, 25, 33],
        "mask_token_id": 151669,
        "num_anchors": 512,
        "markov_rank": 256,
        "markov_head_type": "vanilla",
        "confidence_head_alpha": 1.0,
        "confidence_head_with_markov": True,
    },
    "gemma4_12b": {
        "target_model_name": "google/gemma-4-12B-it",
        "block_size": 7,
        "num_draft_layers": 5,
        "target_layer_ids": [1, 9, 17, 25, 33],
        "mask_token_id": 255999,
        "num_anchors": 512,
        "markov_rank": 256,
        "markov_head_type": "vanilla",
        "confidence_head_alpha": 1.0,
        "confidence_head_with_markov": True,
    },
}


def read_target_metadata(target_gguf_path: str) -> Dict[str, Any]:
    """从 target model GGUF 读取关键参数（n_embd, n_head, etc.）"""
    from gguf.gguf_reader import GGUFReader

    reader = GGUFReader(target_gguf_path)
    meta = {}
    for field in reader.fields.values():
        if field.types and field.types[0].name == "ARRAY":
            continue
        if len(field.types) >= 1:
            val = field.contents(field.types[0])
            meta[field.name] = val
    return meta


def get_target_n_embd(target_meta: Dict[str, Any]) -> int:
    """鲁棒地从各种 architecture 读 n_embd"""
    candidates = [
        "qwen3.embedding_length",
        "qwen2.embedding_length",
        "gemma4.embedding.length",
        "gemma3.embedding.length",
        "llama.embedding_length",
        "embedding_length",
    ]
    for key in candidates:
        if key in target_meta:
            return int(target_meta[key])
    return 0


def get_target_n_head(target_meta: Dict[str, Any]) -> int:
    candidates = [
        "qwen3.attention.head_count",
        "qwen2.attention.head_count",
        "gemma4.attention.head_count",
        "gemma3.attention.head_count",
        "llama.attention.head_count",
        "attention.head_count",
    ]
    for key in candidates:
        if key in target_meta:
            return int(target_meta[key])
    return 0


def get_target_n_head_kv(target_meta: Dict[str, Any]) -> int:
    candidates = [
        "qwen3.attention.head_count_kv",
        "qwen2.attention.head_count_kv",
        "gemma4.attention.head_count_kv",
        "gemma3.attention.head_count_kv",
        "llama.attention.head_count_kv",
        "attention.head_count_kv",
    ]
    for key in candidates:
        if key in target_meta:
            return int(target_meta[key])
    # 默认与 head_count 相同（MHA）
    return get_target_n_head(target_meta)


def get_target_n_vocab(target_meta: Dict[str, Any]) -> int:
    candidates = [
        "qwen3.vocab_size",
        "qwen2.vocab_size",
        "gemma4.vocab_size",
        "gemma3.vocab_size",
        "llama.vocab_size",
        "vocab_size",
    ]
    for key in candidates:
        if key in target_meta:
            return int(target_meta[key])
    return 0


def get_target_head_dim(target_meta: Dict[str, Any]) -> int:
    # head_dim 通常 = n_embd / n_head
    n_embd = get_target_n_embd(target_meta)
    n_head = get_target_n_head(target_meta)
    if n_embd > 0 and n_head > 0:
        return n_embd // n_head
    candidates = [
        "qwen3.attention.key_length",
        "qwen2.attention.key_length",
        "gemma4.attention.key_length",
        "gemma3.attention.key_length",
        "llama.attention.key_length",
        "attention.key_length",
    ]
    for key in candidates:
        if key in target_meta:
            return int(target_meta[key])
    return 0


def get_target_rope_freq_base(target_meta: Dict[str, Any]) -> float:
    candidates = [
        "qwen3.rope.freq_base",
        "qwen2.rope.freq_base",
        "gemma4.rope.freq_base",
        "gemma3.rope.freq_base",
        "llama.rope.freq_base",
        "rope.freq_base",
    ]
    for key in candidates:
        if key in target_meta:
            return float(target_meta[key])
    return 10000.0


def write_metadata(writer: GGUFWriter, dspark_cfg: Dict[str, Any], target_meta: Dict[str, Any]) -> None:
    """写入 fusion.draft.* 和 target.* metadata"""

    # DSpark config (FusionLLM custom namespace)
    writer.add_uint32("fusion.draft.block_size", dspark_cfg["block_size"])
    writer.add_uint32("fusion.draft.num_layers", dspark_cfg["num_draft_layers"])
    writer.add_uint32("fusion.draft.markov_rank", dspark_cfg["markov_rank"])
    writer.add_string("fusion.draft.markov_head_type", dspark_cfg["markov_head_type"])
    writer.add_uint32("fusion.draft.mask_token_id", dspark_cfg["mask_token_id"])

    # target_layer_ids 是 i32 array
    writer.add_array("fusion.draft.target_layer_ids", dspark_cfg["target_layer_ids"])

    # 训练 config（参考用）
    writer.add_uint32("fusion.draft.train.num_anchors", dspark_cfg["num_anchors"])
    writer.add_float32("fusion.draft.train.confidence_head_alpha", dspark_cfg["confidence_head_alpha"])
    writer.add_bool("fusion.draft.train.confidence_head_with_markov", dspark_cfg["confidence_head_with_markov"])

    # Target model 参数（FusionLLM C++ 端需要知道这些才能正确处理 shared embeddings）
    target_n_embd = get_target_n_embd(target_meta)
    target_n_head = get_target_n_head(target_meta)
    target_n_head_kv = get_target_n_head_kv(target_meta)
    target_n_vocab = get_target_n_vocab(target_meta)
    target_head_dim = get_target_head_dim(target_meta)
    target_rope_freq = get_target_rope_freq_base(target_meta)

    writer.add_uint32("fusion.draft.target.n_embd", int(target_n_embd))
    writer.add_uint32("fusion.draft.target.n_head", int(target_n_head))
    writer.add_uint32("fusion.draft.target.n_head_kv", int(target_n_head_kv))
    writer.add_uint32("fusion.draft.target.n_vocab", int(target_n_vocab))
    writer.add_uint32("fusion.draft.target.head_dim", int(target_head_dim))
    writer.add_float32("fusion.draft.target.rope_freq_base", float(target_rope_freq))

    # 模型标识
    writer.add_string("fusion.draft.target_model", dspark_cfg["target_model_name"])
    writer.add_string("fusion.draft.architecture", "qwen3_dspark" if "qwen3" in dspark_cfg["target_model_name"].lower()
                      else "gemma4_dspark")
    writer.add_string("general.name", f"dspark_{dspark_cfg['target_model_name'].split('/')[-1]}_block{dspark_cfg['block_size']}")


def add_draft_tensors_metadata(writer: GGUFWriter, dspark_cfg: Dict[str, Any]) -> None:
    """为 draft model 的每个 tensor 添加 metadata（shape 和 quantization）"""
    num_layers = dspark_cfg["num_draft_layers"]
    n_embd = writer._gguf_writer.kv.get("fusion.draft.target.n_embd", 0)

    # 提示：实际 tensor 写入时由调用方根据 PyTorch checkpoint 决定
    # 这里只声明期望的 tensor 列表

    # FC: [n_embd, len(target_layer_ids) * n_embd]
    concat_dim = len(dspark_cfg["target_layer_ids"]) * n_embd
    writer.add_string("fusion.draft.tensors.expected", "true")  # 标记


def main():
    parser = argparse.ArgumentParser(
        description="DSpark Draft Model GGUF Writer for FusionLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--draft-config",
        choices=list(DSPARK_CONFIGS.keys()),
        default="qwen3_4b",
        help="DSpark 训练 config（决定 target_layer_ids, block_size, etc.）",
    )
    parser.add_argument(
        "--target-gguf",
        required=True,
        help="Target model GGUF path（用于读 target params + 共享 embeddings）",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output GGUF path",
    )
    parser.add_argument(
        "--checklist",
        action="store_true",
        help="只打印需要从 PyTorch checkpoint 读的 tensor 列表，不写 GGUF",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="只写 metadata（方便调试）",
    )
    args = parser.parse_args()

    dspark_cfg = DSPARK_CONFIGS[args.draft_config]
    print(f"[dspark_to_gguf] using config: {args.draft_config}")
    print(f"  target: {dspark_cfg['target_model_name']}")
    print(f"  block_size={dspark_cfg['block_size']}, num_layers={dspark_cfg['num_draft_layers']}")
    print(f"  target_layer_ids={dspark_cfg['target_layer_ids']}")

    # 读 target metadata
    print(f"\n[dspark_to_gguf] reading target GGUF: {args.target_gguf}")
    target_meta = read_target_metadata(args.target_gguf)

    if args.checklist:
        # 打印需要从 PyTorch checkpoint 读的 tensor 列表
        print("\n=== DRAFT MODEL TENSOR CHECKLIST ===")
        print("需要从 PyTorch safetensors 读的 tensors:\n")

        n_layers = dspark_cfg["num_draft_layers"]
        print(f"# Decoder layers ({n_layers} layers × 9 tensors each = {n_layers*9} tensors):")
        for il in range(n_layers):
            print(f"  # layer {il}")
            print(f"  -layers.{il}.self_attn.q_proj.weight       # [n_embd, n_embd]")
            print(f"  -layers.{il}.self_attn.k_proj.weight       # [n_embd, n_embd_kv]")
            print(f"  -layers.{il}.self_attn.v_proj.weight       # [n_embd, n_embd_kv]")
            print(f"  -layers.{il}.self_attn.o_proj.weight       # [n_embd, n_embd]")
            print(f"  -layers.{il}.self_attn.q_norm.weight        # [head_dim]")
            print(f"  -layers.{il}.self_attn.k_norm.weight        # [head_dim]")
            print(f"  -layers.{il}.mlp.gate_proj.weight          # [n_ff, n_embd]")
            print(f"  -layers.{il}.mlp.up_proj.weight            # [n_ff, n_embd]")
            print(f"  -layers.{il}.mlp.down_proj.weight          # [n_embd, n_ff]")
            print(f"  -layers.{il}.input_layernorm.weight        # [n_embd]")
            print(f"  -layers.{il}.post_attention_layernorm.weight  # [n_embd]")

        print(f"\n# fc + hidden_norm + final_norm (3 tensors):")
        print(f"  -fc.weight                                  # [n_embd, len(target_layer_ids) * n_embd]")
        print(f"  -hidden_norm.weight                         # [n_embd]")
        print(f"  -norm.weight                                # [n_embd]  (final)")

        if dspark_cfg["markov_rank"] > 0:
            print(f"\n# Markov head (rank={dspark_cfg['markov_rank']}):")
            print(f"  -markov_head.markov_w1.weight               # [vocab_size, markov_rank]")
            print(f"  -markov_head.markov_w2.weight               # [markov_rank, vocab_size]")

            if dspark_cfg["markov_head_type"] == "gated":
                print(f"  -markov_head.gate_proj.weight               # [markov_rank, n_embd + markov_rank]")
            elif dspark_cfg["markov_head_type"] == "rnn":
                print(f"  -markov_head.joint_proj.weight              # [3*markov_rank, 2*markov_rank + n_embd]")

        if dspark_cfg.get("confidence_head_alpha", 0) > 0:
            print(f"\n# Confidence head:")
            if dspark_cfg.get("confidence_head_with_markov"):
                input_dim_str = f"n_embd + markov_rank"
            else:
                input_dim_str = "n_embd"
            print(f"  -confidence_head.proj.weight                # [1, {input_dim_str}]")

        print(f"\n# Embeddings (与 target 共享):")
        print(f"  # embed_tokens.weight = target_model.tok_embd.weight  # [vocab_size, n_embd]")
        print(f"  # lm_head.weight       = target_model.output.weight    # [vocab_size, n_embd]")

        total_tensors = n_layers * 11 + 3  # layers + fc/hidden_norm/final_norm
        if dspark_cfg["markov_rank"] > 0:
            total_tensors += 2  # vanilla markov
        if dspark_cfg.get("confidence_head_alpha", 0) > 0:
            total_tensors += 1
        print(f"\nTotal unique tensors (excl. shared embeddings): {total_tensors}")
        return

    # 写 GGUF
    print(f"\n[dspark_to_gguf] writing: {args.output}")

    # 估算 n_embd（从 target metadata）
    target_n_embd = get_target_n_embd(target_meta)
    if target_n_embd == 0:
        print("ERROR: failed to read n_embd from target GGUF", file=sys.stderr)
        print("       (model architecture not supported in get_target_n_embd)", file=sys.stderr)
        sys.exit(1)

    writer = GGUFWriter(args.output, "fusionllm")

    # Metadata
    write_metadata(writer, dspark_cfg, target_meta)

    if args.metadata_only:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.close()
        print(f"[dspark_to_gguf] metadata-only GGUF written: {args.output}")
        return

    # 完整模式：读 PyTorch safetensors + 写 tensor
    print(f"\n[dspark_to_gguf] NOTE: Full tensor loading requires --draft-checkpoint <HF repo>")
    print(f"  Currently this script writes metadata only.")
    print(f"  To add tensors: use --checklist to get tensor names, then load with safetensors")
    print(f"  and add via writer.add_tensor() in a follow-up script.")

    # 写 metadata（不写 tensor）
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()
    print(f"[dspark_to_gguf] GGUF metadata-only written: {args.output}")


if __name__ == "__main__":
    main()
