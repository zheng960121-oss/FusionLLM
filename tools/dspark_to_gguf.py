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
    # Extract token count from tokenizer.ggml.tokens ARRAY field.
    # The ReaderField.data is a list of indices into parts containing the actual
    # string data, so len(data) == vocab size.
    if "tokenizer.ggml.tokens" in reader.fields:
        meta["__token_count"] = len(reader.fields["tokenizer.ggml.tokens"].data)
    return meta


def get_target_token_count(target_meta: Dict[str, Any]) -> int:
    """从 target metadata 拿 vocab size（优先 _vocab_size key, 其次 token count, 最后 architecture 默认值）"""
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
            try:
                return int(target_meta[key])
            except (TypeError, ValueError):
                continue
    if "__token_count" in target_meta:
        return int(target_meta["__token_count"])
    arch = target_meta.get("general.architecture", "")
    defaults = {
        "qwen2": 151936,
        "qwen3": 151936,
        "llama": 128000,
        "gemma3": 262144,
        "gemma4": 256000,
    }
    return defaults.get(arch, 0)


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
    return get_target_token_count(target_meta)


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


# ============================================================================
# S9: Actual tensor loading and writing
# ============================================================================

def read_pytorch_checkpoint(checkpoint_path: str) -> Dict[str, "np.ndarray"]:
    """读 PyTorch safetensors checkpoint.

    支持:
    - 本地文件夹包含多个 .safetensors 文件（HF 格式）
    - 单个 .safetensors 文件
    - HF repo ID（如 "deepseek-ai/dspark_qwen3_4b_block7"）— 需要 huggingface_hub
    """
    from safetensors import safe_open

    path = Path(checkpoint_path)
    tensors: Dict[str, np.ndarray] = {}

    if path.is_dir():
        # HF-style checkpoint dir with model-00001-of-N.safetensors + index.json
        st_files = sorted(path.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(f"No .safetensors files in {checkpoint_path}")
        for st_file in st_files:
            with safe_open(str(st_file), framework="np") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
    elif path.is_file() and path.suffix == ".safetensors":
        with safe_open(str(path), framework="np") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    else:
        # 尝试 HF repo ID
        try:
            from huggingface_hub import snapshot_download
            print(f"[dspark_to_gguf] Downloading from HF: {checkpoint_path}")
            local = snapshot_download(
                repo_id=checkpoint_path,
                allow_patterns=["*.safetensors", "*.json"],
            )
            return read_pytorch_checkpoint(local)
        except ImportError:
            raise FileNotFoundError(
                f"{checkpoint_path} is not a local file/dir, and huggingface_hub not installed"
            )

    return tensors


def get_pytorch_tensor(
    state_dict: Dict[str, np.ndarray],
    dspark_key: str,
    candidates: list,
) -> np.ndarray:
    """鲁棒地从 state_dict 拿 tensor，支持多套命名风格。"""
    for key in candidates:
        if key in state_dict:
            t = state_dict[key]
            if dspark_key != key:
                print(f"  [map] {key} -> {dspark_key}  shape={t.shape} dtype={t.dtype}")
            return t
    raise KeyError(
        f"Tensor not found. Tried: {candidates}\n"
        f"Available keys (first 20): {list(state_dict.keys())[:20]}"
    )


def generate_mock_draft_checkpoint(
    output_dir: str,
    dspark_cfg: Dict[str, Any],
    target_meta: Dict[str, Any],
    seed: int = 42,
) -> None:
    """生成 mock DSpark checkpoint (用于本地测试不需 HF 下载).

    结构与真实 PyTorch state_dict 一致：
      embed_tokens, lm_head (shared with target)
      layers.{il}.self_attn.{q,k,v,o}_proj, {q,k}_norm
      layers.{il}.mlp.{gate,up,down}_proj
      layers.{il}.input_layernorm, post_attention_layernorm
      fc, hidden_norm, norm (final)
      markov_head.{markov_w1, markov_w2}
    """
    from safetensors.torch import save_file
    import torch

    target_n_embd = get_target_n_embd(target_meta)
    target_n_head = get_target_n_head(target_meta)
    target_n_vocab = get_target_n_vocab(target_meta)
    if target_n_embd == 0 or target_n_vocab == 0:
        raise ValueError(
            f"target_meta missing n_embd={target_n_embd} or n_vocab={target_n_vocab}"
        )

    head_dim = target_n_embd // target_n_head if target_n_head > 0 else 128
    n_ff = int(target_n_embd * 8 / 3)  # Qwen3 convention
    # Round to multiple of 256 for nicer shapes
    n_ff = ((n_ff + 255) // 256) * 256

    rank = dspark_cfg["markov_rank"]
    block_size = dspark_cfg["block_size"]
    num_layers = dspark_cfg["num_draft_layers"]
    target_layer_ids = dspark_cfg["target_layer_ids"]
    concat_dim = len(target_layer_ids) * target_n_embd

    print(f"[mock] target_n_embd={target_n_embd}, n_head={target_n_head}, n_vocab={target_n_vocab}")
    print(f"[mock] head_dim={head_dim}, n_ff={n_ff}, num_layers={num_layers}")
    print(f"[mock] markov_rank={rank}, concat_dim={concat_dim}")

    torch.manual_seed(seed)
    sd = {}

    # Shared embeddings (placeholder, will be overridden by target in real usage)
    sd["embed_tokens.weight"] = torch.zeros(target_n_vocab, target_n_embd, dtype=torch.float32)
    sd["lm_head.weight"] = torch.zeros(target_n_vocab, target_n_embd, dtype=torch.float32)

    # Decoder layers (Qwen3-style: q/k/v/o_proj + q_norm + k_norm + MLP + 2 norms)
    for il in range(num_layers):
        prefix = f"layers.{il}"
        sd[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(target_n_embd, target_n_embd, dtype=torch.float32) * 0.02
        sd[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(target_n_embd, target_n_embd, dtype=torch.float32) * 0.02
        sd[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(target_n_embd, target_n_embd, dtype=torch.float32) * 0.02
        sd[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(target_n_embd, target_n_embd, dtype=torch.float32) * 0.02
        sd[f"{prefix}.self_attn.q_norm.weight"] = torch.ones(head_dim, dtype=torch.float32)
        sd[f"{prefix}.self_attn.k_norm.weight"] = torch.ones(head_dim, dtype=torch.float32)
        sd[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(n_ff, target_n_embd, dtype=torch.float32) * 0.02
        sd[f"{prefix}.mlp.up_proj.weight"] = torch.randn(n_ff, target_n_embd, dtype=torch.float32) * 0.02
        sd[f"{prefix}.mlp.down_proj.weight"] = torch.randn(target_n_embd, n_ff, dtype=torch.float32) * 0.02
        sd[f"{prefix}.input_layernorm.weight"] = torch.ones(target_n_embd, dtype=torch.float32)
        sd[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(target_n_embd, dtype=torch.float32)

    # FC + norms
    sd["fc.weight"] = torch.randn(target_n_embd, concat_dim, dtype=torch.float32) * 0.02
    sd["hidden_norm.weight"] = torch.ones(target_n_embd, dtype=torch.float32)
    sd["norm.weight"] = torch.ones(target_n_embd, dtype=torch.float32)

    # Markov head
    if rank > 0:
        sd["markov_head.markov_w1.weight"] = torch.randn(target_n_vocab, rank, dtype=torch.float32) * 0.02
        sd["markov_head.markov_w2.weight"] = torch.randn(rank, target_n_vocab, dtype=torch.float32) * 0.02

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out_path / "model.safetensors"))
    print(f"[mock] wrote {len(sd)} tensors to {out_path / 'model.safetensors'}")
    print(f"[mock] total file size: {(out_path / 'model.safetensors').stat().st_size / 1024 / 1024:.1f} MB")


def add_draft_tensors_from_state_dict(
    writer: GGUFWriter,
    state_dict: Dict[str, np.ndarray],
    dspark_cfg: Dict[str, Any],
    target_meta: Dict[str, Any],
    quantize: str = None,
) -> None:
    """从 PyTorch state_dict 提取 draft tensors 并写入 GGUF.

    quantize: None (F16) | "q4_0" | "q4_k_m"
    """
    target_n_embd = get_target_n_embd(target_meta)
    target_n_head = get_target_n_head(target_meta)
    target_n_vocab = get_target_n_vocab(target_meta)
    if target_n_embd == 0:
        raise ValueError(f"target_meta.n_embd == 0")
    head_dim = target_n_embd // target_n_head if target_n_head > 0 else 128

    num_layers = dspark_cfg["num_draft_layers"]
    rank = dspark_cfg["markov_rank"]

    tensors_added = 0
    tensors_missing = []

    def write_tensor(gguf_name: str, pytorch_keys: list, expected_shape: tuple = None):
        nonlocal tensors_added, tensors_missing
        try:
            t = get_pytorch_tensor(state_dict, gguf_name, pytorch_keys)
            t = t.astype(np.float32)  # always load as fp32 then convert below
            if expected_shape is not None and t.shape != expected_shape:
                print(f"  ⚠ {gguf_name}: shape mismatch {t.shape} vs expected {expected_shape}")
            # Convert to fp16 by default, then quantize if requested
            if quantize == "q4_0":
                writer.add_tensor(gguf_name, t.astype(np.float32))  # gguf lib handles quantize
            elif quantize == "q4_k_m":
                writer.add_tensor(gguf_name, t.astype(np.float32))
            else:
                # F16
                writer.add_tensor(gguf_name, t.astype(np.float16))
            tensors_added += 1
        except KeyError as e:
            tensors_missing.append((gguf_name, pytorch_keys))

    # 1. Decoder layers
    for il in range(num_layers):
        write_tensor(
            f"layers.{il}.attn_q.weight",
            [f"layers.{il}.self_attn.q_proj.weight"],
            (target_n_embd, target_n_embd),
        )
        write_tensor(
            f"layers.{il}.attn_k.weight",
            [f"layers.{il}.self_attn.k_proj.weight"],
            (target_n_embd, target_n_embd),
        )
        write_tensor(
            f"layers.{il}.attn_v.weight",
            [f"layers.{il}.self_attn.v_proj.weight"],
            (target_n_embd, target_n_embd),
        )
        write_tensor(
            f"layers.{il}.attn_output.weight",
            [f"layers.{il}.self_attn.o_proj.weight"],
            (target_n_embd, target_n_embd),
        )
        write_tensor(
            f"layers.{il}.attn_q_norm.weight",
            [f"layers.{il}.self_attn.q_norm.weight"],
            (head_dim,),
        )
        write_tensor(
            f"layers.{il}.attn_k_norm.weight",
            [f"layers.{il}.self_attn.k_norm.weight"],
            (head_dim,),
        )
        write_tensor(
            f"layers.{il}.ffn_gate.weight",
            [f"layers.{il}.mlp.gate_proj.weight"],
        )
        write_tensor(
            f"layers.{il}.ffn_up.weight",
            [f"layers.{il}.mlp.up_proj.weight"],
        )
        write_tensor(
            f"layers.{il}.ffn_down.weight",
            [f"layers.{il}.mlp.down_proj.weight"],
        )
        write_tensor(
            f"layers.{il}.input_layernorm.weight",
            [f"layers.{il}.input_layernorm.weight", f"layers.{il}.self_attn_layer_norm.weight"],
            (target_n_embd,),
        )
        write_tensor(
            f"layers.{il}.post_attention_layernorm.weight",
            [f"layers.{il}.post_attention_layernorm.weight", f"layers.{il}.mlp_layer_norm.weight"],
            (target_n_embd,),
        )

    # 2. FC + hidden_norm + final norm
    concat_dim = len(dspark_cfg["target_layer_ids"]) * target_n_embd
    write_tensor(
        "fc.weight",
        ["fc.weight"],
        (target_n_embd, concat_dim),
    )
    write_tensor(
        "hidden_norm.weight",
        ["hidden_norm.weight"],
        (target_n_embd,),
    )
    write_tensor(
        "norm.weight",
        ["norm.weight", "model.norm.weight"],
        (target_n_embd,),
    )

    # 3. Markov head
    if rank > 0:
        write_tensor(
            "markov_head.markov_w1.weight",
            ["markov_head.markov_w1.weight"],
            (target_n_vocab, rank),
        )
        write_tensor(
            "markov_head.markov_w2.weight",
            ["markov_head.markov_w2.weight"],
            (rank, target_n_vocab),
        )

    print(f"\n[draft tensors] added: {tensors_added}")
    if tensors_missing:
        print(f"[draft tensors] missing: {len(tensors_missing)}")
        for name, keys in tensors_missing[:5]:
            print(f"  - {name}: tried {keys}")


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
    parser.add_argument(
        "--draft-checkpoint",
        default=None,
        help="DSpark PyTorch checkpoint 路径（本地 safetensors 文件夹，或 HF repo ID）",
    )
    parser.add_argument(
        "--quantize",
        default=None,
        choices=[None, "q4_0", "q4_k_m"],
        help="量化类型（None = F16, 'q4_0' = Q4_0, 'q4_k_m' = Q4_K_M）",
    )
    parser.add_argument(
        "--from-mock",
        action="store_true",
        help="生成 mock checkpoint (用于本地测试，不需要 HF 下载)",
    )
    parser.add_argument(
        "--mock-output",
        default=None,
        help="mock checkpoint 输出目录（与 --from-mock 配合）",
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
        print(f"  size: {Path(args.output).stat().st_size / 1024 / 1024:.2f} MB")
        return

    # 完整模式：读 PyTorch safetensors + 写 tensor
    if args.from_mock:
        if not args.mock_output:
            print("ERROR: --from-mock requires --mock-output <dir>", file=sys.stderr)
            sys.exit(1)
        print(f"[dspark_to_gguf] generating mock checkpoint to {args.mock_output}")
        generate_mock_draft_checkpoint(args.mock_output, dspark_cfg, target_meta)
        # Use the mock as our checkpoint
        args.draft_checkpoint = args.mock_output

    if args.draft_checkpoint:
        print(f"\n[dspark_to_gguf] reading PyTorch checkpoint: {args.draft_checkpoint}")
        state_dict = read_pytorch_checkpoint(args.draft_checkpoint)
        print(f"[dspark_to_gguf] loaded {len(state_dict)} tensors from checkpoint")
        add_draft_tensors_from_state_dict(writer, state_dict, dspark_cfg, target_meta, quantize=args.quantize)
        if args.quantize:
            print(f"[dspark_to_gguf] tensors written (target quantize={args.quantize})")

    # 写 GGUF (header + KV + tensors)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[dspark_to_gguf] GGUF written: {args.output}")
    print(f"  size: {Path(args.output).stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
