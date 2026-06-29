#!/usr/bin/env python3
"""
DSpark Spec Decode E2E Simulation (Phase 6 S10)
=================================================

Simulates the spec decoding loop without llama.cpp forward passes.
Uses configurable acceptance rates + draft distributions to measure:
  - Acceptance length per verify step
  - Total time vs autoregressive baseline
  - Speedup ratio = effective_tokens / verify_calls

This is the standalone "statistical" E2E — the full version that uses
real DSpark draft model + Qwen3 target lives in fork/llama.cpp-fusionllm
once step_spec() is fully implemented.

Usage:
  python3 tools/spec_decode_simulate.py --accept-rate 0.7 --tokens 1000
  python3 tools/spec_decode_simulate.py --accept-rate 0.5,0.7,0.9,1.0 --compare
"""

import argparse
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SpecStats:
    """Run stats — mirrors fusion::SpecDecodeStats in C++."""
    total_verify_calls: int = 0
    total_draft_proposed: int = 0
    total_draft_accepted: int = 0
    total_output_tokens: int = 0
    bonus_tokens: int = 0
    rejection_steps: int = 0
    acceptance_length: float = 0.0

    # Timing model (relative cost, configurable, calibrated to ~2x speedup at rate=0.7)
    # AR cost_per_token = 1.0 (baseline)
    # Verify: 1 forward on (block_size+1) tokens with KV cache amortization
    #   → ~2x cost of a single AR forward (well-amortized over 8 tokens)
    # Draft: small 5-layer DSpark model vs 40-layer Qwen3 target
    #   → ~1/4 cost of verify (~5x cheaper than AR's per-token cost)
    cost_verify_forward: float = 2.0
    cost_draft_forward: float = 0.25
    cost_ar_forward: float = 1.0


@dataclass
class RunResult:
    label: str
    mode: str  # "spec" | "ar"
    total_tokens: int
    total_time: float
    throughput: float
    stats: SpecStats


def run_autoregressive(target_token_count: int, stats: SpecStats) -> RunResult:
    """Simulate autoregressive: 1 token at a time, each costs cost_ar_forward."""
    stats = SpecStats()
    t = target_token_count * stats.cost_ar_forward
    return RunResult(
        label="Autoregressive",
        mode="ar",
        total_tokens=target_token_count,
        total_time=t,
        throughput=target_token_count / t,
        stats=stats,
    )


def simulate_draft_proposal(
    block_size: int,
    accept_rate: float,
    rng: random.Random,
) -> Tuple[List[int], List[int]]:
    """Simulate draft model proposing `block_size` tokens.

    Returns (draft_tokens, accept_mask).
    accept_mask[i] = 1 if draft position i was accepted, else 0.
    Uses Bernoulli(accept_rate) per position (independent).
    """
    draft_tokens = [rng.randint(0, 100000) for _ in range(block_size)]
    accept_mask = [1 if rng.random() < accept_rate else 0 for _ in range(block_size)]
    return draft_tokens, accept_mask


def run_spec_decode(
    target_token_count: int,
    accept_rate: float,
    block_size: int = 7,
    rng_seed: int = 42,
) -> RunResult:
    """Simulate spec decoding loop end-to-end.

    Each iteration:
      - cost_draft_forward (propose block_size tokens)
      - cost_verify_forward (verify on [1 + block_size] tokens)
      - If accept_count > 0: emit accept_count tokens + 1 bonus = accept_count + 1 total
      - Else: emit 1 bonus (resampled)
    Stop when target_token_count reached.
    """
    rng = random.Random(rng_seed)
    stats = SpecStats()

    total_time = 0.0
    total_emitted = 0

    while total_emitted < target_token_count:
        # 1) Draft model proposes block_size tokens
        _, accept_mask = simulate_draft_proposal(block_size, accept_rate, rng)
        stats.total_draft_proposed += block_size
        stats.total_verify_calls += 1
        total_time += stats.cost_draft_forward + stats.cost_verify_forward

        # 2) Find first rejection (cumulative mask semantics)
        accepted_count = 0
        for m in accept_mask:
            if m:
                accepted_count += 1
            else:
                break

        stats.total_draft_accepted += accepted_count

        # 3) Emit accepted_count tokens + 1 bonus (resampled from target_probs max)
        emitted_this_step = accepted_count + 1  # bonus
        # Cap at remaining budget
        remaining = target_token_count - total_emitted
        if emitted_this_step > remaining:
            emitted_this_step = remaining

        total_emitted += emitted_this_step
        stats.total_output_tokens += emitted_this_step
        stats.bonus_tokens += 1

        if accepted_count < block_size:
            stats.rejection_steps += 1

    # Acceptance length = avg tokens per verify step (including bonus)
    stats.acceptance_length = total_emitted / stats.total_verify_calls
    return RunResult(
        label=f"Spec (accept_rate={accept_rate})",
        mode="spec",
        total_tokens=total_emitted,
        total_time=total_time,
        throughput=total_emitted / total_time,
        stats=stats,
    )


def print_comparison_table(results: List[RunResult]):
    print()
    print(f"{'Mode':<28} {'Tokens':>8} {'Time':>8} {'Tok/s':>8} {'Verify#':>8} {'AvgAccLen':>10} {'RejSteps':>10} {'Speedup':>8}")
    print("-" * 100)
    base_throughput = results[0].throughput if results[0].mode == "ar" else 1.0
    for r in results:
        speedup = r.throughput / base_throughput if base_throughput > 0 else 0
        s = r.stats
        print(
            f"{r.label:<28} "
            f"{r.total_tokens:>8d} "
            f"{r.total_time:>8.2f} "
            f"{r.throughput:>8.2f} "
            f"{s.total_verify_calls:>8d} "
            f"{s.acceptance_length:>10.2f} "
            f"{s.rejection_steps:>10d} "
            f"{speedup:>7.2f}x"
        )


def main():
    parser = argparse.ArgumentParser(
        description="DSpark Spec Decode E2E Simulator (Phase 6 S10)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tokens", type=int, default=1000,
        help="Total tokens to generate (default: 1000)",
    )
    parser.add_argument(
        "--block-size", type=int, default=7,
        help="DSpark block size (default: 7, Qwen3-4B config)",
    )
    parser.add_argument(
        "--accept-rate", type=float, default=None,
        help="Per-position acceptance probability (0.0-1.0)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare multiple acceptance rates + autoregressive baseline",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducibility",
    )
    args = parser.parse_args()

    print(f"=== DSpark Spec Decode E2E Simulator ===")
    print(f"  target tokens: {args.tokens}")
    print(f"  block size: {args.block_size}")
    print(f"  cost model (relative): AR=1.0/tok, verify=2.0/step, draft=0.25/step")
    print(f"  seed: {args.seed}")
    print()

    results = []

    # Autoregressive baseline
    ar = run_autoregressive(args.tokens, SpecStats())
    results.append(ar)
    print(f"[baseline] autoregressive: {ar.total_tokens} tokens in {ar.total_time:.2f}s ({ar.throughput:.2f} tok/s)")

    if args.compare:
        # Sweep acceptance rates
        rates = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
        for rate in rates:
            sr = run_spec_decode(args.tokens, rate, args.block_size, args.seed)
            results.append(sr)
    elif args.accept_rate is not None:
        sr = run_spec_decode(args.tokens, args.accept_rate, args.block_size, args.seed)
        results.append(sr)
    else:
        # Default: one spec run at 0.7
        sr = run_spec_decode(args.tokens, 0.7, args.block_size, args.seed)
        results.append(sr)

    print_comparison_table(results)


if __name__ == "__main__":
    main()