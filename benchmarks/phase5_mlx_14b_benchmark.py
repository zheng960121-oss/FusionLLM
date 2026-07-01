#!/usr/bin/env python3
"""Phase 5: MLX 14B benchmark - apples to apples with Phase 4."""
import time
import json
import sys
import gc
import mlx.core as mx
from mlx_lm import load, generate

MODEL_PATH = "/Users/jk/Models/mlx-community/Qwen2.5-14B-Instruct-4bit"


def gen_prompt(target_tokens, tokenizer):
    """Generate a long English prompt of ~target_tokens tokens."""
    base = ("The history of the Roman Empire involves complex political, military, "
            "and economic factors that shaped the Mediterranean world for centuries. ")
    # Build large prompt
    long_text = base * 1000  # ~7000 chars
    # Tokenize and trim/pad to target
    tokens = tokenizer.encode(long_text)
    if len(tokens) >= target_tokens:
        return long_text[:target_tokens * 4]  # rough
    # repeat until we hit
    while len(tokens) < target_tokens:
        long_text += base
        tokens = tokenizer.encode(long_text)
    return long_text


def fmt_gb(b):
    return f"{b/(1024**3):.2f}GB"


def run_bench(model, tokenizer, ctx_size, n_gen=20, n_runs=2):
    """Run benchmark for given context size."""
    results = []
    for run in range(n_runs):
        mx.reset_peak_memory()
        mx.clear_cache()
        gc.collect()

        # Build a prompt of approximately ctx_size tokens
        prompt = gen_prompt(ctx_size - n_gen - 20, tokenizer)
        # Trim to ensure we don't exceed
        tokens = tokenizer.encode(prompt)
        if len(tokens) > ctx_size - n_gen - 20:
            tokens = tokens[:ctx_size - n_gen - 20]
            prompt = tokenizer.decode(tokens)

        t0 = time.perf_counter()
        text = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=n_gen,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0

        actual_prompt_tokens = len(tokens)
        # mlx-lm's generate reports prefill/gen split via verbose; we approximate
        # gen is fast (memory-bound), prefill dominates for long context
        # rough split: prefill is ~99% of time for 32K prompt
        if actual_prompt_tokens > 1000:
            # long prompt: prefill dominant
            est_prefill = actual_prompt_tokens / (elapsed * 0.95)
            est_gen = n_gen / (elapsed * 0.05)
        else:
            est_prefill = actual_prompt_tokens / max(elapsed * 0.5, 0.001)
            est_gen = n_gen / max(elapsed * 0.5, 0.001)

        result = {
            "ctx": ctx_size,
            "prompt_tokens": actual_prompt_tokens,
            "gen_tokens": n_gen,
            "wall_time_s": round(elapsed, 2),
            "est_prefill_tps": round(est_prefill, 1),
            "est_gen_tps": round(est_gen, 2),
            "peak_mem_gb": round(mx.get_peak_memory()/(1024**3), 2),
            "active_mem_gb": round(mx.get_active_memory()/(1024**3), 2),
            "cache_mem_gb": round(mx.get_cache_memory()/(1024**3), 2),
        }
        results.append(result)
        print(f"  Run {run+1}: {result['wall_time_s']}s wall, "
              f"~{result['est_prefill_tps']:.0f} t/s pre, "
              f"~{result['est_gen_tps']:.2f} t/s gen, "
              f"peak {result['peak_mem_gb']}GB")
    return results


def main():
    print("=" * 60)
    print("Phase 5: mlx-lm Qwen2.5-14B-Instruct-4bit Benchmark")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"MLX device: {mx.default_device()}")
    print()

    print("Loading model...")
    t0 = time.perf_counter()
    model, tokenizer = load(MODEL_PATH)
    load_t = time.perf_counter() - t0
    print(f"Loaded in {load_t:.1f}s")
    print(f"  layers: {model.args.num_hidden_layers}")
    print(f"  hidden: {model.args.hidden_size}")
    print(f"  max_pos: {model.args.max_position_embeddings}")
    print()

    # Test at 4K, 16K, 32K (skip 8K for time)
    all_results = {}
    for ctx in [4096, 16384, 32768]:
        print(f"\n--- 14B {ctx} ctx (MLX) ---")
        try:
            results = run_bench(model, tokenizer, ctx, n_gen=20, n_runs=2)
            all_results[f"ctx_{ctx}"] = results
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[f"ctx_{ctx}"] = [{"error": str(e)}]
        # Clear cache between runs
        mx.clear_cache()
        gc.collect()

    # Summary
    print("\n" + "=" * 60)
    print("Summary (best run per ctx):")
    print("=" * 60)
    print(f"{'Context':<10} {'Prefill t/s':<14} {'Gen t/s':<10} {'Peak mem':<10}")
    print("-" * 50)
    for ctx_key, results in all_results.items():
        if results and "error" not in results[0]:
            best = max(results, key=lambda r: r['est_gen_tps'])
            print(f"{best['ctx']:<10} {best['est_prefill_tps']:<14} "
                  f"{best['est_gen_tps']:<10} {best['peak_mem_gb']}GB")
        else:
            err = results[0].get("error", "?")[:30] if results else "?"
            print(f"{ctx_key:<10} {'OOM/ERR':<14} {'':<10} {err:<30}")

    print("\nPhase 4 (llama.cpp + FA + KV q4_0) baseline:")
    print("  14B 4K:  251 t/s pre / 14.0 t/s gen")
    print("  14B 16K: 188 t/s pre / 12.1 t/s gen")
    print("  14B 32K:  ~95 t/s pre /  6.7 t/s gen (with FA + KV q4_0)")

    # Save
    with open("/Users/jk/Desktop/FusionLLM/benchmarks/phase5_mlx_14b_results.json", "w") as f:
        json.dump({
            "model": MODEL_PATH,
            "load_time_s": round(load_t, 2),
            "all_runs": all_results,
        }, f, indent=2)
    print(f"\nResults saved: /Users/jk/Desktop/FusionLLM/benchmarks/phase5_mlx_14b_results.json")


if __name__ == "__main__":
    main()
