"""End-to-end training-step bench for the GMS overlay.

What it actually measures:
  - Total wall-clock for N train steps (forward + backward + optimizer.step).
  - Peak VRAM (CUDA) or peak resident memory (CPU/MPS).
  - Whether the forward path survives shuffle-boundary crossings.  Sticky
    interval is set deliberately small so multiple reshuffles land inside the
    measured window — if the GMS path is going to crash on a topology change
    mid-run, this is where it shows.

Usage:
    python bench_train.py --synthetic --steps 100 --sticky 25
    python bench_train.py --model Qwen/Qwen2.5-7B-Instruct --steps 100 --sticky 25 --dtype bf16

This is the bench that decides whether the optimization is worth shipping.
Forward-only timing (`bench_gms.py`) is for kernel tuning; this one is for
"does it use less VRAM and finish faster".
"""

from __future__ import annotations

import argparse
import gc
import resource
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.nn as nn

from deep_chaos_gms import disable_gms, enable_gms
from deep_chaos_scheduler import DeepChaosConfig, DeepChaosScheduler


@dataclass
class TrainBenchResult:
    label: str
    total_seconds: float
    per_step_ms: List[float]
    peak_mem_mb: float
    shuffles_seen: int
    survived: bool
    failure: Optional[str] = None
    final_loss: float = float("nan")
    compute_pct_history: List[float] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Memory polling                                                             #
# --------------------------------------------------------------------------- #


def _peak_mem_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024**2
    # CPU/MPS fallback. macOS ru_maxrss is bytes; Linux is KB.
    import sys
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / 1024**2
    return rss / 1024


def _reset_peak_mem(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    gc.collect()


# --------------------------------------------------------------------------- #
#  Model loaders                                                              #
# --------------------------------------------------------------------------- #


class _SyntheticBlock(nn.Module):
    def __init__(self, hidden: int = 512, heads: int = 8, head_dim: int = 64,
                 kv_heads: int = 4, intermediate: int = 2048):
        super().__init__()
        from tests.conftest import TinyAttn, TinyMLP
        self.self_attn = TinyAttn(hidden, heads, head_dim, kv_heads, bias=True)
        self.mlp = TinyMLP(hidden, intermediate)

    def forward(self, x):
        head_view = self.self_attn.q_proj(x)
        x = x + self.self_attn.o_proj(head_view)
        gate = self.mlp.gate_proj(x)
        up = self.mlp.up_proj(x)
        x = x + self.mlp.down_proj(gate * up)
        return x


class _SyntheticModel(nn.Module):
    def __init__(self, n_layers: int = 8, hidden: int = 512):
        super().__init__()
        from types import SimpleNamespace
        self.layers = nn.ModuleList([_SyntheticBlock(hidden=hidden) for _ in range(n_layers)])
        self.config = SimpleNamespace(num_attention_heads=8)
        self.head = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def _load_synthetic(device: torch.device, dtype: torch.dtype, hidden: int = 512):
    model = _SyntheticModel(n_layers=8, hidden=hidden).to(device=device, dtype=dtype)
    inp = torch.randn(2, 32, hidden, device=device, dtype=dtype)
    target = torch.randn(2, 32, hidden, device=device, dtype=dtype)
    return model, inp, target


def _load_hf(model_id: str, device: torch.device, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="sdpa",
    ).to(device)
    text = "Benchmark prompt for GMS overlay training-step timing."
    enc = tok(text, return_tensors="pt", padding="max_length", max_length=64,
              truncation=True)
    inp = enc.input_ids.to(device)
    return model, inp, inp  # use input_ids as labels (causal LM)


# --------------------------------------------------------------------------- #
#  Train loop                                                                 #
# --------------------------------------------------------------------------- #


def _run_train_bench(
    label: str,
    model: nn.Module,
    inp: torch.Tensor,
    target: torch.Tensor,
    scheduler: DeepChaosScheduler,
    device: torch.device,
    steps: int,
    sticky: int,
    is_hf: bool,
    warmup: int = 3,
) -> TrainBenchResult:
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_fn = nn.MSELoss()

    # Warmup (not measured).
    model.train()
    for w in range(warmup):
        scheduler.step(w)
        optim.zero_grad(set_to_none=True)
        if is_hf:
            out = model(input_ids=inp, labels=target)
            loss = out.loss
        else:
            out = model(inp)
            loss = loss_fn(out, target)
        loss.backward()
        optim.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    _reset_peak_mem(device)

    per_step_ms: List[float] = []
    shuffles = 0
    last_shuffle = scheduler.last_shuffle_step
    compute_pct_history: List[float] = []
    final_loss = float("nan")
    failure: Optional[str] = None
    survived = True

    t_start = time.perf_counter()
    for step in range(warmup, warmup + steps):
        try:
            stats = scheduler.step(step)
            if scheduler.last_shuffle_step != last_shuffle:
                shuffles += 1
                last_shuffle = scheduler.last_shuffle_step
                compute_pct_history.append(float(stats.get("compute_pct", 0.0)))

            t0 = time.perf_counter()
            optim.zero_grad(set_to_none=True)
            if is_hf:
                out = model(input_ids=inp, labels=target)
                loss = out.loss
            else:
                out = model(inp)
                loss = loss_fn(out, target)
            loss.backward()
            optim.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            per_step_ms.append((t1 - t0) * 1000.0)
            final_loss = float(loss.detach().item())
        except Exception as exc:
            survived = False
            failure = f"{type(exc).__name__} at step {step} (post-shuffle={shuffles}): {exc}"
            break

    total = time.perf_counter() - t_start
    peak = _peak_mem_mb(device)
    return TrainBenchResult(
        label=label,
        total_seconds=total,
        per_step_ms=per_step_ms,
        peak_mem_mb=peak,
        shuffles_seen=shuffles,
        survived=survived,
        failure=failure,
        final_loss=final_loss,
        compute_pct_history=compute_pct_history,
    )


def _print_result(r: TrainBenchResult) -> None:
    if not r.per_step_ms:
        print(f"  {r.label:<22} CRASHED: {r.failure}")
        return
    median = statistics.median(r.per_step_ms)
    p90 = sorted(r.per_step_ms)[int(0.9 * len(r.per_step_ms))]
    status = "ok" if r.survived else "CRASHED"
    print(
        f"  {r.label:<22} {status:>8}  total={r.total_seconds:>7.2f}s  "
        f"med/step={median:>7.2f}ms  p90={p90:>7.2f}ms  peak={r.peak_mem_mb:>8.1f}MB  "
        f"shuffles={r.shuffles_seen:>2}  loss={r.final_loss:.4f}"
    )
    if r.failure:
        print(f"     → {r.failure}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--sticky", type=int, default=25,
                        help="sticky_interval; default 25 forces 3-4 shuffles in 100 steps")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--dtype", type=str, default="fp32",
                        choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--hidden", type=int, default=512,
                        help="hidden size for the synthetic model")
    parser.add_argument("--skip-triton", action="store_true",
                        help="skip the gms-triton run (e.g. if Triton not installed)")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    print(f"device={device}  dtype={dtype}  steps={args.steps}  sticky={args.sticky}")
    print(f"  → with sticky={args.sticky} and {args.steps} steps we expect "
          f"~{max(1, args.steps // args.sticky)} shuffle boundaries to cross")

    is_hf = bool(args.model and not args.synthetic)

    def _build():
        if is_hf:
            model, inp, target = _load_hf(args.model, device, dtype)
            print(f"model: {args.model}")
        else:
            model, inp, target = _load_synthetic(device, dtype, hidden=args.hidden)
            print(f"model: synthetic 8-layer hidden={args.hidden}")
        config = DeepChaosConfig(sticky_interval=args.sticky, seed=args.seed,
                                 announce_reshuffles=False)
        scheduler = DeepChaosScheduler(model, config)
        return model, inp, target, scheduler

    print("\nbaseline (post-hook):")
    model, inp, target, scheduler = _build()
    r_baseline = _run_train_bench("baseline", model, inp, target, scheduler,
                                  device, args.steps, args.sticky, is_hf)
    _print_result(r_baseline)
    del model, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\ngms-torch:")
    model, inp, target, scheduler = _build()
    enable_gms(scheduler, backend="torch")
    r_torch = _run_train_bench("gms-torch", model, inp, target, scheduler,
                               device, args.steps, args.sticky, is_hf)
    disable_gms(scheduler)
    _print_result(r_torch)
    del model, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    r_triton: Optional[TrainBenchResult] = None
    if not args.skip_triton:
        print("\ngms-triton:")
        model, inp, target, scheduler = _build()
        enable_gms(scheduler, backend="triton")
        r_triton = _run_train_bench("gms-triton", model, inp, target, scheduler,
                                    device, args.steps, args.sticky, is_hf)
        disable_gms(scheduler)
        _print_result(r_triton)

    # Verdict.
    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    survived = [r.survived for r in (r_baseline, r_torch) + ((r_triton,) if r_triton else ())]
    if all(survived):
        print("✓ All backends survived all shuffle boundaries.")
    else:
        print("✗ At least one backend crashed across a shuffle boundary.")

    if r_baseline.total_seconds > 0:
        for r in (r_torch, r_triton):
            if r is None or not r.survived:
                continue
            speedup = r_baseline.total_seconds / r.total_seconds
            mem_delta = r.peak_mem_mb - r_baseline.peak_mem_mb
            mem_pct = 100.0 * mem_delta / max(1.0, r_baseline.peak_mem_mb)
            print(
                f"{r.label}: total {speedup:.2f}x baseline  "
                f"VRAM Δ={mem_delta:+.1f} MB ({mem_pct:+.1f}%)"
            )


if __name__ == "__main__":
    main()
