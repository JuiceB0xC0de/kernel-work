"""Forward-pass profiling harness for the GMS overlay.

Times three configurations on the same model+input:
  1. baseline:     published scheduler with the post-hook mask path
  2. gms-torch:    enable_gms(scheduler, backend="torch")
  3. gms-triton:   enable_gms(scheduler, backend="triton") (falls back to
                    torch path until the real Triton kernel lands)

Uses torch.cuda.Event(enable_timing=True) pairs around each forward call.
On non-CUDA devices it falls back to perf_counter — useful for sanity on
CPU/MPS but the real numbers only come from MI300X.

Usage:
    python bench_gms.py --model Qwen/Qwen2.5-7B-Instruct --steps 50

For a quick local sanity run against a tiny synthetic model:
    python bench_gms.py --synthetic
"""

from __future__ import annotations

import argparse
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.nn as nn

from deep_chaos_gms import disable_gms, enable_gms
from deep_chaos_scheduler import DeepChaosConfig, DeepChaosScheduler


@dataclass
class BenchResult:
    label: str
    fwd_ms: List[float]
    peak_mem_mb: float
    compute_pct: float
    active_layers: int

    @property
    def median(self) -> float:
        return statistics.median(self.fwd_ms)

    @property
    def p90(self) -> float:
        return sorted(self.fwd_ms)[int(0.9 * len(self.fwd_ms))]

    @property
    def best(self) -> float:
        return min(self.fwd_ms)


# --------------------------------------------------------------------------- #
#  Timing primitives                                                          #
# --------------------------------------------------------------------------- #


class _CudaEventTimer:
    def __init__(self):
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)

    @contextmanager
    def measure(self):
        self._start.record()
        yield
        self._end.record()
        torch.cuda.synchronize()

    def elapsed_ms(self) -> float:
        return self._start.elapsed_time(self._end)


class _WallTimer:
    def __init__(self):
        self._start = 0.0
        self._end = 0.0

    @contextmanager
    def measure(self):
        self._start = time.perf_counter()
        yield
        self._end = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (self._end - self._start) * 1000.0


def _make_timer(device: torch.device):
    if device.type == "cuda":
        return _CudaEventTimer()
    return _WallTimer()


# --------------------------------------------------------------------------- #
#  Model loaders                                                              #
# --------------------------------------------------------------------------- #


class _SyntheticBlock(nn.Module):
    def __init__(self, hidden: int = 512, heads: int = 8, head_dim: int = 64,
                 kv_heads: int = 4, intermediate: int = 2048):
        super().__init__()
        from tests.conftest import TinyAttn, TinyMLP  # reuse the test fixtures
        self.self_attn = TinyAttn(hidden, heads, head_dim, kv_heads, bias=True)
        self.mlp = TinyMLP(hidden, intermediate)

    def forward(self, x):
        # Cheap proxy for attention output: project to head-dim space, project back.
        head_view = self.self_attn.q_proj(x)  # [..., heads*head_dim]
        x = x + self.self_attn.o_proj(head_view)
        # SwiGLU-ish MLP: gate * up -> down.
        gate = self.mlp.gate_proj(x)
        up = self.mlp.up_proj(x)
        x = x + self.mlp.down_proj(gate * up)
        return x


class _SyntheticModel(nn.Module):
    def __init__(self, n_layers: int = 8):
        super().__init__()
        from types import SimpleNamespace
        self.layers = nn.ModuleList([_SyntheticBlock() for _ in range(n_layers)])
        self.config = SimpleNamespace(num_attention_heads=8)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _load_synthetic(device: torch.device, dtype: torch.dtype):
    model = _SyntheticModel(n_layers=8).to(device=device, dtype=dtype)
    inp = torch.randn(2, 32, 512, device=device, dtype=dtype)
    return model, inp


def _load_hf(model_id: str, device: torch.device, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="sdpa",
    ).to(device)
    inp = tok("Benchmark prompt for GMS overlay forward timing.",
              return_tensors="pt").input_ids.to(device)
    return model, inp


# --------------------------------------------------------------------------- #
#  Bench loop                                                                 #
# --------------------------------------------------------------------------- #


def _run_bench(
    label: str,
    model: nn.Module,
    inp: torch.Tensor,
    scheduler: DeepChaosScheduler,
    device: torch.device,
    warmup: int,
    steps: int,
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
) -> BenchResult:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Warmup.
    for _ in range(warmup):
        _ = forward_fn(model, inp)
    if device.type == "cuda":
        torch.cuda.synchronize()

    timings: List[float] = []
    for _ in range(steps):
        timer = _make_timer(device)
        with timer.measure():
            _ = forward_fn(model, inp)
        timings.append(timer.elapsed_ms())

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
    else:
        peak = -1.0

    stats = scheduler.cached_stats or {}
    return BenchResult(
        label=label,
        fwd_ms=timings,
        peak_mem_mb=peak,
        compute_pct=float(stats.get("compute_pct", 0.0)),
        active_layers=int(stats.get("active_layers", 0)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model id (omit for --synthetic)")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--sticky", type=int, default=50)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    print(f"device={device}  dtype={dtype}  steps={args.steps}  warmup={args.warmup}")

    if args.synthetic or args.model is None:
        model, inp = _load_synthetic(device, dtype)
        print("model: synthetic 8-layer block stack")
    else:
        model, inp = _load_hf(args.model, device, dtype)
        print(f"model: {args.model}")

    config = DeepChaosConfig(sticky_interval=args.sticky, seed=args.seed)
    scheduler = DeepChaosScheduler(model, config)
    scheduler.step(0)

    def forward(m, x):
        m.train()
        with torch.no_grad():  # bench is forward-only; backward perf is a separate run
            return m(x) if not hasattr(m, "input_ids") else m(input_ids=x)

    results: List[BenchResult] = []

    # 1. Baseline: published post-hook path.
    results.append(_run_bench(
        "baseline (post-hook)", model, inp, scheduler, device,
        args.warmup, args.steps, forward,
    ))

    # 2. GMS torch backend.
    enable_gms(scheduler, backend="torch")
    try:
        results.append(_run_bench(
            "gms-torch", model, inp, scheduler, device,
            args.warmup, args.steps, forward,
        ))
    finally:
        disable_gms(scheduler)

    # 3. GMS triton backend (falls back to torch until the real kernel lands).
    enable_gms(scheduler, backend="triton")
    try:
        results.append(_run_bench(
            "gms-triton", model, inp, scheduler, device,
            args.warmup, args.steps, forward,
        ))
    finally:
        disable_gms(scheduler)

    # Print table.
    header = f"{'backend':<24} {'best (ms)':>10} {'median':>10} {'p90':>10} {'peak MB':>10} {'active':>8} {'compute %':>10}"
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        peak = f"{r.peak_mem_mb:.1f}" if r.peak_mem_mb >= 0 else "  n/a"
        print(
            f"{r.label:<24} {r.best:>10.3f} {r.median:>10.3f} {r.p90:>10.3f} "
            f"{peak:>10} {r.active_layers:>8} {r.compute_pct:>9.1f}%"
        )

    if len(results) >= 2 and results[0].median > 0:
        print()
        for r in results[1:]:
            speedup = results[0].median / r.median
            print(f"{r.label}: {speedup:.2f}x vs baseline (median)")


if __name__ == "__main__":
    main()
