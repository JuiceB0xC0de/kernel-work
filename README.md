# deep-chaos-gms

In-progress kernel-style optimization work for `deep_chaos_scheduler`. **Not for publication.**

## What this is

A runtime overlay that swaps each victim projection's `nn.Linear` for a `GatherMatmulScatterLinear` wrapper. The wrapper:

1. reads the alive-channel index from the scheduler's current `LayerTopology`,
2. slices `weight` to alive output rows (and, for `down_proj`, alive input columns),
3. runs a smaller dense matmul,
4. scatters the result back into a zero-padded full-width output tensor.

The published `deep_chaos_scheduler` package masks dead channels *after* a full-width matmul. This package replaces that with gather-matmul-scatter, paying compute only for the alive channels. With `sticky_interval=50`, the gather cost is amortised over ~50 forward passes.

## Why it lives here, not in the public repo

`~/lucky-pick-scheduler/` is the public GitHub repo. Unfinished optimization code does not get pushed there. This package imports the published `deep_chaos_scheduler` as a read-only dependency. There are zero edits to the public repo.

## Usage

### Explicit

```python
import deep_chaos_scheduler as dcs
import deep_chaos_gms

scheduler = dcs.DeepChaosScheduler(model, dcs.DeepChaosConfig(sticky_interval=50, seed=19))
deep_chaos_gms.enable_gms(scheduler, backend="torch", grad_accum_steps=8)

# ... train loop ...

deep_chaos_gms.disable_gms(scheduler)
```

### Env-var auto-enable (zero edits to existing scripts)

```bash
DEEP_CHAOS_USE_GMS=1 \
DEEP_CHAOS_GMS_BACKEND=torch \
DEEP_CHAOS_GMS_GRAD_ACCUM=8 \
python train_benchmark.py
```

with `import deep_chaos_gms.auto` added once at the top of the script. The auto-import monkey-patches `DeepChaosScheduler.__init__` to call `enable_gms` after construction when the env var is truthy.

## Install (editable)

```bash
pip install -e ~/kernel-work
```

## Status

- Phase 1 (PyTorch reference): in progress
- Phase 2 (Triton kernel for MI300X): stub only, raises `NotImplementedError`

## Plan

Full design: `~/.claude/plans/hey-check-the-mnempalace-stateless-taco.md`
