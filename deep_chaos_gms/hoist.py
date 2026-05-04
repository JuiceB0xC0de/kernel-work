"""Layer hoist — physical surgery on `model.layers`.

Forget wrappers and conditional forwards.  At every reshuffle, rebuild
`model.layers` to contain only:
    - sacred layers (always kept)
    - active victim layers in `mode=both`

Everything else (mode=dead, identity, attn, mlp) is yanked entirely.  The
forward pass literally has fewer blocks.  No saved activations, no FLOPs,
no autograd graph for the absent layers.

Original layer order is restored on the next shuffle, then surgery is
re-applied.

This is much more aggressive than the GMS wrapper: a layer that would have
contributed only its attention (mode=attn) gets removed too.  Rationale:
the chaos scheduler is already perturbing the model in coarse ways, so
treating the four "wasteful" modes (dead/identity/attn/mlp) as just "off"
is a small additional perturbation in exchange for real VRAM and compute
savings.

Public API: `enable_hoist(scheduler)`, `disable_hoist(scheduler)`.
"""

from __future__ import annotations

import warnings
import weakref
from typing import Optional

import torch
import torch.nn as nn

from deep_chaos_scheduler.deep_chaos import resolve_transformer_layers


def _find_layers_parent(model: nn.Module) -> tuple[nn.Module, str]:
    """Locate the parent module that owns the transformer layers ModuleList,
    plus the attribute name. We walk the same paths `resolve_transformer_layers`
    does, but we want the parent so we can `setattr(parent, attr, new_list)`."""

    if hasattr(model, "module") and isinstance(model.module, nn.Module):
        model = model.module

    candidates = [
        ("model.layers", "model", "layers"),
        ("model.model.layers", "model.model", "layers"),
        ("model.decoder.layers", "model.decoder", "layers"),
        ("model.language_model.layers", "model.language_model", "layers"),
        ("language_model.model.layers", "language_model.model", "layers"),
        ("language_model.layers", "language_model", "layers"),
        ("text_model.layers", "text_model", "layers"),
        ("decoder.layers", "decoder", "layers"),
        ("transformer.layers", "transformer", "layers"),
        ("transformer.h", "transformer", "h"),
        ("gpt_neox.layers", "gpt_neox", "layers"),
        ("layers", "", "layers"),
    ]
    for full_path, parent_path, attr in candidates:
        parent = model
        if parent_path:
            ok = True
            for part in parent_path.split("."):
                if not hasattr(parent, part):
                    ok = False
                    break
                parent = getattr(parent, part)
            if not ok:
                continue
        layers = getattr(parent, attr, None)
        if isinstance(layers, (nn.ModuleList,)) and len(layers) > 0:
            return parent, attr
    raise AttributeError(
        "Could not locate transformer layers ModuleList parent for hoisting."
    )


def _apply_surgery(scheduler) -> tuple[int, int]:
    """Rebuild `model.layers` with only sacred + mode=both layers.

    Returns (kept_count, yanked_count).
    """
    parent: nn.Module = scheduler._hoist_parent
    attr: str = scheduler._hoist_attr
    originals: list[nn.Module] = scheduler._hoist_originals

    sacred = scheduler.sacred
    topologies = scheduler.topologies

    kept_indices: list[int] = []
    yanked_indices: list[int] = []
    survivors: list[nn.Module] = []
    for idx, layer in enumerate(originals):
        if idx in sacred:
            survivors.append(layer)
            kept_indices.append(idx)
            continue
        topo = topologies.get(idx)
        # No topology => layer wasn't a victim => keep (defensive).
        if topo is None:
            survivors.append(layer)
            kept_indices.append(idx)
            continue
        if topo.mode == "both":
            survivors.append(layer)
            kept_indices.append(idx)
        else:
            yanked_indices.append(idx)

    new_list = nn.ModuleList(survivors)
    setattr(parent, attr, new_list)
    scheduler._hoist_last_kept = kept_indices
    scheduler._hoist_last_yanked = yanked_indices
    return len(kept_indices), len(yanked_indices)


def enable_hoist(scheduler) -> int:
    """Install layer-hoist surgery.  Patches `scheduler.step` to rebuild
    `model.layers` on every reshuffle event.  Apply once, immediately after
    constructing the scheduler.

    Returns the total number of layers in the original ModuleList.

    Idempotent guard: refuses if hoist is already enabled or if `enable_gms`
    is on — the two paths conflict (GMS swaps projections inside layers; if
    we yank the layer wholesale those swaps are wasted).
    """
    if getattr(scheduler, "_hoist_enabled", False):
        raise RuntimeError("enable_hoist already called on this scheduler")
    if getattr(scheduler, "_gms_enabled", False):
        raise RuntimeError(
            "Cannot enable hoist while GMS is enabled.  "
            "Call disable_gms first."
        )

    sticky = int(getattr(scheduler.config, "sticky_interval", 0))
    if sticky <= 0:
        raise ValueError(
            f"DeepChaosConfig.sticky_interval must be > 0, got {sticky}."
        )

    # Locate and snapshot the layers ModuleList.
    parent, attr = _find_layers_parent(scheduler.model)
    originals = list(getattr(parent, attr))

    scheduler._hoist_enabled = True
    scheduler._hoist_parent = parent
    scheduler._hoist_attr = attr
    scheduler._hoist_originals = originals
    scheduler._hoist_orig_step = scheduler.step
    scheduler._hoist_last_kept = list(range(len(originals)))
    scheduler._hoist_last_yanked = []

    # Drop the post-hooks — they'd run on the original layers (which we keep
    # references to) but the forward only sees the survivors.  Cleaner to
    # remove them and let the hoist do all the work.
    for handle in list(scheduler.hook_handles):
        try:
            handle.remove()
        except Exception:
            pass
    scheduler.hook_handles.clear()

    # Patch step() so surgery runs on every reshuffle.
    orig_step = scheduler.step

    def patched_step(global_step: int):
        stats = orig_step(global_step)
        if stats and stats.get("reshuffle_event", 0.0) == 1.0:
            kept, yanked = _apply_surgery(scheduler)
            stats["hoist_kept_layers"] = float(kept)
            stats["hoist_yanked_layers"] = float(yanked)
        return stats

    scheduler.step = patched_step
    return len(originals)


def disable_hoist(scheduler) -> bool:
    """Restore the original `model.layers` ModuleList and unpatch step."""
    if not getattr(scheduler, "_hoist_enabled", False):
        return False
    parent: nn.Module = scheduler._hoist_parent
    attr: str = scheduler._hoist_attr
    originals: list[nn.Module] = scheduler._hoist_originals
    setattr(parent, attr, nn.ModuleList(originals))

    if hasattr(scheduler, "_hoist_orig_step"):
        scheduler.step = scheduler._hoist_orig_step
    scheduler._hoist_enabled = False
    return True
