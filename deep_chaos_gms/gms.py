"""Gather-matmul-scatter wrapper for DeepChaosScheduler victim projections.

This package is a runtime overlay on top of the published `deep_chaos_scheduler`
package. It does not edit the public repo; it swaps each victim projection's
`nn.Linear` for a `GatherMatmulScatterLinear` that:

    1. reads the alive-channel index from the scheduler's current LayerTopology,
    2. slices `weight` (and `bias`) to the alive output rows (and, for
       `down_proj`, alive input columns),
    3. runs a smaller dense `F.linear`,
    4. scatters the result back into a zero-padded full-width tensor.

The scatter uses the out-of-place `Tensor.scatter`, which is differentiable;
combined with `Tensor.index_select`, autograd handles backward correctly and
dead rows of `weight.grad` and `bias.grad` are exactly zero.

Public API: `enable_gms(scheduler)` and `disable_gms(scheduler)`.
"""

from __future__ import annotations

import os
import weakref
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_enabled() -> bool:
    """Validation mode: check every alive_* index against the projection's
    actual dimensions before each forward.  Triggered by `DEEP_CHAOS_GMS_VALIDATE=1`.

    Why this exists: ROCm 7.2's HIP `index_select` faults hard
    (HSA_STATUS_ERROR_EXCEPTION) when any index >= dim_size, instead of
    raising a clean error like CUDA.  If the upstream scheduler ever
    produces an out-of-bounds index (e.g. via a head_dim/num_heads inference
    mismatch on a model with split QKV dims), it manifests as a
    `vectorized_gather_kernel` crash with no Python-side traceback to the
    real cause.  This switch makes the bound check explicit and Pythonic.
    """
    return os.environ.get("DEEP_CHAOS_GMS_VALIDATE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


# Components hooked by the existing post-hook installer in
# deep_chaos_scheduler/deep_chaos.py::_install_hooks.  We mirror exactly the
# same gating: q/k/v are wrapped only when the layer's binding declares
# `supports_attention_masks` True; o, gate, up, down are wrapped whenever the
# binding has the corresponding projection.
_ATTN_COMPONENTS = ("q", "k", "v", "o")
_MLP_COMPONENTS = ("gate", "up", "down")


def _component_attr_on_parent(component: str) -> str:
    """Attribute name to setattr on the parent attn/mlp module."""
    return {
        "q": "q_proj",
        "k": "k_proj",
        "v": "v_proj",
        "o": "o_proj",
        "gate": "gate_proj",
        "up": "up_proj",
        "down": "down_proj",
    }[component]


def _topo_alive_out(topo, component: str) -> Optional[torch.Tensor]:
    return {
        "q": topo.alive_q_out,
        "k": topo.alive_k_out,
        "v": topo.alive_v_out,
        "o": topo.alive_o_out,
        "gate": topo.alive_gate_out,
        "up": topo.alive_up_out,
        "down": topo.alive_down_out,
    }.get(component)


def _topo_alive_in(topo, component: str) -> Optional[torch.Tensor]:
    if component == "down":
        return topo.alive_down_in
    return None


class GatherMatmulScatterLinear(nn.Module):
    """Drop-in wrapper for an `nn.Linear` projection used inside DeepChaos
    victim layers.  Owns the original `weight` and `bias` parameters (no
    copies).  Uses gather-matmul-scatter when the layer's topology has a
    non-trivial alive set; falls back to dense `F.linear` otherwise.
    """

    def __init__(
        self,
        original: nn.Linear,
        scheduler,
        layer_idx: int,
        component: str,
        backend: str = "torch",
    ):
        super().__init__()
        if not isinstance(original, nn.Linear):
            raise TypeError(
                f"GatherMatmulScatterLinear expects nn.Linear, got {type(original).__name__}"
            )
        if component not in _ATTN_COMPONENTS + _MLP_COMPONENTS:
            raise ValueError(f"Unknown projection component: {component!r}")
        if backend not in ("torch", "triton"):
            raise ValueError(f"Unknown backend: {backend!r}")

        # Re-use the original parameters in-place.  This is what makes the
        # wrapper transparent to the optimizer: weight/bias keep their identity
        # and gradient accumulation lands on the same Parameter objects the
        # optimizer already references.
        self.weight = original.weight
        self.bias = original.bias
        self.in_features = int(original.in_features)
        self.out_features = int(original.out_features)
        self._component = component
        self._layer_idx = int(layer_idx)
        self._backend = backend
        # Weakref so the wrapper -> scheduler edge does not pin the scheduler
        # alive when the user drops their reference.
        self._scheduler_ref = weakref.ref(scheduler)
        # Stashed for disable_gms; not used at forward time.
        self._original_module = original
        # Cache the parent module + attr for fast restore.
        self._parent_ref: Optional[weakref.ReferenceType] = None
        self._parent_attr: Optional[str] = None
        # Cached sticky-block stamp for cheap revalidation.
        self._cached_shuffle_step: Optional[int] = None

    # ----- forward -----------------------------------------------------------

    def _dense_linear(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def _matmul(
        self,
        x_small: torch.Tensor,
        W_small: torch.Tensor,
        b_small: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self._backend == "triton":
            try:
                from .triton_gms import triton_dense_linear

                return triton_dense_linear(x_small, W_small, b_small)
            except Exception:
                # Triton unavailable / non-GPU / kernel error -> torch fallback.
                # We swallow silently here because the user opted into "triton"
                # but the canary tests will catch real numerical regressions.
                pass
        return F.linear(x_small, W_small, b_small)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scheduler = self._scheduler_ref()
        # Eval and detached cases: behave exactly like a plain nn.Linear, so
        # eval-time semantics match the existing post-hook path.  Do NOT wrap
        # in torch.no_grad here — the original hook path doesn't either.
        if scheduler is None or not self.training:
            return self._dense_linear(x)

        topo = scheduler.topologies.get(self._layer_idx)
        if topo is None:
            return self._dense_linear(x)

        # Mode-level early outs that match _install_hook_for_component logic.
        if topo.mode == "dead":
            # Multiply by zero scalar to preserve grad_fn (matches
            # _apply_last_dim_mask's "all dead" branch).
            return self._dense_linear(x) * 0.0
        if topo.mode == "identity":
            return self._dense_linear(x)

        component = self._component
        attn_enabled = topo.mode in ("both", "attn")
        mlp_enabled = topo.mode in ("both", "mlp")
        if component in _ATTN_COMPONENTS and not attn_enabled:
            return self._dense_linear(x) * 0.0
        if component in _MLP_COMPONENTS and not mlp_enabled:
            return self._dense_linear(x) * 0.0

        alive_out = _topo_alive_out(topo, component)
        alive_in = _topo_alive_in(topo, component)

        # If the scheduler chose "no slicing" for this projection (topology
        # didn't populate alive_out), fall back to dense.
        if alive_out is None:
            return self._dense_linear(x)
        if alive_out.numel() == 0:
            # All channels dead for this component -> match _apply_last_dim_mask.
            return self._dense_linear(x) * 0.0

        # Device discipline: alive_* are created on scheduler.layer_device,
        # which under device_map="auto" can differ from this layer's actual
        # device.  Move per-forward; .to(...) on a same-device tensor is a
        # no-op so this is safe to do unconditionally.
        # NOTE: non_blocking=True can fault on ROCm with int64 index tensors
        # if the gather kernel reads before the H2D copy lands.  Use a
        # synchronous copy — index tensors are tiny so the cost is negligible.
        alive_out = alive_out.to(device=x.device).long().contiguous()
        if alive_in is not None:
            alive_in = alive_in.to(device=x.device).long().contiguous()
            if alive_in.numel() == 0:
                # All input channels dead -> output is zero everywhere.
                return self._dense_linear(x) * 0.0

        if _validate_enabled():
            self._validate_indices(alive_out, alive_in, x)

        # ---- Gather --------------------------------------------------------
        W_small = self.weight.index_select(0, alive_out)
        if alive_in is not None:
            W_small = W_small.index_select(1, alive_in)
            x_small = x.index_select(-1, alive_in)
        else:
            x_small = x
        b_small = self.bias.index_select(0, alive_out) if self.bias is not None else None

        # index_select on dim>0 with an unsorted index can produce a
        # non-contiguous tensor on some backends; Triton tile loads demand
        # contiguous inputs.  .contiguous() on an already-contiguous tensor
        # is a no-op so this is safe unconditionally.
        W_small = W_small.contiguous()
        x_small = x_small.contiguous()
        if b_small is not None:
            b_small = b_small.contiguous()

        # ---- Matmul --------------------------------------------------------
        y_small = self._matmul(x_small, W_small, b_small)

        # ---- Scatter -------------------------------------------------------
        # Build a zero tensor of the full output shape and scatter the small
        # result into the alive_out positions.  Tensor.scatter (out-of-place)
        # is differentiable: gradients flow back into y_small at the alive
        # positions and produce zero gradient everywhere else.
        out_shape = list(y_small.shape)
        out_shape[-1] = self.out_features
        out = y_small.new_zeros(out_shape)
        # Broadcast alive_out across the leading dims of y_small.
        idx_shape = [1] * (y_small.ndim - 1) + [alive_out.numel()]
        idx = alive_out.view(*idx_shape).expand_as(y_small)
        return out.scatter(-1, idx, y_small)

    def _validate_indices(
        self,
        alive_out: torch.Tensor,
        alive_in: Optional[torch.Tensor],
        x: torch.Tensor,
    ) -> None:
        """Catch out-of-bounds alive indices BEFORE handing them to
        index_select, with a clean Python error instead of a HIP HSA fault.

        Forces a small device->host sync (.max().item()) — only enabled in
        DEEP_CHAOS_GMS_VALIDATE=1 mode.  Disable once you've confirmed the
        upstream scheduler is producing valid indices for your model.
        """
        out_dim = int(self.out_features)
        in_dim = int(self.in_features)
        out_max = int(alive_out.max().item())
        out_min = int(alive_out.min().item())
        if out_max >= out_dim or out_min < 0:
            raise IndexError(
                f"GMS [{self._component} layer={self._layer_idx}]: "
                f"alive_out range [{out_min}, {out_max}] exceeds out_features={out_dim}. "
                f"Numel={alive_out.numel()}. "
                "This is the upstream scheduler producing an out-of-bounds index "
                "(usually a head_dim/num_heads inference mismatch). On ROCm this "
                "would otherwise manifest as a hard HIP HSA fault in "
                "vectorized_gather_kernel. Fix the index generation in "
                "DeepChaosScheduler.step()."
            )
        if alive_in is not None:
            in_max = int(alive_in.max().item())
            in_min = int(alive_in.min().item())
            x_last = int(x.shape[-1])
            if in_max >= in_dim or in_min < 0:
                raise IndexError(
                    f"GMS [{self._component} layer={self._layer_idx}]: "
                    f"alive_in range [{in_min}, {in_max}] exceeds in_features={in_dim}. "
                    f"Numel={alive_in.numel()}."
                )
            if in_max >= x_last:
                raise IndexError(
                    f"GMS [{self._component} layer={self._layer_idx}]: "
                    f"alive_in max {in_max} exceeds input last-dim {x_last}. "
                    f"This means the upstream output of the previous projection "
                    f"is narrower than alive_in expects."
                )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, component={self._component!r}, "
            f"layer_idx={self._layer_idx}, backend={self._backend!r}"
        )


# ----- install / uninstall ---------------------------------------------------


def _repair_binding_dims(binding) -> list[str]:
    """Fix per-layer dimension metadata that the upstream binding inference
    can get wrong on modern transformers attention modules.

    Specifically: modern `Qwen2Attention` (transformers >= ~4.40) stores
    `num_key_value_heads` only on `attn.config`, not on the module itself.
    The upstream `_first_attr(attn, ("num_key_value_heads", ...))` returns
    None for those modules, and `LayerBindings` then defaults
    `num_kv_heads` to `num_heads`.  On a GQA model
    (num_kv_heads != num_heads) this produces alive_k_out / alive_v_out
    indices that overshoot the actual k_proj / v_proj output dim.

    The published `_apply_last_dim_mask` silently filters
    out-of-bounds indices, so the post-hook path "works" but at a
    much lower effective KV survival rate than configured.
    GMS's `index_select` cannot silently filter — ROCm
    HSA-faults instead — so we have to repair the binding before
    installing wrappers.

    Returns the list of repair messages (empty when nothing was changed).
    """
    messages: list[str] = []
    head_dim = int(binding.head_dim or 0)
    if head_dim <= 0:
        return messages

    # Repair num_heads from q_proj.out_features // head_dim if available.
    q = getattr(binding, "q_proj", None)
    if isinstance(q, nn.Linear) and head_dim > 0:
        derived_q = int(q.out_features // head_dim)
        if derived_q > 0 and derived_q != int(binding.num_heads or 0):
            messages.append(
                f"layer {binding.layer_idx}: num_heads "
                f"{binding.num_heads} -> {derived_q} (from q_proj)"
            )
            binding.num_heads = derived_q

    # Repair num_kv_heads from k_proj.out_features // head_dim.
    k = getattr(binding, "k_proj", None)
    if isinstance(k, nn.Linear) and head_dim > 0:
        derived_kv = int(k.out_features // head_dim)
        if derived_kv > 0 and derived_kv != int(binding.num_kv_heads or 0):
            messages.append(
                f"layer {binding.layer_idx}: num_kv_heads "
                f"{binding.num_kv_heads} -> {derived_kv} (from k_proj)"
            )
            binding.num_kv_heads = derived_kv

    return messages


def _projection_should_wrap(binding, component: str) -> bool:
    """Mirror the existing hook installer's gate on a per-component basis.

    See deep_chaos_scheduler/deep_chaos.py::_install_hooks (L699-727):
      - q/k/v projections: only when binding.supports_attention_masks is True.
        That flag already incorporates the kv_shared and post-proj-norm
        guards, so we don't need to re-check them here.
      - o, gate, up, down: wrap whenever the binding has the corresponding
        module attribute.
    """
    if component in ("q", "k", "v"):
        if not bool(getattr(binding, "supports_attention_masks", False)):
            return False
        proj = getattr(binding, f"{component}_proj", None)
        return isinstance(proj, nn.Linear)
    if component == "o":
        return isinstance(getattr(binding, "o_proj", None), nn.Linear)
    if component == "gate":
        return isinstance(getattr(binding, "gate_proj", None), nn.Linear)
    if component == "up":
        return isinstance(getattr(binding, "up_proj", None), nn.Linear)
    if component == "down":
        return isinstance(getattr(binding, "down_proj", None), nn.Linear)
    return False


def _parent_for_component(binding, component: str):
    if component in _ATTN_COMPONENTS:
        return binding.attn_module
    return binding.mlp_module


def enable_gms(
    scheduler,
    *,
    backend: str = "torch",
    grad_accum_steps: Optional[int] = None,
) -> int:
    """Install GatherMatmulScatterLinear wrappers on every victim projection.

    Removes the existing post-hooks (so we don't pay them on top of the new
    matmul path), then walks `scheduler.bindings` and replaces each eligible
    `nn.Linear` projection with a wrapper.  Originals are stashed on the
    wrapper for `disable_gms`.

    Returns the number of projections wrapped.

    Guards:
      - sticky_interval must be > 0 (existing scheduler code coerces 0 to 1
        silently; we make the misconfig loud).
      - if grad_accum_steps is provided and sticky_interval < grad_accum_steps,
        raise — topology could change mid-accumulation window.
      - if sticky_interval is not a multiple of grad_accum_steps, warn.
    """
    if backend not in ("torch", "triton"):
        raise ValueError(f"Unknown backend: {backend!r}")

    sticky = int(getattr(scheduler.config, "sticky_interval", 0))
    if sticky <= 0:
        raise ValueError(
            f"DeepChaosConfig.sticky_interval must be > 0, got {sticky}. "
            "(The published scheduler coerces this to 1 internally; enable_gms "
            "refuses to install on the misconfig so it surfaces loudly.)"
        )
    if grad_accum_steps is not None:
        gas = int(grad_accum_steps)
        if sticky < gas:
            raise ValueError(
                f"sticky_interval ({sticky}) < grad_accum_steps ({gas}); "
                "topology could reshuffle inside an accumulation window, "
                "producing inconsistent W_small slices across micro-steps."
            )
        if gas > 0 and sticky % gas != 0:
            import warnings

            warnings.warn(
                f"sticky_interval ({sticky}) is not a multiple of "
                f"grad_accum_steps ({gas}). Reshuffles will land near the "
                "edge of accumulation windows. Consider setting "
                f"sticky_interval to {((sticky // gas) + 1) * gas}.",
                stacklevel=2,
            )

    # Idempotency: if already enabled on this scheduler, refuse loudly so the
    # caller notices.  Accidentally double-wrapping would make the indexing
    # math operate on already-sliced tensors.
    if getattr(scheduler, "_gms_enabled", False):
        raise RuntimeError(
            "enable_gms has already been called on this scheduler; call "
            "disable_gms first."
        )

    # Repair per-layer dimension metadata that the upstream binding
    # inference may have gotten wrong (esp. num_kv_heads on modern Qwen2).
    # Must happen BEFORE we drop the post-hooks or build wrappers, and
    # BEFORE the next scheduler.step() generates a fresh topology.
    repairs: list[str] = []
    for binding in scheduler.bindings.values():
        repairs.extend(_repair_binding_dims(binding))
    if repairs:
        # Force the next .step() to regenerate the topology with the corrected
        # head counts.  Don't call .step() ourselves here — the user's first
        # .step(global_step) will produce the right indices.
        scheduler.cached_stats = None
        scheduler.last_shuffle_step = None
        scheduler.topologies.clear()
        import warnings
        warnings.warn(
            "deep_chaos_gms.enable_gms repaired upstream binding metadata: "
            + "; ".join(repairs)
            + ". The published post-hook path silently filters out-of-bounds "
            "alive indices; GMS cannot, so this repair is required for GMS "
            "but also indicates the post-hook path was running at a lower "
            "effective KV survival rate than configured.",
            stacklevel=2,
        )

    # Drop the existing post-hooks but DO NOT call scheduler.remove() — that
    # would also clear topologies/cached_stats/last_shuffle_step and break
    # anything calling .step() afterwards.
    for handle in list(scheduler.hook_handles):
        try:
            handle.remove()
        except Exception:
            pass
    scheduler.hook_handles.clear()

    wrapped = 0
    wrappers: list[GatherMatmulScatterLinear] = []
    for layer_idx, binding in scheduler.bindings.items():
        for component in _ATTN_COMPONENTS + _MLP_COMPONENTS:
            if not _projection_should_wrap(binding, component):
                continue
            parent = _parent_for_component(binding, component)
            if parent is None:
                continue
            attr_name = _component_attr_on_parent(component)
            original = getattr(parent, attr_name, None)
            if not isinstance(original, nn.Linear):
                continue
            # Already wrapped (defensive): skip.
            if isinstance(original, GatherMatmulScatterLinear):
                continue
            wrapper = GatherMatmulScatterLinear(
                original=original,
                scheduler=scheduler,
                layer_idx=layer_idx,
                component=component,
                backend=backend,
            )
            wrapper._parent_ref = weakref.ref(parent)
            wrapper._parent_attr = attr_name
            setattr(parent, attr_name, wrapper)
            wrappers.append(wrapper)
            wrapped += 1

    scheduler._gms_enabled = True
    scheduler._gms_backend = backend
    scheduler._gms_wrappers = wrappers
    return wrapped


def disable_gms(scheduler) -> int:
    """Restore the original `nn.Linear` projections, undoing `enable_gms`.

    Returns the number of projections restored.  Idempotent: returns 0 if not
    enabled.  Does NOT re-install the original post-hooks; the user can call
    `scheduler._install_hooks()` afterwards if they want them back.
    """
    if not getattr(scheduler, "_gms_enabled", False):
        return 0

    restored = 0
    for wrapper in list(getattr(scheduler, "_gms_wrappers", [])):
        parent = wrapper._parent_ref() if wrapper._parent_ref is not None else None
        if parent is None or wrapper._parent_attr is None:
            continue
        # Only restore if the parent still holds our wrapper (don't overwrite
        # something the user installed in the meantime).
        if getattr(parent, wrapper._parent_attr, None) is wrapper:
            setattr(parent, wrapper._parent_attr, wrapper._original_module)
            restored += 1

    scheduler._gms_enabled = False
    scheduler._gms_wrappers = []
    return restored
