"""Forward + backward equivalence tests for GatherMatmulScatterLinear.

Compares the GMS wrapper output against the published scheduler's
post-hook mask-multiply path (`_apply_last_dim_mask` on the dense F.linear
output).  All five topology modes are covered, plus the bias=None case
(Qwen2 reality for o_proj/gate_proj/up_proj/down_proj).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn.functional as F

from deep_chaos_gms import GatherMatmulScatterLinear, disable_gms, enable_gms

from .conftest import (
    TinyBlock,
    make_topology,
    reference_post_hook_output,
)


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #


def _input_for_component(block: TinyBlock, component: str, batch: int, seq: int) -> torch.Tensor:
    """Each projection sees a different in_dim.  Build an input tensor sized
    to that projection's `in_features`."""
    torch.manual_seed(component_hash(component))
    if component in ("q", "k", "v", "gate", "up"):
        in_dim = block.hidden
    elif component == "o":
        in_dim = block.heads * block.head_dim
    elif component == "down":
        in_dim = block.intermediate
    else:
        raise ValueError(component)
    return torch.randn(batch, seq, in_dim, requires_grad=True)


def component_hash(component: str) -> int:
    return abs(hash(component)) % 2**31


def _projection_for(block: TinyBlock, component: str):
    return {
        "q": block.self_attn.q_proj,
        "k": block.self_attn.k_proj,
        "v": block.self_attn.v_proj,
        "o": block.self_attn.o_proj,
        "gate": block.mlp.gate_proj,
        "up": block.mlp.up_proj,
        "down": block.mlp.down_proj,
    }[component]


def _input_for_down_after_gate_mask(
    block: TinyBlock, topo, batch: int, seq: int
) -> torch.Tensor:
    """down_proj's real-world input is already masked at alive_down_in
    positions (gate_proj and up_proj outputs were zeroed at the dead
    intermediate channels).  Mirror that by zeroing the dead columns of a
    random tensor."""
    x = torch.randn(batch, seq, block.intermediate, requires_grad=True)
    if topo.alive_down_in is None:
        return x
    mask = torch.zeros(block.intermediate, dtype=x.dtype)
    mask.scatter_(0, topo.alive_down_in, torch.ones_like(topo.alive_down_in, dtype=x.dtype))
    return (x * mask).detach().requires_grad_(True)


def _components(topo) -> List[str]:
    """Components that have an alive_*_out populated for this topology mode.
    Tests iterate over these."""
    if topo.mode == "dead":
        return ["q", "k", "v", "o", "gate", "up", "down"]
    if topo.mode == "identity":
        return ["q", "k", "v", "o", "gate", "up", "down"]
    out = []
    if topo.mode in ("both", "attn"):
        out.extend(["q", "k", "v", "o"])
    if topo.mode in ("both", "mlp"):
        out.extend(["gate", "up", "down"])
    return out


# --------------------------------------------------------------------------- #
#  Forward equivalence                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["both", "attn", "mlp", "identity", "dead"])
def test_forward_matches_post_hook(block_with_bias, make_scheduler, mode):
    block = block_with_bias
    topo = make_topology(block, mode=mode)
    scheduler = make_scheduler(block, topo)

    # Capture the published-path reference outputs BEFORE wrapping.
    inputs = {c: _input_for_component(block, c, batch=2, seq=3) for c in _components(topo)}
    if "down" in inputs:
        inputs["down"] = _input_for_down_after_gate_mask(block, topo, batch=2, seq=3)

    raw_outputs = {c: _projection_for(block, c)(inputs[c].detach()) for c in inputs}
    expected = reference_post_hook_output(block, topo, raw_outputs)

    # Now wrap and run.
    wrapped = enable_gms(scheduler, backend="torch")
    assert wrapped > 0
    try:
        for projection_module in [
            block.self_attn.q_proj, block.self_attn.k_proj,
            block.self_attn.v_proj, block.self_attn.o_proj,
            block.mlp.gate_proj, block.mlp.up_proj, block.mlp.down_proj,
        ]:
            assert isinstance(projection_module, GatherMatmulScatterLinear)
            projection_module.train()  # GMS wrapper bypasses on .eval()

        actual = {c: _projection_for(block, c)(inputs[c]) for c in inputs}
    finally:
        disable_gms(scheduler)

    for component in inputs:
        torch.testing.assert_close(
            actual[component], expected[component],
            rtol=1e-4, atol=1e-5,
            msg=lambda m, c=component: f"forward mismatch on {c}: {m}",
        )


# --------------------------------------------------------------------------- #
#  Backward equivalence + dead-row gradient zeros                             #
# --------------------------------------------------------------------------- #


def _accumulate_grads_via_post_hook_path(
    block: TinyBlock, topo, inputs: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor | None]]:
    """Run the published mask-multiply path forward+backward, return the
    weight.grad and bias.grad each projection accumulated."""
    raw = {c: _projection_for(block, c)(inputs[c]) for c in inputs}
    masked = reference_post_hook_output(block, topo, raw)
    loss = sum(m.pow(2).sum() for m in masked.values())
    # Zero existing grads first.
    for c in inputs:
        proj = _projection_for(block, c)
        if proj.weight.grad is not None:
            proj.weight.grad.zero_()
        if proj.bias is not None and proj.bias.grad is not None:
            proj.bias.grad.zero_()
    loss.backward()
    weights = {c: _projection_for(block, c).weight.grad.detach().clone() for c in inputs}
    biases = {
        c: (_projection_for(block, c).bias.grad.detach().clone()
            if _projection_for(block, c).bias is not None else None)
        for c in inputs
    }
    return weights, biases


def _accumulate_grads_via_gms(
    block: TinyBlock, scheduler, inputs: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor | None]]:
    """Same loss as above, but with GMS wrappers in place."""
    outputs = {c: _projection_for(block, c)(inputs[c]) for c in inputs}
    loss = sum(o.pow(2).sum() for o in outputs.values())
    for c in inputs:
        proj = _projection_for(block, c)
        if proj.weight.grad is not None:
            proj.weight.grad.zero_()
        if proj.bias is not None and proj.bias.grad is not None:
            proj.bias.grad.zero_()
    loss.backward()
    weights = {c: _projection_for(block, c).weight.grad.detach().clone() for c in inputs}
    biases = {
        c: (_projection_for(block, c).bias.grad.detach().clone()
            if _projection_for(block, c).bias is not None else None)
        for c in inputs
    }
    return weights, biases


@pytest.mark.parametrize("mode", ["both", "attn", "mlp", "identity"])
def test_backward_matches_post_hook(block_with_bias, make_scheduler, mode):
    block = block_with_bias
    topo = make_topology(block, mode=mode)
    scheduler = make_scheduler(block, topo)

    inputs = {c: _input_for_component(block, c, batch=2, seq=3) for c in _components(topo)}
    if "down" in inputs:
        inputs["down"] = _input_for_down_after_gate_mask(block, topo, batch=2, seq=3)

    expected_w, expected_b = _accumulate_grads_via_post_hook_path(block, topo, inputs)

    # Detach + fresh inputs for the GMS run so we don't share grad graph.
    inputs_gms = {c: v.detach().clone().requires_grad_(True) for c, v in inputs.items()}

    enable_gms(scheduler, backend="torch")
    try:
        for projection_module in [
            block.self_attn.q_proj, block.self_attn.k_proj,
            block.self_attn.v_proj, block.self_attn.o_proj,
            block.mlp.gate_proj, block.mlp.up_proj, block.mlp.down_proj,
        ]:
            projection_module.train()
        actual_w, actual_b = _accumulate_grads_via_gms(block, scheduler, inputs_gms)
    finally:
        disable_gms(scheduler)

    for component in inputs:
        torch.testing.assert_close(
            actual_w[component], expected_w[component],
            rtol=1e-4, atol=1e-5,
            msg=lambda m, c=component: f"weight.grad mismatch on {c}: {m}",
        )
        if expected_b[component] is not None:
            torch.testing.assert_close(
                actual_b[component], expected_b[component],
                rtol=1e-4, atol=1e-5,
                msg=lambda m, c=component: f"bias.grad mismatch on {c}: {m}",
            )
        else:
            assert actual_b[component] is None


@pytest.mark.parametrize("mode", ["both", "attn", "mlp"])
def test_dead_rows_have_zero_grad(block_with_bias, make_scheduler, mode):
    """Dead output rows of weight.grad and bias.grad must be exactly zero."""
    block = block_with_bias
    topo = make_topology(block, mode=mode)
    scheduler = make_scheduler(block, topo)

    inputs = {c: _input_for_component(block, c, batch=2, seq=3) for c in _components(topo)}
    if "down" in inputs:
        inputs["down"] = _input_for_down_after_gate_mask(block, topo, batch=2, seq=3)

    enable_gms(scheduler, backend="torch")
    try:
        for projection_module in [
            block.self_attn.q_proj, block.self_attn.k_proj,
            block.self_attn.v_proj, block.self_attn.o_proj,
            block.mlp.gate_proj, block.mlp.up_proj, block.mlp.down_proj,
        ]:
            projection_module.train()
        actual_w, actual_b = _accumulate_grads_via_gms(block, scheduler, inputs)
    finally:
        disable_gms(scheduler)

    for component in inputs:
        proj = _projection_for(block, component)
        out_dim = proj.out_features
        alive = {
            "q": topo.alive_q_out, "k": topo.alive_k_out,
            "v": topo.alive_v_out, "o": topo.alive_o_out,
            "gate": topo.alive_gate_out, "up": topo.alive_up_out,
            "down": topo.alive_down_out,
        }[component]
        if alive is None:
            continue
        all_rows = torch.arange(out_dim)
        alive_set = set(alive.tolist())
        dead_rows = torch.tensor([r for r in all_rows.tolist() if r not in alive_set], dtype=torch.long)
        if dead_rows.numel() == 0:
            continue
        # weight.grad dead rows
        assert actual_w[component].index_select(0, dead_rows).abs().max().item() == 0.0, (
            f"non-zero weight.grad on dead rows of {component}"
        )
        # bias.grad dead rows (if bias exists)
        if actual_b[component] is not None:
            assert actual_b[component].index_select(0, dead_rows).abs().max().item() == 0.0, (
                f"non-zero bias.grad on dead rows of {component}"
            )

    # down_proj's input-dim gather: weight.grad[:, dead_input_cols] should be zero too.
    if topo.alive_down_in is not None and "down" in inputs:
        proj = block.mlp.down_proj
        in_dim = proj.in_features
        all_cols = torch.arange(in_dim)
        alive_in_set = set(topo.alive_down_in.tolist())
        dead_cols = torch.tensor(
            [c for c in all_cols.tolist() if c not in alive_in_set], dtype=torch.long
        )
        if dead_cols.numel() > 0:
            assert actual_w["down"].index_select(1, dead_cols).abs().max().item() == 0.0, (
                "non-zero weight.grad on dead input cols of down_proj"
            )


def test_bias_none_path(block_no_bias, make_scheduler):
    """o_proj, gate_proj, up_proj, down_proj all have bias=None in Qwen2.
    The wrapper must handle that branch on both forward and backward without
    a NoneType error and still zero dead-row weight.grad."""
    block = block_no_bias
    topo = make_topology(block, mode="both")
    scheduler = make_scheduler(block, topo)

    inputs = {c: _input_for_component(block, c, batch=2, seq=3) for c in _components(topo)}
    inputs["down"] = _input_for_down_after_gate_mask(block, topo, batch=2, seq=3)

    enable_gms(scheduler, backend="torch")
    try:
        for projection_module in [
            block.self_attn.q_proj, block.self_attn.k_proj,
            block.self_attn.v_proj, block.self_attn.o_proj,
            block.mlp.gate_proj, block.mlp.up_proj, block.mlp.down_proj,
        ]:
            projection_module.train()
        actual_w, actual_b = _accumulate_grads_via_gms(block, scheduler, inputs)
    finally:
        disable_gms(scheduler)

    # block_no_bias has bias=False on q/k/v too -> all biases None.
    for component, b in actual_b.items():
        assert b is None, f"unexpected non-None bias.grad on {component}"

    # weight.grad still zero on dead rows.
    for component in inputs:
        proj = _projection_for(block, component)
        alive = {
            "q": topo.alive_q_out, "k": topo.alive_k_out, "v": topo.alive_v_out,
            "o": topo.alive_o_out, "gate": topo.alive_gate_out,
            "up": topo.alive_up_out, "down": topo.alive_down_out,
        }[component]
        if alive is None:
            continue
        out_dim = proj.out_features
        alive_set = set(alive.tolist())
        dead_rows = torch.tensor(
            [r for r in range(out_dim) if r not in alive_set], dtype=torch.long
        )
        if dead_rows.numel() > 0:
            assert actual_w[component].index_select(0, dead_rows).abs().max().item() == 0.0


# --------------------------------------------------------------------------- #
#  Install/uninstall + safety guards                                          #
# --------------------------------------------------------------------------- #


def test_disable_restores_originals(block_with_bias, make_scheduler):
    block = block_with_bias
    topo = make_topology(block, mode="both")
    scheduler = make_scheduler(block, topo)

    originals = {
        "q": block.self_attn.q_proj, "k": block.self_attn.k_proj,
        "v": block.self_attn.v_proj, "o": block.self_attn.o_proj,
        "gate": block.mlp.gate_proj, "up": block.mlp.up_proj,
        "down": block.mlp.down_proj,
    }
    enable_gms(scheduler, backend="torch")
    for component, orig in originals.items():
        assert _projection_for(block, component) is not orig
    restored = disable_gms(scheduler)
    assert restored == 7
    for component, orig in originals.items():
        assert _projection_for(block, component) is orig


def test_double_enable_raises(block_with_bias, make_scheduler):
    block = block_with_bias
    topo = make_topology(block, mode="both")
    scheduler = make_scheduler(block, topo)
    enable_gms(scheduler, backend="torch")
    try:
        with pytest.raises(RuntimeError, match="already been called"):
            enable_gms(scheduler, backend="torch")
    finally:
        disable_gms(scheduler)


def test_grad_accum_guard(block_with_bias, make_scheduler):
    block = block_with_bias
    topo = make_topology(block, mode="both")
    scheduler = make_scheduler(block, topo)
    scheduler.config.sticky_interval = 4
    with pytest.raises(ValueError, match="sticky_interval"):
        enable_gms(scheduler, backend="torch", grad_accum_steps=8)


def test_sticky_zero_refused(block_with_bias, make_scheduler):
    block = block_with_bias
    topo = make_topology(block, mode="both")
    scheduler = make_scheduler(block, topo)
    scheduler.config.sticky_interval = 0
    with pytest.raises(ValueError, match="sticky_interval"):
        enable_gms(scheduler, backend="torch")


def test_eval_mode_falls_back_to_dense(block_with_bias, make_scheduler):
    """In eval(), the wrapper should produce the same output as the original
    nn.Linear regardless of topology — matching post-hook eval semantics."""
    block = block_with_bias
    topo = make_topology(block, mode="both")
    scheduler = make_scheduler(block, topo)

    x = torch.randn(2, 3, block.hidden)
    expected = block.self_attn.q_proj(x)
    enable_gms(scheduler, backend="torch")
    try:
        # eval() bypasses GMS path.
        block.self_attn.q_proj.eval()
        actual = block.self_attn.q_proj(x)
    finally:
        disable_gms(scheduler)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_kv_shared_layer_q_proj_not_wrapped(make_scheduler):
    """When supports_attention_masks=False (e.g. Gemma-style post-proj norm
    or Gemma-4 KV-shared layer), q_proj/k_proj/v_proj must NOT be wrapped."""
    torch.manual_seed(0)
    block = TinyBlock()
    # Build binding manually with supports_attention_masks=False.
    from .conftest import _make_binding
    binding = _make_binding(0, block, supports_attention_masks=False)
    from .conftest import MockScheduler, MockConfig
    topo = make_topology(block, mode="both")
    scheduler = MockScheduler(bindings={0: binding}, topologies={0: topo})

    enable_gms(scheduler, backend="torch")
    try:
        # q/k/v left alone; o + mlp wrapped.
        assert not isinstance(block.self_attn.q_proj, GatherMatmulScatterLinear)
        assert not isinstance(block.self_attn.k_proj, GatherMatmulScatterLinear)
        assert not isinstance(block.self_attn.v_proj, GatherMatmulScatterLinear)
        assert isinstance(block.self_attn.o_proj, GatherMatmulScatterLinear)
        assert isinstance(block.mlp.gate_proj, GatherMatmulScatterLinear)
        assert isinstance(block.mlp.up_proj, GatherMatmulScatterLinear)
        assert isinstance(block.mlp.down_proj, GatherMatmulScatterLinear)
    finally:
        disable_gms(scheduler)
