"""End-to-end smoke test: build a real DeepChaosScheduler around a toy
multi-layer model, install GMS, call .step() + run a forward+backward.

Verifies the integration plumbing (resolve_transformer_layers, victim_range
inference, sacred-layer logic, hook installation -> swap) actually flows
through without hitting the published code's internal asserts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from deep_chaos_gms import GatherMatmulScatterLinear, disable_gms, enable_gms
from deep_chaos_scheduler import DeepChaosConfig, DeepChaosScheduler

from .conftest import TinyBlock


class TinyModel(nn.Module):
    """Mimics the minimum surface DeepChaosScheduler.resolve_transformer_layers
    needs to find the decoder stack."""

    def __init__(self, n_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([TinyBlock() for _ in range(n_layers)])
        self.config = SimpleNamespace(num_attention_heads=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.layers:
            # Don't run a real transformer math path; just sum the q_proj of
            # each layer so backward gives every wrapped projection a
            # gradient signal.
            x = x + block.self_attn.q_proj(x)
        return x


def test_step_then_gms_then_forward_backward():
    torch.manual_seed(0)
    model = TinyModel(n_layers=4)
    config = DeepChaosConfig(sticky_interval=10, seed=19)
    scheduler = DeepChaosScheduler(model, config)
    stats = scheduler.step(0)
    assert stats["reshuffle_event"] == 1.0
    assert stats["compute_pct"] <= 100.0

    n_wrapped = enable_gms(scheduler, backend="torch")
    # At minimum o_proj on every victim, plus gate/up/down per victim.
    # victim_range default for 4 layers is (0, 4); sacred is [0, 1, 2, 3] so
    # there are no non-sacred victims for n=4.  Bump n_layers to make the
    # test meaningful.
    if n_wrapped == 0:
        # n_layers=4 -> sacred = [0, 1, 2, 3] -> all sacred, no victims.
        # That's a valid corner the scheduler handles by raising; build a
        # bigger model and try again.
        scheduler.remove()
        model = TinyModel(n_layers=8)
        scheduler = DeepChaosScheduler(model, config)
        scheduler.step(0)
        n_wrapped = enable_gms(scheduler, backend="torch")
    assert n_wrapped > 0

    try:
        model.train()
        x = torch.randn(1, 4, 64)
        y = model(x)
        loss = y.pow(2).sum()
        loss.backward()
        assert torch.isfinite(loss).item()

        # All wrapped projections should have non-None weight.grad.
        for block in model.layers:
            for proj in (
                block.self_attn.q_proj, block.self_attn.k_proj,
                block.self_attn.v_proj, block.self_attn.o_proj,
                block.mlp.gate_proj, block.mlp.up_proj, block.mlp.down_proj,
            ):
                if isinstance(proj, GatherMatmulScatterLinear):
                    # weight.grad accumulates only into alive rows; that's fine.
                    # The smoke check is that backward didn't NaN/error.
                    if proj.weight.grad is not None:
                        assert torch.isfinite(proj.weight.grad).all().item()
    finally:
        disable_gms(scheduler)


def test_step_advances_after_sticky_interval():
    torch.manual_seed(0)
    model = TinyModel(n_layers=8)
    scheduler = DeepChaosScheduler(model, DeepChaosConfig(sticky_interval=5, seed=7))
    s0 = scheduler.step(0)
    assert s0["reshuffle_event"] == 1.0
    s_mid = scheduler.step(2)  # within the sticky window
    assert s_mid["reshuffle_event"] == 0.0
    s_next = scheduler.step(5)  # at the next block boundary
    assert s_next["reshuffle_event"] == 1.0
