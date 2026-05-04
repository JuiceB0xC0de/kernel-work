"""Shared fixtures for the deep_chaos_gms test suite.

Builds a minimal stand-in scheduler + transformer-style block so we can run
GatherMatmulScatterLinear against the same `_apply_last_dim_mask` post-hook
path the published scheduler uses, without spinning up an HF model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import pytest
import torch
import torch.nn as nn

from deep_chaos_scheduler.deep_chaos import (
    LayerBindings,
    LayerTopology,
    _apply_last_dim_mask,
)


@dataclass
class MockConfig:
    sticky_interval: int = 50


@dataclass
class MockScheduler:
    """Minimal stand-in for DeepChaosScheduler used by enable_gms."""

    bindings: Dict[int, LayerBindings]
    topologies: Dict[int, LayerTopology]
    config: MockConfig = field(default_factory=MockConfig)
    hook_handles: List = field(default_factory=list)
    layer_device: torch.device = field(default_factory=lambda: torch.device("cpu"))


class TinyAttn(nn.Module):
    def __init__(self, hidden: int, heads: int, head_dim: int, kv_heads: int, bias: bool):
        super().__init__()
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=bias)
        # Qwen2 reality: o_proj has no bias.
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.num_heads = heads
        self.num_key_value_heads = kv_heads
        self.head_dim = head_dim


class TinyMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)


class TinyBlock(nn.Module):
    """Mimics a Qwen2 decoder layer's projection topology — q/k/v/o + gate/up/down,
    no attention math, no norms.  Each projection runs independently on the
    same input.  We don't compose them into a real transformer because the
    GMS wrapper is a per-projection swap and doesn't care about the rest."""

    def __init__(
        self,
        hidden: int = 64,
        heads: int = 4,
        head_dim: int = 16,
        kv_heads: int = 2,
        intermediate: int = 128,
        attn_bias: bool = False,
    ):
        super().__init__()
        self.self_attn = TinyAttn(hidden, heads, head_dim, kv_heads, bias=attn_bias)
        self.mlp = TinyMLP(hidden, intermediate)
        self.hidden = hidden
        self.heads = heads
        self.head_dim = head_dim
        self.kv_heads = kv_heads
        self.intermediate = intermediate

    def run_all_projections(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "q": self.self_attn.q_proj(x),
            "k": self.self_attn.k_proj(x),
            "v": self.self_attn.v_proj(x),
            "o": self.self_attn.o_proj(
                # Just feed an x-shaped tensor scaled to o_proj's input dim.
                x.new_empty(*x.shape[:-1], self.heads * self.head_dim).uniform_(-1, 1)
            ),
            "gate": self.mlp.gate_proj(x),
            "up": self.mlp.up_proj(x),
            "down": self.mlp.down_proj(
                x.new_empty(*x.shape[:-1], self.intermediate).uniform_(-1, 1)
            ),
        }


def _make_binding(layer_idx: int, block: TinyBlock, supports_attention_masks: bool = True) -> LayerBindings:
    binding = LayerBindings(layer_idx=layer_idx)
    binding.attn_module = block.self_attn
    binding.mlp_module = block.mlp
    binding.q_proj = block.self_attn.q_proj
    binding.k_proj = block.self_attn.k_proj
    binding.v_proj = block.self_attn.v_proj
    binding.o_proj = block.self_attn.o_proj
    binding.gate_proj = block.mlp.gate_proj
    binding.up_proj = block.mlp.up_proj
    binding.down_proj = block.mlp.down_proj
    binding.hidden_size = block.hidden
    binding.intermediate_size = block.intermediate
    binding.num_heads = block.heads
    binding.num_kv_heads = block.kv_heads
    binding.head_dim = block.head_dim
    binding.supports_attention_masks = supports_attention_masks
    binding.supports_mlp_masks = True
    binding.kv_shared = False
    return binding


def _heads_to_indices(heads: Sequence[int], head_dim: int) -> torch.Tensor:
    out: List[int] = []
    for h in heads:
        out.extend(range(h * head_dim, (h + 1) * head_dim))
    return torch.tensor(out, dtype=torch.long)


def make_topology(
    block: TinyBlock,
    *,
    mode: str = "both",
    alive_q_heads: Sequence[int] = (0, 2),
    alive_kv_heads: Sequence[int] = (0,),
    alive_o_out: Sequence[int] = (0, 1, 2, 5, 7, 9),
    alive_gate: Sequence[int] = tuple(range(0, 64, 2)),  # half the intermediate dim
    alive_down_out: Sequence[int] = (0, 3, 5, 8, 11),
) -> LayerTopology:
    topo = LayerTopology(mode=mode)
    if mode in ("both", "attn"):
        topo.alive_q_heads = list(alive_q_heads)
        topo.alive_kv_heads = list(alive_kv_heads)
        topo.alive_q_out = _heads_to_indices(alive_q_heads, block.head_dim)
        topo.alive_k_out = _heads_to_indices(alive_kv_heads, block.head_dim)
        topo.alive_v_out = _heads_to_indices(alive_kv_heads, block.head_dim)
        topo.alive_o_out = torch.tensor(alive_o_out, dtype=torch.long)
    if mode in ("both", "mlp"):
        gate = torch.tensor(alive_gate, dtype=torch.long)
        topo.alive_gate_out = gate
        topo.alive_up_out = gate
        topo.alive_down_in = gate
        topo.alive_down_out = torch.tensor(alive_down_out, dtype=torch.long)
    return topo


@pytest.fixture
def block_with_bias():
    torch.manual_seed(0)
    return TinyBlock(attn_bias=True)


@pytest.fixture
def block_no_bias():
    torch.manual_seed(0)
    return TinyBlock(attn_bias=False)


@pytest.fixture
def make_scheduler():
    """Returns a factory that builds a MockScheduler around a given block + topology."""

    def _factory(block: TinyBlock, topology: LayerTopology, layer_idx: int = 0):
        binding = _make_binding(layer_idx=layer_idx, block=block)
        return MockScheduler(
            bindings={layer_idx: binding},
            topologies={layer_idx: topology},
        )

    return _factory


def reference_post_hook_output(
    block: TinyBlock,
    topo: LayerTopology,
    inputs: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Reproduce what the published _install_hooks path produces: full
    F.linear, then _apply_last_dim_mask on the output.  This is the ground
    truth the GMS wrapper must match."""

    # Per-component: dead-mode = output*0; identity-mode = output unchanged;
    # mode-mismatch (attn while mode==mlp, etc.) = output*0; otherwise apply
    # _apply_last_dim_mask with the relevant alive_*_out tensor.
    attn_enabled = topo.mode in ("both", "attn")
    mlp_enabled = topo.mode in ("both", "mlp")

    def _component_alive(component: str):
        return {
            "q": topo.alive_q_out,
            "k": topo.alive_k_out,
            "v": topo.alive_v_out,
            "o": topo.alive_o_out,
            "gate": topo.alive_gate_out,
            "up": topo.alive_up_out,
            "down": topo.alive_down_out,
        }[component]

    # IMPORTANT: the published path masks down_proj OUTPUT only (it has no
    # input-dim gather).  Our GMS path additionally gathers down_proj INPUT
    # by alive_down_in.  But the inputs to down_proj in the real flow are
    # already produced by gate_proj/up_proj outputs, both of which are
    # masked by alive_gate_out / alive_up_out (== alive_down_in).  So feeding
    # an input that's already masked at the alive_down_in positions makes the
    # two paths algebraically identical.
    out: Dict[str, torch.Tensor] = {}
    for component, base_out in inputs.items():
        if topo.mode == "dead":
            out[component] = base_out * 0.0
            continue
        if topo.mode == "identity":
            out[component] = base_out
            continue
        if component in ("q", "k", "v", "o") and not attn_enabled:
            out[component] = base_out * 0.0
            continue
        if component in ("gate", "up", "down") and not mlp_enabled:
            out[component] = base_out * 0.0
            continue
        alive = _component_alive(component)
        out[component] = _apply_last_dim_mask(base_out, alive)
    return out
