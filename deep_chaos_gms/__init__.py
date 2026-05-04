"""Runtime gather-matmul-scatter overlay for `deep_chaos_scheduler`.

Standalone package: depends on the published `deep_chaos_scheduler`; does NOT
edit it.  Install wrappers with `enable_gms(scheduler)`, restore with
`disable_gms(scheduler)`.

For auto-enable via the `DEEP_CHAOS_USE_GMS=1` environment variable, import
`deep_chaos_gms.auto` once at the top of your script.
"""

from .gms import GatherMatmulScatterLinear, disable_gms, enable_gms
from .hoist import disable_hoist, enable_hoist

__all__ = [
    "GatherMatmulScatterLinear",
    "disable_gms",
    "disable_hoist",
    "enable_gms",
    "enable_hoist",
]
