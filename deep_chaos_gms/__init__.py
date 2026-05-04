"""Runtime gather-matmul-scatter overlay for `deep_chaos_scheduler`.

Standalone package: depends on the published `deep_chaos_scheduler`; does NOT
edit it.  Install wrappers with `enable_gms(scheduler)`, restore with
`disable_gms(scheduler)`.

For auto-enable via the `DEEP_CHAOS_USE_GMS=1` environment variable, import
`deep_chaos_gms.auto` once at the top of your script.
"""

from .gms import GatherMatmulScatterLinear, disable_gms, enable_gms

__all__ = ["GatherMatmulScatterLinear", "enable_gms", "disable_gms"]
