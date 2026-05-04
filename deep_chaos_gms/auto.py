"""Auto-enable `enable_gms` on every newly-constructed `DeepChaosScheduler`.

Triggered when the `DEEP_CHAOS_USE_GMS` env var is truthy.  Backend defaults
to `"torch"`; override with `DEEP_CHAOS_GMS_BACKEND=triton`.  Optional
`DEEP_CHAOS_GMS_GRAD_ACCUM` env var passes `grad_accum_steps` to the
sanity-check guard.

Usage: `import deep_chaos_gms.auto` once, before constructing the scheduler.
"""

from __future__ import annotations

import os
from typing import Any

from .gms import enable_gms


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


def _maybe_install():
    if not _truthy(os.environ.get("DEEP_CHAOS_USE_GMS")):
        return

    try:
        import deep_chaos_scheduler as upstream
    except ImportError:
        return

    target_cls = getattr(upstream, "DeepChaosScheduler", None)
    if target_cls is None:
        return
    if getattr(target_cls, "_gms_auto_patched", False):
        return

    backend = os.environ.get("DEEP_CHAOS_GMS_BACKEND", "torch").strip().lower()
    grad_accum_env = os.environ.get("DEEP_CHAOS_GMS_GRAD_ACCUM")
    grad_accum: int | None
    try:
        grad_accum = int(grad_accum_env) if grad_accum_env else None
    except ValueError:
        grad_accum = None

    original_init = target_cls.__init__

    def patched_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        try:
            enable_gms(self, backend=backend, grad_accum_steps=grad_accum)
        except Exception as exc:
            import warnings

            warnings.warn(
                f"deep_chaos_gms.auto: enable_gms failed ({exc!r}); "
                "scheduler is running with the published post-hook path.",
                stacklevel=2,
            )

    target_cls.__init__ = patched_init
    target_cls._gms_auto_patched = True


_maybe_install()
