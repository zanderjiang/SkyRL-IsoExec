"""A memoized Triton launcher for pik's pinned kernels, gated by ``SKYRL_ISOEXEC_PIK_FASTLAUNCH`` (default off).

``JITFunction.run`` re-derives the argument binding, specialization, cache key, and grid on every call. pik's
kernels are pinned -- one generated kernel per leaf count, one autotuned config per shape class, fixed strides
and BLOCK -- so for a given shape class those steps always produce the same answer, and the answer is cached.
The fast path then calls Triton's own ``CompiledKernel.__getitem__`` with the same kernel object, grid tuple,
and argument values the slow path would have used, so there is no arithmetic here and no bit can move. The pin
key is a strict superset of everything Triton's specialization consults: kernel identity, each tensor arg's
type/dtype/16-byte pointer alignment, every non-tensor arg verbatim, and the sorted launch kwargs. The CUDA
stream lookup and the launch hooks are deliberately not skipped; only ``used_global_vals`` re-validation is,
which can fail to raise but cannot change a value. Any exception permanently rejects that shape class and
falls back to the untouched call.
"""

from __future__ import annotations

import os

import torch

_ENABLED: bool | None = None


def _enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("SKYRL_ISOEXEC_PIK_FASTLAUNCH", "0") == "1"
    return _ENABLED


class _Pin:
    """One pinned launch: the resolved kernel runner plus the argument vector template."""

    __slots__ = ("runner", "argv", "slots", "kern")

    def __init__(self, runner, argv, slots, kern):
        self.runner = runner  # CompiledKernel.__getitem__(grid) closure -- Triton's own tail
        self.argv = argv  # the slow path's bound_args.values(), as a list
        self.slots = slots  # [(argv_index, call_args_index)] for the tensor arguments
        # A strong reference, because ``_key`` keys on ``id(kern)``: a collected wrapper whose address was
        # reused would alias two shape classes.
        self.kern = kern


_PINS: dict = {}  # key -> _Pin | str(rejection reason)
_COUNTS = {
    "served": 0,  # launches issued through a pin -- THE engagement number
    "admitted": 0,  # shape classes pinned
    "rejected": 0,  # shape classes permanently refused
    "fallback": 0,  # calls that took the original path after a rejection
    "capture_skip": 0,  # admissions declined because a CUDA graph was capturing
}


def fastlaunch_counts() -> dict:
    """Launch census; ``served`` counts launches that actually went through a pin."""
    out = dict(_COUNTS)
    out["enabled"] = _enabled()
    out["pins"] = sum(1 for v in _PINS.values() if isinstance(v, _Pin))
    out["rejections"] = {k: v for k, v in _PINS.items() if isinstance(v, str)}
    return out


def _key(kern, args, kwargs):
    """(kernel identity, per-tensor specialization inputs, every scalar verbatim, launch kwargs).

    Tensors contribute ``(type, dtype, data_ptr() & 15)``, which is everything about a tensor that reaches the
    compiler; shapes and strides arrive as separate scalars and are keyed by value, strictly finer than the
    ``== 1`` / ``% 16 == 0`` predicates Triton derives from them.
    """
    parts = [id(kern)]
    for a in args:
        if isinstance(a, torch.Tensor):
            parts.append((type(a), a.dtype, a.data_ptr() & 15))
        else:
            parts.append(a)
    if kwargs:
        parts.append(tuple(sorted(kwargs.items())))
    return tuple(parts)


def _underlying_jit(kern):
    """The ``JITFunction`` under an ``Autotuner``/``Heuristics`` wrapper (or the kernel itself)."""
    seen = 0
    fn = kern
    while hasattr(fn, "fn") and not hasattr(fn, "device_caches"):
        fn = fn.fn
        seen += 1
        if seen > 8:  # pragma: no cover -- a wrapper chain this deep is not ours
            raise RuntimeError("could not reach the JITFunction under this kernel wrapper")
    if not hasattr(fn, "device_caches"):
        raise RuntimeError(f"{type(kern).__name__} exposes no JITFunction to pin")
    return fn


def _canon_grid(g):
    """Triton's own 3-D grid canonicalization: missing dims are 1.

    ``CompiledKernel.__getitem__``'s runner indexes ``grid[0..2]`` unconditionally while pik call sites pass a
    1-D grid, so the fast path must canonicalize before handing the grid over.
    """
    g = tuple(g)
    n = len(g)
    return (g[0], g[1] if n > 1 else 1, g[2] if n > 2 else 1)


def _resolve(jit, cap_args, cap_kwargs, grid):
    """Re-run Triton's own derivation once at admission and return the objects it resolved to."""
    from triton.runtime.driver import driver
    from triton.runtime.jit import compute_cache_key

    device = driver.active.get_current_device()
    kernel_cache, kernel_key_cache, _target, _backend, binder = jit.device_caches[device]
    bound_args, specialization, options = binder(*cap_args, **cap_kwargs)
    ckey = compute_cache_key(kernel_key_cache, specialization, options)
    ck = kernel_cache.get(ckey, None)
    if ck is None:  # pragma: no cover -- the real launch just ran, so it must be there
        raise RuntimeError("the compiled kernel is not in Triton's cache after a real launch")
    argv = list(bound_args.values())
    g = grid(bound_args) if callable(grid) else grid
    return ck, argv, _canon_grid(g)


def _run_and_capture(kern, grid, args, kwargs):
    """Issue the launch on Triton's untouched path, capturing what its binder was fed.

    ``pre_run_hooks`` fire inside ``JITFunction.run`` with the fully expanded arguments, including the
    autotuner's chosen config, which is exactly the input to the derivation being memoized.
    """
    jit = _underlying_jit(kern)
    cap: list = []

    def _hook(*a, **kw):
        cap.append((a, dict(kw)))

    jit.add_pre_run_hook(_hook)
    try:
        kern[grid](*args, **kwargs)
    finally:
        try:
            jit.pre_run_hooks.remove(_hook)
        except ValueError:  # pragma: no cover
            pass
    if not cap:  # pragma: no cover -- run() always fires its pre_run hooks
        raise RuntimeError("the pre-run hook never fired; cannot see what Triton bound")
    # An Autotuner on a cache miss benchmarks every config through the same JITFunction, so the last fire is
    # the one carrying the winning config.
    return jit, cap[-1]


def _pin_from(jit, kern, grid, args, key, cap):
    """Pin what the just-completed real launch resolved to. Never issues a launch itself."""
    cap_args, cap_kwargs = cap
    ck, argv, g = _resolve(jit, cap_args, cap_kwargs, grid)

    # Map each tensor slot of the resolved argument vector back to the caller's positional args by identity;
    # a tensor that cannot be placed refuses the pin rather than being guessed at.
    slots = []
    for i, v in enumerate(argv):
        if not isinstance(v, torch.Tensor):
            continue
        for j, a in enumerate(args):
            if a is v:
                slots.append((i, j))
                break
        else:
            raise RuntimeError(f"bound tensor at argv[{i}] is not one of the positional arguments")

    runner = ck[g]  # Triton's own launch closure: stream, launch_metadata, hooks, kernel.run

    # The pin only templates the scalar half of the argument vector: every tensor position is refilled from
    # the live call in `launch`, and the loop above refuses the pin unless `slots` covers every tensor
    # position. Storing the admission-time tensors would be unbounded retention with no reader -- `_PINS` has
    # no eviction, and a retained activation can pin a whole micro-batch's autograd graph for the process's
    # life. Dropping them changes no launch.
    stored_argv = list(argv)
    for _i, _ in slots:
        stored_argv[_i] = None
    pin = _Pin(runner, stored_argv, tuple(slots), kern)
    _PINS[key] = pin
    _COUNTS["admitted"] += 1
    print(
        f"[ISOEXEC-PIK] fastlaunch PINNED {getattr(ck, 'name', '?')} grid={g} "
        f"({len(argv)} args, {len(slots)} tensor slots): Triton's per-call launch derivation "
        f"(binder + specialization + cache key + grid) is now paid once for this shape class. "
        f"Same CompiledKernel, same grid, same arguments -> bitwise-neutral by construction.",
        flush=True,
    )
    return pin


def _reject(key, reason: str) -> None:
    _PINS[key] = reason
    _COUNTS["rejected"] += 1
    print(
        f"[ISOEXEC-PIK] fastlaunch REFUSED a shape class: {reason}. Triton's own launch path "
        f"stays in charge for it (correct, just not memoized).",
        flush=True,
    )


def launch(kern, grid, *args, **kwargs):
    """``kern[grid](*args, **kwargs)``, through a memoized launcher when this shape class is pinned."""
    if not _enabled():
        return kern[grid](*args, **kwargs)

    key = _key(kern, args, kwargs)
    pin = _PINS.get(key)

    if pin is None:
        # First sight of this shape class: the launch is issued on Triton's untouched path, and only the
        # pinning afterwards is ours.
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            # Do not touch Triton's caches mid-capture; the next eager call at this shape will pin.
            _COUNTS["capture_skip"] += 1
            return kern[grid](*args, **kwargs)
        jit, cap = _run_and_capture(kern, grid, args, kwargs)  # a real error here IS fatal
        try:
            _pin_from(jit, kern, grid, args, key, cap)
        except Exception as exc:  # noqa: BLE001 -- a pin that cannot be built is never fatal
            _reject(key, f"{type(exc).__name__}: {exc}")
        return None

    if isinstance(pin, str):
        _COUNTS["fallback"] += 1
        return kern[grid](*args, **kwargs)

    argv = pin.argv.copy()
    for i, j in pin.slots:
        argv[i] = args[j]

    try:
        pin.runner(*argv)
    except Exception as exc:  # noqa: BLE001 -- demote and let Triton do it
        _reject(key, f"demoted after a fast-path error: {type(exc).__name__}: {exc}")
        _COUNTS["fallback"] += 1
        return kern[grid](*args, **kwargs)
    _COUNTS["served"] += 1
    return None
