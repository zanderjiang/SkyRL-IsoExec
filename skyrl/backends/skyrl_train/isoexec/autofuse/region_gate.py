"""The region gate: a compiled region runs only after proving bit-equality against eager.

The ladder is trace (under the same inductor config the compile will use, since
``emulate_precision_casts`` is read at trace time) -> classify every node with ``safe_ops`` ->
refuse regions with too few fusable ops -> compile under ``REGION_INDUCTOR_CONFIG`` -> scan the
generated code for banned constructs and for at least one emitted kernel -> compare compiled
against eager bit for bit. Every rung fails closed.

The comparison operands are the gate's own hazard populations, not just the caller's inputs: a
region can be bit-identical on randn at every shape and diverge on signed zeros. The returned
artifact is a :class:`PinnedCompiledRegion`, which runs every call under the admitted config patch
and refuses shapes outside the admitted key, so a guard trip cannot silently regenerate code under
ambient config. The verdict is a record, not a side effect: nothing here mutates any model.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

from .cfgpin import make_config_pin
from .safe_ops import OpClass, classify_graph, fusable_op_count

#: Strictness version of this gate, folded into ``config_fingerprint``. Bump it whenever the gate
#: gets stricter; verdicts recorded by the weaker gate then stop resolving until re-admitted.
GATE_VERSION = 2

#: ``torch._inductor.config`` keys pinned for a region-direct compile. Shares every numerics key
#: with ``compile_guard.REQUIRED_INDUCTOR_CONFIG`` and differs on the lite-mode scoping keys
#: deliberately: with ``fallback_by_default`` on, a region compile emits no fused kernel at all.
REGION_INDUCTOR_CONFIG: dict[str, Any] = {
    # Eager numerics, identical to the compile guard's.
    "emulate_precision_casts": True,  # per-op rounding preserved; fp fusion off; toolkit libdevice
    "eager_numerics.division_rounding": True,  # Triton's div.full vs eager's div_rn
    # No graph-rewriting passes.
    "use_pre_grad_passes": False,
    "use_joint_graph_passes": False,
    "use_post_grad_passes": False,
    # Determinism of codegen choices.
    "deterministic": True,  # stable codegen choices
    "fx_graph_cache": False,  # artifact identity comes from the ledger, not a mutable cache
    "fx_graph_remote_cache": False,
}

#: Banned constructs in generated code. The ``tl.*`` entries carry the opening paren deliberately:
#: ``tl.max(``/``tl.min(`` are the reduction calls, while the elementwise forms are spelled
#: ``tl.maximum(``/``tl.minimum(`` (inductor's clamp lowering) and must not be caught by substring.
_BANNED_CODE_TOKENS: tuple[str, ...] = (
    "extern_kernels.",  # an aten fallback GEMM/op reached codegen
    "tl.dot(",  # a matmul was lowered into a Triton kernel
    "triton_red_",  # a looped reduction kernel was emitted
    "triton_per_",  # a persistent reduction kernel was emitted
    "triton_mix_",  # mix-order reduction
    "tl.sum(",  # reduction primitive inside any emitted kernel
    "tl.max(",
    "tl.min(",
    "tl.cumsum(",
    "tl.reduce(",
    "tl.associative_scan(",
)


ADMITTED = "ADMITTED"
REFUSED = "REFUSED"  # a safety rung failed -- never retry without a changed composition
DECLINED_SHORT = "DECLINED_SHORT"  # policy: too few fusable ops to be worth an artifact
VACUOUS = "VACUOUS"  # nothing compiled; a pass would be meaningless
INCONCLUSIVE = "INCONCLUSIVE"  # environment could not run the gate; treated as refusal


@dataclass
class RegionVerdict:
    site: str
    verdict: str
    reason: str
    region_sig: str = ""
    shape_key: str = ""
    fusable_ops: int = 0
    kernels_emitted: int = 0
    n_diff_bits: int = -1
    populations: list[str] = field(default_factory=list)  # bit-compare populations that passed
    op_names: list[str] = field(default_factory=list)
    decomps: list[str] = field(default_factory=list)  # bitwise-decomp entries applied (decomp.py)
    #: EAGER_SPEC (compiled == eager, per pinned shape) or PAIR_EQUALITY (trainer bits == engine
    #: bits via one shared symbolic artifact; see autofuse/pair.py). The class changes the resolution
    #: keying (exact shape vs shape class) and the absence semantics (eager fallback vs refusal).
    admission_class: str = "EAGER_SPEC"
    config_fp: str = ""
    torch_fp: str = ""
    arch: str = ""
    wall_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def bit_equal(a, b) -> tuple[bool, int]:
    """Bit-pattern equality and mismatch count. torch.equal is blind to signed zero."""
    import torch

    if a.shape != b.shape or a.dtype != b.dtype:
        return False, max(a.numel(), b.numel())
    if a.dtype.is_floating_point:
        width = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.dtype.itemsize]
        ai, bi = a.contiguous().view(width), b.contiguous().view(width)
    else:
        ai, bi = a.contiguous(), b.contiguous()
    n = int((ai != bi).sum().item())
    return n == 0, n


def _flatten_outputs(out) -> list:
    import torch

    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (list, tuple)):
        flat: list = []
        for o in out:
            flat.extend(_flatten_outputs(o))
        return flat
    return []


def torch_fingerprint() -> str:
    """Identity of the compiler that produced an artifact, so a changed compiler cannot silently serve
    an old verdict."""
    import torch

    parts = [torch.__version__, getattr(torch.version, "cuda", None) or "cpu"]
    try:
        import triton

        parts.append(f"triton-{triton.__version__}")
    except Exception:
        parts.append("triton-none")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        parts.append(f"sm{cap[0]}{cap[1]}")
    return "|".join(parts)


def arch_tag() -> str:
    import torch

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        return f"sm{cap[0]}{cap[1]}"
    return "cpu"


def config_fingerprint(config: dict[str, Any] | None = None, decomps: dict | None = None) -> str:
    """Identity of the (config, gate strictness, decomposition table) triple a verdict was produced
    under. An active decomp table folds its entry names in; an empty table leaves the hash
    unchanged, so existing ledger entries stay valid and the with/without distinction is exact."""
    cfg = REGION_INDUCTOR_CONFIG if config is None else config
    payload = {"_gate_version": GATE_VERSION, **cfg}
    if decomps:
        from .decomp import decomp_names

        payload["_bitwise_decomps"] = decomp_names(decomps)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def region_signature(gm) -> str:
    """Shape-free identity of a region: the ordered aten op sequence plus tensor dtypes.

    Two sites whose glue is the same chain share a signature -- that is the property-keying.
    Shapes live in ``shape_key``; a signature licenses nothing by itself, admission is always
    per (signature, shape_key, arch, torch_fp).
    """
    items: list[str] = []
    for node in gm.graph.nodes:
        if node.op in ("call_function", "call_method"):
            tname = str(node.target)
            # sym_* nodes exist only in symbolic traces; skipping them keeps one region's signature
            # identical between a symbolic admission trace and a static resolution trace.
            if "sym_size" in tname or "sym_numel" in tname or "sym_stride" in tname:
                continue
            items.append(tname)
        elif node.op == "placeholder":
            val = node.meta.get("val")
            dt = getattr(val, "dtype", None)
            if dt is None:
                continue  # SymInt placeholders exist only in symbolic traces; not part of identity
            items.append(f"in:{dt}")
    return hashlib.sha256("&".join(items).encode()).hexdigest()[:16]


def shape_key_of(example_inputs: Sequence) -> str:
    import torch

    parts = []
    for t in example_inputs:
        if isinstance(t, torch.Tensor):
            parts.append(f"{tuple(t.shape)}:{t.dtype}:{tuple(t.stride())}")
        else:
            parts.append(repr(t))
    return ";".join(parts)


def _trace_aten(fn: Callable, example_inputs: Sequence):
    """Trace to an aten-level graph with fake tensors, so no device work happens at trace time."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    fakes = [fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t for t in example_inputs]
    with fake_mode:
        return make_fx(fn, tracing_mode="real", _allow_non_fake_inputs=True)(*fakes)


# Hazard populations: the bit-compare's own operands; the caller's inputs are only the base row.


def _hazard_variant(example_inputs, mutate) -> list:
    """Clone the inputs and apply ``mutate`` to every float tensor; ints, indices and scalars pass
    through. Shapes, dtypes and strides are preserved, so the artifact serves the variant as is."""
    import torch

    out = []
    for t in example_inputs:
        if isinstance(t, torch.Tensor) and t.dtype.is_floating_point:
            c = t.clone()
            mutate(c)
            out.append(c)
        else:
            out.append(t)
    return out


def _mut_pm_zero_lanes(t) -> None:
    """Every 4th lane +0.0, every 7th lane -0.0: the signed-zero hazard, where Triton's unary minus
    yields +0.0 for the ``-x`` inductor lowers aten.neg to but torch's neg gives -0.0."""
    flat = t.view(-1)
    flat[::4] = 0.0
    flat[::7] = -0.0


def _mut_all_zeros(t) -> None:
    """All zeros, alternating signs."""
    flat = t.view(-1)
    flat.zero_()
    flat[1::2] = -0.0


def _mut_subnormals(t) -> None:
    """Subnormal magnitudes with mixed signs, interleaved with tiny normals; exercises flush-to-zero
    divergence in multiplies and casts."""
    import torch

    flat = t.view(-1)
    n = flat.numel()
    g = torch.Generator().manual_seed(29)
    signs = (torch.randint(0, 2, (n,), generator=g, dtype=torch.int8).float() * 2 - 1).to(flat.dtype)
    flat.copy_((signs * (2.0**-130)).to(flat.dtype))
    flat[::5] = (2.0**-126) * 1.5  # a tiny normal, so the population is not degenerate


def _mut_wide_exponent(t) -> None:
    """Wide exponent spread, so any reordering or carried-precision difference shows up loudly."""
    import torch

    g = torch.Generator().manual_seed(31)
    r = torch.randn(t.shape, generator=g, dtype=torch.float32)
    e = torch.exp(torch.randn(t.shape, generator=g, dtype=torch.float32) * 4)
    t.copy_((r * e).to(t.dtype))


#: name -> mutator. Order is the report order; every population must pass or the region refuses.
_HAZARD_BUILDERS: dict[str, Callable] = {
    "pm_zero_lanes": _mut_pm_zero_lanes,
    "all_zeros": _mut_all_zeros,
    "subnormals": _mut_subnormals,
    "wide_exponent": _mut_wide_exponent,
}


#: A fresh ``torch.compile`` wrapper is not a fresh dynamo frame: dynamo caches on the wrapped
#: function's ``__code__`` object and counts recompiles per code object, so many shape keys over one
#: region function exhaust the default cap and are then served eager, silently. :func:`fresh_frame`
#: is the fix; this raised cap is a backstop for the ``decomps`` path (whose compile target is a
#: GraphModule with no code object to clone). Raising it is safe because the compile count is bounded
#: by the fusion ledger, not by dynamo.
PINNED_DYNAMO_CONFIG: dict[str, Any] = {
    "recompile_limit": 4096,
    "accumulated_recompile_limit": 65536,
}


def pinned_dynamo_patch():
    """Context manager raising :data:`PINNED_DYNAMO_CONFIG` for a pinned compile.

    Returns a no-op context on any torch that does not expose these knobs, so the caller never
    has to branch (older spellings were ``cache_size_limit`` / ``accumulated_cache_size_limit``;
    both are patched when present).
    """
    import contextlib

    import torch._dynamo.config as dcfg

    keys = dict(PINNED_DYNAMO_CONFIG)
    # legacy spellings, patched too when this torch still carries them
    legacy = (
        ("recompile_limit", "cache_size_limit"),
        ("accumulated_recompile_limit", "accumulated_cache_size_limit"),
    )
    for new, old in legacy:
        if hasattr(dcfg, old):
            keys[old] = keys[new]
    keys = {k: v for k, v in keys.items() if hasattr(dcfg, k)}
    if not keys:
        return contextlib.nullcontext()
    return dcfg.patch(keys)


def fresh_frame(fn: Callable) -> Callable:
    """A callable with ``fn``'s exact bytecode, globals and closure but a distinct code object.

    This is what makes "one artifact per shape key" true at dynamo's granularity; see
    :data:`PINNED_DYNAMO_CONFIG`. Anything that is not a plain Python function is returned unchanged.
    """
    import types

    if not isinstance(fn, types.FunctionType):
        return fn
    try:
        clone = types.FunctionType(fn.__code__.replace(), fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)
        clone.__kwdefaults__ = fn.__kwdefaults__
        clone.__qualname__ = fn.__qualname__
        clone.__dict__.update(fn.__dict__)
        return clone
    except Exception:  # noqa: BLE001 -- never let the clone break a compile
        return fn


def _dynamo_compile_count() -> int:
    """Frames dynamo has compiled successfully so far, or -1 if unreadable.

    This is exact only because :func:`fresh_frame` guarantees the compile cannot be served from an
    existing cache entry: with a shared code object a cache hit and a permanently-skipped frame are
    indistinguishable, since neither moves these counters.
    """
    try:
        from torch._dynamo.utils import counters

        return int(counters["frames"]["ok"])
    except Exception:  # noqa: BLE001 -- a probe must never break a compile
        return -1


def _is_dynamo_product(fn: Callable) -> bool:
    """True iff ``fn`` came out of ``torch.compile``; the engagement probe only speaks about dynamo,
    so a stubbed or monkeypatched compile is never called disengaged on its evidence."""
    return hasattr(fn, "_torchdynamo_orig_callable") or hasattr(fn, "_torchdynamo_inline")


class RegionShapeMismatch(RuntimeError):
    """A pinned artifact was invoked at a shape it was never admitted for."""


class PinnedCompiledRegion:
    """The only compiled-region artifact the gate hands out.

    Every ``__call__`` runs under the admitted inductor config patch, so a dynamo recompile
    regenerates code under the same numerics rather than ambient config; inputs whose shape key
    differs from the admitted one raise :class:`RegionShapeMismatch`, since one artifact serves one
    shape. The patch re-enters per call from Python, but under CUDA-graph replay the call happens
    once at capture.
    """

    __slots__ = ("_fn", "_cfg", "_pin", "shape_key", "site", "engaged")

    def __init__(self, fn: Callable, cfg: dict[str, Any], shape_key: str, site: str = "", engaged: bool = True):
        self._fn = fn
        self._cfg = dict(cfg)
        # Built once, entered per call -- see autofuse/cfgpin.py. The per-call re-entry, and the
        # recompile guarantee above, are unchanged; only the rebuild of the patch object is removed.
        self._pin = make_config_pin(self._cfg)
        self.shape_key = shape_key
        self.site = site
        #: False when dynamo did not compile the region on the warm call: the artifact is then
        #: semantically eager and a caller must not report it as COMPILED.
        self.engaged = engaged

    def __call__(self, *args):
        key = shape_key_of(args)
        if key != self.shape_key:
            raise RegionShapeMismatch(
                f"pinned region {self.site or '<region>'} admitted at {self.shape_key} was "
                f"invoked at {key}; a recompile outside the admitted envelope is refused, not "
                f"served (defect 2)"
            )
        revert = self._pin()
        try:
            return self._fn(*args)
        finally:
            revert()


def compile_pinned(
    fn: Callable,
    example_inputs: Sequence,
    *,
    site: str = "",
    config: dict[str, Any] | None = None,
    decomps: dict | None = None,
) -> PinnedCompiledRegion:
    """Compile ``fn`` fresh under the pinned config and wrap it, one dynamo wrapper per artifact.

    With ``decomps`` the compile target is the traced, table-rewritten graph, so recompiles cannot
    lose the repair. This does not run the admission ladder; callers must hold a ledger verdict for
    (site, shape) first. The compile target is :func:`fresh_frame`'s clone of ``fn`` under
    :func:`pinned_dynamo_patch`. The returned region carries ``engaged``, which is False when dynamo
    produced no compiled frame; an unengaged artifact is eager and must not be reported as COMPILED.
    """
    import torch._inductor.config as icfg

    from .decomp import compile_rewritten

    cfg = dict(REGION_INDUCTOR_CONFIG if config is None else config)
    target = fresh_frame(fn) if not decomps else fn
    with icfg.patch(cfg), pinned_dynamo_patch():
        compiled, _ = compile_rewritten(target, example_inputs, decomps, dynamic=False)
        before = _dynamo_compile_count()
        compiled(*example_inputs)  # warm inside the pin: the first real compile happens here
        after = _dynamo_compile_count()
    # Only dynamo products are judged on dynamo's counter; a stubbed compile is left to the caller's
    # self-check.
    engaged = before < 0 or not _is_dynamo_product(compiled) or after > before
    return PinnedCompiledRegion(compiled, cfg, shape_key_of(example_inputs), site=site, engaged=engaged)


def admit_region(
    fn: Callable,
    example_inputs: Sequence,
    *,
    site: str,
    min_fusable: int = 3,
    admitted_transcendentals: frozenset[str] = frozenset(),
    config: dict[str, Any] | None = None,
    decomps: dict | None = None,
) -> tuple[RegionVerdict, Callable | None]:
    """Run the full admission ladder on one region at one shape.

    Returns ``(verdict, compiled_fn_or_None)``; the callable comes back only on ADMITTED, so a caller
    cannot install a refused artifact. Never raises -- any internal failure is INCONCLUSIVE.
    """
    t0 = time.time()
    v = RegionVerdict(site=site, verdict=INCONCLUSIVE, reason="gate did not complete")
    try:
        return _admit_region(fn, example_inputs, v, min_fusable, admitted_transcendentals, config, decomps)
    except Exception as e:  # noqa: BLE001 -- the fail-closed promise, kept structurally
        v.verdict = INCONCLUSIVE
        v.reason = f"gate error: {type(e).__name__}: {e}"
        return v, None
    finally:
        v.wall_s = round(time.time() - t0, 3)


def _admit_region(fn, example_inputs, v, min_fusable, admitted_transcendentals, config, decomps=None):
    import torch
    import torch._inductor.config as icfg

    cfg = dict(REGION_INDUCTOR_CONFIG if config is None else config)
    v.config_fp = config_fingerprint(cfg, decomps)
    v.torch_fp = torch_fingerprint()
    v.arch = arch_tag()
    v.shape_key = shape_key_of(example_inputs)

    # Trace under the same config: emulate_precision_casts is read at trace time.
    with icfg.patch(cfg):
        gm = _trace_aten(fn, example_inputs)
    v.region_sig = region_signature(gm)

    # Classify, failing closed on any non-SAFE op.
    verdicts, violations = classify_graph(gm.graph, admitted_transcendentals=admitted_transcendentals)
    v.op_names = [x.op_name for x in verdicts]
    if violations:
        v.verdict = REFUSED
        v.reason = "; ".join(f"{x.op_name}: {x.reason}" for x in violations[:6]) + (
            "" if len(violations) <= 6 else f" (+{len(violations) - 6} more)"
        )
        return v, None
    conditional_left = [x for x in verdicts if x.op_class is OpClass.CONDITIONAL]
    assert not conditional_left  # classify_graph puts these in violations

    # Length policy.
    v.fusable_ops = fusable_op_count(verdicts)
    if v.fusable_ops < min_fusable:
        v.verdict = DECLINED_SHORT
        v.reason = f"{v.fusable_ops} fusable ops < min {min_fusable}; not worth an artifact"
        return v, None

    # Compile and scan the generated code. With a decomp table the compile target is the
    # table-rewritten graph, so the repair is baked into what dynamo traces.
    from torch._inductor.utils import run_and_get_code

    from .decomp import compile_rewritten

    # fresh_frame + pinned_dynamo_patch for the reason compile_pinned uses them: an offline sweep
    # walks a whole shape ladder through one region function, and on a shared code object the later
    # rungs would emit no kernel and be recorded VACUOUS -- a refusal manufactured by dynamo's cap.
    with icfg.patch(cfg), pinned_dynamo_patch():
        compiled, applied = compile_rewritten(
            fresh_frame(fn) if not decomps else fn, example_inputs, decomps, dynamic=False
        )
        v.decomps = applied
        result, code_blobs = run_and_get_code(compiled, *example_inputs)

    joined = "\n".join(code_blobs)
    hits = [tok for tok in _BANNED_CODE_TOKENS if tok in joined]
    if hits:
        v.verdict = REFUSED
        v.reason = f"generated code contains banned constructs: {hits}"
        return v, None
    on_cuda = any(isinstance(t, torch.Tensor) and t.is_cuda for t in example_inputs)
    if on_cuda:
        v.kernels_emitted = joined.count("@triton.jit")
    else:
        v.kernels_emitted = joined.count("cpp_fused") or joined.count("async_compile.cpp")
    if v.kernels_emitted == 0:
        v.verdict = VACUOUS
        v.reason = "no kernel was emitted; a bitwise pass over nothing proves nothing"
        return v, None

    # Bit compare, on the caller's operands and on the gate's own hazard populations. Every
    # invocation goes through the pin, so a guard trip regenerates under the same numerics.
    pinned = PinnedCompiledRegion(compiled, cfg, v.shape_key, site=v.site)
    rows: list[tuple[str, list]] = [("base", list(example_inputs))]
    rows += [(name, _hazard_variant(example_inputs, m)) for name, m in _HAZARD_BUILDERS.items()]

    total_diff_base = -1
    for name, inputs in rows:
        eager_out = _flatten_outputs(fn(*inputs))
        compiled_out = _flatten_outputs(pinned(*inputs))
        if len(eager_out) != len(compiled_out):
            v.verdict = REFUSED
            v.reason = (
                f"population {name!r}: output arity differs "
                f"(eager {len(eager_out)} vs compiled {len(compiled_out)})"
            )
            return v, None
        total = 0
        for e, c in zip(eager_out, compiled_out):
            _, n = bit_equal(e, c)
            total += n
        if name == "base":
            total_diff_base = total
        if total:
            v.n_diff_bits = total
            v.verdict = REFUSED
            v.reason = f"compiled differs from eager in {total} elements on population {name!r} at " f"this shape" + (
                " -- a randn-only check would have PASSED this region (the defect-1 class: "
                "e.g. inductor's aten.neg lowering flips signed zeros)"
                if name != "base" and total_diff_base == 0
                else ""
            )
            return v, None
        v.populations.append(name)
    v.n_diff_bits = total_diff_base

    run1 = _flatten_outputs(pinned(*example_inputs))
    run2 = _flatten_outputs(pinned(*example_inputs))
    for c1, c2 in zip(run1, run2):
        ok, n = bit_equal(c1, c2)
        if not ok:
            v.verdict = REFUSED
            v.reason = f"compiled region is nondeterministic ({n} differing elements run-to-run)"
            return v, None
    v.populations.append("determinism")

    v.verdict = ADMITTED
    v.reason = (
        "bit-equal to eager on "
        + "/".join(v.populations[:-1])
        + ", deterministic, no banned constructs, kernel emitted"
    )
    return v, pinned


def positive_control(device: str = "cuda") -> dict:
    """Compile a bf16 chain without ``emulate_precision_casts`` and demand it differs from eager.

    If it does not differ, this environment cannot distinguish fused-with-fp32-carry from
    eager-rounding and every green verdict is suspect; the caller must then refuse everything. The
    chain is three dependent bf16 multiplies, chosen to lose bits on the intermediate round.
    """
    import torch
    import torch._inductor.config as icfg

    def chain(x, y):
        a = x * y  # eager rounds to bf16 here
        b = a * y  # and here
        return b * x

    gen = torch.Generator(device=device).manual_seed(7)
    x = torch.rand(4096, generator=gen, device=device, dtype=torch.float32).to(torch.bfloat16)
    x = x + 1.0
    y = x * 1.0009765625  # exactly representable multiplier
    eager = chain(x, y)

    cfg = dict(REGION_INDUCTOR_CONFIG)
    cfg["emulate_precision_casts"] = False
    with icfg.patch(cfg):
        compiled = torch.compile(chain, dynamic=False)(x, y)
    ok, n = bit_equal(eager, compiled)
    return {
        "control_failed_as_required": not ok,
        "n_diff": n,
        "note": "control must DIFFER; if it matches, the gate cannot detect a fused fp32 carry",
    }


# The transcendental sweep: the CONDITIONAL class's admission.

#: The primitives candidate regions use, swept in this order. Each is compiled as a single-op region
#: and compared against ATen over the full bf16 domain and a deterministic fp32 sample.
SWEEP_PRIMITIVES: tuple[str, ...] = (
    "exp",
    "log",
    "sigmoid",
    "tanh",
    "erf",
    "rsqrt",
    "reciprocal",
    "silu",
    "gelu",
    "exp2",
    "log2",
    "log1p",
    "expm1",
    "sin",
    "cos",
    "pow",
)


def transcendental_sweep(device: str = "cuda", primitives: Sequence[str] = SWEEP_PRIMITIVES) -> dict:
    """Per-primitive bitwise comparison of inductor's lowering vs ATen, per dtype.

    The admitted set is what ``safe_ops.classify_graph`` receives as ``admitted_transcendentals``;
    nothing else may promote a CONDITIONAL op.
    """
    import torch
    import torch._inductor.config as icfg

    def _apply(name: str, t):
        if name == "pow":
            return torch.pow(t, 3.0)
        if name == "silu":
            return torch.nn.functional.silu(t)
        if name == "gelu":
            return torch.nn.functional.gelu(t, approximate="tanh")
        return getattr(torch, name)(t)

    # Full bf16 domain: every one of the 65,536 bit patterns, NaNs excluded from the compare.
    all_bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    bf16_domain = all_bits.view(torch.bfloat16).to(device)
    gen = torch.Generator(device=device).manual_seed(11)
    fp32_sample = (torch.rand(1 << 20, generator=gen, device=device) - 0.5) * 200.0

    admitted: list[str] = []
    rejected: dict[str, str] = {}
    for name in primitives:
        detail: list[str] = []
        ok_all = True
        for dtype, dom in (("bf16", bf16_domain), ("fp32", fp32_sample)):
            try:
                x = dom.clone()
                eager = _apply(name, x)
                fn = lambda t: _apply(name, t)  # noqa: E731
                with icfg.patch(dict(REGION_INDUCTOR_CONFIG)):
                    comp = torch.compile(fn, dynamic=False)(x)
                finite = torch.isfinite(eager) | torch.isfinite(comp)
                nan_both = torch.isnan(eager) & torch.isnan(comp)
                mask = finite & ~nan_both
                e_m, c_m = eager[mask], comp[mask]
                ok, n = bit_equal(e_m, c_m)
                if not ok:
                    ok_all = False
                    detail.append(f"{dtype}: {n}/{int(mask.sum())} differ")
            except Exception as e:  # noqa: BLE001
                ok_all = False
                detail.append(f"{dtype}: sweep error {type(e).__name__}: {e}")
        if ok_all:
            admitted.append(name)
        else:
            rejected[name] = "; ".join(detail)
    return {
        "admitted": admitted,
        "rejected": rejected,
        "torch_fp": torch_fingerprint(),
        "arch": arch_tag(),
        "config_fp": config_fingerprint(),
    }
