"""PAIR_EQUALITY: the admission class whose contract is trainer bits == engine bits.

Where both sides run the same compiled artifact, pair-equality holds by construction, so this class
admits transcendentals that EAGER_SPEC cannot. The trainer never compiles its own forward -- it
wraps the inference artifact in :class:`PairRegionFn`, whose forward runs the shared artifact under
``no_grad`` and whose backward is an independent eager VJP with no bitwise contract. The two sides
never see the same leading dim, so entries are keyed by a shape class with dim 0 wildcarded and
admission requires size-genericity (no new graph across the M grid, no size-derived scalar in the
float dataflow) plus self-consistency across arrangements.

Unlike EAGER_SPEC, absence is a mismatch rather than a bit-neutral fallback: a pair region missing
on one side is the divergence, so ``install_autofuse_sites`` raises in every process that must
carry the region.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from .cfgpin import make_config_pin
from .region_gate import (
    _BANNED_CODE_TOKENS,
    _HAZARD_BUILDERS,
    ADMITTED,
    INCONCLUSIVE,
    REFUSED,
    REGION_INDUCTOR_CONFIG,
    VACUOUS,
    RegionVerdict,
    _flatten_outputs,
    _hazard_variant,
    _trace_aten,
    arch_tag,
    bit_equal,
    config_fingerprint,
    region_signature,
    torch_fingerprint,
)
from .safe_ops import OpClass, classify_graph, fusable_op_count

PAIR_EQUALITY = "PAIR_EQUALITY"
EAGER_SPEC = "EAGER_SPEC"


def shape_class_of(example_inputs: Sequence) -> str:
    """Like ``shape_key_of`` with dim 0 wildcarded -- the dim that differs between the trainer's
    shards and the engine's ladder. Remaining dims, dtype and contiguity class stay exact."""
    import torch

    parts = []
    for t in example_inputs:
        if isinstance(t, torch.Tensor):
            dims = ("*",) + tuple(t.shape[1:])
            contig = "C" if t.is_contiguous() else "S"
            parts.append(f"{dims}:{t.dtype}:{contig}")
        else:
            parts.append(repr(t))
    return "PAIR;" + ";".join(parts)


def trace_region_symbolic(fn: Callable, example_inputs: Sequence):
    """Symbolic aten trace, used by both admission and resolution so region signatures match.

    Falls back to the static trace when symbolic tracing is unavailable; the sym-scalar scan is then
    vacuous, which is safe because a static trace has no sym nodes to miss.
    """
    from torch.fx.experimental.proxy_tensor import make_fx

    try:
        # Real tensors straight in: symbolic mode fakeifies internally, and pre-fakeified inputs
        # raise, which would silently fall back to a static trace whose scan misses every sym node.
        return make_fx(fn, tracing_mode="symbolic", _allow_non_fake_inputs=True)(*example_inputs)
    except Exception:  # noqa: BLE001
        return _trace_aten(fn, example_inputs)


def size_scalar_feeds_float(gm) -> list[str]:
    """Nodes where a size-derived scalar enters the float dataflow.

    Sym ints as shape args (view/chunk/slice) are legal; a SymInt/SymFloat as an arithmetic operand
    is not, since the per-element expression could then differ between specializations.
    """
    import torch

    offenders: list[str] = []
    sym_producers = set()
    for node in gm.graph.nodes:
        tname = str(node.target)
        if "sym_size" in tname or "sym_numel" in tname or "_local_scalar_dense" in tname or tname.endswith("item"):
            sym_producers.add(node)
            continue
        val = node.meta.get("val") if hasattr(node, "meta") else None
        if isinstance(val, (torch.SymInt, torch.SymFloat)):
            sym_producers.add(node)
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        val = node.meta.get("val") if hasattr(node, "meta") else None
        is_float_out = getattr(getattr(val, "dtype", None), "is_floating_point", False)
        if not is_float_out:
            continue
        tname = str(node.target)
        if any(v in tname for v in ("view", "reshape", "slice", "select", "split", "chunk", "expand", "as_strided")):
            continue  # shape args are the legal use
        for arg in node.args:
            if arg in sym_producers:
                offenders.append(f"{tname} consumes {arg}")
    return offenders


class PairShapeClassMismatch(RuntimeError):
    """A pair artifact was invoked outside its admitted shape class."""


class PairArtifact:
    """The shared forward artifact: one ``dynamic=True`` dynamo wrapper, every call under the admitted
    config patch, inputs locked to the admitted shape class (dim 0 free)."""

    __slots__ = ("_fn", "_cfg", "_pin", "shape_class", "site")

    def __init__(self, fn: Callable, cfg: dict[str, Any], shape_class: str, site: str = ""):
        self._fn = fn
        self._cfg = dict(cfg)
        # Built once, entered per call: a dynamo recompile must still regenerate under the admitted
        # numerics, so the pin cannot be dropped -- only its rebuild. See autofuse/cfgpin.py.
        self._pin = make_config_pin(self._cfg)
        self.shape_class = shape_class
        self.site = site

    def __call__(self, *args):
        cls = shape_class_of(args)
        if cls != self.shape_class:
            raise PairShapeClassMismatch(
                f"pair artifact {self.site or '<region>'} admitted for class {self.shape_class} "
                f"was invoked at {cls}"
            )
        revert = self._pin()
        try:
            return self._fn(*args)
        finally:
            revert()


def compile_pair(fn: Callable, example_inputs: Sequence, *, site: str = "", config=None) -> PairArtifact:
    """One fresh symbolic artifact under the pin. Callers must hold a PAIR ledger verdict."""
    import torch
    import torch._inductor.config as icfg

    cfg = dict(REGION_INDUCTOR_CONFIG if config is None else config)
    with icfg.patch(cfg):
        compiled = torch.compile(fn, dynamic=True)
        compiled(*example_inputs)
    return PairArtifact(compiled, cfg, shape_class_of(example_inputs), site=site)


class PairRegionFn:
    """Namespace for the autograd wrapper (built lazily; torch import stays out of module scope)."""

    _fn_cls = None

    @classmethod
    def _cls(cls):
        if cls._fn_cls is None:
            import torch

            class _PairRegionFn(torch.autograd.Function):
                """Forward runs the shared artifact under no_grad, so it is bit-identical to the
                engine; backward is an independent eager VJP with no bitwise contract."""

                @staticmethod
                def forward(ctx, artifact, region_fn, n_tensors, *args):
                    ctx.region_fn = region_fn
                    ctx.n_tensors = n_tensors
                    tensors = args[:n_tensors]
                    ctx.extras = args[n_tensors:]
                    ctx.save_for_backward(*tensors)
                    with torch.no_grad():
                        out = artifact(*tensors, *ctx.extras)
                    return out

                @staticmethod
                def backward(ctx, *grad_outputs):
                    tensors = ctx.saved_tensors
                    with torch.enable_grad():
                        leaves = [t.detach().requires_grad_(t.is_floating_point()) for t in tensors]
                        out = ctx.region_fn(*leaves, *ctx.extras)
                    outs = out if isinstance(out, (tuple, list)) else (out,)
                    diff_leaves = [x for x in leaves if x.requires_grad]
                    grads = torch.autograd.grad(
                        [o for o in outs if o.requires_grad],
                        diff_leaves,
                        [g for o, g in zip(outs, grad_outputs) if o.requires_grad],
                        allow_unused=True,
                    )
                    it = iter(grads)
                    leaf_grads = tuple(next(it) if x.requires_grad else None for x in leaves)
                    return (None, None, None) + leaf_grads + (None,) * len(ctx.extras)

            cls._fn_cls = _PairRegionFn
        return cls._fn_cls

    @classmethod
    def apply(cls, artifact: PairArtifact, region_fn: Callable, *args):
        """Split tensor args (autograd-tracked) from python scalars, then apply."""
        import torch

        tensors = [a for a in args if isinstance(a, torch.Tensor)]
        extras = [a for a in args if not isinstance(a, torch.Tensor)]
        if [id(a) for a in args] != [id(a) for a in tensors + extras]:
            # tensors must be a prefix for the (tensors, extras) split to reconstruct the call
            raise TypeError("pair region call: tensor arguments must precede scalar arguments")
        return cls._cls().apply(artifact, region_fn, len(tensors), *tensors, *extras)


#: Extra hazard populations for transcendental-bearing regions: saturation (the flush boundary where
#: implementations disagree) and NaN payloads.
def _mut_saturation(t) -> None:
    import torch

    flat = t.view(-1)
    n = flat.numel()
    g = torch.Generator().manual_seed(37)
    mags = torch.tensor([20.0, 30.0, 60.0, 88.0, 100.0, 700.0])
    pick = mags[torch.randint(0, len(mags), (n,), generator=g)]
    sign = torch.randint(0, 2, (n,), generator=g, dtype=torch.int8).float() * 2 - 1
    flat.copy_((pick * sign).to(flat.dtype))
    flat[::11] = 0.0  # keep a live band too


def _mut_nan_payloads(t) -> None:
    import torch

    flat = t.view(-1)
    if flat.dtype == torch.bfloat16:
        # Distinct payloads, one with the sign bit; the second is written as a negative literal
        # because 0xFFC7 overflows int16.
        nan_bits = torch.tensor([0x7FC1, 0xFFC7 - 0x10000], dtype=torch.int16)
        flat[::13] = nan_bits[0].view(torch.bfloat16)
        flat[1::13] = nan_bits[1].view(torch.bfloat16)
    else:
        flat[::13] = float("nan")


_PAIR_EXTRA_HAZARDS = {"saturation": _mut_saturation, "nan_payloads": _mut_nan_payloads}


def pair_admit(
    fn: Callable,
    inputs_grid: Sequence[Sequence],
    *,
    site: str,
    min_fusable: int = 2,
    config: dict[str, Any] | None = None,
) -> tuple[RegionVerdict, PairArtifact | None]:
    """The PAIR admission ladder over ``inputs_grid``, a list of example-input tuples across M.

    All entries must share a shape class; an M=1 entry may be included last and is bit-checked but
    exempt from the no-new-graph gate. No eager reference is consulted: transcendentals are
    admissible, while reductions, matmuls, scans and RNG are refused as in EAGER_SPEC.
    """
    import time

    t0 = time.time()
    v = RegionVerdict(site=site, verdict=INCONCLUSIVE, reason="pair gate did not complete")
    v.admission_class = PAIR_EQUALITY
    try:
        return _pair_admit(fn, list(inputs_grid), v, min_fusable, config)
    except Exception as e:  # noqa: BLE001 -- fail closed
        v.verdict = INCONCLUSIVE
        v.reason = f"pair gate error: {type(e).__name__}: {e}"
        return v, None
    finally:
        v.wall_s = round(time.time() - t0, 3)


def _pair_admit(fn, grid, v, min_fusable, config):
    import torch
    import torch._dynamo
    import torch._inductor.config as icfg
    from torch._dynamo.utils import counters

    cfg = dict(REGION_INDUCTOR_CONFIG if config is None else config)
    v.config_fp = config_fingerprint(cfg)
    v.torch_fp = torch_fingerprint()
    v.arch = arch_tag()
    base_inputs = list(grid[0])
    v.shape_key = shape_class_of(base_inputs)
    for inputs in grid[1:]:
        if shape_class_of(inputs) != v.shape_key:
            v.verdict = REFUSED
            v.reason = f"grid inputs span shape classes: {shape_class_of(inputs)} != {v.shape_key}"
            return v, None

    # Trace and classify. Transcendentals are legal here since the pair contract compares the
    # artifact against itself. The trace is symbolic: a static trace would make the sym-scalar scan
    # below vacuous.
    with icfg.patch(cfg):
        gm = trace_region_symbolic(fn, base_inputs)
    v.region_sig = region_signature(gm)
    verdicts, violations = classify_graph(gm.graph)
    v.op_names = [x.op_name for x in verdicts]
    hard = [x for x in violations if x.op_class is OpClass.BANNED]
    if hard:
        v.verdict = REFUSED
        v.reason = "; ".join(f"{x.op_name}: {x.reason}" for x in hard[:6])
        return v, None
    # CONDITIONAL (transcendental) ops count as fusable here: they are what this class exists to admit.
    v.fusable_ops = fusable_op_count(verdicts) + sum(1 for x in verdicts if x.op_class is OpClass.CONDITIONAL)
    if v.fusable_ops < min_fusable:
        v.verdict = REFUSED
        v.reason = f"{v.fusable_ops} fusable ops < min {min_fusable}"
        return v, None

    # Sym-scalar scan, on the symbolic trace so size uses appear.
    offenders = size_scalar_feeds_float(gm)
    if offenders:
        v.verdict = REFUSED
        v.reason = f"size-derived scalar in float dataflow (expression may differ per specialization): {offenders[:3]}"
        return v, None

    # One symbolic compile under the pin; code scan on its output.
    from torch._inductor.utils import run_and_get_code

    torch._dynamo.reset()
    counters.clear()
    with icfg.patch(cfg):
        compiled = torch.compile(fn, dynamic=True)
        _, code_blobs = run_and_get_code(compiled, *base_inputs)
    joined = "\n".join(code_blobs)
    hits = [tok for tok in _BANNED_CODE_TOKENS if tok in joined]
    if hits:
        v.verdict = REFUSED
        v.reason = f"generated code contains banned constructs: {hits}"
        return v, None
    on_cuda = any(isinstance(t, torch.Tensor) and t.is_cuda for t in base_inputs)
    v.kernels_emitted = joined.count("@triton.jit") if on_cuda else (joined.count("cpp_fused") or 1)
    if on_cuda and v.kernels_emitted == 0:
        v.verdict = VACUOUS
        v.reason = "no kernel emitted"
        return v, None

    artifact = PairArtifact(compiled, cfg, v.shape_key, site=v.site)

    # Size-genericity: no new graph across the M>=2 grid.
    m1_rows = []
    for inputs in grid:
        first = next(t for t in inputs if isinstance(t, torch.Tensor))
        (m1_rows if first.shape[0] == 1 else []).append(inputs)
    graphs_before = counters["stats"]["unique_graphs"]
    for inputs in grid:
        first = next(t for t in inputs if isinstance(t, torch.Tensor))
        if first.shape[0] == 1:
            continue
        artifact(*inputs)
    if counters["stats"]["unique_graphs"] != graphs_before:
        v.verdict = REFUSED
        v.reason = (
            "a size in the M>=2 grid took a NEW graph: the artifact is not size-generic and "
            "pair-equality across trainer/engine shapes cannot be claimed"
        )
        return v, None
    for inputs in m1_rows:  # the 0/1 edge may respecialize; still bit-checked below
        artifact(*inputs)

    # Self-consistency: the same logical elements at different M must give identical bits, whether
    # passed as one batch or as two chunks.
    def _rows(inputs):
        return next(t for t in inputs if isinstance(t, torch.Tensor)).shape[0]

    populations = [("base", base_inputs)]
    populations += [(name, _hazard_variant(base_inputs, m)) for name, m in _HAZARD_BUILDERS.items()]
    populations += [(name, _hazard_variant(base_inputs, m)) for name, m in _PAIR_EXTRA_HAZARDS.items()]
    for name, inputs in populations:
        m = _rows(inputs)
        if m < 2:
            continue
        whole = _flatten_outputs(artifact(*inputs))
        cut = max(1, m // 2)
        first_args, second_args = [], []
        for a in inputs:
            if isinstance(a, torch.Tensor) and a.shape[0] == m:
                first_args.append(a[:cut].contiguous())
                second_args.append(a[cut:].contiguous())
            else:
                first_args.append(a)
                second_args.append(a)
        parts1 = _flatten_outputs(artifact(*first_args))
        parts2 = _flatten_outputs(artifact(*second_args))
        ndiff = 0
        for w, p1, p2 in zip(whole, parts1, parts2):
            if w.shape[0] == m:
                rejoined = torch.cat([p1, p2], dim=0)
                _, n = bit_equal(w, rejoined)
                ndiff += n
        if ndiff:
            v.n_diff_bits = ndiff
            v.verdict = REFUSED
            v.reason = (
                f"SELF-CONSISTENCY failed on population {name!r}: the same logical elements at "
                f"different M differ in {ndiff} elements -- per-element expression is NOT "
                f"M-independent; pair-equality across trainer/engine shapes is falsified"
            )
            return v, None
        v.populations.append(name)

    # determinism (artifact vs itself, twice)
    r1 = _flatten_outputs(artifact(*base_inputs))
    r2 = _flatten_outputs(artifact(*base_inputs))
    for a, b in zip(r1, r2):
        ok, n = bit_equal(a, b)
        if not ok:
            v.verdict = REFUSED
            v.reason = f"nondeterministic ({n} differing elements run-to-run)"
            return v, None
    v.populations.append("determinism")
    v.n_diff_bits = 0

    v.verdict = ADMITTED
    v.reason = (
        "PAIR_EQUALITY: size-generic symbolic artifact, self-consistent across arrangements on "
        + "/".join(v.populations[:-1])
        + ", deterministic, no banned constructs. No eager reference consulted (by design)."
    )
    return v, artifact
