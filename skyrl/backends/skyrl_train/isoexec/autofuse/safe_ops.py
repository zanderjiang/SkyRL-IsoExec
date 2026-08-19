"""The property classifier: which aten ops may appear inside a bitwise-fused region.

An op is admissible iff it moves bytes or evaluates a single correctly-rounded IEEE-754 primitive
per element; float reductions are banned because their association tree is inductor's, not ATen's.
The allowlist is fail-closed -- unknown ops are BANNED -- and transcendentals stay CONDITIONAL
until the per-primitive sweep proves libdevice-vs-ATen bit equality on the running (arch, torch).
Division is admitted only because the region config pins ``eager_numerics.division_rounding``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class OpClass(Enum):
    SAFE = "safe"  # exactly-rounded pointwise, or pure byte movement
    CONDITIONAL = "conditional"  # transcendental: needs the per-primitive sweep's admission
    BANNED = "banned"  # reduction / matmul / unknown -- never fusable


# Exactly-rounded pointwise primitives, keyed by the aten op's overload-packet base name.
_EXACT_POINTWISE: frozenset[str] = frozenset(
    {
        "add",
        "sub",
        "rsub",
        "mul",
        "div",  # admitted only under eager_numerics.division_rounding -- see module docstring
        "neg",
        "abs",
        "sqrt",  # IEEE-exact, unlike rsqrt
        "minimum",
        "maximum",
        "clamp",
        "clamp_min",
        "clamp_max",
        "relu",
        "where",
        "copysign",
        "sign",
        "signbit",
        "isnan",
        "isinf",
        "isfinite",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "logical_and",
        "logical_or",
        "logical_not",
        "logical_xor",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_not",
        "bitwise_left_shift",
        "bitwise_right_shift",
        "floor",
        "ceil",
        "round",
        "trunc",
        "frac",
        "remainder",
        "fmod",
        "nan_to_num",
        "threshold",
        "hardtanh",  # clamp by another name
        "leaky_relu",  # mul + where; both exact
    }
)

# Transcendentals: one rounding per element, but libdevice-vs-ATen equality is a measured property
# of the (arch, torch) pair, not a spec guarantee. CONDITIONAL until swept.
_TRANSCENDENTAL: frozenset[str] = frozenset(
    {
        "exp",
        "exp2",
        "expm1",
        "log",
        "log2",
        "log10",
        "log1p",
        "sigmoid",
        "tanh",
        "sinh",
        "cosh",
        "erf",
        "erfc",
        "erfinv",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "pow",
        "rsqrt",  # not IEEE-exact; grouped with the transcendentals deliberately
        "reciprocal",
        "gelu",
        "silu",
        "logit",
        "softplus",
        "elu",
        "hardswish",
        "hardsigmoid",
        "lgamma",
        "digamma",
    }
)

# Byte movement: copies, views, concatenation, gather-style indexed reads. No float arithmetic,
# so no rounding and no order.
_MOVEMENT: frozenset[str] = frozenset(
    {
        "_to_copy",
        "to",
        # The prims alias AOT/functionalized graphs use for the dtype cast make_fx spells `_to_copy`.
        "convert_element_type",
        "clone",
        "contiguous",
        "copy",
        "copy_",
        "cat",
        "stack",
        "view",
        "_unsafe_view",
        "reshape",
        "permute",
        "transpose",
        "t",
        "squeeze",
        "unsqueeze",
        "expand",
        "expand_as",
        "slice",
        "select",
        "split",
        "split_with_sizes",
        "chunk",
        "narrow",
        "flip",
        "roll",
        "repeat",
        "repeat_interleave",
        "tile",
        "gather",
        "index_select",
        "index",
        "take",
        "masked_fill",
        "masked_select",
        "tril",
        "triu",
        "diagonal",
        "as_strided",
        "unfold",
        "constant_pad_nd",
        "full",
        "full_like",
        "zeros",
        "zeros_like",
        "ones",
        "ones_like",
        "empty",
        "empty_like",
        "arange",
        "fill",
        "fill_",
        "detach",
        "alias",
        "lift_fresh_copy",
        "type_as",
        "pin_memory",
        "flatten",
        "unbind",
        "unflatten",
        "movedim",
        "broadcast_tensors",
        "broadcast_to",
    }
)

# Safe only when not accumulating: the classifier checks arguments before admitting these.
_MOVEMENT_IF_NOT_ACCUMULATE: frozenset[str] = frozenset(
    {"index_put", "index_put_", "scatter", "scatter_", "index_copy", "index_copy_", "put"}
)

# Named bans, so a refusal says why rather than "unknown". Absence from this table still bans;
# presence only improves the message.
_KNOWN_BANNED: dict[str, str] = {
    "sum": "float reduction: association tree is inductor's, not ATen's (axis 18)",
    "mean": "float reduction",
    "prod": "float reduction",
    "cumsum": "float scan: same association hazard as a reduction",
    "cumprod": "float scan",
    "amax": "reduction (order-independent for max, but the fused softmax it belongs to is not)",
    "amin": "reduction",
    "max": "reduction",
    "min": "reduction",
    "argmax": "reduction",
    "argmin": "reduction",
    "topk": "reduction (selection requires owning the comparison order)",
    "sort": "reduction-class (tie order is a bitwise property)",
    "var": "float reduction",
    "std": "float reduction",
    "var_mean": "float reduction",
    "logsumexp": "float reduction",
    "_softmax": "folds a float reduction; decomposition rewrites the arithmetic",
    "softmax": "folds a float reduction",
    "_log_softmax": "folds a float reduction",
    "native_layer_norm": "folds a float reduction; inductor's schedule is a third schedule",
    "layer_norm": "folds a float reduction",
    "_fused_rms_norm": "folds a float reduction",
    "rms_norm": "folds a float reduction",
    "group_norm": "folds a float reduction",
    "native_group_norm": "folds a float reduction",
    "mm": "matmul: batch-invariant dispatcher override must not be lowered past (compile_guard)",
    "addmm": "matmul: dispatcher override",
    "bmm": "matmul: dispatcher override",
    "baddbmm": "matmul: dispatcher override",
    "matmul": "matmul: dispatcher override",
    "linear": "matmul: dispatcher override",
    "einsum": "matmul-class",
    "mv": "matmul-class",
    "dot": "matmul-class (a dot product IS a float reduction)",
    "addr": "matmul-class",
    "convolution": "matmul-class",
    "conv1d": "matmul-class",
    "conv2d": "matmul-class",
    "scaled_dot_product_attention": "attention: owns reductions and its own kernel choice",
    "_scaled_dot_product_flash_attention": "attention",
    "grid_sampler_2d": "interpolation: multi-term FMA per element, association not pinned",
    "upsample_bilinear2d": "interpolation: multi-term accumulation per element",
    "addcmul": "double rounding vs fused: a*b+c*d has two products and one add-tree",
    "addcdiv": "double rounding hazard",
    "lerp": "a + w*(b-a): contraction-sensitive composite",
    "dropout": "RNG: philox offset bookkeeping differs eager vs compiled",
    "native_dropout": "RNG",
    "rand": "RNG",
    "randn": "RNG",
    "rand_like": "RNG",
    "randn_like": "RNG",
    "bernoulli": "RNG",
    "multinomial": "RNG + reduction",
}


@dataclass(frozen=True)
class OpVerdict:
    op_name: str  # overload-packet base name, e.g. "add"
    op_class: OpClass
    reason: str


def _base_name(target: Any) -> str | None:
    """Base (overload-packet) name of an fx call target, or None for non-aten targets."""
    packet = getattr(target, "overloadpacket", None)
    if packet is not None:
        name = getattr(packet, "__name__", None)
        if name:
            return name
    name = getattr(target, "__name__", None)
    if isinstance(name, str) and name:
        return name.split(".", 1)[0]
    return None


def _accumulates(node_target_name: str, args: tuple, kwargs: dict) -> bool:
    """Does this index_put/scatter-family call accumulate, making it a float reduction?"""
    if kwargs.get("accumulate"):
        return True
    if kwargs.get("reduce"):
        return True
    if node_target_name.startswith("index_put") and len(args) >= 4 and bool(args[3]):
        return True
    # scatter.reduce passes reduce as the 5th positional
    if node_target_name.startswith("scatter") and len(args) >= 5 and args[4]:
        return True
    return False


def classify_target(target: Any, args: tuple = (), kwargs: dict | None = None) -> OpVerdict:
    """Classify one fx call target. Fail-closed: anything unrecognized is BANNED."""
    kwargs = kwargs or {}
    name = _base_name(target)
    if name is None:
        return OpVerdict("<unresolvable>", OpClass.BANNED, "target has no resolvable aten name")

    # getitem on a multi-output node moves no bytes at all.
    if name in ("getitem", "getattr"):
        return OpVerdict(name, OpClass.SAFE, "tuple/attribute access, no data movement")

    # Reading a size rounds nothing; a size entering the float dataflow is caught by
    # pair.size_scalar_feeds_float, not here.
    if name.startswith("sym_"):
        return OpVerdict(name, OpClass.SAFE, "symbolic shape metadata, no data movement")

    # An in-place arithmetic op is not normalized to its functional spelling: mutation carries
    # storage ownership a local pointwise bit proof does not cover.
    if name.endswith("_") and name not in _MOVEMENT and name not in _MOVEMENT_IF_NOT_ACCUMULATE:
        return OpVerdict(
            name,
            OpClass.BANNED,
            "externally visible mutation requires an explicit ownership adapter",
        )

    base = name.rstrip("_") if name not in _MOVEMENT and name not in _MOVEMENT_IF_NOT_ACCUMULATE else name
    if base in _KNOWN_BANNED:
        return OpVerdict(base, OpClass.BANNED, _KNOWN_BANNED[base])
    if name in _MOVEMENT_IF_NOT_ACCUMULATE:
        if _accumulates(name, args, kwargs):
            return OpVerdict(name, OpClass.BANNED, "accumulating scatter: a float reduction in atomic order")
        return OpVerdict(name, OpClass.SAFE, "indexed copy, no accumulation")
    if name in _MOVEMENT or base in _MOVEMENT:
        return OpVerdict(name, OpClass.SAFE, "byte movement: no float arithmetic")
    if base in _EXACT_POINTWISE:
        return OpVerdict(base, OpClass.SAFE, "exactly-rounded IEEE-754 pointwise primitive")
    if base in _TRANSCENDENTAL:
        return OpVerdict(
            base,
            OpClass.CONDITIONAL,
            "transcendental: libdevice-vs-ATen bit equality must be proven by the sweep "
            "for this (arch, torch) before admission",
        )
    return OpVerdict(name, OpClass.BANNED, "not on the allowlist (fail-closed)")


def classify_graph(
    graph, *, admitted_transcendentals: frozenset[str] = frozenset()
) -> tuple[list[OpVerdict], list[OpVerdict]]:
    """Classify every call node of an fx graph, returning ``(all_verdicts, violations)``.

    A CONDITIONAL op named in ``admitted_transcendentals`` counts as SAFE; otherwise it is a
    violation, like a BANNED op. Any violation refuses the whole region -- there is no partial
    admission.
    """
    verdicts: list[OpVerdict] = []
    violations: list[OpVerdict] = []
    for node in graph.nodes:
        if node.op not in ("call_function", "call_method", "call_module"):
            continue
        if node.op == "call_module":
            v = OpVerdict(str(node.target), OpClass.BANNED, "opaque submodule call inside a region")
            verdicts.append(v)
            violations.append(v)
            continue
        v = classify_target(node.target, tuple(node.args), dict(node.kwargs))
        if v.op_class is OpClass.CONDITIONAL and v.op_name in admitted_transcendentals:
            v = OpVerdict(v.op_name, OpClass.SAFE, "transcendental admitted by sweep")
        verdicts.append(v)
        if v.op_class is not OpClass.SAFE:
            violations.append(v)
    return verdicts, violations


def fusable_op_count(verdicts: list[OpVerdict]) -> int:
    """How many ops in the region earn a fused kernel; views and getitem are free and do not count."""
    free = {
        "getitem",
        "getattr",
        "view",
        "_unsafe_view",
        "reshape",
        "permute",
        "transpose",
        "t",
        "squeeze",
        "unsqueeze",
        "expand",
        "expand_as",
        "slice",
        "select",
        "alias",
        "detach",
        "as_strided",
        "flatten",
        "unflatten",
        "movedim",
        "broadcast_to",
    }
    return sum(1 for v in verdicts if v.op_class is OpClass.SAFE and v.op_name not in free)
