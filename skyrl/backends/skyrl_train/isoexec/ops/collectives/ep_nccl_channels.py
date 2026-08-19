"""Per-communicator NCCL channel counts: the Megatron YAML, and the budget that gates writing it.

``SKYRL_ISOEXEC_NCCL_CHANNEL_PLAN`` (e.g. ``"ep:8"``) is the only production entry point; see
``nccl_channel_budget.py`` for the pricing model. A per-group ``max_ctas`` overrides the process-wide
``NCCL_MAX_NCHANNELS``, which is what makes "narrow process, wide EP group" expressible, and widening is
bit-safe only where the group performs no arithmetic -- an all-to-all copies each output byte from exactly one
input byte, so channel count changes routing and never values. MOVE_ONLY entries never enter the YAML: they
become dedicated same-membership duplicate groups, because Megatron's incumbent EP group also carries IsoExec's
safety MIN reduction. Rank-asymmetric plans, topology, group creation, readback, memory, service, wrapper
ownership, or teardown evidence all kill the run rather than degrading silently.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
from dataclasses import replace

from .nccl_channel_budget import (
    MOVE_ONLY,
    Verdict,
    admit,
    parse_plan,
)

_ENV_CHANNELS = "SKYRL_ISOEXEC_EP_A2A_CHANNELS"
_ENV_GROUPS = "SKYRL_ISOEXEC_EP_A2A_CHANNEL_GROUPS"
_ENV_PLAN = "SKYRL_ISOEXEC_NCCL_CHANNEL_PLAN"
_ENV_BUDGET = "SKYRL_ISOEXEC_NCCL_CHANNEL_BUDGET_GIB"
_ENV_ACK_REDUCE = "SKYRL_ISOEXEC_NCCL_CHANNEL_ACK_REDUCE"

# Only groups that permute bytes may be widened here. `ep_tp` (expert TP reduction) and `ep_dp` (expert
# gradient reduction) are not pure permutations, so widening them is a reduction-order question.
_DEFAULT_GROUPS = ("ep",)
_REDUCING_GROUPS = frozenset({"ep_tp", "ep_dp", "tp", "dp", "dp_cp", "mp", "embd", "pos_embd"})

_WRITTEN: str | None = None

#: Set by `nccl_config_path` when the budgeted plan path is taken, read by `verify_group_channels`
#: to turn the projection into a measurement. `None` means the plan path was not used at all.
_VERDICT: Verdict | None = None

_GROUP_GETTERS = {
    "ep": "get_expert_model_parallel_group",
    "tp_ep": "get_expert_tensor_and_model_parallel_group",
    "tp": "get_tensor_model_parallel_group",
}
_MOVE_GUARD_INSTALLED = False
_MOVE_GUARD_GROUPS: dict[int, str] = {}
_MOVE_GUARD_COUNTS: dict[tuple[str, str], int] = {}
_MOVE_OPERATIONS = {
    "all_to_all_single": 4,
    "all_to_all": 2,
    "all_gather_into_tensor": 2,
    "_all_gather_base": 2,
    "all_gather": 2,
    "broadcast": 2,
}
_REDUCE_OPERATIONS = {
    "all_reduce": 2,
    "all_reduce_coalesced": 2,
    "reduce": 3,
    "reduce_scatter_tensor": 3,
    "_reduce_scatter_base": 3,
    "reduce_scatter": 3,
}


def move_only_operation_verdict(operation: str) -> str:
    """Classify a torch.distributed operation for a widened move-only communicator."""
    return "serve" if operation in _MOVE_OPERATIONS else "refuse"


def validate_postflight_evidence(
    verdict: Verdict,
    evidence: dict[str, tuple[int | None, int, float | None]],
) -> str:
    """Validate ``(max_ctas readback, prewarm serves, non-torch GiB)`` per admitted group."""
    errors: list[str] = []
    measured_total = 0.0
    rows: list[str] = []
    for plan in verdict.admitted:
        readback, served, measured_gib = evidence.get(plan.name, (None, 0, None))
        if readback != plan.channels:
            errors.append(f"{plan.name}: requested max_ctas={plan.channels}, read back {readback!r}")
        if served < 1:
            errors.append(f"{plan.name}: lazy-pattern prewarm served={served}; expected >=1")
        if measured_gib is None or measured_gib < 0:
            errors.append(f"{plan.name}: non-torch communicator charge unavailable")
        else:
            measured_total += measured_gib
        rows.append(
            f"{plan.name}:requested={plan.channels} readback={readback} "
            f"prewarm_served={served} measured={measured_gib!r}GiB"
        )
    if measured_total > verdict.budget_gib:
        errors.append(
            f"measured full lazy-pattern charge {measured_total:.3f} GiB/rank exceeds "
            f"declared budget {verdict.budget_gib:.3f} GiB/rank"
        )
    head = "[ISOEXEC-NCCL-BUDGET] POSTFLIGHT " + " | ".join(rows)
    if errors:
        raise RuntimeError(head + "\n[ISOEXEC-NCCL-BUDGET] REFUSING TO CONTINUE: " + "; ".join(errors))
    return head + f" total={measured_total:.3f}GiB <= budget={verdict.budget_gib:.3f}GiB OK"


def validate_plan_signatures(signatures: list[tuple[tuple[str, int, int], ...]]) -> None:
    """Refuse if ranks did not construct the exact same named channel plan."""
    unique = set(signatures)
    if len(unique) != 1:
        raise RuntimeError(f"rank-nonunanimous plan signatures: {sorted(unique)!r}")


def channel_plan_requested() -> bool:
    """Whether either per-communicator configuration path was explicitly requested."""
    return bool(os.environ.get(_ENV_PLAN, "").strip() or ep_channels() > 0)


def preinit_request_signature(worlds: dict[str, int] | None) -> tuple:
    """Every configuration input that must agree before Megatron constructs process groups."""
    return (
        os.environ.get(_ENV_PLAN, ""),
        os.environ.get(_ENV_BUDGET, "0"),
        os.environ.get(_ENV_ACK_REDUCE, "0"),
        os.environ.get(_ENV_CHANNELS, "0"),
        os.environ.get(_ENV_GROUPS, ""),
        os.environ.get("NCCL_MAX_NCHANNELS", ""),
        os.environ.get("NCCL_MIN_NCHANNELS", ""),
        os.environ.get("NCCL_ALGO", ""),
        os.environ.get("SKYRL_ISOEXEC_NCCL_PIN", "1"),
        tuple(sorted((worlds or {}).items())),
    )


def config_content_hash(path: str | None) -> str | None:
    """Path-independent identity of YAML plus duplicate-transport plans."""
    if path is None and (_VERDICT is None or not _VERDICT.any_admitted):
        return None
    payload = {
        "yaml_sha256": None if path is None else hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest(),
        "admitted": (
            []
            if _VERDICT is None
            else [(plan.name, plan.channels, plan.world, plan.baseline_channels) for plan in _VERDICT.admitted]
        ),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def channel_plan_admitted() -> bool:
    """Whether the budget admitted any existing or duplicate communicator."""
    return _VERDICT is not None and _VERDICT.any_admitted


def validate_preinit_reports(reports: list[dict]) -> str:
    """Require unanimous request, verdict, and YAML content before PG construction."""
    errors = [report["error"] for report in reports if report.get("error")]
    signatures = {(report.get("request"), report.get("admitted"), report.get("content_hash")) for report in reports}
    if len(signatures) != 1:
        errors.append(f"rank-nonunanimous preinit reports: {sorted(map(repr, signatures))!r}")
    if errors:
        raise RuntimeError("[ISOEXEC-NCCL-BUDGET] PREINIT REFUSAL: " + "; ".join(errors))
    request, admitted, content_hash = next(iter(signatures))
    return (
        f"[ISOEXEC-NCCL-BUDGET] PREINIT unanimous ranks={len(reports)} admitted={admitted} "
        f"content_sha256={content_hash} request={request}"
    )


def move_only_group_names() -> frozenset[str]:
    """Admitted groups whose only legal runtime role is byte movement."""
    if _VERDICT is None:
        return frozenset()
    return frozenset(plan.name for plan in _VERDICT.admitted if plan.contract == MOVE_ONLY)


def move_only_group_getters() -> frozenset[str]:
    """Incumbent groups are never move-only; movement uses dedicated duplicates."""
    return frozenset()


def derive_worlds(
    *,
    world_size: int,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    expert_tensor_parallel_size: int | None = None,
) -> dict[str, int]:
    """Megatron's process-group name -> world size, from the parallelism config.

    This reproduces Megatron's own arithmetic rather than importing it, so a plan can be priced on CPU and
    before ``initialize_model_parallel`` has run.
    """
    tp = max(1, tensor_model_parallel_size)
    pp = max(1, pipeline_model_parallel_size)
    cp = max(1, context_parallel_size)
    ep = max(1, expert_model_parallel_size)
    etp = tp if expert_tensor_parallel_size is None else max(1, expert_tensor_parallel_size)
    dp = max(1, world_size // (tp * pp * cp))
    return {
        "tp": tp,
        "pp": pp,
        "cp": cp,
        "ep": ep,
        "ep_tp": etp,
        # The expert data-parallel group is what is left of the world after ep x etp x pp.
        "ep_dp": max(1, world_size // (ep * etp * pp)),
        # The dispatcher's preprocess AllGather rides this one (`EXPERT_TENSOR_AND_MODEL_PARALLEL`).
        "tp_ep": etp * ep,
        "dp": dp,
        "dp_cp": dp * cp,
        "mp": tp * pp,
        "embd": pp,
        "pos_embd": pp,
    }


def _budget_gib() -> float:
    try:
        return max(0.0, float(os.environ.get(_ENV_BUDGET, "0").strip() or "0"))
    except ValueError:
        return 0.0


def _free_gib() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return free / 1024**3
    except Exception:  # noqa: BLE001 - a missing CUDA context must not kill the guard
        return None


def budgeted_plan(worlds: dict[str, int]) -> Verdict | None:
    """Parse, price, and admit ``SKYRL_ISOEXEC_NCCL_CHANNEL_PLAN``; ``None`` means the flag is not set.

    Separated from ``nccl_config_path`` so the decision depends only on the environment, the world sizes, and
    optionally a free-memory reading.
    """
    spec = os.environ.get(_ENV_PLAN, "").strip()
    if not spec:
        return None
    plans = parse_plan(spec, worlds=worlds)
    # A MOVE_ONLY purchase is a fresh communicator, so none of the incumbent group's channels offset it.
    plans = tuple(replace(plan, baseline_channels=0) if plan.contract == MOVE_ONLY else plan for plan in plans)
    verdict = admit(
        plans,
        budget_gib=_budget_gib(),
        free_gib=_free_gib(),
        # The reserve is left to the operator: only the launcher knows the KV pool size.
        reserve_gib=0.0,
        ack_reduce=os.environ.get(_ENV_ACK_REDUCE, "0") == "1",
    )
    print(verdict.banner, flush=True)
    return verdict


def ep_channels() -> int:
    """Requested channel count for the expert groups. 0 (default) = do nothing, exactly today."""
    try:
        return max(0, int(os.environ.get(_ENV_CHANNELS, "0")))
    except ValueError:
        return 0


def ep_channel_groups() -> tuple[str, ...]:
    spec = os.environ.get(_ENV_GROUPS, "").strip()
    if not spec:
        return _DEFAULT_GROUPS
    return tuple(g.strip() for g in spec.split(",") if g.strip())


def _write_yaml(cfg: dict[str, dict[str, int]], tag: str) -> str | None:
    """Write the per-group NCCL YAML megatron reads, or ``None`` (= today) if anything goes wrong."""
    global _WRITTEN
    try:
        import yaml  # megatron reads the file with yaml.safe_load, so require the same parser

        d = pathlib.Path(os.environ.get("SKYRL_ISOEXEC_EP_NCCL_DIR", tempfile.gettempdir()))
        d.mkdir(parents=True, exist_ok=True)
        # One file per content, so all local ranks agree on the path.
        path = d / f"isoexec_nccl_comm_{tag}.yaml"
        # The write must be atomic: every local rank writes this same path, and a truncating write lets a
        # concurrent reader see an empty file, which Megatron loads as None rather than {} and then indexes.
        body = yaml.safe_dump(cfg)
        if not body.strip() or yaml.safe_load(body) is None:
            raise ValueError(f"refusing to write a YAML megatron would load as None: {body!r}")
        # mkstemp, not a pid-derived name: two writers sharing a pid would collide on the temp file.
        fd, tmp_name = tempfile.mkstemp(dir=str(d), prefix=f".isoexec_nccl_comm_{tag}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(body)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            pathlib.Path(tmp_name).unlink(missing_ok=True)
            raise
        if yaml.safe_load(path.read_text()) is None:  # read-back: never hand megatron a None
            raise ValueError(f"{path} parses as None after write")
        _WRITTEN = str(path)
        print(
            f"[ISOEXEC-EP-NCCL] per-group NCCL channels ON: {cfg} -> {path}. The process-wide "
            f"NCCL_MAX_NCHANNELS={os.environ.get('NCCL_MAX_NCHANNELS', '(unset)')} still governs "
            f"every OTHER communicator. What each widened group's channel count can and cannot "
            f"move is the [ISOEXEC-NCCL-BUDGET] contract line above, per group -- not a blanket "
            f"claim here.",
            flush=True,
        )
        return _WRITTEN
    except Exception as e:  # noqa: BLE001
        print(
            f"[ISOEXEC-EP-NCCL] DISABLED: could not write the per-group NCCL config "
            f"({type(e).__name__}: {e}). Falling back to the process-wide setting -- i.e. exactly "
            f"today's behaviour, no bits and no shapes move.",
            flush=True,
        )
        return None


def nccl_config_path(worlds: dict[str, int] | None = None) -> str | None:
    """Write (once) the per-group NCCL YAML and return its path, or ``None`` to keep the default.

    MOVE_ONLY entries do not enter the YAML; they are built as dedicated duplicates after Megatron initializes
    its logical groups. ``worlds`` comes from ``derive_worlds``; without it nothing can be priced and the
    budgeted path declines. Idempotent, so every rank in the process reads one file.
    """
    global _VERDICT
    if _WRITTEN is not None:
        return _WRITTEN

    if worlds:
        verdict = budgeted_plan(worlds)
        if verdict is not None:
            _VERDICT = verdict
            if not verdict.any_admitted:
                return None
            if os.environ.get(_ENV_CHANNELS, "0") not in ("", "0"):
                print(
                    f"[ISOEXEC-NCCL-BUDGET] NOTE: {_ENV_PLAN} is set, so {_ENV_CHANNELS} is IGNORED. "
                    f"One of the two decides the config, never both.",
                    flush=True,
                )
            # Never put MOVE_ONLY plans in this YAML: that would widen the incumbent logical group, which
            # carries the bf16-wire safety MIN vote.
            existing = tuple(p for p in verdict.admitted if p.contract != MOVE_ONLY)
            if not existing:
                print(
                    "[ISOEXEC-NCCL-MOVE] plan admitted duplicate transport only; Megatron's "
                    "nccl_communicator_config_path remains None so logical groups stay pinned",
                    flush=True,
                )
                return None
            cfg = {p.name: {"max_ctas": p.channels, "min_ctas": p.channels} for p in existing}
            tag = "plan_" + "_".join(f"{p.name}{p.channels}" for p in existing)
            return _write_yaml(cfg, tag)

    n = ep_channels()
    if n <= 0:
        return None

    raise RuntimeError(
        f"{_ENV_CHANNELS} directly widens the incumbent EP communicator, which carries the "
        f"moe_a2a_wire safety reduction. Use {_ENV_PLAN}=ep:<channels> to create a dedicated "
        "same-membership movement communicator."
    )


def reset_channel_plan_state() -> None:
    """Clear per-process admission state when Megatron tears down for a fresh lifecycle."""
    global _WRITTEN, _VERDICT
    _WRITTEN = None
    _VERDICT = None


def _readback_max_ctas(group) -> int | None:
    import torch

    candidates = (group, getattr(group, "_get_backend", lambda _device: None)(torch.device("cuda")))
    for candidate in candidates:
        config = getattr(getattr(candidate, "options", None), "config", None)
        value = getattr(config, "max_ctas", None)
        if value is not None:
            return int(value)
    return None


def _report_charge_gib(prewarm_report: dict | None, getter: str) -> float | None:
    if not prewarm_report:
        return None
    matches = [row for row in prewarm_report.get("nontorch_groups", ()) if row[0] == getter]
    if len(matches) != 1:
        return None
    return float(matches[0][1]) / 1024


def _group_from_call(args, kwargs, position: int):
    if "group" in kwargs:
        return kwargs["group"]
    return args[position] if len(args) > position else None


def _install_move_only_guard(groups: dict[str, object]) -> None:
    """Count copy service and reject any later arithmetic collective on a move-only group."""
    global _MOVE_GUARD_INSTALLED
    if not groups or _MOVE_GUARD_INSTALLED:
        return

    import torch.distributed as dist

    _MOVE_GUARD_GROUPS.update((id(group), name) for name, group in groups.items())

    def wrap_move(name: str, group_position: int) -> None:
        original = getattr(dist, name, None)
        if original is None:
            return

        def guarded(*args, **kwargs):
            group_name = _MOVE_GUARD_GROUPS.get(id(_group_from_call(args, kwargs, group_position)))
            if group_name is not None:
                key = (group_name, name)
                count = _MOVE_GUARD_COUNTS.get(key, 0) + 1
                _MOVE_GUARD_COUNTS[key] = count
                if count == 1 or count & (count - 1) == 0:
                    print(
                        f"[ISOEXEC-NCCL-MOVE] CENSUS group={group_name} op={name} served={count} "
                        "arithmetic_declined=0",
                        flush=True,
                    )
            return original(*args, **kwargs)

        setattr(dist, name, guarded)

    def wrap_reduce(name: str, group_position: int) -> None:
        original = getattr(dist, name, None)
        if original is None:
            return

        def guarded(*args, **kwargs):
            group_name = _MOVE_GUARD_GROUPS.get(id(_group_from_call(args, kwargs, group_position)))
            if group_name is not None:
                print(
                    f"[ISOEXEC-NCCL-MOVE] DRIFT group={group_name} op={name} "
                    "arithmetic_declined=1 -- REFUSING TO CONTINUE",
                    flush=True,
                )
                raise RuntimeError(f"move-only NCCL group {group_name!r} received arithmetic collective {name!r}")
            return original(*args, **kwargs)

        setattr(dist, name, guarded)

    for name, position in _MOVE_OPERATIONS.items():
        wrap_move(name, position)
    for name, position in _REDUCE_OPERATIONS.items():
        wrap_reduce(name, position)
    _MOVE_GUARD_INSTALLED = True
    print(
        f"[ISOEXEC-NCCL-MOVE] installed move-only operation guard groups={sorted(groups)}; "
        "copy calls are counted and any torch.distributed reduction fails closed",
        flush=True,
    )


def verify_group_channels(prewarm_report: dict | None = None) -> None:
    """Verify requested channels and memory after every lazy collective pattern has run."""
    if _VERDICT is None or not _VERDICT.any_admitted:
        return

    import torch.distributed as dist
    from megatron.core import parallel_state as mpu

    evidence: dict[str, tuple[int | None, int, float | None]] = {}
    local_error = ""
    try:
        if os.environ.get("SKYRL_ISOEXEC_NCCL_PREWARM", "0") != "1":
            raise RuntimeError("an admitted channel plan requires SKYRL_ISOEXEC_NCCL_PREWARM=1")
        for plan in _VERDICT.admitted:
            getter = _GROUP_GETTERS.get(plan.name)
            if getter is None:
                evidence[plan.name] = (None, 0, None)
                continue
            if plan.contract == MOVE_ONLY:
                from .ep_move_group import transport_evidence

                evidence[plan.name] = transport_evidence(plan.name, prewarm_report)
            else:
                group = getattr(mpu, getter)()
                charge = _report_charge_gib(prewarm_report, getter)
                evidence[plan.name] = (_readback_max_ctas(group), int(charge is not None), charge)
        banner = validate_postflight_evidence(_VERDICT, evidence)
    except Exception as exc:  # noqa: BLE001 - every rank must reach the unanimous refusal
        banner = ""
        local_error = f"rank={dist.get_rank()} {type(exc).__name__}: {exc}"

    signature = tuple((plan.name, plan.channels, plan.world) for plan in _VERDICT.admitted)
    local_report = {"rank": dist.get_rank(), "signature": signature, "error": local_error}
    reports: list[dict | None] = [None] * dist.get_world_size()
    dist.all_gather_object(reports, local_report)
    errors = [report["error"] for report in reports if report and report["error"]]
    try:
        validate_plan_signatures([report["signature"] for report in reports if report])
    except RuntimeError as exc:
        errors.append(str(exc))
    if errors:
        raise RuntimeError("[ISOEXEC-NCCL-BUDGET] unanimous postflight refusal:\n" + "\n".join(errors))
    print(banner, flush=True)
    from .ep_move_group import install_router

    install_router()


def initialize_movement_groups(control_gather) -> None:
    """Construct admitted duplicate transport groups after Megatron creates logical groups."""
    if _VERDICT is None or not _VERDICT.any_admitted:
        return
    from .ep_move_group import initialize_movement_groups as initialize

    initialize(_VERDICT.admitted, control_gather)
