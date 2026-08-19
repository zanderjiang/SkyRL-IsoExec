"""Admission for an explicit NCCL config on Megatron's incumbent process groups."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
from typing import Any, Callable

import torch
import torch.distributed as dist

CONFIG_ENV = "SKYRL_ISOEXEC_NCCL_INCUMBENT_CONFIG"
CONTRACT_ENV = "SKYRL_ISOEXEC_NCCL_INCUMBENT_CONTRACT"

_LEGACY_PLAN_ENVS = (
    "SKYRL_ISOEXEC_NCCL_CHANNEL_PLAN",
    "SKYRL_ISOEXEC_EP_A2A_CHANNELS",
)
_CAPABILITY_ENV = "SKYRL_ISOEXEC_NCCL_CAPABILITY_MODE"
_CAPABILITY_MANIFEST_ENV = "SKYRL_ISOEXEC_NCCL_CAPABILITY_MANIFEST"
_TRANSPORT_REQUIREMENTS_ENV = "SKYRL_ISOEXEC_NCCL_TRANSPORT_BOUNDARY_REQUIREMENTS"


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("an incumbent NCCL config requires the yaml package") from exc
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict) or not document:
        raise RuntimeError("incumbent NCCL config must be a non-empty YAML mapping")
    return document


def _load_contract(path: pathlib.Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    allowed = {
        "config_sha256",
        "capability_manifest_sha256",
        "transport_requirements_sha256",
        "process_channel_env",
        "groups",
        "memory_mib",
    }
    if not isinstance(document, dict) or set(document) != allowed:
        raise RuntimeError(f"incumbent NCCL contract keys must be exactly {sorted(allowed)!r}")
    return document


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{name} must be a finite non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0:
        raise RuntimeError(f"{name} must be a finite non-negative number")
    return result


def _canonical_contract(config_path: pathlib.Path, contract_path: pathlib.Path) -> tuple[dict, tuple]:
    config = _load_yaml(config_path)
    contract = _load_contract(contract_path)
    if _sha256(config_path) != contract["config_sha256"]:
        raise RuntimeError("incumbent NCCL config hash does not match its contract")

    capability_path = pathlib.Path(os.environ.get(_CAPABILITY_MANIFEST_ENV, "").strip())
    requirements_path = pathlib.Path(os.environ.get(_TRANSPORT_REQUIREMENTS_ENV, "").strip())
    if os.environ.get(_CAPABILITY_ENV, "off").strip().lower() != "enforce":
        raise RuntimeError(f"{CONFIG_ENV} requires {_CAPABILITY_ENV}=enforce")
    if not str(capability_path) or not capability_path.is_file():
        raise RuntimeError(f"{CONFIG_ENV} requires a readable {_CAPABILITY_MANIFEST_ENV}")
    if not str(requirements_path) or not requirements_path.is_file():
        raise RuntimeError(f"{CONFIG_ENV} requires a readable {_TRANSPORT_REQUIREMENTS_ENV}")
    if _sha256(capability_path) != contract["capability_manifest_sha256"]:
        raise RuntimeError("capability manifest hash does not match the incumbent contract")
    if _sha256(requirements_path) != contract["transport_requirements_sha256"]:
        raise RuntimeError("transport requirements hash does not match the incumbent contract")

    expected_env = contract["process_channel_env"]
    if not isinstance(expected_env, dict) or set(expected_env) != {"NCCL_MAX_NCHANNELS", "NCCL_MIN_NCHANNELS"}:
        raise RuntimeError("process_channel_env must declare exact NCCL MAX and MIN values")
    observed_env = {name: os.environ.get(name) for name in expected_env}
    if observed_env != expected_env:
        raise RuntimeError(f"process channel env mismatch observed={observed_env!r} expected={expected_env!r}")

    groups = contract["groups"]
    if not isinstance(groups, list) or not groups:
        raise RuntimeError("incumbent NCCL contract groups must be a non-empty list")
    canonical_groups = []
    config_keys = set()
    for index, row in enumerate(groups):
        required = {"config_key", "getter", "kwargs", "raw_min_ctas", "raw_max_ctas"}
        if not isinstance(row, dict) or set(row) != required:
            raise RuntimeError(f"invalid incumbent group row={index}: {row!r}")
        config_key = str(row["config_key"]).strip()
        getter = str(row["getter"]).strip()
        kwargs = row["kwargs"]
        raw_min = row["raw_min_ctas"]
        raw_max = row["raw_max_ctas"]
        if not config_key or not getter or not isinstance(kwargs, dict):
            raise RuntimeError(f"invalid incumbent group row={index}: {row!r}")
        if isinstance(raw_min, bool) or isinstance(raw_max, bool):
            raise RuntimeError(f"invalid incumbent group CTA row={index}: {row!r}")
        if not isinstance(raw_min, int) or not isinstance(raw_max, int) or raw_min < 1 or raw_max < raw_min:
            raise RuntimeError(f"invalid incumbent group CTA row={index}: {row!r}")
        if config_key in config_keys:
            raise RuntimeError(f"duplicate incumbent config_key={config_key!r}")
        config_keys.add(config_key)
        config_row = config.get(config_key)
        if config_row != {"min_ctas": raw_min, "max_ctas": raw_max}:
            raise RuntimeError(
                f"incumbent YAML row {config_key!r}={config_row!r} does not match "
                f"contract min/max={raw_min}/{raw_max}"
            )
        canonical_groups.append((config_key, getter, tuple(sorted(kwargs.items())), raw_min, raw_max))
    if set(config) != config_keys:
        raise RuntimeError(
            f"every incumbent YAML key must have one group contract: {sorted(set(config) ^ config_keys)!r}"
        )

    memory = contract["memory_mib"]
    memory_keys = {
        "max_incremental_nontorch",
        "max_absolute_nontorch_after",
        "min_device_free_after",
    }
    if not isinstance(memory, dict) or set(memory) != memory_keys:
        raise RuntimeError(f"memory_mib keys must be exactly {sorted(memory_keys)!r}")
    canonical_memory = tuple(sorted((key, _positive_finite(value, key)) for key, value in memory.items()))
    signature = (
        str(config_path.resolve()),
        _sha256(config_path),
        str(contract_path.resolve()),
        _sha256(contract_path),
        tuple(sorted(expected_env.items())),
        tuple(sorted(canonical_groups)),
        canonical_memory,
    )
    return contract, signature


def resolve_preinit_config(control_gather: Callable[[str, Any], list[Any]]) -> str | None:
    """WORLD-vote and return an explicit incumbent-group YAML, or ``None`` when off."""

    config_raw = os.environ.get(CONFIG_ENV, "").strip()
    contract_raw = os.environ.get(CONTRACT_ENV, "").strip()
    requested = bool(config_raw or contract_raw)
    local_error = ""
    signature = ()
    resolved = None
    if requested:
        try:
            if not config_raw or not contract_raw:
                raise RuntimeError(f"{CONFIG_ENV} and {CONTRACT_ENV} must be set together")
            conflicts = [name for name in _LEGACY_PLAN_ENVS if os.environ.get(name, "").strip() not in ("", "0")]
            if conflicts:
                raise RuntimeError(f"incumbent config cannot compose with legacy movement plans: {conflicts!r}")
            config_path = pathlib.Path(config_raw)
            contract_path = pathlib.Path(contract_raw)
            if not config_path.is_file() or not contract_path.is_file():
                raise RuntimeError("incumbent config and contract paths must be readable files")
            _, signature = _canonical_contract(config_path, contract_path)
            resolved = str(config_path.resolve())
        except Exception as exc:  # noqa: BLE001 - vote before any subgroup exists
            local_error = f"rank={dist.get_rank()} {type(exc).__name__}: {exc}"
    reports = control_gather(
        "nccl-incumbent-config-preinit-v1",
        {
            "rank": dist.get_rank(),
            "requested": requested,
            "signature": signature,
            "error": local_error,
        },
    )
    world = dist.get_world_size()
    errors = [str(report.get("error")) for report in reports if report.get("error")]
    ranks = sorted(int(report.get("rank", -1)) for report in reports)
    signatures = {(bool(report.get("requested")), tuple(report.get("signature", ()))) for report in reports}
    if ranks != list(range(world)):
        errors.append(f"incumbent preinit ranks must be WORLD exactly once, got {ranks!r}")
    if len(signatures) != 1:
        errors.append(f"rank-nonunanimous incumbent request {sorted(map(repr, signatures))!r}")
    if errors:
        raise RuntimeError("[ISOEXEC-NCCL-INCUMBENT] PREINIT REFUSAL: " + "; ".join(errors))
    if not requested:
        return None
    print(
        f"[ISOEXEC-NCCL-INCUMBENT] PREINIT unanimous ranks={world} config={resolved} "
        f"signature_sha256={signature[1]}",
        flush=True,
    )
    return resolved


def _readback(group, field: str) -> int | None:
    candidates = (group, getattr(group, "_get_backend", lambda _device: None)(torch.device("cuda")))
    for candidate in candidates:
        value = getattr(getattr(getattr(candidate, "options", None), "config", None), field, None)
        if value is not None:
            return int(value)
    return None


def verify_postflight(mpu, prewarm_report: dict | None, control_gather: Callable[[str, Any], list[Any]]) -> dict:
    """Hard-gate requested PG readbacks and post-prewarm residency on every rank."""

    config_raw = os.environ.get(CONFIG_ENV, "").strip()
    contract_raw = os.environ.get(CONTRACT_ENV, "").strip()
    if not config_raw and not contract_raw:
        return {}
    local_error = ""
    local = {"rank": dist.get_rank(), "groups": (), "memory": {}}
    try:
        if not config_raw or not contract_raw:
            raise RuntimeError(f"{CONFIG_ENV} and {CONTRACT_ENV} must be set together")
        contract, _ = _canonical_contract(pathlib.Path(config_raw), pathlib.Path(contract_raw))
        if not isinstance(prewarm_report, dict) or not prewarm_report:
            raise RuntimeError("incumbent config requires a completed ownership-aware prewarm report")
        group_rows = []
        for row in contract["groups"]:
            getter = getattr(mpu, row["getter"], None)
            if getter is None or not callable(getter):
                raise RuntimeError(f"unknown Megatron group getter={row['getter']!r}")
            kwargs = dict(row["kwargs"])
            if row["getter"] in ("get_embedding_group", "get_position_embedding_group"):
                kwargs = {"check_initialized": False, **kwargs}
            group = getter(**kwargs)
            if group is None or dist.get_world_size(group) < 2:
                raise RuntimeError(f"requested incumbent group did not resolve active: {row['getter']}")
            raw_min = _readback(group, "min_ctas")
            raw_max = _readback(group, "max_ctas")
            if raw_min != row["raw_min_ctas"] or raw_max != row["raw_max_ctas"]:
                raise RuntimeError(
                    f"channel readback mismatch getter={row['getter']} raw_min/max={raw_min}/{raw_max} "
                    f"expected={row['raw_min_ctas']}/{row['raw_max_ctas']}"
                )
            group_rows.append(
                {
                    "config_key": row["config_key"],
                    "getter": row["getter"],
                    "members": tuple(dist.get_process_group_ranks(group)),
                    "raw_min_ctas": raw_min,
                    "raw_max_ctas": raw_max,
                }
            )
        memory_contract = contract["memory_mib"]
        free_mib = torch.cuda.mem_get_info()[0] / (1024**2)
        memory = {
            "incremental_nontorch_mib": float(prewarm_report["incremental_nontorch_mib"]),
            "absolute_nontorch_after_mib": float(prewarm_report["absolute_nontorch_after_mib"]),
            "device_free_after_mib": free_mib,
        }
        if memory["incremental_nontorch_mib"] > memory_contract["max_incremental_nontorch"]:
            raise RuntimeError("incremental non-Torch residency exceeds incumbent contract")
        if memory["absolute_nontorch_after_mib"] > memory_contract["max_absolute_nontorch_after"]:
            raise RuntimeError("absolute non-Torch residency exceeds incumbent contract")
        if memory["device_free_after_mib"] < memory_contract["min_device_free_after"]:
            raise RuntimeError("device headroom is below incumbent contract")
        local.update(groups=tuple(group_rows), memory=memory)
    except Exception as exc:  # noqa: BLE001 - all ranks vote their local readback/memory verdict
        local_error = f"rank={dist.get_rank()} {type(exc).__name__}: {exc}"
        local["error"] = local_error
    reports = control_gather("nccl-incumbent-postflight-v1", local)
    world = dist.get_world_size()
    errors = [str(report.get("error")) for report in reports if report.get("error")]
    ranks = sorted(int(report.get("rank", -1)) for report in reports)
    if ranks != list(range(world)):
        errors.append(f"incumbent postflight ranks must be WORLD exactly once, got {ranks!r}")
    if not errors:
        # Memory is rank-local and need not be byte-identical. Group contracts and each subgroup's
        # member readbacks are checked below; only configuration identity was exact at preinit.
        by_group: dict[tuple, dict[int, tuple[int, int]]] = {}
        for report in reports:
            rank = int(report["rank"])
            for row in report["groups"]:
                key = (row["config_key"], row["getter"], tuple(row["members"]))
                by_group.setdefault(key, {})[rank] = (row["raw_min_ctas"], row["raw_max_ctas"])
        for key, readbacks in by_group.items():
            members = key[2]
            if tuple(sorted(readbacks)) != tuple(sorted(members)):
                errors.append(f"group={key!r} readback reporters={sorted(readbacks)!r} != members={members!r}")
            elif len(set(readbacks.values())) != 1:
                errors.append(f"group={key!r} rank-nonunanimous readback={readbacks!r}")
    if errors:
        raise RuntimeError("[ISOEXEC-NCCL-INCUMBENT] POSTFLIGHT REFUSAL: " + "; ".join(errors))
    result = {
        "groups": tuple(report["groups"] for report in reports),
        "memory": tuple(report["memory"] for report in reports),
    }
    if dist.get_rank() == 0:
        print(
            f"[ISOEXEC-NCCL-INCUMBENT] POSTFLIGHT_PASS ranks={world} " f"requested_groups={len(contract['groups'])}",
            flush=True,
        )
    return result
