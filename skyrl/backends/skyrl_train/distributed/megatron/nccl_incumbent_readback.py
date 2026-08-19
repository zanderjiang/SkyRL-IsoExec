"""Local post-init census for MCore's original TP and EP NCCL groups.

This module deliberately does less than the optional incumbent admission
contract.  It creates no process groups, sends no collective or Store traffic,
warms no transport, and makes no memory decision.  It only verifies that the
``min_ctas``/``max_ctas`` values retained by already-created ProcessGroupNCCL
objects match the rows MCore was asked to consume.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from typing import Any

import torch
import torch.distributed as dist

CONFIG_SHA256_ENV = "SKYRL_ISOEXEC_NCCL_INCUMBENT_CONFIG_SHA256"

_UNSET_CTA = -(2**31)
_TARGETS = {
    "tp": "get_tensor_model_parallel_group",
    "ep": "get_expert_model_parallel_group",
}


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_targets(path: pathlib.Path) -> dict[str, dict[str, int]]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("NCCL communicator readback requires the yaml package") from exc

    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict):
        raise RuntimeError("NCCL communicator config must be a YAML mapping")
    targets: dict[str, dict[str, int]] = {}
    for key in _TARGETS:
        if key not in document:
            continue
        row = document[key]
        if not isinstance(row, dict):
            raise RuntimeError(f"configured target {key!r} must be a mapping")
        requested: dict[str, int] = {}
        for field in ("min_ctas", "max_ctas"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise RuntimeError(f"configured target {key!r} requires positive integer {field}")
            requested[field] = value
        if requested["max_ctas"] < requested["min_ctas"]:
            raise RuntimeError(f"configured target {key!r} has max_ctas below min_ctas")
        targets[key] = requested
    return targets


def _readback(group: Any, field: str) -> int | None:
    """Read the raw ProcessGroupNCCL option without initializing any transport."""

    candidates = [group]
    get_backend = getattr(group, "_get_backend", None)
    if callable(get_backend):
        backend = get_backend(torch.device("cuda"))
        if backend is not None and backend is not group:
            candidates.insert(0, backend)
    for candidate in candidates:
        value = getattr(getattr(getattr(candidate, "options", None), "config", None), field, None)
        if value is not None:
            return int(value)
    return None


def _effective(raw: int | None, env_name: str) -> int | None:
    if raw is not None and raw != _UNSET_CTA:
        return raw
    env_value = os.environ.get(env_name, "").strip()
    if not env_value:
        return None
    try:
        return int(env_value)
    except ValueError:
        return None


def census_original_tp_ep(mpu: Any, config_path: str | None) -> tuple[dict[str, Any], ...]:
    """Verify configured original TP/EP groups locally, immediately after PG init.

    Every trainer process calls this function and emits its own complete census.
    Cross-rank agreement follows from each member checking the same YAML request
    against its own retained ProcessGroupNCCL options; intentionally no new
    synchronization or communication is introduced here.
    """

    if not config_path:
        return ()
    path = pathlib.Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"[ISOEXEC-NCCL-MCORE] configured YAML is not readable: {path}")

    observed_sha = _sha256(path)
    expected_sha = os.environ.get(CONFIG_SHA256_ENV, "").strip().lower()
    if expected_sha and observed_sha != expected_sha:
        raise RuntimeError(f"[ISOEXEC-NCCL-MCORE] config hash mismatch observed={observed_sha} expected={expected_sha}")

    try:
        targets = _load_targets(path)
    except Exception as exc:  # noqa: BLE001 - normalize the startup refusal banner
        raise RuntimeError(f"[ISOEXEC-NCCL-MCORE] configured target is invalid: {exc}") from exc

    rows: list[dict[str, Any]] = []
    for key, requested in targets.items():
        getter_name = _TARGETS[key]
        getter = getattr(mpu, getter_name, None)
        if not callable(getter):
            raise RuntimeError(f"[ISOEXEC-NCCL-MCORE] missing configured target getter={getter_name}")
        try:
            group = getter()
        except Exception as exc:  # noqa: BLE001 - report a missing configured target uniformly
            raise RuntimeError(f"[ISOEXEC-NCCL-MCORE] missing configured target key={key}: {exc}") from exc
        if group is None:
            raise RuntimeError(f"[ISOEXEC-NCCL-MCORE] missing configured target key={key}")

        raw_min = _readback(group, "min_ctas")
        raw_max = _readback(group, "max_ctas")
        members = tuple(int(rank) for rank in dist.get_process_group_ranks(group))
        row = {
            "key": key,
            "getter": getter_name,
            "members": members,
            "requested_min_ctas": requested["min_ctas"],
            "requested_max_ctas": requested["max_ctas"],
            "raw_min_ctas": raw_min,
            "raw_max_ctas": raw_max,
            "effective_min_ctas": _effective(raw_min, "NCCL_MIN_NCHANNELS"),
            "effective_max_ctas": _effective(raw_max, "NCCL_MAX_NCHANNELS"),
        }
        if raw_min != requested["min_ctas"] or raw_max != requested["max_ctas"]:
            raise RuntimeError(
                "[ISOEXEC-NCCL-MCORE] channel readback mismatch "
                + json.dumps(row, sort_keys=True, separators=(",", ":"))
            )
        rows.append(row)

    rank = dist.get_rank()
    print(
        "[ISOEXEC-NCCL-MCORE] CENSUS_PASS "
        + json.dumps(
            {
                "rank": rank,
                "config": str(path),
                "config_sha256": observed_sha,
                "groups": rows,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return tuple(rows)
