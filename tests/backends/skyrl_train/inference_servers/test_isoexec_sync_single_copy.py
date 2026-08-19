"""CPU tests for the IsoExec weight-sync single-copy path (2026-08-13).

These pin the behaviours the sync fix depends on, none of which needs a GPU. They are
SOURCE-level assertions on purpose: the code they guard lives in a vLLM worker-extension class and
a Megatron worker, neither of which is importable without those runtimes, and the properties at
stake are structural (which default a flag reads, which branch a call sits in) rather than
numerical.

1. **The double-copy path stays retired.** ``SKYRL_ISOEXEC_LAYERWISE_RELOAD`` and
   ``SKYRL_ISOEXEC_REAPPLY_CACHE`` used to select it and have since been removed outright, so the
   single-copy path is unconditional rather than merely the default. Enabling either was a latent
   correctness hazard *and* a double copy of every synced byte: the layerwise bracket re-registered
   the pre-sync parameter objects over the whole model, and the CPU cache existed only to repair
   that (a full-model pinned D2H at apply plus a full-model H2D at reapply). These tests fail if
   either switch is reintroduced -- in the code or in the flag census.

2. **The drift fingerprint has no silent-empty hole.** The witness selector's ``numel > 1<<20``
   floor is a preference, not a gate: a model with no param that large used to select nothing, and
   ``if _sig:`` turned the whole post-sync check into a no-op that reported stale as success.

3. **The once-per-sync receiver seam runs on BOTH dispatch branches.** ``bump_sync_epoch`` is the
   only invalidation seam for the MoE router fp32 cast cache (a fixed-address buffer a captured
   decode graph reads directly) and the memoized expert-weight stack; it used to fire only under
   ``if self.colocate_all:``.

4. **The sync-time force-fresh skip is evidence-based and collective.**

(The sleep-skip coverage record -- the other half of the single-copy path -- is tested in
``skyrl/backends/skyrl_train/isoexec/runtimes/vllm/tests/test_sleep_skip_backup.py``, which can load
that module standalone.)

Run:
    uv run --isolated --extra dev --extra fsdp pytest \
        tests/backends/skyrl_train/inference_servers/test_isoexec_sync_single_copy.py -q
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]

LAYERWISE = REPO / "skyrl/backends/skyrl_train/inference_servers/layerwise_reload.py"
VLLM_WORKER = REPO / "skyrl/backends/skyrl_train/inference_servers/vllm_worker.py"
DISPATCH = REPO / "skyrl/backends/skyrl_train/workers/worker_dispatch.py"
MEGATRON_WORKER = REPO / "skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py"
FLAGS = REPO / "skyrl/backends/skyrl_train/isoexec/core/flags.py"


def _func_source(path: Path, name: str) -> str:
    """Source of a top-level or method definition named ``name`` (first match)."""
    src = path.read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(src, node)
    raise AssertionError(f"{name} not found in {path}")


# ==================================================================================================
# 1. code defaults
# ==================================================================================================
def _env_default(source: str, var: str):
    """The literal default every ``os.environ.get(var, <default>)`` in ``source`` uses.

    Asserts the file is self-consistent -- two reads of the same flag with different defaults is
    exactly the bug class this test exists for.
    """
    seen = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or len(node.args) != 2:
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "get"):
            continue
        key, dflt = node.args
        if not (isinstance(key, ast.Constant) and key.value == var):
            continue
        if isinstance(dflt, ast.Constant):
            seen.add(dflt.value)
    assert len(seen) <= 1, f"{var} is read with conflicting defaults {seen}"
    return seen.pop() if seen else None


RETIRED_DOUBLE_COPY_SWITCHES = (
    "SKYRL_ISOEXEC_LAYERWISE_RELOAD",
    "SKYRL_ISOEXEC_REAPPLY_CACHE",
)


@pytest.mark.parametrize("path", [LAYERWISE, VLLM_WORKER])
@pytest.mark.parametrize("var", RETIRED_DOUBLE_COPY_SWITCHES)
def test_retired_double_copy_switches_are_not_read(path, var):
    """Reintroducing a read of either switch reintroduces the double-copy path behind a flag."""
    assert _env_default(path.read_text(), var) is None, (
        f"{path.name} reads the retired {var}. Selecting it is the double-copy path: the layerwise "
        "bracket clobbers the native sync and the CPU cache re-copies the whole model to repair it."
    )


def test_registry_does_not_advertise_the_retired_switches():
    """The flags registry is the documentation of record, and a table that still lists a removed
    lever is a lie in it. Read structurally (the module needs megatron to import)."""
    src = FLAGS.read_text()
    catalogued = {
        node.args[0].value
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Flag"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    still_listed = sorted(v for v in RETIRED_DOUBLE_COPY_SWITCHES if v in catalogued)
    assert not still_listed, f"flags.py still catalogues retired switches: {still_listed}"


# ==================================================================================================
# 2. the drift fingerprint's selection floor is a preference, not a gate
# ==================================================================================================
def test_witness_floor_is_adaptive_not_hardcoded():
    src = _func_source(VLLM_WORKER, "load_weights")
    assert "_ix_sig_floor" in src, "the witness floor must be adaptive state on the worker"
    assert "src.numel() > _sig_floor" in src, "the selector must compare against the adaptive floor"
    assert "src.numel() > (1 << 20)" not in src, "the hardcoded 1 MiB gate is the silent-empty hole"


def test_empty_witness_set_is_reported_and_self_heals():
    src = _func_source(VLLM_WORKER, "isoexec_reapply_cached_weights")
    assert "NO WITNESS" in src, (
        "a sync with no drift witness must SAY so -- silently skipping the check is how a small "
        "model reported stale weights as clean"
    )
    assert "self._ix_sig_floor = 0" in src, "the floor must drop so the next sync is checked"


def test_coverage_record_is_written_unconditionally():
    """``_ix_synced_meta`` is what keeps the ~15 GiB engine-sleep skip alive now that the redundant
    reapply cache is gone. It must be written on the unconditional path: gating it was exactly what
    made the sleep skip and the single-copy path mutually exclusive."""
    src = _func_source(VLLM_WORKER, "load_weights")
    tree = ast.parse(src.lstrip())
    writes = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr == "_ix_synced_meta" and isinstance(n.ctx, ast.Load)
    ]
    assert writes, "_ix_synced_meta must be recorded in load_weights"

    # The master switch and a lazy-init hasattr are fine; a per-feature env gate is not -- that
    # coupling is what made the sleep skip and the single-copy path mutually exclusive.
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.dump(node.test)
        if "environ" not in test or "'SKYRL_ISOEXEC'" in test:
            continue
        body = ast.dump(ast.Module(body=node.body + node.orelse, type_ignores=[]))
        assert "_ix_synced_meta" not in body, (
            f"the coverage record is gated on a per-feature env flag ({ast.unparse(node.test)}) -- "
            "that coupling is exactly what made the sleep skip and the single-copy path mutually "
            "exclusive"
        )


# ==================================================================================================
# 3. the once-per-sync seam is unconditional
# ==================================================================================================
def test_reapply_seam_runs_on_both_dispatch_branches():
    """``isoexec_reapply_cached_weights`` must be called exactly once, OUTSIDE the colocate_all
    branch. It carries the post-sync drift check and ``bump_sync_epoch`` -- the only invalidation
    seam for the MoE router fp32 cast cache and the memoized expert-weight stack. Under the old
    placement a non-colocated IsoExec run routed every step's tokens with the FIRST sync's router
    weights, silently, and was never drift-checked at all."""
    src = DISPATCH.read_text()
    fn = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "save_weights_for_sampler"
    )

    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "isoexec_reapply_cached_weights"
    ]
    assert len(calls) == 1, f"expected exactly one reapply call site, found {len(calls)}"
    call = calls[0]

    # Find the enclosing `if` chain and prove none of them tests colocate_all.
    def _guards_of(target, node, stack):
        if node is target:
            return list(stack)
        for child in ast.iter_child_nodes(node):
            pushed = False
            if isinstance(node, ast.If) and any(child is c for c in node.body + node.orelse):
                stack.append(node.test)
                pushed = True
            got = _guards_of(target, child, stack)
            if pushed:
                stack.pop()
            if got is not None:
                return got
        return None

    guards = _guards_of(call, fn, [])
    assert guards is not None
    dumped = " ".join(ast.dump(g) for g in guards)
    assert (
        "colocate_all" not in dumped
    ), "the reapply call is still gated on colocate_all; it must run on the non-colocated path too"
    assert "SKYRL_ISOEXEC" in dumped, "it must still be gated on the IsoExec master switch"


# ==================================================================================================
# 4. the sync-time force-fresh skip
# ==================================================================================================
def test_sync_time_force_fresh_is_skippable_and_the_optim_time_one_is_not():
    """``_isoexec_force_fresh_model_params`` runs twice per step. The optim-time call IS the fix and
    must stay unconditional; the sync-time call is belt-and-braces and cost 4.66 s of a 34.5 s
    ``sync_weights`` on the v10 Qwen3.5-35B-A3B arm while every ``[ISOEXEC-POSTSTEP]`` line in that
    run read ``pre == post`` bitwise."""
    src = MEGATRON_WORKER.read_text()
    tree = ast.parse(src)
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_isoexec_force_fresh_model_params"
    ]
    assert len(calls) == 2, f"expected exactly 2 call sites (optim + sync), found {len(calls)}"
    kwsets = [{k.arg for k in c.keywords} for c in calls]
    assert kwsets.count({"skip_if_unchanged"}) == 1, "exactly one call site may pass skip_if_unchanged"
    assert kwsets.count(set()) == 1, "the optim-time call must remain unconditional"


def test_force_fresh_skip_decision_is_collective_and_fail_closed():
    """The routine it gates contains collectives (``start_param_sync``), so a per-rank decision
    would deadlock; and a guard that errors must run the fix, never skip it."""
    src = _func_source(MEGATRON_WORKER, "_isoexec_params_provably_unchanged")
    assert "all_reduce" in src and "ReduceOp.MIN" in src, "the skip decision must be all-reduced (MIN)"
    assert "return False" in src, "the exception path must fail closed (run the refresh)"

    witness = _func_source(MEGATRON_WORKER, "_isoexec_param_buffer_witness")
    assert (
        witness.count("return None") >= 3
    ), "the witness must return None (== 'assume changed') on every surface it cannot fingerprint"


def test_replicated_param_broadcast_is_coalesced():
    """One NCCL broadcast per replicated param is hundreds of latency-bound collectives inside the
    sync critical path. Flattening is bitwise-neutral (a broadcast installs rank-0's bytes either
    way) and the dest copy must use view_as, not reshape, or a non-contiguous param writes nowhere."""
    src = _func_source(MEGATRON_WORKER, "_isoexec_sync_replicated_params")
    assert src.count("torch.distributed.broadcast(") == 1, "the broadcast must be issued once per dtype group"
    assert "view_as(p.data)" in src, "the scatter-back must view_as the dest, not reshape it"
