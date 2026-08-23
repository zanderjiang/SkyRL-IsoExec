"""Topology claims: grounded derivation, refusing enforcement, and the engine pik-assert ordering.

Three obligations, one file:
  1. The production contract's TopologyClaims are the profile's DECLARED facts (no invented
     literals) and every proof ref resolves to a real colocated gate file.
  2. ``assert_topology_within_claims`` accepts inside a claimed envelope, REFUSES outside it,
     and demotes to warn-only under SKYRL_ISOEXEC_MANIFEST_STRICT=0.
  3. The engine install path builds the contract BEFORE the pik install, so
     ``_assert_plan_matches_manifest`` reads a real view (the audit found it saw None and skipped).
"""

import os
import pathlib
from types import SimpleNamespace

from skyrl.backends.skyrl_train.isoexec.core import process_contract as pc
from skyrl.backends.skyrl_train.isoexec.core.contract_build import derive_topology_claims
from skyrl.backends.skyrl_train.isoexec.core.registry_build import build_registry
from skyrl.backends.skyrl_train.isoexec.models import qwen3_5
from skyrl.backends.skyrl_train.isoexec.models.profile import ProfileError, TopologyAxisFact

STRICT_ENV = "SKYRL_ISOEXEC_MANIFEST_STRICT"
_ISOEXEC_DIR = pathlib.Path(__file__).resolve().parents[2]

# The production deployment (run_qwen35_dapo_isoexec.sh): trainer TP4/SP1, engine TP8/SP0.
TRAINER_ACTUAL = {"TP": 4, "SP": 1, "PP": 1, "CP": 1}
ENGINE_ACTUAL = {"TP": 8, "SP": 0, "PP": 1, "CP": 1}


def _contract():
    reg = build_registry(strict=True)
    return qwen3_5.build(reg, arch="sm90", profile=qwen3_5.PROFILE)


def _refuses(fn, *a, **kw) -> str:
    try:
        fn(*a, **kw)
    except (RuntimeError, ProfileError) as e:
        return str(e)
    raise AssertionError(f"{getattr(fn, '__name__', fn)} should have refused")


def _strict_env(value):
    if value is None:
        os.environ.pop(STRICT_ENV, None)
    else:
        os.environ[STRICT_ENV] = value


def test_claims_are_the_declared_profile_facts():
    c = _contract()
    facts = {t.axis: t for t in qwen3_5.PROFILE.topology}
    claims = {t.axis: t for t in c.claims.topology}
    assert set(claims) == set(facts) == {"TP", "SP", "PP", "CP"}
    for axis, f in facts.items():
        t = claims[axis]
        assert (t.kind, t.degree, t.collective_plan, t.domain, t.proof) == (
            f.kind,
            f.degree,
            f.collective_plan,
            tuple(f.domain),
            f.proof,
        ), f"claim {axis} is not the declared fact"
    # The production axes, verbatim: values grounded in the colocated gates, not invented here.
    assert claims["TP"].kind == "invariant" and claims["TP"].domain == (1, 2, 4, 8)
    assert claims["SP"].kind == "invariant" and claims["SP"].domain == (0, 1)
    assert claims["PP"].kind == "pinned" and claims["PP"].degree == 1
    assert claims["CP"].kind == "pinned" and claims["CP"].degree == 1


def test_proof_refs_resolve_to_existing_gate_files():
    for t in qwen3_5.PROFILE.topology:
        if t.proof is None:
            continue
        p = _ISOEXEC_DIR / t.proof
        assert p.is_file(), f"claim {t.axis}: proof ref {t.proof!r} does not resolve under {_ISOEXEC_DIR}"


def test_tp_domain_members_divide_pik_leaves():
    # The TP invariance proof is the pik leaf tree: only divisors of G are even expressible.
    (tp,) = [t for t in qwen3_5.PROFILE.topology if t.axis == "TP"]
    g = qwen3_5.PROFILE.pik_leaves
    assert all(g % d == 0 for d in tp.domain), f"TP domain {tp.domain} vs pik_leaves={g}"


def test_derivation_is_order_independent():
    facts = qwen3_5.PROFILE.topology
    assert derive_topology_claims(facts) == derive_topology_claims(tuple(reversed(facts)))


def test_profile_refuses_ungrounded_invariant():
    msg = _refuses(TopologyAxisFact, axis="EP", kind="invariant", domain=(1, 8))
    assert "proof" in msg
    msg = _refuses(TopologyAxisFact, axis="EP", kind="invariant", proof="somewhere")
    assert "domain" in msg
    msg = _refuses(TopologyAxisFact, axis="PP", kind="pinned")
    assert "degree" in msg


def test_enforcement_accepts_inside_domains():
    c = _contract()
    saved = os.environ.get(STRICT_ENV)
    try:
        _strict_env(None)  # default strict
        assert pc.assert_topology_within_claims(c, TRAINER_ACTUAL, side="TRAINER") is True
        assert pc.assert_topology_within_claims(c, ENGINE_ACTUAL, side="ENGINE") is True
    finally:
        _strict_env(saved)


def test_enforcement_refuses_outside_domains():
    c = _contract()
    saved = os.environ.get(STRICT_ENV)
    try:
        _strict_env(None)
        msg = _refuses(pc.assert_topology_within_claims, c, dict(ENGINE_ACTUAL, TP=16), side="ENGINE")
        assert "TP" in msg and "16" in msg and "proven domain" in msg
        msg = _refuses(pc.assert_topology_within_claims, c, dict(TRAINER_ACTUAL, PP=2), side="TRAINER")
        assert "PP" in msg and "pins 1" in msg
        msg = _refuses(pc.assert_topology_within_claims, c, dict(TRAINER_ACTUAL, CP=2), side="TRAINER")
        assert "CP" in msg
    finally:
        _strict_env(saved)


def test_enforcement_warn_only_when_strict_off():
    c = _contract()
    saved = os.environ.get(STRICT_ENV)
    try:
        _strict_env("0")
        assert pc.assert_topology_within_claims(c, dict(ENGINE_ACTUAL, TP=16), side="ENGINE") is False
    finally:
        _strict_env(saved)


def test_enforcement_skips_unobtainable_axes_and_claimless_contracts():
    c = _contract()
    saved = os.environ.get(STRICT_ENV)
    try:
        _strict_env(None)
        # An axis the caller could not obtain is skipped (visibly), never guessed fatal.
        assert pc.assert_topology_within_claims(c, {"TP": 8}, side="ENGINE") is True
        assert pc.assert_topology_within_claims(None, ENGINE_ACTUAL, side="ENGINE") is True
        # A synthetic contract with no claims has nothing to check -- capability tests unaffected.
        import dataclasses

        from skyrl.backends.skyrl_train.isoexec.contract import Claims
        from skyrl.backends.skyrl_train.isoexec.contract.identity import compute_identities

        bare = dataclasses.replace(c, claims=Claims())
        bare = dataclasses.replace(bare, identities=compute_identities(bare))
        assert pc.assert_topology_within_claims(bare, {"TP": 999}, side="ENGINE") is True
    finally:
        _strict_env(saved)


def test_engine_pik_assert_sees_real_view():
    """The audit's dead arm: with the process contract built (as the reordered engine install now
    does), ``_assert_plan_matches_manifest`` compares the env-built plan against the REAL
    production pins and refuses a split -- no stub view, the actual qwen35 contract."""
    from skyrl.backends.skyrl_train.isoexec.ops.collectives import pik_tp_invariant as pik

    saved_c, saved_v, saved_env = pc._CONTRACT, pc._VIEW, os.environ.get(STRICT_ENV)
    try:
        _strict_env(None)
        pc._CONTRACT, pc._VIEW = None, None
        c = pc.get_process_contract(qwen3_5.MODEL, arch="sm90")
        assert c is not None and pc.cached_contract_view() is not None
        pins = None
        for (op, _site), entry in pc.cached_contract_view().items():
            if op == "collectives.tree_all_reduce":
                pins = entry["pinned_constants"]
                break
        assert pins and int(pins["leaves"]) == qwen3_5.PROFILE.pik_leaves
        good = SimpleNamespace(
            num_leaves=int(pins["leaves"]), bf16_leaves=(str(pins["leaf_dtype"]) == "bf16")
        )
        pik._assert_plan_matches_manifest("ENGINE", good)  # reachable AND passing
        bad = SimpleNamespace(num_leaves=int(pins["leaves"]) * 2, bf16_leaves=good.bf16_leaves)
        msg = _refuses(pik._assert_plan_matches_manifest, "ENGINE", bad)
        assert "plan/manifest SPLIT" in msg and "leaves" in msg
    finally:
        pc._CONTRACT, pc._VIEW = saved_c, saved_v
        _strict_env(saved_env)


def test_engine_contract_build_precedes_pik_install():
    """Static ordering gate on the engine adapter: the audit found the contract built 110 lines
    AFTER the pik install, making the engine arm of the pin assert silently dead. The
    ContractAdapter now owns the ordering: run_install builds the contract and checks the claims
    BEFORE install(), and the pik install lives inside the engine's install closure."""
    import inspect

    from skyrl.backends.skyrl_train.isoexec.core.adapter import ContractAdapter

    run_src = inspect.getsource(ContractAdapter.run_install)
    assert (
        run_src.index("self.build_contract()") < run_src.index("check_all_claims(") < run_src.index("self.install()")
    ), "the adapter must build its contract and check claims before any install"
    src = (_ISOEXEC_DIR / "runtimes" / "vllm" / "gptmodel_vllm.py").read_text()
    closure_at = src.index("def _isoexec_install")
    pik_at = src.index('apply_pik_tp_invariant(side="ENGINE")')
    run_at = src.index(".run_install()")
    assert "VLLMContractAdapter(" in src, "the engine must construct its ContractAdapter"
    assert closure_at < pik_at < run_at, "the pik install must live inside the adapter-run install closure"


def _run():
    import traceback

    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed.append(name)
            traceback.print_exc()
            print(f"FAIL {name}")
    print(f"{len(fns) - len(failed)}/{len(fns)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    _run()
