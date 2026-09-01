"""Static alignment gate between ops/*/_register.py claims and the colocated ops/*/tests/.

Checks that (a) every ``bitwise_equal_to`` referent names an impl on the same op (FAILS otherwise,
minus the documented external-baseline allowlist), (b) every equivalence claim has a colocated test
naming the claiming impl id, and (c) every declared hazard is referenced in the family's tests.
(b)/(c) are report-only until ``_STRICT_COVERAGE`` is flipped. CPU-only, no kernel imports.
"""

from __future__ import annotations

import importlib
import pathlib
import re

from skyrl.backends.skyrl_train.isoexec.core.registry import HAZARDS, Registry
from skyrl.backends.skyrl_train.isoexec.core.registry_build import (
    _FAMILIES,
    _ISOEXEC_ROOT,
    build_registry,
)

_OPS_DIR = pathlib.Path(__file__).resolve().parents[2] / "ops"

# Flip to True once every claim has a colocated test and every hazard is referenced.
_STRICT_COVERAGE = False

# bitwise_equal_to referents that deliberately name an UPSTREAM baseline, not a registered impl.
# Each entry is itself a standing finding: the claim is proven against code this repo does not
# register (megatron's stock non-fused ``sort_chunks_by_idxs`` cat/chunk loop), so no registry
# cross-check can validate it -- only the colocated bitwise battery can. Remove an entry to make
# the corresponding claim hard-fail.
_EXTERNAL_BASELINES = {
    ("moe.dispatch", "chunk_sort_gather", "megatron_cat_chunk_loop"),
}

# String-level hazard spellings accepted as "this test talks about that hazard". Deliberately
# narrow: the point is greppability from claim to test, not proof of exercise (the tests own that).
_HAZARD_ALIASES = {
    "null_lanes": ("null_lanes", "null lanes", "null_row", "null row", "null_block", "NULL lane"),
    "t_zero": ("t_zero", "T=0", "T = 0", "t == 0"),
    "non_contiguous": ("non_contiguous", "non-contiguous", "noncontiguous"),
    "e_mismatch": ("e_mismatch", "expert-count mismatch", "expert count mismatch", "E mismatch"),
    "profiling_shapes": ("profiling_shapes", "profiling shapes", "profile_run", "profiling shape"),
    "tie_boundaries": ("tie_boundaries", "tie boundaries", "tie boundary", "tie_rule", "ties"),
    "subnormals": ("subnormals", "subnormal"),
    "signed_zero": ("signed_zero", "signed zero", "signed-zero", "-0.0"),
}
assert set(_HAZARD_ALIASES) == set(HAZARDS)


def _family_registries():
    """{family: Registry with only that family's ops} -- attributes each op to its directory."""
    out = {}
    for fam in _FAMILIES:
        try:
            mod = importlib.import_module(f"{_ISOEXEC_ROOT}.ops.{fam}._register")
        except ModuleNotFoundError:
            continue
        reg = Registry()
        mod.register(reg)
        out[fam] = reg
    return out


def _tests_sources(fam: str) -> dict:
    """{filename: source text} for the family's colocated tests (helpers included)."""
    d = _OPS_DIR / fam / "tests"
    if not d.is_dir():
        return {}
    return {p.name: p.read_text(errors="replace") for p in sorted(d.glob("*.py"))}


def _iter_impls():
    for fam, reg in _family_registries().items():
        for op_name, op in sorted(reg.ops.items()):
            for impl_id, impl in sorted(op.impls.items()):
                yield fam, op_name, impl_id, impl


def _findings():
    """(violations, reports): (a) entries in violations; (b)/(c) entries in reports."""
    violations, reports = [], []
    fams = _family_registries()
    sources = {fam: _tests_sources(fam) for fam in fams}

    for fam, reg in fams.items():
        blob = "\n".join(sources[fam].values())
        fam_has_tests = bool(sources[fam])
        for op_name, op in sorted(reg.ops.items()):
            for impl_id, impl in sorted(op.impls.items()):
                bit = impl.capabilities.get("bitwise_equal_to")
                proof = impl.capabilities.get("equivalence_proof")

                # (a) bitwise_equal_to referent resolves on the same op
                if bit is not None and bit not in op.impls:
                    row = (fam, op_name, impl_id, f"bitwise_equal_to={bit!r} names no impl on this op")
                    if (op_name, impl_id, bit) in _EXTERNAL_BASELINES:
                        reports.append(row + ("allowlisted external baseline",))
                    else:
                        violations.append(row)

                # (b) any equivalence claim needs a colocated test naming the claiming impl
                if bit is not None or proof is not None:
                    if not fam_has_tests:
                        reports.append((fam, op_name, impl_id, "family has no tests/ dir", "claim untested"))
                    elif not re.search(re.escape(impl_id), blob):
                        reports.append(
                            (fam, op_name, impl_id, f"no colocated test names impl {impl_id!r}", "claim untested")
                        )

                # (c) every declared hazard is referenced somewhere in the family's tests
                for hz in impl.hazards:
                    if not any(alias in blob for alias in _HAZARD_ALIASES[hz]):
                        reports.append(
                            (
                                fam,
                                op_name,
                                impl_id,
                                f"hazard {hz!r} never referenced in {fam}/tests",
                                "hazard unreferenced",
                            )
                        )
    return violations, reports


def _print_table(rows, title):
    if not rows:
        return
    print(f"\n== {title} ({len(rows)}) ==")
    for row in rows:
        print("  " + " | ".join(str(c) for c in row))


def test_registry_builds():
    reg = build_registry()
    assert reg.ops, "registry is empty"


def test_bitwise_referents_resolve():
    violations, reports = _findings()
    _print_table(
        [r for r in reports if r[-1] == "allowlisted external baseline"], "external-baseline referents (allowlisted)"
    )
    _print_table(violations, "DANGLING bitwise_equal_to referents")
    assert not violations, f"{len(violations)} bitwise_equal_to referent(s) name no registered impl"


def test_claims_have_colocated_tests():
    _, reports = _findings()
    rows = [r for r in reports if r[-1] == "claim untested"]
    _print_table(rows, "equivalence claims without a colocated test")
    if _STRICT_COVERAGE:
        assert not rows, f"{len(rows)} equivalence claim(s) have no colocated test"


def test_declared_hazards_are_referenced():
    _, reports = _findings()
    rows = [r for r in reports if r[-1] == "hazard unreferenced"]
    _print_table(rows, "declared hazards never referenced in colocated tests")
    if _STRICT_COVERAGE:
        assert not rows, f"{len(rows)} declared hazard(s) are never referenced"


def test_every_family_with_claims_has_tests_dir():
    fams = _family_registries()
    missing = []
    for fam, reg in fams.items():
        claims = any(
            impl.capabilities.get("bitwise_equal_to") is not None
            or impl.capabilities.get("equivalence_proof") is not None
            for op in reg.ops.values()
            for impl in op.impls.values()
        )
        if claims and not (_OPS_DIR / fam / "tests").is_dir():
            missing.append(fam)
    _print_table([(f, "no tests/ directory") for f in missing], "claiming families without tests/")
    if _STRICT_COVERAGE:
        assert not missing


def _run():
    import traceback

    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
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
