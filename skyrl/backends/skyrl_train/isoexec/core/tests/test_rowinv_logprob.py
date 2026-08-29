"""The composition the rowinv leaf-tree logprob IS -- unconditionally, at all four sites.

This was a flag test. ``SKYRL_ISOEXEC_ROWINV_LOGPROB`` is gone: the leaf-tree denominator is the
composed default, so what is left to assert is not that a lever selects it but that nothing else
can be selected. Two halves, both CPU:

  * the CONTRACT: ``rowinv_leaftree`` at ALL FOUR sites (one function everywhere is the design),
    FUNCTION-half at every one of them, carrying the leaves/block/accum pins, and no aten-order
    entry anywhere in the composition. The aten impls stay REGISTERED -- they are still the
    structural-decline fallback -- so this asserts the composition, not the registry;
  * DETERMINISM: the same code composes the same identities every time. The hash-rotation test
    that used to live here proved a one-sided flag flip could not hide; with no flag there is no
    one-sided flip to catch, and the property that replaces it is that the two runtimes derive the
    identical numerical_policy from identical code.

Like test_manifest_qwen35, every composition test builds under a cleared environment so it asserts
what the code composes, not whatever shell invoked it.
"""

import os
from contextlib import contextmanager

from skyrl.backends.skyrl_train.isoexec.core.process_contract import build_contract_view
from skyrl.backends.skyrl_train.isoexec.core.registry_build import build_registry
from skyrl.backends.skyrl_train.isoexec.models import qwen3_5
from skyrl.backends.skyrl_train.isoexec.models.policy import (
    ROWINV_BLOCK,
    policy_matches_registry_capabilities,
)

OP = "logprobs.log_softmax"
_SITES = ("trainer_fwd", "trainer_score", "engine_prefill", "engine_decode")
_SUPERSEDED = ("aten_reference",)


@contextmanager
def _code_defaults(**overrides):
    """Strip every IsoExec override so the composition reflects code defaults, not the caller's shell."""
    saved = {k: v for k, v in os.environ.items() if k.startswith("SKYRL_ISOEXEC")}
    for k in saved:
        del os.environ[k]
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k in list(os.environ):
            if k.startswith("SKYRL_ISOEXEC"):
                del os.environ[k]
        os.environ.update(saved)


def _build():
    reg = build_registry(strict=True)
    c = qwen3_5.build(reg, arch="sm90")
    return reg, c, build_contract_view(c, reg)


def test_rowinv_serves_all_four_sites_with_the_pins():
    with _code_defaults():
        reg, _, view = _build()
        problems = policy_matches_registry_capabilities(qwen3_5.PROFILE, reg)
    assert problems == [], problems
    for site in _SITES:
        e = view[(OP, site)]
        assert e["impl_id"] == "rowinv_leaftree", f"{site}: {e['impl_id']}"
        # FUNCTION-half at every site: this selection moves bits and is hashed, never proof-carried.
        assert e["half"] == "function", f"{site}: {e['half']}"
        assert e["pinned_constants"] == {
            "leaves": qwen3_5.PROFILE.pik_leaves,
            "block": ROWINV_BLOCK,
            "accum": "kahan_fp32",
        }, f"{site}: {e['pinned_constants']}"


def test_no_superseded_impl_survives_in_the_composition():
    """The aten-order denominator is unreachable by composition, at every site and both variants.

    It is still REGISTERED (it is the structural-decline fallback), so the check that matters is
    that nothing SELECTS it: a reintroduced branch would put trainer_score back on a schedule the
    engine cannot reproduce, which is the exact defect rowinv exists to close.
    """
    for profile in (qwen3_5.PROFILE, qwen3_5.CPR_PROFILE):
        with _code_defaults():
            reg = build_registry(strict=True)
            c = qwen3_5.build(reg, arch="sm90", profile=profile)
        selected = {e.impl.id for e in c.composition if OP in e.region}
        assert selected == {"rowinv_leaftree"}, f"{profile.gdn_kernel}: {selected}"
        # and the registry still carries them, so a decline has a name
        assert set(_SUPERSEDED) <= set(reg.get_op(OP).impls)


def test_composition_is_deterministic():
    """Identical code composes identical identities -- what the handshake compares on both sides."""
    with _code_defaults():
        _, c1, _ = _build()
        _, c2, _ = _build()
    assert c1.identities == c2.identities
