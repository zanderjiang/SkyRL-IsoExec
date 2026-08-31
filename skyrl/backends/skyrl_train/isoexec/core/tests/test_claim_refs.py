"""Claim refs resolve to real in-tree code, or they refuse.

Guards two easily-missed faults: a ref that escapes the package via ``../``, and a ``def`` that is
only text inside a docstring.
"""

import os
import tempfile

from skyrl.backends.skyrl_train.isoexec.core import claim_refs
from skyrl.backends.skyrl_train.isoexec.models import qwen3_5

_DOCSTRING_ONLY = '"""\ndef ghost_hook(state, ctx):\n    only text inside a docstring\n"""\n'


def test_declared_state_hooks_resolve():
    for s in qwen3_5.PROFILE.states:
        assert claim_refs.hook_ref_problem(s.ref) is None, s.ref


def test_missing_file_symbol_and_bare_path_refuse():
    assert "missing" in claim_refs.hook_ref_problem("lifecycle/does_not_exist.py::nope")
    assert "not defined" in claim_refs.hook_ref_problem("lifecycle/ordering.py::not_a_real_symbol")
    assert "not defined" in claim_refs.hook_ref_problem("lifecycle/ordering.py")


def test_ref_escaping_the_package_refuses():
    with tempfile.TemporaryDirectory() as outside:
        path = os.path.join(outside, "fake_hook.py")
        with open(path, "w") as fh:
            fh.write("def ghost_hook(state, ctx):\n    return True\n")
        ref = os.path.relpath(path, claim_refs.ISOEXEC_DIR) + "::ghost_hook"
        assert "escapes the isoexec package" in claim_refs.hook_ref_problem(ref)


def test_def_inside_a_docstring_is_not_a_hook():
    with tempfile.TemporaryDirectory(dir=claim_refs.ISOEXEC_DIR) as inside:
        path = os.path.join(inside, "fake_hook.py")
        with open(path, "w") as fh:
            fh.write(_DOCSTRING_ONLY)
        ref = os.path.relpath(path, claim_refs.ISOEXEC_DIR) + "::ghost_hook"
        assert "not defined" in claim_refs.hook_ref_problem(ref)


def test_declared_topology_proofs_resolve():
    for t in qwen3_5.PROFILE.topology:
        if t.proof is not None:
            assert claim_refs.proof_ref_problem(t.proof) is None, t.proof


def test_proof_ref_forms():
    # A gate name followed by what it shows, as ops/gdn/_register.py declares it.
    assert claim_refs.proof_ref_problem("gdn_native_kernel_parity_test: split-exactness") is None
    assert "resolves to no gate" in claim_refs.proof_ref_problem("TODO(nobody): write this")
    assert "empty proof ref" in claim_refs.proof_ref_problem("   ")
