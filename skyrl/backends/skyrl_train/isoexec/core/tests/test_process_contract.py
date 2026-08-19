import os
import pickle
import tempfile
from types import SimpleNamespace

from skyrl.backends.skyrl_train.isoexec.core import process_contract as pc
from skyrl.backends.skyrl_train.isoexec.core import process_manifest as pm
from skyrl.backends.skyrl_train.isoexec.core.contract_delivery import load_contract
from skyrl.backends.skyrl_train.isoexec.core.registry_build import build_registry
from skyrl.backends.skyrl_train.isoexec.models import qwen3_5

# torch-first, as in production: engine workers load torch long before any isoexec module.
from skyrl.backends.skyrl_train.weight_sync.cuda_ipc_strategy import CudaIpcInitInfo

STRICT_ENV = "SKYRL_ISOEXEC_MANIFEST_STRICT"


def _clear():
    pm._MANIFEST, pm._HASH = None, None
    pc._CONTRACT = None
    for var in (STRICT_ENV, "ISOEXEC_CONTRACT_PATH", "ISOEXEC_CONTRACT_HASH"):
        os.environ.pop(var, None)


def _seed():
    _clear()
    reg = build_registry(strict=True)
    m = qwen3_5.build(reg, arch="sm90").freeze()
    pm._MANIFEST, pm._HASH = m, m.hash()
    return reg, m


def test_caching_idempotent():
    _seed()
    c1 = pc.get_process_contract()
    c2 = pc.get_process_contract()
    assert c1 is not None and c1 is c2 and pc.cached_contract() is c1


def test_hash_is_composite_of_numerical_policy():
    _seed()
    saved = dict(pm._EXTENSIONS)
    pm._EXTENSIONS.clear()
    try:
        c = pc.get_process_contract()
        h = pc.contract_hash()
        assert h == c.identities.numerical_policy  # no extensions: the identity, byte-for-byte
        pm.register_manifest_extension("dummy", lambda: "digest-1")
        h2 = pc.contract_hash()
        assert h2 != h  # an extension folds into the contract handshake exactly as the manifest's
    finally:
        pm._EXTENSIONS.clear()
        pm._EXTENSIONS.update(saved)


def test_agreement_match_mismatch_and_warn_only():
    _seed()
    h = pc.contract_hash()
    assert pc.assert_contract_agreement(h, other_side="peer") is True
    try:
        pc.assert_contract_agreement("0" * 64)
        raise AssertionError("strict mismatch must raise")
    except RuntimeError:
        pass
    os.environ[STRICT_ENV] = "0"
    try:
        assert pc.assert_contract_agreement("0" * 64) is False
    finally:
        os.environ.pop(STRICT_ENV, None)


def test_agreement_skips_when_not_built():
    _clear()
    assert pc.assert_contract_agreement("anything") is True


def test_contract_path_write_and_round_trip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "contract.json")
        _seed()
        os.environ["ISOEXEC_CONTRACT_PATH"] = path
        try:
            c = pc.get_process_contract()
            assert os.path.exists(path)
            loaded = load_contract(path)
            assert loaded.identities == c.identities
        finally:
            os.environ.pop("ISOEXEC_CONTRACT_PATH", None)


def test_receiver_helper_with_pickled_init_info():
    # Real strategy class, pickled exactly as collective_rpc delivers it.
    _seed()
    h = pc.contract_hash()
    info = pickle.loads(
        pickle.dumps(CudaIpcInitInfo(override_existing_receiver=False, model_dtype_str="bfloat16", contract_hash=h))
    )
    assert info.contract_hash == h
    assert pc.assert_init_info_contract(info, other_side="trainer") is True

    bad = pickle.loads(
        pickle.dumps(
            CudaIpcInitInfo(override_existing_receiver=False, model_dtype_str="bfloat16", contract_hash="0" * 64)
        )
    )
    try:
        pc.assert_init_info_contract(bad, other_side="trainer")
        raise AssertionError("mismatched stamped hash must raise under strict")
    except RuntimeError:
        pass

    unstamped = CudaIpcInitInfo(override_existing_receiver=False, model_dtype_str="bfloat16")
    assert pc.assert_init_info_contract(unstamped) is True  # old sender: no stamp -> skip
    assert pc.assert_init_info_contract(SimpleNamespace()) is True  # no attr at all -> skip
