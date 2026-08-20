import os
import pickle
import tempfile
from types import SimpleNamespace

from skyrl.backends.skyrl_train.isoexec.core import process_contract as pc
from skyrl.backends.skyrl_train.isoexec.core.contract_delivery import load_contract

# torch-first, as in production: engine workers load torch long before any isoexec module.
from skyrl.backends.skyrl_train.weight_sync.cuda_ipc_strategy import CudaIpcInitInfo

STRICT_ENV = "SKYRL_ISOEXEC_MANIFEST_STRICT"

# Resolves to models/qwen3_5.py by name pattern; no config.json needed.
MODEL_PATH = "qwen3.5-35b-a3b"


def _clear():
    pc._CONTRACT, pc._VIEW = None, None
    for var in (STRICT_ENV, "ISOEXEC_CONTRACT_PATH", "ISOEXEC_CONTRACT_HASH"):
        os.environ.pop(var, None)


def _seed():
    _clear()
    return pc.get_process_contract(MODEL_PATH, arch="sm90")


def test_caching_idempotent():
    c1 = _seed()
    c2 = pc.get_process_contract()
    assert c1 is not None and c1 is c2 and pc.cached_contract() is c1
    assert pc.cached_contract_view(), "the (op, site) view is built alongside the contract"


def test_hash_is_composite_of_numerical_policy():
    c = _seed()
    saved = dict(pc._EXTENSIONS)
    pc._EXTENSIONS.clear()
    try:
        h = pc.contract_hash()
        assert h == c.identities.numerical_policy  # no extensions: the identity, byte-for-byte
        pc.register_contract_extension("dummy", lambda: "digest-1")
        h2 = pc.contract_hash()
        assert h2 != h  # an extension folds into the handshake hash
    finally:
        pc._EXTENSIONS.clear()
        pc._EXTENSIONS.update(saved)


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
        _clear()
        os.environ["ISOEXEC_CONTRACT_PATH"] = path
        try:
            c = pc.get_process_contract(MODEL_PATH, arch="sm90")
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
