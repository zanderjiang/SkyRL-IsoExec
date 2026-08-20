import os
import tempfile

from skyrl.backends.skyrl_train.isoexec.contract import to_canonical_json
from skyrl.backends.skyrl_train.isoexec.core.contract_build import (
    build_execution_contract,
)
from skyrl.backends.skyrl_train.isoexec.core.contract_delivery import (
    CONTRACT_HASH_ENV,
    ContractDeliveryError,
    expected_installed_keys,
    load_contract,
    validate_contract_against_installed,
    write_contract_file,
)
from skyrl.backends.skyrl_train.isoexec.core.registry_build import build_registry
from skyrl.backends.skyrl_train.isoexec.models import qwen3_5
from skyrl.backends.skyrl_train.isoexec.models.policy import build_selections


def _build():
    reg = build_registry(strict=True)
    sel = build_selections(qwen3_5.PROFILE, qwen3_5.EXCEPTIONS)
    return reg, sel, build_execution_contract(reg, sel, arch="sm90", model=qwen3_5.MODEL)


def _no_env():
    os.environ.pop(CONTRACT_HASH_ENV, None)


def test_write_load_round_trip():
    _, _, c = _build()
    _no_env()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "contract.json")
        h = write_contract_file(c, path)
        assert h == c.identities.numerical_policy
        loaded = load_contract(path)
    assert loaded.identities == c.identities
    assert to_canonical_json(loaded) == to_canonical_json(c)


def test_hash_env_cross_check():
    _, _, c = _build()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "contract.json")
        h = write_contract_file(c, path)
        try:
            os.environ[CONTRACT_HASH_ENV] = h
            load_contract(path)  # correct env passes
            os.environ[CONTRACT_HASH_ENV] = "0" * 64
            try:
                load_contract(path)
                raise AssertionError("wrong env hash must refuse")
            except ContractDeliveryError:
                pass
        finally:
            _no_env()
        load_contract(path)  # absent env: self-consistency only


def test_tampered_file_refuses():
    _, _, c = _build()
    _no_env()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "contract.json")
        write_contract_file(c, path)
        raw = open(path, "rb").read()
        tampered = raw.replace(b'"version":1', b'"version":2', 1)
        assert tampered != raw
        open(path, "wb").write(tampered)
        try:
            load_contract(path)
            raise AssertionError("tampered file must refuse")
        except ContractDeliveryError:
            pass  # decodes cleanly; the identity recompute is what catches it


def test_installed_validation():
    reg, sel, c = _build()
    exact = set(sel.keys())
    dropped = exact - {sorted(exact)[0]}
    foreign = exact | {("moe.router", "bogus_site")}

    def verdict(keys):
        try:
            validate_contract_against_installed(c, reg, keys)
            return True
        except ContractDeliveryError:
            return False

    assert verdict(exact) is True
    assert verdict(dropped) is False
    assert verdict(foreign) is False


def test_expected_keys_match_selection_keys():
    reg, sel, c = _build()
    assert expected_installed_keys(c, reg) == frozenset(sel.keys())
