import os
import tempfile

from skyrl.backends.skyrl_train.isoexec.contract import to_canonical_json
from skyrl.backends.skyrl_train.isoexec.core.composition import CompositionError
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


def _build():
    reg = build_registry(strict=True)
    m = qwen3_5.build(reg, arch="sm90").freeze()
    return reg, m, build_execution_contract(reg, m, profile=qwen3_5.PROFILE)


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
        load_contract(path)  # absent env: self-consistency only, like load_manifest


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


def test_installed_validation_parity():
    reg, m, c = _build()
    exact = set(m.keys())
    dropped = exact - {sorted(exact)[0]}
    foreign = exact | {("moe.router", "bogus_site")}

    def verdicts(keys):
        try:
            m.validate_against_installed(keys)
            mv = True
        except CompositionError:
            mv = False
        try:
            validate_contract_against_installed(c, reg, keys)
            cv = True
        except ContractDeliveryError:
            cv = False
        return mv, cv

    assert verdicts(exact) == (True, True)
    assert verdicts(dropped) == (False, False)
    assert verdicts(foreign) == (False, False)


def test_expected_keys_match_manifest_keys():
    reg, m, c = _build()
    assert expected_installed_keys(c, reg) == frozenset(m.keys())
