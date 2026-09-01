"""One parser for SKYRL_ISOEXEC_GDN_KERNEL, and the contract build that follows it.

The fault this closes: the declaration site matched the env value case-sensitively against one
literal and fell back silently, so ``SKYRL_ISOEXEC_GDN_KERNEL=CPR`` made BOTH runtimes
derive the same wrong contract -- identical hashes, handshake green, cpr executing.
"""

import os
from contextlib import contextmanager

import pytest

from skyrl.backends.skyrl_train.isoexec.core.contract_build import ContractBuildError
from skyrl.backends.skyrl_train.isoexec.core.gdn_kernel_env import (
    DEFAULT_KERNEL,
    KERNEL_ENV,
    TRAINER_KERNEL_ENV,
    gdn_kernel_mode,
    gdn_trainer_kernel_override,
    parse_gdn_kernel,
)
from skyrl.backends.skyrl_train.isoexec.core.registry_build import build_registry
from skyrl.backends.skyrl_train.isoexec.models import qwen3_5


@contextmanager
def _env(**pairs):
    saved = {k: os.environ.get(k) for k in pairs}
    try:
        for k, v in pairs.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        yield
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def _refuses(fn, *a, **kw):
    with pytest.raises(ContractBuildError) as e:
        fn(*a, **kw)
    return str(e.value)


def test_case_insensitive_and_defaulted():
    assert parse_gdn_kernel("CPR") == "cpr"
    assert parse_gdn_kernel("  Recurrent ") == "recurrent"
    assert parse_gdn_kernel(None) == DEFAULT_KERNEL == "recurrent"


def test_unknown_value_refuses():
    for bad in ("cp-r", "cprr", "recurent"):
        with pytest.raises(ValueError) as e:
            parse_gdn_kernel(bad)
        assert "vocabulary" in str(e.value)


def test_retired_kernel_name_refuses_and_names_its_successor():
    with pytest.raises(ValueError) as e:
        parse_gdn_kernel("chunk_synced")
    assert "'chunk_synced' was renamed to 'cpr'" in str(e.value)


def test_env_reads_go_through_the_parser():
    with _env(**{KERNEL_ENV: "CPR", TRAINER_KERNEL_ENV: None}):
        assert gdn_kernel_mode() == "cpr"
        assert gdn_trainer_kernel_override() is None
    with _env(**{TRAINER_KERNEL_ENV: "Chunk"}):
        assert gdn_trainer_kernel_override() == "chunk"


def test_case_variant_selects_the_declared_variant():
    reg = build_registry(strict=True)
    with _env(**{KERNEL_ENV: "CPR", TRAINER_KERNEL_ENV: None}):
        c = qwen3_5.build(reg, arch="sm90")
    want = qwen3_5.build(reg, arch="sm90", profile=qwen3_5.CPR_PROFILE)
    assert c.identities == want.identities


def test_kernel_without_a_declared_variant_refuses():
    reg = build_registry(strict=True)
    with _env(**{KERNEL_ENV: "chunk", TRAINER_KERNEL_ENV: None}):
        msg = _refuses(qwen3_5.build, reg, arch="sm90")
    assert "declares no profile variant" in msg


def test_trainer_only_kernel_override_refuses():
    reg = build_registry(strict=True)
    with _env(**{KERNEL_ENV: None, TRAINER_KERNEL_ENV: "chunk"}):
        msg = _refuses(qwen3_5.build, reg, arch="sm90", profile=qwen3_5.PROFILE)
    assert TRAINER_KERNEL_ENV in msg and "ONE gdn.core kernel" in msg
    # Set to what the contract already declares, it asks for no split and is accepted.
    with _env(**{KERNEL_ENV: None, TRAINER_KERNEL_ENV: "RECURRENT"}):
        assert qwen3_5.build(reg, arch="sm90", profile=qwen3_5.PROFILE).identities.numerical_policy
