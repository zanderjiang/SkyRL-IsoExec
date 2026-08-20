"""CPU-only contract tests for PIK's pinned cuBLASLt extension."""

import importlib.util
from pathlib import Path

import pytest

_SOURCE = Path(__file__).parents[1] / "pik/cublas.py"
_SPEC = importlib.util.spec_from_file_location("isoexec_test_pik_cublas", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_CUBLAS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CUBLAS)


def test_pinned_extension_requires_cuda13_needed_runpath_and_resolution(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    extension = tmp_path / "pik_cublaslt.so"
    readelf = f"""
 0x0000000000000001 (NEEDED) Shared library: [libcublasLt.so.13]
 0x000000000000001d (RUNPATH) Library runpath: [{runtime}]
"""
    ldd = f"""
 libcublasLt.so.13 => {runtime / "libcublasLt.so.13"} (0x1)
 libcublas.so.13 => {runtime / "libcublas.so.13"} (0x2)
"""
    _CUBLAS._validate_pinned_extension(
        extension, runtime, readelf_output=readelf, ldd_output=ldd
    )


def test_pinned_extension_refuses_cuda12_cache_entry(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    readelf = f"""
 0x0000000000000001 (NEEDED) Shared library: [libcublasLt.so.12]
 0x000000000000001d (RUNPATH) Library runpath: [{runtime}]
"""
    with pytest.raises(RuntimeError, match="NEEDED lacks libcublasLt.so.13.*CUDA-12"):
        _CUBLAS._validate_pinned_extension(
            tmp_path / "pik_cublaslt.so",
            runtime,
            readelf_output=readelf,
            ldd_output="libcublasLt.so.12 => /usr/local/cuda/libcublasLt.so.12 (0x1)",
        )
