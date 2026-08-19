"""cuBLASLt leaf-GEMM backend (JIT-compiled on first use)."""

from __future__ import annotations

import functools
import os
import pathlib
import re
import subprocess

import torch

_HERE = pathlib.Path(__file__).parent


def _validate_pinned_extension(
    extension: pathlib.Path,
    runtime: pathlib.Path,
    *,
    readelf_output: str | None = None,
    ldd_output: str | None = None,
) -> None:
    """Refuse a cached PIK extension whose ELF ABI escaped the selected CUDA runtime."""
    if readelf_output is None:
        completed = subprocess.run(
            ["readelf", "-d", str(extension)],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if completed.returncode:
            raise RuntimeError(f"[ISOEXEC-PIK] readelf failed for {extension}")
        readelf_output = completed.stdout
    needed = set(re.findall(r"Shared library: \[([^]]+)\]", readelf_output))
    runpaths = re.findall(r"Library (?:runpath|rpath): \[([^]]+)\]", readelf_output, flags=re.IGNORECASE)
    errors: list[str] = []
    if "libcublasLt.so.13" not in needed:
        errors.append(f"NEEDED lacks libcublasLt.so.13: {sorted(needed)}")
    incompatible = sorted(name for name in needed if name.startswith(("libcublas", "libcudart")) and ".so.12" in name)
    if incompatible:
        errors.append(f"NEEDED contains CUDA-12 owners: {incompatible}")
    runtime = runtime.resolve()
    if str(runtime) not in runpaths:
        errors.append(f"RUNPATH does not pin {runtime}: {runpaths}")

    if ldd_output is None:
        completed = subprocess.run(
            ["ldd", str(extension)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode:
            errors.append(f"ldd failed with exit code {completed.returncode}")
        ldd_output = completed.stdout + completed.stderr
    owners: dict[str, pathlib.Path] = {}
    for name, owner in re.findall(r"^\s*(libcublas(?:Lt)?\.so\.\d+)\s+=>\s+(\S+)", ldd_output, flags=re.MULTILINE):
        owners[name] = pathlib.Path(owner).resolve()
    for name in ("libcublasLt.so.13", "libcublas.so.13"):
        owner = owners.get(name)
        if owner is None:
            errors.append(f"ldd did not resolve {name}")
            continue
        try:
            owner.relative_to(runtime)
        except ValueError:
            errors.append(f"ldd resolved {name} outside pinned runtime: {owner}")
    if errors:
        raise RuntimeError("[ISOEXEC-PIK] pinned cuBLASLt extension refusal: " + "; ".join(errors))
    print(
        f"[ISOEXEC-PIK] cublaslt_extension={extension.resolve()} needed=libcublasLt.so.13 "
        f"cublaslt_owner={owners['libcublasLt.so.13']} cublas_owner={owners['libcublas.so.13']} ",
        flush=True,
    )


@functools.lru_cache(maxsize=1)
def _ext():
    # cuBLASLt ships with the torch wheel's nvidia deps; fall back to the toolkit.
    import nvidia  # noqa: F401
    from torch.utils.cpp_extension import load

    pinned_runtime = os.environ.get("SKYRL_TE_CUDA_RUNTIME_LIB")
    extra_ldflags: list[str]
    if pinned_runtime:
        # A bare ``-lcublasLt`` can silently skip a wheel's versioned-only ``.so.13`` and bind a
        # system ``libcublasLt.so.12``. In a pinned runtime arm, name the exact objects and record
        # their directory in RUNPATH so a cache entry can never acquire a different CUDA owner.
        runtime = pathlib.Path(pinned_runtime).expanduser().resolve()
        include = runtime.parent / "include"
        cublaslt = runtime / "libcublasLt.so.13"
        cublas = runtime / "libcublas.so.13"
        if not (include / "cublasLt.h").is_file() or not (include / "cublas_v2.h").is_file():
            raise RuntimeError(f"[ISOEXEC-PIK] pinned CUDA runtime has no coherent cuBLAS headers: {include}")
        if not cublaslt.is_file() or not cublas.is_file():
            raise RuntimeError(
                "[ISOEXEC-PIK] pinned CUDA runtime must provide libcublasLt.so.13 and libcublas.so.13: " f"{runtime}"
            )
        inc = [str(include)]
        extra_ldflags = [str(cublaslt), str(cublas), f"-Wl,-rpath,{runtime}"]
    else:
        inc, lib = [], []
        for base in (
            pathlib.Path(torch.__file__).parent.parent / "nvidia" / "cu13",
            pathlib.Path(torch.__file__).parent.parent / "nvidia" / "cublas",
            pathlib.Path("/usr/local/cuda"),
        ):
            if (base / "include" / "cublasLt.h").exists():
                inc.append(str(base / "include"))
            for sub in ("lib", "lib64"):
                if list((base / sub).glob("libcublasLt.so*")):
                    lib.append(str(base / sub))
        extra_ldflags = [f"-L{p}" for p in lib] + ["-lcublasLt", "-lcublas"]

    extension = load(
        name="pik_cublaslt",
        sources=[str(_HERE / "csrc" / "cublaslt_leaf.cpp")],
        extra_include_paths=inc,
        extra_ldflags=extra_ldflags,
        extra_cflags=["-O3"],
        build_directory=_ensure_build_dir(),
        verbose=bool(os.environ.get("PIK_VERBOSE")),
    )
    if pinned_runtime:
        _validate_pinned_extension(pathlib.Path(extension.__file__), runtime)
    return extension


def _ensure_build_dir() -> str:
    d = pathlib.Path(os.environ.get("PIK_CACHE", pathlib.Path.home() / ".cache" / "pik")) / "ext"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def available() -> bool:
    try:
        _ext()
        return True
    except Exception:  # noqa: BLE001
        return False


def leaf_gemm(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, beta: float = 0.0) -> torch.Tensor:
    """out = x @ w.T + beta*out.   x:[M,K] bf16  w:[N,K] bf16  out:[M,N] fp32.

    beta=1 accumulates, which lets the caller fuse the bottom level of the combine
    tree: leaf 2i with beta=0 then leaf 2i+1 with beta=1 into the same buffer is
    exactly the fp32 tree node (l_2i + l_2i+1).
    """
    _ext().leaf_gemm(x, w, out, beta)
    return out


def leaf_gemm_batched(x: torch.Tensor, w: torch.Tensor, p: torch.Tensor, leaf_k: int) -> torch.Tensor:
    """All m leaf partials in ONE strided-batch cuBLASLt call (batch dim = leaf index).

    x:[M, m*leaf_k] bf16  w:[N, m*leaf_k] bf16  p:[m, M, N] fp32 (or bf16 under a bf16-leaf
    plan). One pinned non-split-K algo per (leaf_k, N, lda, ldb, m) -- the key excludes M, so a
    single kernel serves every batch size. The caller folds p with tree_reduce_kernel; the
    admission bit-compare in pik/gemm.py is what licenses this path per shape.
    """
    _ext().leaf_gemm_batched(x, w, p, leaf_k)
    return p
