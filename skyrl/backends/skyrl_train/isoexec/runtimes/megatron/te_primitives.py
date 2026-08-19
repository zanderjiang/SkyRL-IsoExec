"""Transformer Engine primitives for an otherwise exact local-spec trainer.

No TE model modules are installed: only implementation-neutral trainer primitives, and only after
the instantiated model is proven to belong entirely to the existing local-spec composition. The
first such primitive is Megatron's TE-backed multi-tensor L2 used for gradient-norm calculation.
Default-off, requested with ``SKYRL_ISOEXEC_TE_PRIMITIVES=1``.
"""

from __future__ import annotations

import importlib.metadata
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ENABLE_ENV = "SKYRL_ISOEXEC_TE_PRIMITIVES"
VALIDATED_TE_VERSION = "2.17.1"
VALIDATED_FLASHINFER_VERSION = "0.6.12"
JIT_CUDA_HOME_ENV = "SKYRL_FLASHINFER_JIT_CUDA_HOME"
RUNTIME_CUDA_LIB_ENV = "SKYRL_TE_CUDA_RUNTIME_LIB"
PIK_CACHE_ENV = "PIK_CACHE"
_VALIDATED_JIT_PACKAGES = {
    "nvidia-cuda-nvcc": "13.0.88",
    "nvidia-cuda-runtime": "13.0.96",
    "nvidia-cuda-crt": "13.0.88",
    "nvidia-cuda-cccl": "13.0.85",
    "nvidia-nvvm": "13.0.88",
    "nvidia-curand": "10.4.0.35",
}
_TE_DISTRIBUTIONS = (
    "transformer-engine",
    "transformer-engine-cu13",
    "transformer-engine-torch",
)

_COUNTS = {
    "admitted": 0,
    "models_validated": 0,
    "model_modules_scanned": 0,
    "optimizer_censuses": 0,
    "grad_norm_served": 0,
    "grad_norm_tensors": 0,
    "grad_norm_elements": 0,
}
_ORIGINAL_L2_NORM = None


def enabled() -> bool:
    """Whether the default-off TE-primitives mode was explicitly requested."""
    return os.environ.get(ENABLE_ENV, "0") == "1"


def stats() -> dict[str, int]:
    """Return a copy of the process-local engagement counters."""
    return dict(_COUNTS)


def _class_id(value) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{getattr(cls, '__module__', '?')}.{getattr(cls, '__qualname__', cls.__name__)}"


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _distribution_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _TE_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def compatibility_errors(*, env=None, versions=None) -> list[str]:
    """Return configuration/package reasons to refuse primitives-only TE."""
    env = os.environ if env is None else env
    versions = _distribution_versions() if versions is None else versions
    errors: list[str] = []
    if env.get("SKYRL_ISOEXEC", "0") != "1":
        errors.append("SKYRL_ISOEXEC=1 required")
    if env.get("SKYRL_ISOEXEC_LOCAL_SPEC", "0") != "1":
        errors.append("SKYRL_ISOEXEC_LOCAL_SPEC=1 required")
    if env.get("SKYRL_ISOEXEC_SELECTIVE_TE", "0") != "0":
        errors.append("SKYRL_ISOEXEC_SELECTIVE_TE=0 required")
    for name in _TE_DISTRIBUTIONS:
        found = versions.get(name)
        if found != VALIDATED_TE_VERSION:
            errors.append(f"{name}=={VALIDATED_TE_VERSION} required, found {found or 'absent'}")
    return errors


def _mapped_paths(maps_text: str) -> set[Path]:
    paths: set[Path] = set()
    for line in maps_text.splitlines():
        path = line.rsplit(maxsplit=1)[-1]
        if path.startswith("/"):
            paths.add(Path(path).resolve())
    return paths


def _metadata_versions(site_packages: Path) -> dict[str, str | None]:
    """Read distribution versions from a toolchain-only site-packages directory."""
    versions: dict[str, str | None] = {}
    for name in _VALIDATED_JIT_PACKAGES:
        normalized = name.replace("-", "_")
        metadata_files = list(site_packages.glob(f"{normalized}-*.dist-info/METADATA"))
        if len(metadata_files) != 1:
            versions[name] = None
            continue
        match = re.search(r"(?m)^Version:\s*(\S+)\s*$", metadata_files[0].read_text())
        versions[name] = match.group(1) if match else None
    return versions


def cuda_runtime_surface_identity(*, env=None) -> dict[str, object]:
    """Prove unversioned CUDA consumers cannot escape the pinned CUDA-13 runtime.

    NVIDIA's component wheels intentionally ship versioned cuBLAS objects only. Some generic
    consumers (including FlashInfer's autotuner metadata probe) use ``dlopen("libcublas.so")``.
    Without explicit aliases, that lookup can fall through to a system CUDA toolkit even when
    CUDA-13 objects are already mapped by PyTorch.
    """
    env = os.environ if env is None else env
    selected = env.get(RUNTIME_CUDA_LIB_ENV)
    if not selected:
        raise RuntimeError(f"[ISOEXEC-TE-PRIMITIVES] CUDA runtime surface refusal: {RUNTIME_CUDA_LIB_ENV} is required")
    runtime_lib = Path(selected).expanduser().resolve()
    errors: list[str] = []
    if not runtime_lib.is_dir():
        errors.append(f"missing CUDA runtime library directory: {runtime_lib}")

    aliases: dict[str, str] = {}
    for alias_name, versioned_name in (
        ("libcublas.so", "libcublas.so.13"),
        ("libcublasLt.so", "libcublasLt.so.13"),
    ):
        alias = runtime_lib / alias_name
        target = runtime_lib / versioned_name
        if not alias.is_symlink():
            errors.append(f"missing unversioned CUDA-13 alias: {alias}")
            continue
        try:
            selected_target = alias.resolve(strict=True)
            expected_target = target.resolve(strict=True)
        except FileNotFoundError:
            errors.append(f"broken unversioned CUDA-13 alias: {alias}")
            continue
        aliases[alias_name] = str(selected_target)
        if selected_target != expected_target:
            errors.append(f"{alias} resolves to {selected_target}, expected {expected_target}")

    ld_entries = [Path(value).expanduser().resolve() for value in env.get("LD_LIBRARY_PATH", "").split(":") if value]
    if runtime_lib not in ld_entries:
        errors.append(f"{runtime_lib} missing from LD_LIBRARY_PATH")
    else:
        for earlier in ld_entries[: ld_entries.index(runtime_lib)]:
            for alias_name in aliases:
                if (earlier / alias_name).exists():
                    errors.append(f"{earlier / alias_name} precedes pinned runtime alias in LD_LIBRARY_PATH")

    if errors:
        raise RuntimeError("[ISOEXEC-TE-PRIMITIVES] CUDA runtime surface refusal: " + "; ".join(errors))
    result: dict[str, object] = {"runtime_lib": str(runtime_lib), "aliases": aliases}
    print(
        "[ISOEXEC-TE-PRIMITIVES] CUDA-RUNTIME-SURFACE "
        f"runtime_lib={runtime_lib} " + " ".join(f"{name}={target}" for name, target in aliases.items()),
        flush=True,
    )
    return result


def _elf_needed(path: Path) -> set[str]:
    completed = subprocess.run(
        ["readelf", "-d", str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode:
        raise RuntimeError(f"readelf failed for {path} with exit code {completed.returncode}")
    return set(re.findall(r"Shared library: \[([^]]+)\]", completed.stdout))


def pik_extension_identity(*, env=None, maps_text: str | None = None, needed_by_path=None) -> dict[str, object]:
    """Census loaded/cached PIK extensions and reject a cross-CUDA cache entry."""
    env = os.environ if env is None else env
    mapped = _mapped_paths(Path("/proc/self/maps").read_text() if maps_text is None else maps_text)
    mapped_pik = {path for path in mapped if path.name.startswith("pik_") and path.suffix == ".so"}
    selected = env.get(PIK_CACHE_ENV)
    if not selected:
        if mapped_pik:
            raise RuntimeError(
                f"[ISOEXEC-TE-PRIMITIVES] PIK extension refusal: mapped PIK extensions require {PIK_CACHE_ENV}"
            )
        return {}
    cache = Path(selected).expanduser().resolve()
    errors: list[str] = []
    if not cache.is_absolute() or not cache.is_dir() or not os.access(cache, os.W_OK):
        errors.append(f"PIK_CACHE must be an existing writable absolute directory: {cache}")

    cached_pik = set(cache.rglob("pik_*.so")) if cache.is_dir() else set()
    extensions = cached_pik | mapped_pik
    needed_census: dict[str, list[str]] = {}
    for extension in sorted(extensions):
        resolved = extension.resolve()
        try:
            resolved.relative_to(cache)
        except ValueError:
            errors.append(f"mapped PIK extension escaped PIK_CACHE: {resolved}")
        try:
            needed = set(needed_by_path[str(resolved)]) if needed_by_path is not None else _elf_needed(resolved)
        except (KeyError, RuntimeError) as exc:
            errors.append(str(exc))
            continue
        needed_census[str(resolved)] = sorted(needed)
        incompatible = sorted(
            name for name in needed if name in ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12")
        )
        if incompatible:
            errors.append(f"{resolved} requires incompatible CUDA-12 libraries: {', '.join(incompatible)}")
        if resolved.name == "pik_cublaslt.so" and "libcublasLt.so.13" not in needed:
            errors.append(f"{resolved} does not require pinned libcublasLt.so.13")

    if errors:
        raise RuntimeError("[ISOEXEC-TE-PRIMITIVES] PIK extension refusal: " + "; ".join(errors))
    result: dict[str, object] = {
        "cache": str(cache),
        "cached": len(cached_pik),
        "mapped": len(mapped_pik),
        "needed": needed_census,
    }
    print(
        f"[ISOEXEC-TE-PRIMITIVES] PIK-EXTENSIONS cache={cache} "
        f"cached={len(cached_pik)} mapped={len(mapped_pik)} cuda12_needed=0",
        flush=True,
    )
    return result


def flashinfer_cublas_identity() -> dict[str, str]:
    """Exercise FlashInfer's unversioned cuBLAS probe and match it to its wheel owner."""
    from flashinfer.autotuner import _get_cublas_version

    found = _get_cublas_version()
    try:
        wheel = importlib.metadata.version("nvidia-cublas")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("[ISOEXEC-TE-PRIMITIVES] nvidia-cublas distribution is absent") from exc
    expected = ".".join(wheel.split(".")[:3])
    if found != expected:
        raise RuntimeError(
            "[ISOEXEC-TE-PRIMITIVES] FlashInfer cuBLAS refusal: "
            f"autotuner loaded {found}, nvidia-cublas wheel owns {expected}"
        )
    result = {"flashinfer": found, "nvidia_cublas": wheel}
    print(
        f"[ISOEXEC-TE-PRIMITIVES] FLASHINFER-CUBLAS loaded={found} nvidia_cublas={wheel}",
        flush=True,
    )
    return result


def jit_toolchain_identity(
    *,
    env=None,
    package_versions=None,
    flashinfer_version: str | None = None,
    nvcc_output: str | None = None,
    runtime_header_text: str | None = None,
    torch_cuda: str | None = None,
) -> dict[str, object]:
    """Prove FlashInfer JIT uses one compiler/header ABI, separate from runtime libraries.

    NVIDIA component wheels share the ``nvidia/cu13`` namespace, so an unconstrained install can
    silently combine a newer ``nvcc`` with older CUDA runtime headers. CCCL correctly refuses that
    composition during JIT. This census runs before model allocation and again in Ray actors.
    """
    env = os.environ if env is None else env
    selected = env.get(JIT_CUDA_HOME_ENV)
    if not selected:
        return {}
    cuda_home = Path(selected).expanduser().resolve()
    configured_cuda_home = Path(env.get("CUDA_HOME", "")).expanduser().resolve()
    configured_cuda_lib = Path(env.get("CUDA_LIB_PATH", "")).expanduser().resolve()
    workspace = Path(env.get("FLASHINFER_WORKSPACE_BASE", "")).expanduser()
    nvcc = cuda_home / "bin/nvcc"
    runtime_header = cuda_home / "include/cuda_runtime_api.h"
    errors: list[str] = []
    if configured_cuda_home != cuda_home:
        errors.append(f"CUDA_HOME={configured_cuda_home} differs from {JIT_CUDA_HOME_ENV}={cuda_home}")
    expected_cuda_lib = (cuda_home / "lib64").resolve()
    if configured_cuda_lib != expected_cuda_lib:
        errors.append(f"CUDA_LIB_PATH={configured_cuda_lib} differs from JIT runtime surface {expected_cuda_lib}")
    elif (configured_cuda_lib / "libcudart.so.12").exists():
        errors.append(f"CUDA_LIB_PATH exposes incompatible libcudart.so.12: {configured_cuda_lib}")
    if not nvcc.is_file() or not os.access(nvcc, os.X_OK):
        errors.append(f"missing executable nvcc: {nvcc}")
    if not runtime_header.is_file():
        errors.append(f"missing CUDA runtime header: {runtime_header}")
    for link_name in (
        "lib64/libcudart.so",
        "lib64/libcurand.so",
        "lib64/stubs/libcuda.so",
    ):
        if not (cuda_home / link_name).is_file():
            errors.append(f"missing FlashInfer JIT link owner: {cuda_home / link_name}")
    if not workspace.is_absolute() or not workspace.is_dir() or not os.access(workspace, os.W_OK):
        errors.append(f"FLASHINFER_WORKSPACE_BASE must be an existing writable absolute directory: {workspace}")

    site_packages = cuda_home.parent.parent
    found_packages = _metadata_versions(site_packages) if package_versions is None else package_versions
    for name, expected in _VALIDATED_JIT_PACKAGES.items():
        if found_packages.get(name) != expected:
            errors.append(f"{name}=={expected} required in JIT toolchain, found {found_packages.get(name) or 'absent'}")

    if flashinfer_version is None:
        try:
            flashinfer_version = importlib.metadata.version("flashinfer-python")
        except importlib.metadata.PackageNotFoundError:
            flashinfer_version = None
    if flashinfer_version != VALIDATED_FLASHINFER_VERSION:
        errors.append(
            f"flashinfer-python=={VALIDATED_FLASHINFER_VERSION} required, found {flashinfer_version or 'absent'}"
        )

    if nvcc_output is None and nvcc.is_file():
        completed = subprocess.run(
            [str(nvcc), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        nvcc_output = completed.stdout + completed.stderr
        if completed.returncode:
            errors.append(f"nvcc --version failed with exit code {completed.returncode}")
    nvcc_match = re.search(r"\brelease\s+(\d+)\.(\d+)\b", nvcc_output or "")
    compiler_version = tuple(map(int, nvcc_match.groups())) if nvcc_match else None
    if compiler_version is None:
        errors.append("could not parse nvcc major.minor")

    if runtime_header_text is None and runtime_header.is_file():
        runtime_header_text = runtime_header.read_text()
    header_match = re.search(r"(?m)^#define\s+CUDART_VERSION\s+(\d+)\s*$", runtime_header_text or "")
    header_code = int(header_match.group(1)) if header_match else None
    header_version = (header_code // 1000, (header_code % 1000) // 10) if header_code is not None else None
    if header_version is None:
        errors.append("could not parse CUDART_VERSION from cuda_runtime_api.h")
    if compiler_version is not None and header_version is not None and compiler_version != header_version:
        errors.append(f"CUDA compiler/header mismatch: nvcc={compiler_version} CUDART={header_version}")

    if torch_cuda is None:
        import torch

        torch_cuda = torch.version.cuda
    torch_match = re.fullmatch(r"(\d+)\.(\d+)", torch_cuda or "")
    torch_version = tuple(map(int, torch_match.groups())) if torch_match else None
    if torch_version is None:
        errors.append(f"could not parse torch CUDA version: {torch_cuda!r}")
    elif header_version is not None and torch_version != header_version:
        errors.append(f"CUDA header/torch build mismatch: CUDART={header_version} torch={torch_version}")

    if errors:
        raise RuntimeError("[ISOEXEC-TE-PRIMITIVES] FlashInfer JIT toolchain refusal: " + "; ".join(errors))
    result: dict[str, object] = {
        "cuda_home": str(cuda_home),
        "nvcc": ".".join(map(str, compiler_version)),
        "cudart_headers": ".".join(map(str, header_version)),
        "torch_cuda": torch_cuda,
        "flashinfer": flashinfer_version,
        "packages": dict(found_packages),
        "workspace": str(workspace.resolve()),
    }
    print(
        "[ISOEXEC-TE-PRIMITIVES] FLASHINFER-JIT "
        f"cuda_home={cuda_home} nvcc={result['nvcc']} cudart_headers={result['cudart_headers']} "
        f"torch_cuda={torch_cuda} flashinfer={flashinfer_version} workspace={workspace.resolve()}",
        flush=True,
    )
    return result


def runtime_identity(role: str, *, maps_text: str | None = None) -> dict[str, object]:
    """Fail closed if a Ray actor escaped the production arm's pinned interpreter/runtime."""
    selected = os.environ.get("SKYRL_RAY_PY_EXECUTABLE")
    if not selected:
        return {}
    expected = Path(selected).expanduser().absolute()
    actual = Path(sys.executable).absolute()
    expected_prefix = expected.parent.parent
    actual_prefix = Path(sys.prefix).absolute()
    if actual != expected or actual_prefix != expected_prefix:
        raise RuntimeError(
            f"[ISOEXEC-TE-PRIMITIVES] {role} interpreter drift: expected executable/prefix "
            f"{expected}/{expected_prefix}, found {actual}/{actual_prefix}"
        )
    versions = _distribution_versions()
    errors = [
        f"{name}=={VALIDATED_TE_VERSION} required, found {version or 'absent'}"
        for name, version in versions.items()
        if version != VALIDATED_TE_VERSION
    ]
    flashinfer_cublas = flashinfer_cublas_identity() if role.startswith("engine") else {}
    prefix = Path(sys.prefix).resolve()
    mapped = _mapped_paths(Path("/proc/self/maps").read_text() if maps_text is None else maps_text)

    def under_prefix(path: Path) -> bool:
        try:
            path.relative_to(prefix)
        except ValueError:
            return False
        return True

    required_runtime = (
        "libcudart.so.13",
        "libcublas.so.13",
        "libcublasLt.so.13",
        "libcudnn.so.9",
        "libnccl.so.2",
    )
    selected_runtime: dict[str, str] = {}
    for name in required_runtime:
        matches = {path for path in mapped if name in path.name}
        if len(matches) != 1:
            errors.append(f"{role} expected one mapped {name}, found {len(matches)}")
            continue
        path = next(iter(matches))
        selected_runtime[name] = str(path)
        if not under_prefix(path):
            errors.append(f"{role} selected {name} outside environment prefix: {path}")
    escaped = sorted(
        path
        for path in mapped
        if path.name.startswith(("libcudart", "libcublas", "libcudnn", "libnccl", "libnvshmem"))
        and not under_prefix(path)
    )
    if escaped:
        errors.append(f"{role} selected mixed runtime libraries: " + ", ".join(map(str, escaped)))
    if errors:
        raise RuntimeError(f"[ISOEXEC-TE-PRIMITIVES] {role} runtime refusal: " + "; ".join(errors))
    result: dict[str, object] = {
        "role": role,
        "python": str(actual),
        "prefix": str(prefix),
        "versions": versions,
        "runtime_libraries": selected_runtime,
        "runtime_surface": cuda_runtime_surface_identity(),
        "flashinfer_jit": jit_toolchain_identity(),
        "flashinfer_cublas": flashinfer_cublas,
        "pik_extensions": pik_extension_identity(maps_text=maps_text),
    }
    print(
        f"[ISOEXEC-TE-PRIMITIVES] RUNTIME role={role} python={actual} "
        + " ".join(f"{name}={version}" for name, version in versions.items())
        + " "
        + " ".join(f"{name}={path}" for name, path in selected_runtime.items()),
        flush=True,
    )
    return result


def loader_errors(*, extension_file: str, maps_text: str, prefix: str | None = None) -> list[str]:
    """Validate the pinned extension and CUDA-13 libraries selected by the loader."""
    prefix_path = Path(sys.prefix if prefix is None else prefix).resolve()
    extension_path = Path(extension_file).resolve()
    mapped = _mapped_paths(maps_text)
    errors: list[str] = []

    def under_prefix(path: Path) -> bool:
        try:
            path.relative_to(prefix_path)
        except ValueError:
            return False
        return True

    if not under_prefix(extension_path):
        errors.append(f"TE torch extension escaped environment prefix: {extension_path}")
    if "transformer_engine_torch" not in extension_path.name or extension_path.suffix != ".so":
        errors.append(f"unexpected TE torch extension path: {extension_path}")

    required_names = (
        "transformer_engine_torch",
        "libtransformer_engine.so",
        "libcudart.so.13",
        "libcublas.so.13",
        "libcublasLt.so.13",
        "libcudnn.so.9",
        "libnccl.so.2",
    )
    for name in required_names:
        matches = {path for path in mapped if name in path.name}
        if len(matches) != 1:
            errors.append(f"loader expected one mapped {name}, found {len(matches)}")
            continue
        selected = next(iter(matches))
        if not under_prefix(selected):
            errors.append(f"loader selected {name} outside environment prefix: {selected}")
    incompatible = sorted(
        path for path in mapped if path.name in ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12")
    )
    if incompatible:
        errors.append("loader selected CUDA-12 runtime: " + ", ".join(map(str, incompatible)))
    # The cluster preamble points at system CUDA/cuDNN/NCCL installations.  A partial
    # LD_LIBRARY_PATH can therefore produce a process with the pinned TE extension but a second,
    # incompatible runtime library mapped from /opt or the base environment.  Refuse every such
    # mixed owner, not merely the first required basename found above.
    runtime_prefixes = ("libcudart", "libcublas", "libcudnn", "libnccl", "libnvshmem")
    escaped_runtime = sorted(
        path for path in mapped if path.name.startswith(runtime_prefixes) and not under_prefix(path)
    )
    if escaped_runtime:
        errors.append(
            "loader selected CUDA runtime libraries outside environment prefix: " + ", ".join(map(str, escaped_runtime))
        )
    return errors


def _model_roots(model) -> list[object]:
    if isinstance(model, (list, tuple)):
        return list(model)
    return [model]


def validate_local_model(model) -> int:
    """Refuse any TE-owned module in the final local-spec model tree."""
    forbidden: list[str] = []
    scanned = 0
    for root in _model_roots(model):
        for name, module in root.named_modules():
            scanned += 1
            owner = _class_id(module)
            owner_lower = owner.lower()
            if owner_lower.startswith("transformer_engine.") or ".extensions.transformer_engine" in owner_lower:
                forbidden.append(f"{name or '<root>'}:{owner}")
    if forbidden:
        raise RuntimeError(
            "[ISOEXEC-TE-PRIMITIVES] local-spec model contains TE module owner(s): " + ", ".join(forbidden[:16])
        )
    if scanned == 0:
        raise RuntimeError("[ISOEXEC-TE-PRIMITIVES] local-spec model census scanned zero modules")
    _COUNTS["models_validated"] += 1
    _COUNTS["model_modules_scanned"] += scanned
    print(
        "[ISOEXEC-TE-PRIMITIVES] "
        f"models_validated={_COUNTS['models_validated']} "
        f"model_modules_scanned={_COUNTS['model_modules_scanned']} te_model_modules=0",
        flush=True,
    )
    return scanned


def _tensor_totals(tensor_lists: object) -> tuple[int, int]:
    tensors = elements = 0

    def visit(value) -> None:
        nonlocal tensors, elements
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
            return
        numel = getattr(value, "numel", None)
        if callable(numel):
            tensors += 1
            elements += int(numel())

    visit(tensor_lists)
    return tensors, elements


def _instrument_l2_norm(clip_grads, te_l2_norm) -> None:
    """Instrument the actual Megatron binding while delegating to the same TE callable."""
    global _ORIGINAL_L2_NORM
    current = clip_grads.l2_norm_impl
    if getattr(current, "_isoexec_te_primitives_wrapper", False):
        return
    if current is not te_l2_norm:
        raise RuntimeError(
            "[ISOEXEC-TE-PRIMITIVES] grad-norm ABI drift: Megatron binding "
            f"{getattr(current, '__module__', '?')}.{getattr(current, '__name__', type(current).__name__)} "
            "is not transformer_engine_torch.multi_tensor_l2norm"
        )
    if (
        getattr(te_l2_norm, "__module__", None) != "transformer_engine_torch"
        or getattr(te_l2_norm, "__name__", None) != "multi_tensor_l2norm"
    ):
        raise RuntimeError("[ISOEXEC-TE-PRIMITIVES] unexpected TE 2.17.1 multi-tensor L2 ABI")

    _ORIGINAL_L2_NORM = te_l2_norm

    def counted_l2_norm(*args, **kwargs):
        tensor_lists = kwargs.get("tensor_lists", args[2] if len(args) > 2 else ())
        tensors, elements = _tensor_totals(tensor_lists)
        result = te_l2_norm(*args, **kwargs)
        _COUNTS["grad_norm_served"] += 1
        _COUNTS["grad_norm_tensors"] += tensors
        _COUNTS["grad_norm_elements"] += elements
        if _is_power_of_two(_COUNTS["grad_norm_served"]):
            print(
                "[ISOEXEC-TE-PRIMITIVES] "
                f"grad_norm_served={_COUNTS['grad_norm_served']} "
                f"grad_norm_tensors={_COUNTS['grad_norm_tensors']} "
                f"grad_norm_elements={_COUNTS['grad_norm_elements']}",
                flush=True,
            )
        return result

    counted_l2_norm._isoexec_te_primitives_wrapper = True
    counted_l2_norm._isoexec_te_original = te_l2_norm
    clip_grads.l2_norm_impl = counted_l2_norm


def preflight(*, env=None, versions=None, maps_text: str | None = None, prefix: str | None = None):
    """Prove the requested TE package/loader identity before model allocation."""
    if not enabled():
        raise RuntimeError(f"[ISOEXEC-TE-PRIMITIVES] preflight requires {ENABLE_ENV}=1")
    errors = compatibility_errors(env=env, versions=versions)
    if errors:
        raise RuntimeError("[ISOEXEC-TE-PRIMITIVES] compatibility refusal: " + "; ".join(errors))
    cuda_runtime_surface_identity(env=env)
    jit_toolchain_identity(env=env)
    pik_extension_identity(env=env, maps_text=maps_text or "")

    import transformer_engine.pytorch  # noqa: F401
    import transformer_engine_torch
    from megatron.core.optimizer import clip_grads

    if maps_text is None:
        maps_text = Path("/proc/self/maps").read_text()
    errors = loader_errors(
        extension_file=transformer_engine_torch.__file__,
        maps_text=maps_text,
        prefix=prefix,
    )
    if errors:
        raise RuntimeError("[ISOEXEC-TE-PRIMITIVES] loader refusal: " + "; ".join(errors))

    found_versions = _distribution_versions() if versions is None else versions
    print(
        "[ISOEXEC-TE-PRIMITIVES] PREFLIGHT "
        f"python={Path(sys.executable).absolute()} prefix={Path(sys.prefix).absolute()} "
        + " ".join(f"{name}={found_versions.get(name)}" for name in _TE_DISTRIBUTIONS)
        + f" extension={Path(transformer_engine_torch.__file__).resolve()} loader=CUDA13-only",
        flush=True,
    )
    return clip_grads, transformer_engine_torch.multi_tensor_l2norm


def admit(
    model,
    *,
    env=None,
    versions=None,
    maps_text: str | None = None,
    prefix: str | None = None,
) -> None:
    """Admit TE primitives without allowing TE to own any model module or forward."""
    if not enabled():
        return
    clip_grads, te_l2_norm = preflight(env=env, versions=versions, maps_text=maps_text, prefix=prefix)
    runtime_identity("trainer")

    validate_local_model(model)
    _instrument_l2_norm(clip_grads, te_l2_norm)
    _COUNTS["admitted"] += 1
    print(
        "[ISOEXEC-TE-PRIMITIVES] "
        f"admitted={_COUNTS['admitted']} te={VALIDATED_TE_VERSION} local_spec=1 selective_te=0 "
        f"te_model_modules=0 grad_norm_served={_COUNTS['grad_norm_served']} default_off=1",
        flush=True,
    )


def _optimizer_children(optimizer) -> Iterable[object]:
    state = vars(optimizer)
    yield from (state.get("chained_optimizers", ()) or ())
    yield from (state.get("cpu_optimizers", ()) or ())
    gpu_optimizer = state.get("gpu_optimizer")
    if gpu_optimizer is not None:
        yield gpu_optimizer
    inner = state.get("optimizer")
    if inner is not None:
        yield inner


def _optimizer_leaves(optimizer) -> tuple[list[object], list[object]]:
    nodes: list[object] = []
    leaves: list[object] = []
    seen: set[int] = set()

    def visit(node) -> None:
        if node is None or id(node) in seen:
            return
        seen.add(id(node))
        nodes.append(node)
        children = list(_optimizer_children(node))
        if children:
            for child in children:
                visit(child)
        else:
            leaves.append(node)

    visit(optimizer)
    return nodes, leaves


def _optimizer_param_numel(optimizer) -> tuple[int, int]:
    cpu = cuda = 0
    seen: set[int] = set()
    for group in getattr(optimizer, "param_groups", ()):
        for param in group.get("params", ()):
            if id(param) in seen:
                continue
            seen.add(id(param))
            if getattr(param, "is_cuda", False):
                cuda += int(param.numel())
            else:
                cpu += int(param.numel())
    return cpu, cuda


def census_optimizer(optimizer, config) -> dict[str, object]:
    """Report the concrete leaf optimizers and their real CPU/CUDA ownership."""
    if not enabled():
        return {}
    nodes, leaves = _optimizer_leaves(optimizer)
    if not leaves:
        raise RuntimeError("[ISOEXEC-TE-PRIMITIVES] optimizer census found no executed leaf owner")

    # Configuration is not execution ownership.  Require the wrapper that implements offload and
    # its resolved fraction to agree with OptimizerConfig before reporting any offload fact.
    hybrid = [node for node in nodes if type(node).__name__ == "HybridDeviceOptimizer"]
    offload_enabled = bool(getattr(config, "optimizer_cpu_offload", False))
    expected_fraction = float(getattr(config, "optimizer_offload_fraction", 0.0) or 0.0)
    if offload_enabled != bool(hybrid):
        raise RuntimeError(
            "[ISOEXEC-TE-PRIMITIVES] optimizer offload owner drift: "
            f"config={offload_enabled} hybrid_owners={len(hybrid)}"
        )
    for owner in hybrid:
        actual_fraction = float(getattr(owner, "offload_fraction", -1.0))
        if actual_fraction != expected_fraction:
            raise RuntimeError(
                "[ISOEXEC-TE-PRIMITIVES] optimizer offload fraction drift: "
                f"config={expected_fraction} actual={actual_fraction}"
            )

    cpu_numel = cuda_numel = te_gpu_numel = 0
    owner_totals: dict[tuple[str, str], int] = {}
    for leaf in leaves:
        owner = _class_id(leaf)
        leaf_cpu, leaf_cuda = _optimizer_param_numel(leaf)
        cpu_numel += leaf_cpu
        cuda_numel += leaf_cuda
        if leaf_cpu:
            owner_totals[(owner, "cpu")] = owner_totals.get((owner, "cpu"), 0) + leaf_cpu
        if leaf_cuda:
            owner_totals[(owner, "cuda")] = owner_totals.get((owner, "cuda"), 0) + leaf_cuda
            if owner.lower().startswith("transformer_engine."):
                te_gpu_numel += leaf_cuda

    full_cpu_offload = offload_enabled and expected_fraction == 1.0
    if full_cpu_offload and cuda_numel:
        raise RuntimeError(
            "[ISOEXEC-TE-PRIMITIVES] full CPU optimizer offload owns CUDA parameters: " f"cuda_param_numel={cuda_numel}"
        )

    _COUNTS["optimizer_censuses"] += 1
    owners = ",".join(f"{owner}@{device}:{numel}" for (owner, device), numel in sorted(owner_totals.items()))
    te_gpu_traffic: int | str = 0 if te_gpu_numel == 0 else "UNVERIFIED"
    result = {
        "leaf_owners": len(leaves),
        "cpu_param_numel": cpu_numel,
        "cuda_param_numel": cuda_numel,
        "te_optimizer_gpu_param_numel": te_gpu_numel,
        "te_optimizer_gpu_traffic": te_gpu_traffic,
        "full_cpu_offload": full_cpu_offload,
    }
    print(
        "[ISOEXEC-TE-PRIMITIVES] "
        f"optimizer_censuses={_COUNTS['optimizer_censuses']} leaf_owners={len(leaves)} "
        f"cpu_param_numel={cpu_numel} cuda_param_numel={cuda_numel} "
        f"te_optimizer_gpu_param_numel={te_gpu_numel} te_optimizer_gpu_traffic={te_gpu_traffic} "
        f"full_cpu_offload={int(full_cpu_offload)} owners={owners or '<empty>'} "
        f"grad_norm_served={_COUNTS['grad_norm_served']}",
        flush=True,
    )
    return result


if __name__ == "__main__":
    preflight()
