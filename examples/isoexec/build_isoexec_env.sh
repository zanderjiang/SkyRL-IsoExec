#!/usr/bin/env bash
# Build and verify the two CUDA-13 environments used by the IsoExec production launcher.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
local_root=${ISOEXEC_LOCAL_ROOT:-${HOME}/isoexec}   # venvs and caches; override per site
runtime_root=${ISOEXEC_RUNTIME_VENV:-${local_root}/venvs/skyrl-isoexec-cu130}
jit_root=${ISOEXEC_JIT_VENV:-${local_root}/venvs/skyrl-cuda130-jit}
wheel_root=${ISOEXEC_WHEEL_ROOT:-${repo}/pinned-wheels}   # see pinned-wheels/README.md
mode=build
replace=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verify-only) mode=verify; shift ;;
    --replace) replace=1; shift ;;
    -h|--help)
      echo "usage: $0 [--verify-only] [--replace]"
      exit 0
      ;;
    *) echo "usage: $0 [--verify-only] [--replace]" >&2; exit 2 ;;
  esac
done

command -v sha256sum >/dev/null || { echo "REFUSAL: sha256sum is required" >&2; exit 1; }

torch_wheel=${wheel_root}/torch-2.14.0.dev20260620+cu130-cp312-cp312-manylinux_2_28_x86_64.whl
triton_wheel=${wheel_root}/triton-3.7.1+git5d6048aa-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
vllm_wheel=${wheel_root}/vllm-1.0.0.dev20260620+cu130-cp312-cp312-linux_x86_64.whl
fa3_wheel=${wheel_root}/flash_attn_3-3.0.0-cp310-abi3-linux_x86_64.whl
te_torch_wheel=${wheel_root}/transformer_engine_torch-2.17.1-cp312-cp312-linux_x86_64.whl

require_hash() {
  local path=$1 expected=$2 observed
  [[ -f "${path}" ]] || { echo "REFUSAL: missing wheel ${path}" >&2; exit 1; }
  observed=$(sha256sum "${path}" | awk '{print $1}')
  [[ "${observed}" == "${expected}" ]] || {
    echo "REFUSAL: wheel drift ${path}: ${observed} != ${expected}" >&2
    exit 1
  }
}

verify_envs() {
  local runtime=$1 jit=$2
  [[ -x "${runtime}/bin/python" ]] || { echo "REFUSAL: runtime Python missing: ${runtime}" >&2; return 1; }
  [[ -x "${jit}/bin/python" ]] || { echo "REFUSAL: JIT Python missing: ${jit}" >&2; return 1; }
  "${runtime}/bin/python" - "${repo}" "${runtime}" <<'PY' || return 1
import importlib.metadata as md
import json
from pathlib import Path
import sys

repo = Path(sys.argv[1]).resolve()
runtime = Path(sys.argv[2]).resolve()
expected = {
    "torch": "2.14.0.dev20260620+cu130",
    "triton": "3.7.1+git5d6048aa",
    "vllm": "1.0.0.dev20260620+cu130",
    "vllm-router": "0.1.14.post1",
    "transformer-engine": "2.17.1",
    "transformer-engine-cu13": "2.17.1",
    "transformer-engine-torch": "2.17.1",
    "ray": "2.51.1",
    "flash-attn-3": "3.0.0",
    "flashinfer-python": "0.6.12",
    "megatron-core": "0.19.0+71e418ea7",
    "megatron-bridge": "0.6.0+91a15142",
    "nvidia-cublas": "13.2.0.9",
    "nvidia-cuda-runtime": "13.0.96",
    "nvidia-nccl-cu13": "2.30.7",
    "nvidia-cudnn-cu13": "9.23.1.3",
    "nvidia-nvshmem-cu13": "3.4.5",
}
observed = {name: md.version(name) for name in expected}
bad = {name: (expected[name], observed[name]) for name in expected if observed[name] != expected[name]}
if bad:
    raise SystemExit(f"REFUSAL: runtime package drift: {bad}")
import torch
import skyrl
if torch.version.cuda != "13.0":
    raise SystemExit(f"REFUSAL: torch CUDA drift: {torch.version.cuda!r}")
cuda_lib = runtime / "lib/python3.12/site-packages/nvidia/cu13/lib"
for alias_name, soname in {
    "libcublas.so": "libcublas.so.13",
    "libcublasLt.so": "libcublasLt.so.13",
}.items():
    alias = cuda_lib / alias_name
    if not alias.is_symlink() or alias.readlink() != Path(soname):
        raise SystemExit(f"REFUSAL: CUDA-13 loader alias drift: {alias} -> {alias.readlink() if alias.is_symlink() else '<missing>'}")
skyrl_path = Path(skyrl.__file__).resolve()
if not skyrl_path.is_relative_to(repo):
    raise SystemExit(f"REFUSAL: skyrl is not this checkout: {skyrl_path}")
if not Path(sys.executable).absolute().is_relative_to(runtime):
    raise SystemExit(f"REFUSAL: wrong runtime interpreter: {sys.executable}")
stub = repo / "examples/isoexec/nightly/_torchvision_stub"
sys.path.insert(0, str(stub))
import torchvision
if torchvision.__version__ != "0.0.0-isoexec-stub":
    raise SystemExit(f"REFUSAL: text-only torchvision surface drift: {torchvision.__file__}")
import vllm.model_executor.models.qwen3_5  # noqa: F401
print(json.dumps({"runtime": str(runtime), "packages": observed, "skyrl": str(skyrl_path)}, sort_keys=True))
PY

  "${jit}/bin/python" - "${jit}" <<'PY' || return 1
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import sys

jit = Path(sys.argv[1]).resolve()
expected = {
    "nvidia-cuda-cccl": "13.0.85",
    "nvidia-cuda-crt": "13.0.88",
    "nvidia-cuda-nvcc": "13.0.88",
    "nvidia-cuda-runtime": "13.0.96",
    "nvidia-curand": "10.4.0.35",
    "nvidia-nvvm": "13.0.88",
}
observed = {name: md.version(name) for name in expected}
bad = {name: (expected[name], observed[name]) for name in expected if observed[name] != expected[name]}
if bad:
    raise SystemExit(f"REFUSAL: JIT package drift: {bad}")
nvcc = jit / "lib/python3.12/site-packages/nvidia/cu13/bin/nvcc"
if not nvcc.is_file():
    raise SystemExit("REFUSAL: CUDA 13 JIT nvcc is missing")
cuda_root = jit / "lib/python3.12/site-packages/nvidia/cu13"
for relative, target in {
    "lib64": Path("lib"),
    "lib/libcudart.so": Path("libcudart.so.13"),
    "lib/libcurand.so": Path("libcurand.so.10"),
    "lib/stubs/libcuda.so": Path("/usr/lib64/libcuda.so.1"),
}.items():
    alias = cuda_root / relative
    if not alias.is_symlink() or alias.readlink() != target:
        raise SystemExit(f"REFUSAL: CUDA-13 JIT link-owner drift: {alias} -> {alias.readlink() if alias.is_symlink() else '<missing>'}")
version = subprocess.check_output([str(nvcc), "--version"], text=True)
if "release 13.0" not in version:
    raise SystemExit(f"REFUSAL: nvcc drift: {version.strip()}")
print(json.dumps({"jit": str(jit), "nvcc": str(nvcc), "packages": observed}, sort_keys=True))
PY
}

if [[ "${mode}" == verify ]]; then
  verify_envs "${runtime_root}" "${jit_root}"
  echo "ISOEXEC_ENV_VERIFY_PASS=1"
  exit 0
fi

command -v uv >/dev/null || { echo "REFUSAL: uv is required to build environments" >&2; exit 1; }

# flash-linear-attention is checked out, not installed -- see ops/gdn/gdn_fla_backward.py for why.
# Pinned: these kernels are part of the bitwise contract.
fla_commit=ebf3a0cff2be3e6f2b2f99820b8fe4e28855ced0
fla_dir=${ISOEXEC_FLA_SOURCE:-${repo}/third_party/flash-linear-attention}
if [[ ! -d "${fla_dir}/fla" ]]; then
  command -v git >/dev/null || { echo "REFUSAL: git is required to fetch flash-linear-attention" >&2; exit 1; }
  mkdir -p "$(dirname "${fla_dir}")"
  git clone --quiet https://github.com/fla-org/flash-linear-attention.git "${fla_dir}"
fi
observed_fla=$(git -C "${fla_dir}" rev-parse HEAD)
if [[ "${observed_fla}" != "${fla_commit}" ]]; then
  git -C "${fla_dir}" fetch --quiet origin "${fla_commit}" 2>/dev/null || git -C "${fla_dir}" fetch --quiet origin
  git -C "${fla_dir}" checkout --quiet "${fla_commit}"
fi
observed_fla=$(git -C "${fla_dir}" rev-parse HEAD)
[[ "${observed_fla}" == "${fla_commit}" ]] || {
  echo "REFUSAL: flash-linear-attention drift: ${observed_fla} != ${fla_commit}" >&2; exit 1; }
require_hash "${torch_wheel}" 063edf3d548cf57ddf0a6294121c815dec98477307890fff40dff29401c0154d
require_hash "${triton_wheel}" fd64476dcf7a3d56ce252bde145bb6cdb8a5d3785c6491428cf3899a0c0df544
require_hash "${vllm_wheel}" ad0f919d8f21f1f3b7c4f2b480d1c9372410ec64fbbb19fa62cf47e966f4fa20
require_hash "${fa3_wheel}" b802f459bc0bd8642ffeecf965990476b35775da3dd2fd6a5a6b24f982e712aa
require_hash "${te_torch_wheel}" 2a06bc89229bece34fb3bea384cb5a0655e60bd3eb664835478cf36ed8277921

if [[ -e "${runtime_root}" || -e "${jit_root}" ]]; then
  if verify_envs "${runtime_root}" "${jit_root}" >/dev/null 2>&1; then
    verify_envs "${runtime_root}" "${jit_root}"
    echo "ISOEXEC_ENV_ALREADY_CURRENT=1"
    exit 0
  fi
  [[ "${replace}" == 1 ]] || {
    echo "REFUSAL: target exists but is not exact; rerun with --replace to preserve it as a backup" >&2
    exit 1
  }
fi

runtime_parent=$(dirname "${runtime_root}")
jit_parent=$(dirname "${jit_root}")
mkdir -p "${runtime_parent}" "${jit_parent}"
runtime_stage=$(mktemp -d "${runtime_parent}/.isoexec-runtime.XXXXXX")
jit_stage=$(mktemp -d "${jit_parent}/.isoexec-jit.XXXXXX")
cleanup() {
  [[ -z "${runtime_stage:-}" || ! -d "${runtime_stage}" ]] || rm -rf -- "${runtime_stage}"
  [[ -z "${jit_stage:-}" || ! -d "${jit_stage}" ]] || rm -rf -- "${jit_stage}"
}
trap cleanup EXIT

uv venv --python 3.12 --relocatable "${runtime_stage}"
VIRTUAL_ENV="${runtime_stage}" UV_PROJECT_ENVIRONMENT="${runtime_stage}" \
  uv sync --project "${repo}" --frozen --extra isoexec --no-dev --active
uv pip install --no-config --python "${runtime_stage}/bin/python" --no-deps \
  "${torch_wheel}" \
  'nvidia-cublas==13.2.0.9' \
  'transformer-engine==2.17.1' \
  'transformer-engine-cu13==2.17.1' \
  "${te_torch_wheel}"
cuda13_runtime_lib=${runtime_stage}/lib/python3.12/site-packages/nvidia/cu13/lib
ln -s libcublas.so.13 "${cuda13_runtime_lib}/libcublas.so"
ln -s libcublasLt.so.13 "${cuda13_runtime_lib}/libcublasLt.so"

uv venv --python 3.12 --relocatable "${jit_stage}"
uv pip install --python "${jit_stage}/bin/python" \
  'nvidia-cuda-cccl==13.0.85' \
  'nvidia-cuda-crt==13.0.88' \
  'nvidia-cuda-nvcc==13.0.88' \
  'nvidia-cuda-runtime==13.0.96' \
  'nvidia-curand==10.4.0.35' \
  'nvidia-nvvm==13.0.88'
jit_cuda_root=${jit_stage}/lib/python3.12/site-packages/nvidia/cu13
mkdir -p "${jit_cuda_root}/lib/stubs"
ln -s lib "${jit_cuda_root}/lib64"
ln -s libcudart.so.13 "${jit_cuda_root}/lib/libcudart.so"
ln -s libcurand.so.10 "${jit_cuda_root}/lib/libcurand.so"
ln -s /usr/lib64/libcuda.so.1 "${jit_cuda_root}/lib/stubs/libcuda.so"
uv pip install --python "${jit_stage}/bin/python" --no-deps --editable "${repo}"
verify_envs "${runtime_stage}" "${jit_stage}"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
[[ ! -e "${runtime_root}" ]] || mv -- "${runtime_root}" "${runtime_root}.backup.${stamp}"
[[ ! -e "${jit_root}" ]] || mv -- "${jit_root}" "${jit_root}.backup.${stamp}"
mv -- "${runtime_stage}" "${runtime_root}"
mv -- "${jit_stage}" "${jit_root}"
runtime_stage=
jit_stage=
verify_envs "${runtime_root}" "${jit_root}"
echo "ISOEXEC_ENV_BUILD_PASS=1"
echo "runtime=${runtime_root}"
echo "jit=${jit_root}"
