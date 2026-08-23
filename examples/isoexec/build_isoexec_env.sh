#!/usr/bin/env bash
# Build and verify the two CUDA-13 environments used by the IsoExec launcher.
#
#   runtime  <local_root>/venvs/skyrl-isoexec-cu130   uv sync of the `isoexec` extra (+ TE 2.17.1)
#   jit      <local_root>/venvs/skyrl-cuda130-jit     nvcc toolchain for FlashInfer/TE JIT
#
# Stages: wheels -> fla -> build (staged, atomic promote) -> verify. Every stage refuses on drift.
# The wheels stage fills pinned-wheels/ — the directory uv.lock resolves the isoexec extra from —
# with symlinks into ISOEXEC_WHEEL_CACHE. Each wheels.txt entry is fetched (ISOEXEC_WHEEL_MIRROR,
# then upstream) and sha256-checked; one that cannot be fetched is built, as the last resort, from
# its sources.txt recipe in a third venv, <local_root>/venvs/skyrl-cuda130-build (torch+nvcc+cmake+rust).
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
local_root=${ISOEXEC_LOCAL_ROOT:-${HOME}/isoexec}
runtime_root=${ISOEXEC_RUNTIME_VENV:-${local_root}/venvs/skyrl-isoexec-cu130}
jit_root=${ISOEXEC_JIT_VENV:-${local_root}/venvs/skyrl-cuda130-jit}
build_root=${ISOEXEC_BUILD_VENV:-${local_root}/venvs/skyrl-cuda130-build}
wheel_dir=${repo}/pinned-wheels
wheel_cache=${ISOEXEC_WHEEL_CACHE:-${local_root}/wheels}   # fetched and built wheels live here
wheel_mirror=${ISOEXEC_WHEEL_MIRROR-https://github.com/zanderjiang/SkyRL-IsoExec/releases/download/wheels-20260620}  # serves every wheels.txt filename ('+' as '.'); empty disables
build_jobs=${ISOEXEC_BUILD_JOBS:-$(nproc)}
fla_dir=${ISOEXEC_FLA_SOURCE:-${repo}/third_party/flash-linear-attention}
fla_commit=ebf3a0cff2be3e6f2b2f99820b8fe4e28855ced0
py=3.12

usage() { echo "usage: $0 [--wheels-only | --verify-only] [--replace]"; }
refuse() { echo "REFUSAL: $*" >&2; exit 1; }
log() { echo "[build_isoexec_env] $*" >&2; }

mode=build replace=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --wheels-only) mode=wheels ;;
    --verify-only) mode=verify ;;
    --replace) replace=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done

# ---------------------------------------------------------------- wheels
fetch() {  # fetch <name> <dest> <upstream-url|->
  local name=$1 dest=$2 upstream=$3 url
  for url in ${wheel_mirror:+${wheel_mirror%/}/${name//+/.}} ${upstream#-}; do
    log "fetching ${name} from ${url}"
    if curl -fsSL --retry 3 -o "${dest}.part" "${url}"; then mv -- "${dest}.part" "${dest}"; return 0; fi
    rm -f -- "${dest}.part"
  done
  return 1
}

source_built=()  # wheels.txt names whose file was built here (lock hash does not apply)

ensure_wheels() {
  local sha name upstream dest cached observed
  mkdir -p "${wheel_cache}"
  while read -r sha name upstream; do
    [[ -n "${sha}" && "${sha}" != \#* ]] || continue
    dest=${wheel_dir}/${name} cached=${wheel_cache}/${name}
    if [[ ! -f "${dest}" ]]; then
      [[ -f "${cached}" ]] || fetch "${name}" "${cached}" "${upstream}" || build_wheel "${name}" ||
        refuse "cannot obtain ${name}: not in pinned-wheels/, ${wheel_cache}, ISOEXEC_WHEEL_MIRROR, upstream, or sources.txt (see pinned-wheels/README.md)"
      ln -sf -- "${cached}" "${dest}"
    fi
    observed=$(sha256sum "${dest}" | cut -d' ' -f1)
    if [[ "${observed}" == "${sha}" ]]; then
      continue
    elif [[ -f "${wheel_cache}/${name}.sha256" && "$(cat "${wheel_cache}/${name}.sha256")" == "${observed}" ]]; then
      log "${name}: source-built (sha256 ${observed}); identity rests on its sources.txt recipe"
      source_built+=("${name}")
    else
      refuse "wheel drift ${name}: sha256 ${observed} != ${sha}"
    fi
  done < "${wheel_dir}/wheels.txt"
  log "pinned wheels present and hash-verified"
}

wheel_path() { awk -v n="$1" '$2 ~ n {print $2}' "${wheel_dir}/wheels.txt" | sed "s|^|${wheel_dir}/|"; }
dist_name() { local stem=${1%%-*}; echo "${stem//_/-}"; }
is_source_built() { local w; for w in "${source_built[@]}"; do [[ "${w}" == "$1" ]] && return 0; done; return 1; }

# ---------------------------------------------------------------- source builds
install_cuda13() {  # install_cuda13 <venv>: nvcc 13.0 from wheels, laid out as a CUDA_HOME
  local venv=$1 cuda=$1/lib/python${py}/site-packages/nvidia/cu13 f
  uv pip install --python "${venv}/bin/python" \
    'nvidia-cuda-cccl==13.0.85' 'nvidia-cuda-crt==13.0.88' 'nvidia-cuda-nvcc==13.0.88' \
    'nvidia-cuda-runtime==13.0.96' 'nvidia-curand==10.4.0.35' 'nvidia-nvvm==13.0.88'
  mkdir -p "${cuda}/lib/stubs"
  [[ -e "${cuda}/lib64" ]] || ln -s lib "${cuda}/lib64"
  for f in "${cuda}"/lib/lib*.so.*; do
    [[ -e "${f%%.so.*}.so" ]] || ln -s "$(basename "${f}")" "${f%%.so.*}.so"
  done
  ln -sf /usr/lib64/libcuda.so.1 "${cuda}/lib/stubs/libcuda.so"
}

build_toolchain() {  # build_toolchain <stage>: pinned torch + its CUDA-13 libraries, nvcc, build tools
  local stage=$1
  uv venv --python "${py}" --relocatable "${stage}"
  uv pip install --python "${stage}/bin/python" "$(wheel_path '^torch-')" "$(wheel_path '^triton-')" \
    cmake ninja setuptools setuptools-scm setuptools-rust wheel packaging jinja2 pybind11 numpy einops
  install_cuda13 "${stage}"
  curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs |
    RUSTUP_HOME="${stage}/rust" CARGO_HOME="${stage}/rust" sh -s -- -y --no-modify-path --profile minimal --default-toolchain none >/dev/null
}

ensure_toolchain() {
  local stage
  [[ -x "${build_root}/bin/python" && -x "${build_root}/rust/bin/cargo" ]] && return 0
  [[ -f "$(wheel_path '^torch-')" ]] || refuse "pinned torch wheel is required before anything can be built from source"
  mkdir -p "$(dirname "${build_root}")"
  stage=$(mktemp -d "$(dirname "${build_root}")/.isoexec-build.XXXXXX")
  build_toolchain "${stage}"
  rm -rf -- "${build_root}"
  mv -- "${stage}" "${build_root}"
}

checkout_source() {  # checkout_source <source> <dir>
  local source=$1 dir=$2 url ref sha
  rm -rf -- "${dir}"; mkdir -p "${dir}"
  case "${source}" in
    git+*)
      url=${source#git+}; url=${url%%#*}; ref=${url##*@}; url=${url%@*}
      git -C "${dir}" init --quiet
      git -C "${dir}" fetch --quiet --depth 1 "${url}" "${ref}"
      git -C "${dir}" checkout --quiet FETCH_HEAD
      [[ "$(git -C "${dir}" rev-parse HEAD)" == "${ref}" ]] || refuse "source drift: ${url} is not at ${ref}" ;;
    *)
      url=${source%%#*}; sha=${source##*#sha256=}
      curl -fsSL --retry 3 -o "${dir}/sdist.tar.gz" "${url}"
      [[ "$(sha256sum "${dir}/sdist.tar.gz" | cut -d' ' -f1)" == "${sha}" ]] || refuse "sdist drift: ${url}"
      tar -xzf "${dir}/sdist.tar.gz" -C "${dir}" --strip-components=1 ;;
  esac
}

build_wheel() {  # build_wheel <name>: sources.txt recipe -> ${wheel_cache}/<name> (+ .sha256)
  local name=$1 source build_env subdir src out cuda
  read -r source build_env < <(awk -v n="${name}" '$1 == n {$1 = ""; print substr($0, 2)}' "${wheel_dir}/sources.txt")
  [[ -n "${source:-}" ]] || return 1
  subdir=${source##*#subdirectory=}; [[ "${subdir}" != "${source}" ]] || subdir=
  ensure_toolchain
  src=${local_root}/src/$(dist_name "${name}")
  out=$(mktemp -d "${wheel_cache}/.build.XXXXXX")
  cuda=${build_root}/lib/python${py}/site-packages/nvidia/cu13
  log "building ${name} from ${source} (jobs=${build_jobs})"
  checkout_source "${source}" "${src}"
  SECONDS=0
  env -C "${src}" ${build_env} \
    PATH="${build_root}/bin:${build_root}/rust/bin:${cuda}/bin:${PATH}" \
    CUDA_HOME="${cuda}" CUDACXX="${cuda}/bin/nvcc" CUDAToolkit_ROOT="${cuda}" \
    RUSTUP_HOME="${build_root}/rust" CARGO_HOME="${build_root}/rust" \
    MAX_JOBS="${build_jobs}" CMAKE_BUILD_PARALLEL_LEVEL="${build_jobs}" NVCC_THREADS="${NVCC_THREADS:-2}" \
    uv build --wheel --no-build-isolation --python "${build_root}/bin/python" --out-dir "${out}" \
      "${src}${subdir:+/${subdir}}" &> "${wheel_cache}/${name}.log" ||
    refuse "source build of ${name} failed; see ${wheel_cache}/${name}.log"
  [[ -f "${out}/${name}" ]] || refuse "source build of ${name} produced '$(ls "${out}")' instead; see ${wheel_cache}/${name}.log"
  mv -- "${out}/${name}" "${wheel_cache}/${name}"
  sha256sum "${wheel_cache}/${name}" | cut -d' ' -f1 > "${wheel_cache}/${name}.sha256"
  rm -rf -- "${out}"
  log "built ${name} in ${SECONDS}s: sha256 $(cat "${wheel_cache}/${name}.sha256")"
}

# ---------------------------------------------------------------- flash-linear-attention
# Checked out, not installed -- see ops/gdn/gdn_fla_backward.py. Its kernels are part of the contract.
ensure_fla() {
  if [[ ! -d "${fla_dir}/fla" ]]; then
    mkdir -p "$(dirname "${fla_dir}")"
    git clone --quiet https://github.com/fla-org/flash-linear-attention.git "${fla_dir}"
  fi
  if [[ "$(git -C "${fla_dir}" rev-parse HEAD)" != "${fla_commit}" ]]; then
    git -C "${fla_dir}" fetch --quiet origin "${fla_commit}" || git -C "${fla_dir}" fetch --quiet origin
    git -C "${fla_dir}" checkout --quiet "${fla_commit}"
  fi
  [[ "$(git -C "${fla_dir}" rev-parse HEAD)" == "${fla_commit}" ]] || refuse "flash-linear-attention drift"
}

# ---------------------------------------------------------------- verify
verify_envs() {  # verify_envs <runtime> <jit>
  local runtime=$1 jit=$2
  [[ -x "${runtime}/bin/python" ]] || { echo "REFUSAL: runtime Python missing: ${runtime}" >&2; return 1; }
  [[ -x "${jit}/bin/python" ]] || { echo "REFUSAL: JIT Python missing: ${jit}" >&2; return 1; }
  "${runtime}/bin/python" - "${repo}" "${runtime}" <<'PY' || return 1
import importlib.metadata as md, json, sys
from pathlib import Path

repo, runtime = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
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
bad = {n: (expected[n], observed[n]) for n in expected if observed[n] != expected[n]}
if bad:
    raise SystemExit(f"REFUSAL: runtime package drift: {bad}")
import torch, skyrl
if torch.version.cuda != "13.0":
    raise SystemExit(f"REFUSAL: torch CUDA drift: {torch.version.cuda!r}")
cuda_lib = runtime / "lib/python3.12/site-packages/nvidia/cu13/lib"
for alias, soname in {"libcublas.so": "libcublas.so.13", "libcublasLt.so": "libcublasLt.so.13"}.items():
    link = cuda_lib / alias
    if not link.is_symlink() or link.readlink() != Path(soname):
        raise SystemExit(f"REFUSAL: CUDA-13 loader alias drift: {link}")
skyrl_path = Path(skyrl.__file__).resolve()
if not skyrl_path.is_relative_to(repo):
    raise SystemExit(f"REFUSAL: skyrl is not this checkout: {skyrl_path}")
if not Path(sys.executable).absolute().is_relative_to(runtime):
    raise SystemExit(f"REFUSAL: wrong runtime interpreter: {sys.executable}")
sys.path.insert(0, str(repo / "examples/isoexec/nightly/_torchvision_stub"))
import torchvision
if torchvision.__version__ != "0.0.0-isoexec-stub":
    raise SystemExit(f"REFUSAL: text-only torchvision surface drift: {torchvision.__file__}")
import vllm.model_executor.models.qwen3_5  # noqa: F401
print(json.dumps({"runtime": str(runtime), "packages": observed, "skyrl": str(skyrl_path)}, sort_keys=True))
PY

  "${jit}/bin/python" - "${jit}" <<'PY' || return 1
import importlib.metadata as md, json, subprocess, sys
from pathlib import Path

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
bad = {n: (expected[n], observed[n]) for n in expected if observed[n] != expected[n]}
if bad:
    raise SystemExit(f"REFUSAL: JIT package drift: {bad}")
cuda_root = jit / "lib/python3.12/site-packages/nvidia/cu13"
nvcc = cuda_root / "bin/nvcc"
if not nvcc.is_file():
    raise SystemExit("REFUSAL: CUDA 13 JIT nvcc is missing")
for rel, target in {
    "lib64": Path("lib"),
    "lib/libcudart.so": Path("libcudart.so.13"),
    "lib/libcurand.so": Path("libcurand.so.10"),
    "lib/stubs/libcuda.so": Path("/usr/lib64/libcuda.so.1"),
}.items():
    link = cuda_root / rel
    if not link.is_symlink() or link.readlink() != target:
        raise SystemExit(f"REFUSAL: CUDA-13 JIT link drift: {link}")
if "release 13.0" not in subprocess.check_output([str(nvcc), "--version"], text=True):
    raise SystemExit("REFUSAL: nvcc drift")
print(json.dumps({"jit": str(jit), "nvcc": str(nvcc), "packages": observed}, sort_keys=True))
PY
}

# ---------------------------------------------------------------- build
build_runtime() {  # build_runtime <stage>
  local stage=$1 lib skip=() extra=() name
  # Source-built wheels cannot match the byte hashes frozen in uv.lock: sync skips them and they are
  # installed on top, the same way TE (outside the lock) always is.
  for name in "${source_built[@]}"; do
    skip+=(--no-install-package "$(dist_name "${name}")"); extra+=("${wheel_dir}/${name}")
  done
  uv venv --python "${py}" --relocatable "${stage}"
  VIRTUAL_ENV="${stage}" UV_PROJECT_ENVIRONMENT="${stage}" \
    uv sync --project "${repo}" --frozen --extra isoexec --no-dev --active "${skip[@]}"
  # TE is outside the lock: [tool.uv] override-dependencies pins TE 2.11 for the megatron extra.
  is_source_built "$(basename "$(wheel_path '^transformer_engine_torch-')")" || extra+=("$(wheel_path '^transformer_engine_torch-')")
  uv pip install --no-config --python "${stage}/bin/python" --no-deps \
    'nvidia-cublas==13.2.0.9' 'transformer-engine==2.17.1' 'transformer-engine-cu13==2.17.1' "${extra[@]}"
  lib=${stage}/lib/python${py}/site-packages/nvidia/cu13/lib
  ln -s libcublas.so.13 "${lib}/libcublas.so"
  ln -s libcublasLt.so.13 "${lib}/libcublasLt.so"
}

build_jit() {  # build_jit <stage>
  local stage=$1
  uv venv --python "${py}" --relocatable "${stage}"
  install_cuda13 "${stage}"
  uv pip install --python "${stage}/bin/python" --no-deps --editable "${repo}"
}

# ---------------------------------------------------------------- main
for tool in sha256sum curl git uv; do command -v "${tool}" >/dev/null || refuse "${tool} is required"; done

case "${mode}" in
  verify)
    verify_envs "${runtime_root}" "${jit_root}"
    echo "ISOEXEC_ENV_VERIFY_PASS=1"; exit 0 ;;
  wheels)
    ensure_wheels
    echo "ISOEXEC_WHEELS_PASS=1"; exit 0 ;;
esac

ensure_wheels
ensure_fla

if [[ "${replace}" == 0 && ( -e "${runtime_root}" || -e "${jit_root}" ) ]]; then
  if verify_envs "${runtime_root}" "${jit_root}"; then
    echo "ISOEXEC_ENV_ALREADY_CURRENT=1"; exit 0
  fi
  refuse "target exists but is not exact; rerun with --replace to rebuild (the old env is kept as a .backup)"
fi

mkdir -p "$(dirname "${runtime_root}")" "$(dirname "${jit_root}")"
runtime_stage=$(mktemp -d "$(dirname "${runtime_root}")/.isoexec-runtime.XXXXXX")
jit_stage=$(mktemp -d "$(dirname "${jit_root}")/.isoexec-jit.XXXXXX")
cleanup() { for d in "${runtime_stage:-}" "${jit_stage:-}"; do [[ -n "${d}" && -d "${d}" ]] && rm -rf -- "${d}"; done; return 0; }
trap cleanup EXIT

build_runtime "${runtime_stage}"
build_jit "${jit_stage}"
verify_envs "${runtime_stage}" "${jit_stage}"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
[[ ! -e "${runtime_root}" ]] || mv -- "${runtime_root}" "${runtime_root}.backup.${stamp}"
[[ ! -e "${jit_root}" ]] || mv -- "${jit_root}" "${jit_root}.backup.${stamp}"
mv -- "${runtime_stage}" "${runtime_root}"
mv -- "${jit_stage}" "${jit_root}"
runtime_stage= jit_stage=
verify_envs "${runtime_root}" "${jit_root}"
echo "ISOEXEC_ENV_BUILD_PASS=1"
echo "runtime=${runtime_root}"
echo "jit=${jit_root}"
