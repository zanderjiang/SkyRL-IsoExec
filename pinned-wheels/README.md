# Pinned wheels

The `isoexec` extra resolves torch, triton, vllm and flash-attn-3 from this directory
(`[tool.uv.sources]` path entries, hashes frozen in `uv.lock`); `transformer_engine_torch` is
installed on top of the lock by the build script. They are pinned as exact binaries because the
bitwise trainer/engine identity IsoExec asserts is a property of exact binaries: a different torch
or triton build changes reduction orders and the identity gate goes red for reasons that have
nothing to do with your model.

`wheels.txt` is the source of truth: one line per wheel with its sha256 and, where one exists, an
upstream URL. `examples/isoexec/build_isoexec_env.sh` fills this directory from it before `uv sync`
runs. Wheels live in `ISOEXEC_WHEEL_CACHE` (default `<local_root>/wheels`) and are symlinked here;
each is looked for in order: a file already here, the cache, `ISOEXEC_WHEEL_MIRROR`, the upstream
URL, and finally a source build from `sources.txt`. The mirror defaults to the project's GitHub
release `wheels-20260620`, which carries all five (asset names spell `+` as `.`); point it at any
base URL serving the same filenames, or set it empty to skip. Fetched files are sha256-checked
against `wheels.txt`; nothing is installed on a mismatch. `--wheels-only` runs just this stage,
which is also how a mirror is populated.

| Wheel | Source |
|---|---|
| `torch-2.14.0.dev20260620+cu130` | Delisted from the PyTorch nightly index (pip/uv resolution fails) but still served at the pinned URL. Prefer a mirror: nightly blobs disappear without notice. |
| `triton-3.7.1+git5d6048aa` | Still on the nightly index; fetched from the pinned URL. |
| `vllm-1.0.0.dev20260620+cu130` | No public index. vLLM commit `6f573f486` (`compat.VALIDATED_VLLM_VERSION`) against the torch above, `TORCH_CUDA_ARCH_LIST=9.0a` (sm80 PTX fallbacks are emitted by its CMake), Release build. |
| `flash_attn_3-3.0.0` | No public index. `Dao-AILab/flash-attention` `hopper/` at `14c37795` against the torch above (sm90a kernels plus the sm80 fallbacks the default build emits). Provides `flash_attn_interface`. |
| `transformer_engine_torch-2.17.1` | No public index. The PyPI `transformer_engine_torch-2.17.1` sdist, C++ extension only, built against the torch above with `NVTE_WITH_NCCL_EP=0`. |

## Building from source

When a wheel cannot be fetched from anywhere, the build script compiles it from its
`sources.txt` line (`filename  source  build env`):
a git URL pinned to a commit, or a PyPI sdist pinned by sha256, plus the environment the build
runs under. It happens automatically inside the wheels stage; nothing to pass.

The toolchain is a third venv, `<local_root>/venvs/skyrl-cuda130-build`: the pinned torch wheel
with its CUDA-13 libraries, the same `nvidia-cuda-nvcc==13.0.88` wheel set the JIT venv uses
(laid out as `CUDA_HOME`, so `/usr/local/cuda` is never consulted), cmake/ninja/setuptools, and a
rustup under `rust/` for vLLM's Rust tool parser (the 1.95 toolchain named by its
`rust-toolchain.toml` is installed on first use; the optional `vllm-rs` frontend binary needs
`protoc` and is skipped without it, as in the original wheel). Host requirements are a C++
compiler nvcc 13 accepts (gcc 11 to 14), git, curl and network access for the sources and CMake's
fetched dependencies. `ISOEXEC_BUILD_JOBS` (default `nproc`) bounds parallelism; with 64 jobs on
a 192-core node vLLM took 20 min, FA3 19 min and TE under a minute. The toolchain venv is about
7 GB and the build trees under `<local_root>/src` about 9 GB; each build writes
`<wheel>.log` next to the cached wheel.

Source-built wheels are not byte-reproducible, so their `wheels.txt` hashes cannot be expected to
match; the script instead records the sha256 it observed as `<wheel>.sha256` in the cache and
accepts that file from then on. The lock's own hash cannot match either, so `uv sync` skips those
packages and they are installed on top with `--no-deps`, the way TE always is. Their identity
therefore rests on the pinned commit or sdist, the pinned torch, and nvcc 13.0.88, not on a
byte hash, and the step-1 identity gate is the oracle for whether the result is equivalent. The
rebuilt vLLM wheel uses `VLLM_VERSION_OVERRIDE` so that its dist-info carries the version the lock
and `verify` expect; the one visible difference from the original is that `vllm.__version__`
reports `1.0.0.dev20260620+cu130` instead of `0.22.1rc1.dev436+g6f573f486`, which
`compat.VALIDATED_VLLM_VERSION` uses as a label only.

Only `wheels.txt`, `sources.txt` and this README are tracked.
