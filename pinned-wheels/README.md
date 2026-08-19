# Pinned wheels

The `isoexec` extra resolves four wheels from this directory. They are pinned rather than
resolved from an index because the bitwise trainer/engine identity that IsoExec asserts is a
property of *exact* binaries: a different torch or triton build changes reduction orders and the
identity gate goes red for reasons that have nothing to do with your model.

Drop the files here (or point `ISOEXEC_WHEEL_ROOT` at a directory that already holds them and run
`examples/isoexec/build_isoexec_env.sh`, which verifies each sha256 before installing anything).

| Wheel | sha256 | How to obtain |
|---|---|---|
| `torch-2.14.0.dev20260620+cu130-cp312-cp312-manylinux_2_28_x86_64.whl` | `063edf3d548cf57ddf0a6294121c815dec98477307890fff40dff29401c0154d` | Published on the PyTorch nightly cu130 index; fetch that exact build. |
| `triton-3.7.1+git5d6048aa-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` | `fd64476dcf7a3d56ce252bde145bb6cdb8a5d3785c6491428cf3899a0c0df544` | Published on the PyTorch nightly cu130 index (matches torch 2.14's pin). |
| `vllm-1.0.0.dev20260620+cu130-cp312-cp312-linux_x86_64.whl` | `ad0f919d8f21f1f3b7c4f2b480d1c9372410ec64fbbb19fa62cf47e966f4fa20` | Not published to any index. Build from vLLM at commit `6f573f486` (`compat.VALIDATED_VLLM_VERSION`) against the torch wheel above. `setuptools_scm` reproduces `0.22.1rc1.dev436+g6f573f486`, which is what `vllm.__version__` self-reports; the `1.0.0.dev20260620+cu130` label lives only in the dist-info. |
| `flash_attn_3-3.0.0-cp310-abi3-linux_x86_64.whl` | `b802f459bc0bd8642ffeecf965990476b35775da3dd2fd6a5a6b24f982e712aa` | Not published to any index. Build from `Dao-AILab/flash-attention` `hopper/` (`__version__` 3.0.0) against the torch wheel above, sm90a only. Provides `flash_attn_interface` for the varlen backend. |

`transformer_engine_torch-2.17.1-cp312-cp312-linux_x86_64.whl`
(`2a06bc89229bece34fb3bea384cb5a0655e60bd3eb664835478cf36ed8277921`) is installed by the build
script into the JIT environment only; it is not a resolver input and so has no `[tool.uv.sources]`
entry.
