# IsoExec

Make the **Megatron trainer** and the **vLLM rollout engine** compute **bitwise-identical** token
logprobs, so the PPO/GRPO importance ratio `r_t = π_train / π_rollout` is exactly 1 and no KL
correction term is needed.

## The idea

Trainer and engine normally run different implementations of the same mathematics — a chunked scan
versus a recurrent one, a fused epilogue versus an unfused one, a tree all-reduce versus a ring.
These are algebraically equal and numerically different, and the difference shows up as a spurious
importance ratio that the algorithm then has to correct for.

IsoExec removes the difference instead of correcting for it. Every op on the logprob path is
required to evaluate the *same function* at every site — trainer forward, trainer scoring forward,
engine prefill, engine decode — where "same function" means the same rounding schedule, not just
the same mathematics.

Because "we made them identical" is easy to believe and hard to verify, the composition is not left
implicit. Each op declares which implementation it installs at each site; those declarations are
frozen into a content-hashed **execution contract**, and the two processes exchange hashes before
weight sync. A composition that differs across the two runtimes refuses to start rather than
silently producing a wrong ratio.

## Layout

| Directory | What lives there |
|---|---|
| `contract/` | The versioned, content-hashed execution contract: types, identity hashes, canonical serialization, validation. See [`contract/README.md`](contract/README.md). |
| `core/` | The registry every op registers into, manifest and contract construction, the process-level handshake, the flag census (`flags.py`), runtime fingerprinting. |
| `ops/` | The kernels and installers, by family: `attention`, `collectives` (including the `pik` tree-reduction collective), `gdn`, `logprobs`, `mm`, `moe`, `norms`, `rope`. |
| `models/` | Per-model composition policy — which implementation each op takes at each site (`qwen3_5.py`, `policy.py`). |
| `runtimes/` | The two host runtimes: `megatron/` patches and spec providers, `vllm/` plugin, model classes and engine patches. |
| `autofuse/` | Bitwise-safe `torch.compile` region selection, with a fusion ledger that is pinned into the handshake. |
| `sync/` | Native (no-HuggingFace) weight sync for the unified-GPTModel route. |
| `lifecycle/` | Install-ordering assertions. |

## Wiring it

Importing the package runs a small number of installers at module top — before anything can import
`megatron.bridge` — and that ordering is load-bearing; see the docstring in `__init__.py`. From
there:

- vLLM worker, before engine creation: `apply_vllm_isoexec_env()`
- Megatron worker, after model build: `apply_megatron_isoexec_patches()`

The whole stack is inert unless `SKYRL_ISOEXEC=1`. Individual ops are selected by
`SKYRL_ISOEXEC_*` environment variables, every one of which is catalogued in
[`core/flags.py`](core/flags.py) with its code default, the side that reads it, the channel that
forwards it, and whether it is bit-relevant (hashed into the manifest) or proven neutral.

Supported models: Qwen3 dense, Megatron MoE, and hybrid GatedDeltaNet (Qwen3.5).

## Running it

`examples/isoexec/` holds a matched pair — `run_qwen35_dapo_isoexec.sh` and
`run_qwen35_dapo_native.sh` — identical in every task-side knob so that any difference between the
two runs is attributable to the stack. Build the pinned runtime first with
`examples/isoexec/build_isoexec_env.sh` (see [`pinned-wheels/README.md`](../../../../pinned-wheels/README.md) for
the binaries it expects).

The quantity to read is `policy/rollout_train_logprobs_abs_diff_mean`, at full precision from your
logger rather than from the rounded console line. Note that `policy_kl == 0.0` is *not* evidence of
anything here: both arms set `use_kl_loss=false`, so the native baseline logs it too.

## Tests

```bash
pytest skyrl/backends/skyrl_train/isoexec/
```

These run on CPU and cover the contract (identity hashes, canonical serialization round-trips,
validation) and the composition manifest. They build the registry and the Qwen3.5 manifest under a
cleared environment, so they assert what the code composes by default rather than whatever the
invoking shell happened to export.
