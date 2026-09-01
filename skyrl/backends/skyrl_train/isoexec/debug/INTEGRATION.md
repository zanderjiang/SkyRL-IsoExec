# Debug-mode integration (changes owned by files this package must not touch)

Debug mode is gated by `SKYRL_ISOEXEC_DEBUG_TRACE` (trace dir). Items 1-6 are APPLIED: flags in
`core/flags.py`, hook installation via `ContractAdapter._install_debug_trace()` (both sides,
fail-soft), and demotion at EVERY refusal point through `enforce.refuse()`/`enforce.demoted()` —
claims, the unregistered-claim-kind refusal, every `close_phase` boundary, the handshake and the
delivery cross-check. `set_step` is wired at trainer `optim_step` and at the engine's effective
`load_weights` (item 5). Item 4 is no longer a run-config *request*: debug tracing on the engine
with `SKYRL_ISOEXEC_ENABLE_CUDAGRAPH=1` is now REFUSED at init. Pinned by
`core/tests/test_debug_integration.py`, whose cases drive the real adapters end to end.

**Still owed by files this package does not own** — each is a one-liner, listed with its exact
site in "Owed call-site changes" at the end. Original specification kept below.

1. **Flags** (`core/flags.py`): register `SKYRL_ISOEXEC_DEBUG_TRACE`, `SKYRL_ISOEXEC_DEBUG_SIDE`,
   `SKYRL_ISOEXEC_DEBUG_SAMPLE`, `SKYRL_ISOEXEC_DEBUG_LADDER`, `SKYRL_ISOEXEC_DEBUG_RING` as
   `DIAGNOSTIC`, sides `("both",)`, forwarded by `(TRAIN, ENGINE)` — without this the envs never
   reach the Ray actors. `_SIDE` should be stamped per-process by
   the launcher (`trainer` in the Megatron worker path, `engine` in the vLLM path), not by hand.
   APPLIED for those five and for `_SEGMENTS` (item 6). **NOT YET REGISTERED**, same treatment
   owed: `SKYRL_ISOEXEC_DEBUG_REGIONS` (region allow list — without it a multi-region trace
   cannot be narrowed from the launcher, and `mm` cannot be enabled in a Ray actor at all) and
   `SKYRL_ISOEXEC_DEBUG_DIGEST` (`auto`/`eager`/`triton`; both sides must agree, and the
   manifest records what each side used).

2. **Hook installation**: trainer — in `workers/megatron/megatron_worker.py`, right after
   `apply_megatron_isoexec_patches()` / model build, call
   `isoexec.debug.install_debug_hooks(model)`. Engine — in `runtimes/vllm/gptmodel_vllm.py`,
   after `swap_gdn_core(...)`, call `install_debug_hooks(self.gpt)`. Optionally call
   `isoexec.debug.set_step(n)` per training step / scoring pass for step-keyed alignment and true
   every-Nth-forward sampling.

   **The `model` argument matters and the trainer does not pass one today.** It installs
   `install_layer_context_hooks`, which wraps each decoder layer's `forward` to publish
   `layer_number - 1` (megatron's global, pipeline-offset-included index) to every region door
   inside it. A side that has it records `layer_src="module"`; a side that does not falls back to
   `layer_src="call_order"` (the region's call ordinal within the step). Those two index spaces
   do NOT agree on a model with mixed layer types — Qwen3.5-35B-A3B's 30 GDN layers have module
   ids `0,1,2,4,5,6,8,...` and call ordinals `0..29` — so per-layer keys will not align and the
   comparator reports a `layer_src_mismatch` warning next to the resulting absences. Pass the
   model on BOTH sides.

   `install_gdn_layer_hooks(gpt)` still exists as an alias of `install_layer_context_hooks`, so
   the current engine call site keeps working, but it no longer RECORDS anything of its own. It
   used to wrap `GatedDeltaNet.forward`, i.e. the post-`out_proj` `[T,1,2048]` tensor, while the
   trainer recorded the kernel door's `[1,T,H,D]` — two different quantities under one region
   name, which made every no-fault engine/trainer `gdn.core` comparison structurally divergent.
   Both sides now record at the kernel door only.

3. **Non-enforcement semantics**: when `SKYRL_ISOEXEC_DEBUG_TRACE` is set, every enforcement
   refusal demotes to logged-and-continue while the ledger still records the violation, so the
   artifact is exactly as red as strict mode would leave it and only the raise is suppressed.
   `enforce.demoted()` is the one condition (debug tracing OR
   `SKYRL_ISOEXEC_MANIFEST_STRICT=0`) and `enforce.refuse()` the one site that acts on it —
   `grep -rn 'enforce.refuse'` enumerates what demotes. Demoting the handshake alone was
   self-defeating: `on_weight_sync` closes WEIGHT_SYNC on the record the demotion just wrote.
   NOT demoted: the delivered file's self-consistency check, because a tampered artifact is not a
   kernel mix. Users may then run any kernel mix on either side; the trace, not the gate, reports
   the disagreement.

4. **Execution config for debug runs**: run the engine eager (no CUDA-graph decode). APPLIED as
   a REFUSAL, not a request: `ContractAdapter._install_debug_trace` raises when the engine arms
   tracing with `SKYRL_ISOEXEC_ENABLE_CUDAGRAPH=1` (`ContractAdapter.CUDAGRAPH_REFUSAL`), before
   any hook is installed. It deliberately does NOT go through `enforce.refuse()`: `enforce.demoted()`
   is true precisely because debug tracing is armed, so that refusal would demote itself to a log
   line every time it fired. It is also not the failure fail-soft exists for — the run would not
   crash, it would produce a trace that reads CLEAN because the decode half of the forward was
   never observed, which is worse than no trace.

   Two backstops behind the refusal, for capture reached by any other route: `wrap_region` checks
   `torch.cuda.is_current_stream_capturing()` and skips the record rather than poisoning the
   capture with the digest's D2H copy (the skips are counted into `capture_skipped` in the
   manifest and warned about by the comparator), and the comparator still reports a replayed
   graph's missing records as an `absent` divergence (exit 2), naming this cause when the missing
   side is the engine.

5. **`set_step` must be wired on BOTH sides, or `SKYRL_ISOEXEC_DEBUG_SAMPLE` must stay 1.**
   APPLIED: `GPTModelVLLMWrapper._isoexec_debug_set_step` calls `set_step` on every *effective*
   `load_weights` (vLLM also calls it at build with HF names, which all miss). The engine has no
   step of its own, so the key is the weight-sync count; the train loop syncs once before the first
   `optim_step`, so the Nth sync carries the weights of `optim_step` N-1 and that constant offset is
   applied at the call site. The first sync stays unkeyed, matching the trainer's own pre-`optim_step`
   window, so both sides switch from call-ordinal to step sampling at the same moment.

   Original text: `set_step` was called only in the trainer's `optim_step`; the engine
   records carry `step: null` and therefore sample by *call ordinal* while the trainer samples by
   *step*. With `SAMPLE > 1` the two sides then select disjoint record sets. The comparator no
   longer calls that clean: it reports `COMPARISON INCONCLUSIVE: side-disjoint sampling` and exits
   3. The engine-side fix is one call — `isoexec.debug.set_step(n)` once per engine forward /
   generation step in `runtimes/vllm/gptmodel_vllm.py`, with the same `n` the trainer uses.

6. **Register `SKYRL_ISOEXEC_DEBUG_SEGMENTS`** in `core/flags.py` (same `DIAGNOSTIC`,
   `("both",)`, `(TRAIN, ENGINE)` treatment as the other debug envs). APPLIED. Set it to a row
   count to record one digest per that many rows of the tensor's FIRST NON-UNIT dim (dim 0 was
   useless on the shapes the GDN door produces: `[1,T,H,D]` is a single segment); the comparator
   then reports which row
   segments differ, and — the part that matters for triage — distinguishes a fault localized to
   some rows from a whole-tensor round-off/reduction-order difference, which the k-ladder cannot.
   Unregistered, it works only for single-process runs.

## Trace format version 3

`trace.FORMAT_VERSION` / `compare.FORMAT_VERSION` is **3**. `compare.load_dir` refuses records of
any other version with an explicit message rather than mis-reading them, so traces captured before
this change must be re-captured — and they must be, because v3 also changes the DIGEST: the
position weight is now one splitmix64 per group of 8 positions (see `thash`), so v2 digests and
v3 digests of the same tensor differ. What v3 adds on top of v2:

| field | why |
|---|---|
| `layer_src` | `"module"` / `"call_order"` / `null` — which rung answered for `layer`. Two sides using different rungs are not comparing the same key space, and the comparator now says so instead of reporting the absences it causes |
| `seg_axis` | which dim `segments` sliced: the first NON-UNIT dim, so a `[1,T,H,D]` GDN output is segmented over T instead of yielding one useless segment |
| manifest `capture_skipped` | records skipped because a hook was reached under CUDA-graph capture |
| manifest `regions_hooked` | which regions actually had a door in this process, so "no records for region X" separates "clean" from "never hooked here" |
| manifest `digest_backend` | `auto`/`eager`/`triton`; the two sides must agree |

What v2 added:

| field | why |
|---|---|
| `rank`, `rank_src` | grouping key, so many processes writing one trace dir align per rank instead of by pid sort order (`rank_src` is `dist` / `env:RANK` / `env:LOCAL_RANK` / `pid`) |
| `seq` | per-process record counter; with `ts` (now microsecond) it gives the exact causal order the "FIRST DIVERGENCE" claim depends on |
| `out` (now a dotted string path) | nested tuple and dict outputs are descended, so `"1.0"` and `"h"` no longer collide with `"0"` or vanish |
| `unrecordable` | an output that could not be digested (unsupported dtype, no tensors, too deeply nested) is recorded with a reason instead of dropped |
| `segments`, `seg_rows` | row-segment digests when `SKYRL_ISOEXEC_DEBUG_SEGMENTS` is set |
| `manifest-*.json` sidecar | per process: sample rate, whether `set_step` drove it, steps seen vs recorded, ladder/segment settings — this is what lets the report distinguish "verified clean" from "not observed" |

## Running the comparator offline

The comparator is stdlib-only and is meant to run where there is no torch, no CUDA and no
TransformerEngine. Supported invocations:

```bash
python <repo>/skyrl/backends/skyrl_train/isoexec/debug/compare.py TRAINER_DIR ENGINE_DIR
PYTHONPATH=<repo>/skyrl/backends/skyrl_train/isoexec python -m debug.compare TRAINER_DIR ENGINE_DIR
PYTHONPATH=<repo>/skyrl/backends/skyrl_train/isoexec python -m debug TRAINER_DIR ENGINE_DIR
```

`python -m skyrl.backends.skyrl_train.isoexec.debug.compare` is **not** supported offline: `-m`
imports every parent package, and `isoexec/__init__.py` calls `install_no_te_guard()` →
`import transformer_engine`, which raises `OSError: libcublas.so.13: cannot open shared object
file` on a machine without the CUDA/TE stack. Nothing inside `debug/` can prevent that. If the
fully qualified form should work, the one-line fix belongs to `isoexec/__init__.py` or
`runtimes/megatron/no_te_guard.py`: widen the TE probe's `except ImportError` to
`except (ImportError, OSError)` so a broken/partial TE install degrades to "TE absent" instead of
propagating. APPLIED in `runtimes/megatron/no_te_guard.py`, so the fully qualified `-m` form
works on a machine with no CUDA/TE stack.

Exit codes: `0` clean, `2` divergence (value, shape/dtype, absent record/region, rank-set
mismatch, one-sided unrecordable), `3` inconclusive (side-disjoint sampling), `1` bad input
(no traces, or a refused format version).

`--json` is capped: `--json-max-per-region N` (default 200, `0` = uncapped) keeps the first N
divergences PER REGION — they are causally sorted, so those are the earliest, which is what
triage reads — and records what it dropped under `truncated`. The per-region counts,
`first_divergence` and `origins` are never trimmed. A fully divergent 110k-record comparison
serialized 66MB before; at the default it is under 1MB.

## Owed call-site changes (files this package does not own)

Each is one line, and each is currently a real hole in the trace, not a nicety:

| where | change | why |
|---|---|---|
| `core/flags.py` | register `SKYRL_ISOEXEC_DEBUG_REGIONS` and `SKYRL_ISOEXEC_DEBUG_DIGEST`, `DIAGNOSTIC`, sides `("both",)`, forwarded by `(TRAIN, ENGINE)` | without forwarding, neither reaches a Ray actor, so a multi-region trace cannot be narrowed and the two sides cannot be pinned to the same digest backend |
| `workers/megatron/megatron_worker.py` | `install_debug_hooks()` → `install_debug_hooks(model)` | trainer records fall back to `layer_src="call_order"`, which does not align with the engine's module indices on a mixed-layer-type model |
| `runtimes/vllm/gptmodel_vllm.py` | `install_debug_hooks()` → `install_debug_hooks(self.gpt)`; the separate `install_gdn_layer_hooks(self.gpt)` call is then redundant (kept working as an alias) | same, engine side |
| `runtimes/vllm/gptmodel_vllm.py` (~:732) | delete the `SKYRL_ISOEXEC_ENABLE_CUDAGRAPH` warning block | it is now a hard refusal in `ContractAdapter._install_debug_trace`, raised before this point; leaving a "warning" that the run is degraded contradicts a refusal that already happened, and its text ("Run the engine eager for full traces") understates it |
| `ops/rope/rope_fused.py:~356` (`revert_engine_fused_rope`) and `runtimes/vllm/compat.py:~58` | compare against `getattr(binding, "__wrapped__", binding)` instead of the binding itself | both hard-check the *identity* of the live `_apply_rotary_pos_emb_bshd` binding, so `rope.rope` cannot be traced at all today; with the unwrap, its door moves from `install.NOT_HOOKED` into `install.DOORS` and coverage goes 20/22 → 21/22 |
