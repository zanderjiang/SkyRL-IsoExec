# logprobs/tests

Claim -> test:
- `logprobs.lm_head_slice:sampled_rows` extract seam, and the `SKYRL_ISOEXEC_EXACT_SAMPLED_LOGPROBS` source (both sides via `--fixed-source-mode`, separate processes) -> `logprob_extract_dist.py` (SKYRL_ISOEXEC=1, torchrun world>=2; bitwise torch.equal over the live TP gather)
- exact-vocab TP transport (`exact_vocab_transport.py`) -> `exact_vocab_transport_dist.py` (torchrun; correctness + wire contract)
- `rowinv.py` row-count invariance (bitwise, N=1..1024 vs the 1024-row chunk, with the ATen non-invariance as a live positive control), fp64 accuracy dominance over ATen, backward allclose, decline paths -> `rowinv_gpu.py` (single GPU, run as a file)
- `rowinv.py` TP-invariance (bitwise at TP=1/2/4/8 on identical inputs, with the rank-partial anti-pattern as a live negative control) -> `rowinv_tp_dist.py` (torchrun --nproc-per-node=8; degrees cap at world)
- `rowinv.py` ENGAGEMENT preflight (served>0 per call shape under the composed contract, the enforcement boundary live in both directions, negative control included) -> `rowinv_preflight.py` (single GPU, minutes; run BEFORE committing to a long job -- its docstring states exactly what a full engine+trainer bring-up still owes)
- `rowinv.py` dispatch discipline (one dispatch point, both grad modes served, chunk walk == unchunked, no mid-tensor function mix, unimportable-rowinv is byte-for-byte the incumbent) -> `test_rowinv_dispatch_cpu.py`
- `rowinv.py` admission split (payload dtype is per-call eligibility; env/vocab-partition drift still raises) -> `test_rowinv_admit_cpu.py`

`rowinv_leaftree` is the SELECTED impl at all four sites and carries no `bitwise_equal_to` claim:
it replaces the incumbent on both runtimes at once, so there is no asymmetric pairing to discharge.
`aten_reference` stays REGISTERED as the structural-decline fallback: nothing selects it, but it is
what the engine and trainer run when rowinv declines a call.

The `*_dist.py` files run under torchrun and skip gracefully off-CI; `rowinv_gpu.py` skips without
CUDA. The `test_*_cpu.py` files run under plain pytest. The full-wrapper qualification stays in the
private nightly tree.
