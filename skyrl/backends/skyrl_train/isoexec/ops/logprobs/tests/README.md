# logprobs/tests

Claim -> test:
- `logprobs.log_softmax:aten_reference_fused_exp` `bitwise_equal_to: aten_reference` -> `logprob_extract_dist.py` (torchrun world>=2; bitwise torch.equal over the live TP gather)
- `logprobs.lm_head_slice:sampled_rows` extract seam -> `logprob_extract_dist.py`
- exact-vocab TP transport (`exact_vocab_transport.py`) -> `exact_vocab_transport_dist.py` (torchrun; correctness + wire contract)

Both are run as files under torchrun (they skip gracefully off-CI); single-process pytest has no
coverage here yet. The full-wrapper qualification stays in the private nightly tree.
