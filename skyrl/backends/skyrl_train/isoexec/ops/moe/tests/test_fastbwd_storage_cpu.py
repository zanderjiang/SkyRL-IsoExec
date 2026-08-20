"""CPU proofs for live analytic-expert-backward chunking, recycling, and engagement."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.isoexec.ops.moe import moe_backward_kernel as B


def _cfg(clamp=None):
    return SimpleNamespace(
        activation_func=torch.nn.functional.silu,
        activation_func_clamp_value=clamp,
        glu_linear_offset=0.125,
    )


def _full_epilogue(inter, probs, dhs, cfg):
    gate, up, gate_c, sg, silu, lin, h, hs = B._epilogue_values(inter, probs, cfg, inter.dtype)
    dhs_f = dhs.float()
    dprobs = (dhs_f * h).sum(-1).to(probs.dtype)
    dh = dhs_f * probs.unsqueeze(-1).float()
    dgate = dh * lin * (sg * (1.0 + gate_c.float() * (1.0 - sg)))
    dup = dh * silu
    if cfg.activation_func_clamp_value is not None:
        dgate = dgate * (gate <= cfg.activation_func_clamp_value).float()
        dup = dup * (up.abs() <= cfg.activation_func_clamp_value).float()
    return hs, torch.cat([dgate, dup], dim=-1).to(inter.dtype), dprobs


@pytest.mark.parametrize("clamp", [None, 1.25])
def test_fastbwd_row_chunks_match_full_expression_and_reuse_saved_storage(monkeypatch, clamp):
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_BWD_POINTWISE_ROWS", "7")
    cfg = _cfg(clamp)
    inter = torch.randn(31, 26, dtype=torch.bfloat16)
    probs = torch.randn(31, dtype=torch.float32)
    dhs = torch.randn(31, 13, dtype=torch.bfloat16)
    hs_ref, dinter_ref, dprobs_ref = _full_epilogue(inter.clone(), probs, dhs, cfg)

    seen = []

    def eager_spy(site, fn, *args):
        seen.append(site)
        return fn(*args)

    monkeypatch.setattr(B, "call_region", eager_spy)

    hs = B._build_hs_chunked(inter, probs, cfg, inter.dtype)
    private_saved = inter.clone()
    saved_ptr = private_saved.data_ptr()
    dinter, dprobs = B._build_dinter_chunked(private_saved, probs, dhs, cfg, inter.dtype)

    assert torch.equal(hs, hs_ref)
    assert torch.equal(dinter, dinter_ref)
    torch.testing.assert_close(dprobs, dprobs_ref, rtol=2e-6, atol=2e-6)
    assert dinter.data_ptr() == saved_ptr
    chunks = (inter.shape[0] + 6) // 7
    assert seen.count("moe.fastbwd.epilogue_hs_chunk") == chunks
    assert seen.count("moe.fastbwd.epilogue_vjp_chunk") == chunks


def test_fastbwd_compile_shape_classes_share_full_chunks_and_expose_ragged_tails(monkeypatch):
    """The ledger is finite per chunk shape; arbitrary total rows must not cause full-shape churn."""
    from skyrl.backends.skyrl_train.isoexec.autofuse.region_gate import shape_key_of

    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_BWD_POINTWISE_ROWS", "16")
    cfg = _cfg()
    observed = {}

    def spy(site, fn, *args):
        observed.setdefault(total, {}).setdefault(site, set()).add(shape_key_of(args))
        return fn(*args)

    monkeypatch.setattr(B, "call_region", spy)
    for total in (32, 37, 45):
        inter = torch.randn(total, 26, dtype=torch.bfloat16)
        probs = torch.randn(total, dtype=torch.float32)
        dhs = torch.randn(total, 13, dtype=torch.bfloat16)
        B._build_hs_chunked(inter, probs, cfg, inter.dtype)
        B._build_dinter_chunked(inter.clone(), probs, dhs, cfg, inter.dtype)

    for site in ("moe.fastbwd.epilogue_hs_chunk", "moe.fastbwd.epilogue_vjp_chunk"):
        full_shape_keys = set.intersection(*(observed[total][site] for total in observed))
        assert full_shape_keys, f"{site}: fixed 16-row chunks did not share an artifact class"
        # One full class plus at most one ragged-tail class per total.
        assert all(len(observed[total][site]) <= 2 for total in observed)


@pytest.mark.parametrize("clamp", [None, 1.25])
def test_fastbwd_regions_have_stable_aten_graph_digests(clamp):
    """The real ledger drift probe must be able to trace both scalar-specialized branches."""
    from skyrl.backends.skyrl_train.isoexec.autofuse.bwd_compile import graph_digest

    inter = torch.randn(7, 26, dtype=torch.bfloat16)
    probs = torch.randn(7, dtype=torch.float32)
    dhs = torch.randn(7, 13, dtype=torch.bfloat16)
    hs_args = (inter, probs, clamp, 0.125, inter.dtype)
    vjp_args = (inter, probs, dhs, clamp, 0.125, inter.dtype)
    hs_digest = graph_digest(B._fastbwd_epilogue_hs_region, hs_args)
    vjp_digest = graph_digest(B._fastbwd_epilogue_vjp_region, vjp_args)
    assert len(hs_digest) == len(vjp_digest) == 16
    assert hs_digest == graph_digest(B._fastbwd_epilogue_hs_region, hs_args)
    assert vjp_digest == graph_digest(B._fastbwd_epilogue_vjp_region, vjp_args)
    assert hs_digest != vjp_digest


def _reference_experts(x, probs, counts, w1, w2, cfg):
    outputs = []
    start = 0
    for expert, count in enumerate(counts.tolist()):
        stop = start + count
        inter = x[start:stop] @ w1[expert].T
        gate, up = inter.chunk(2, dim=-1)
        if cfg.activation_func_clamp_value is not None:
            gate = gate.clamp(max=cfg.activation_func_clamp_value)
            up = up.clamp(min=-cfg.activation_func_clamp_value, max=cfg.activation_func_clamp_value)
        hidden = cfg.activation_func(gate) * (up + cfg.glu_linear_offset)
        hidden = (hidden * probs[start:stop].unsqueeze(-1)).to(x.dtype)
        outputs.append(hidden @ w2[expert].T)
        start = stop
    return torch.cat(outputs) if outputs else x.new_empty((0, x.shape[1]))


@pytest.mark.parametrize("counts", [[4, 3, 3], [8, 1, 1]])
def test_fastbwd_custom_function_matches_reference_cpu(counts):
    torch.manual_seed(sum(counts))
    cfg = _cfg()
    counts_t = torch.tensor(counts, dtype=torch.long)
    # hidden/ffn were 6/4 privately; the public grouped-only backward runs aten._grouped_mm,
    # which requires 16-byte-multiple strides, so bf16 needs multiples of 8
    rows, experts, hidden, ffn = sum(counts), len(counts), 8, 8
    x = torch.randn(rows, hidden, dtype=torch.bfloat16, requires_grad=True)
    probs = torch.randn(rows, dtype=torch.float32, requires_grad=True)
    w1 = [torch.randn(2 * ffn, hidden, dtype=torch.bfloat16, requires_grad=True) for _ in range(experts)]
    w2 = [torch.randn(hidden, ffn, dtype=torch.bfloat16, requires_grad=True) for _ in range(experts)]
    cotangent = torch.randn(rows, hidden, dtype=torch.bfloat16)

    got = B.moe_experts_fastbwd(x, probs, counts_t, experts, cfg, w1, w2, pik_fc2=False, n_leaves=1)
    got.backward(cotangent)

    # Forward is gate-critical. On CPU, padded bmm and per-expert mm can choose different reduction
    # schedules, so this portable test checks accuracy; the CUDA nightly owns torch.equal at production dims.
    with torch.no_grad():
        ref = _reference_experts(x, probs, counts_t, torch.stack(w1), torch.stack(w2), cfg)
    torch.testing.assert_close(got, ref, rtol=1e-2, atol=3e-2)

    for grad in (x.grad, probs.grad, *(weight.grad for weight in w1), *(weight.grad for weight in w2)):
        assert grad is not None and torch.isfinite(grad).all()


def test_fastbwd_scoring_forward_does_not_materialize_inter_rows(monkeypatch):
    """The wrapper captures caller grad mode before autograd.Function disables it internally."""
    torch.manual_seed(11)
    counts = torch.tensor([3, 2], dtype=torch.long)
    rows, experts, hidden, ffn = int(counts.sum()), counts.numel(), 8, 8  # grouped_mm stride rule
    x = torch.randn(rows, hidden, dtype=torch.bfloat16, requires_grad=True)
    probs = torch.randn(rows, dtype=torch.float32, requires_grad=True)
    w1 = [torch.randn(2 * ffn, hidden, dtype=torch.bfloat16, requires_grad=True) for _ in range(experts)]
    w2 = [torch.randn(hidden, ffn, dtype=torch.bfloat16, requires_grad=True) for _ in range(experts)]
    seen = []
    original = B._forward_core

    def wrapped(*args, **kwargs):
        seen.append(kwargs["want_inter"])
        return original(*args, **kwargs)

    monkeypatch.setattr(B, "_forward_core", wrapped)
    with torch.no_grad():
        scoring = B.moe_experts_fastbwd(x, probs, counts, experts, _cfg(), w1, w2, pik_fc2=False, n_leaves=1)
    assert seen[-1] is False
    assert scoring.grad_fn is None

    training = B.moe_experts_fastbwd(x, probs, counts, experts, _cfg(), w1, w2, pik_fc2=False, n_leaves=1)
    assert seen[-1] is True
    training.sum().backward()
    assert x.grad is not None


def test_fastbwd_chunk_size_refuses_nonpositive(monkeypatch):
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_BWD_POINTWISE_ROWS", "0")
    with pytest.raises(ValueError, match="must be positive"):
        B._pointwise_chunk_rows()
