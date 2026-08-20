"""CPU guards that engagement counters sit on admitted implementation branches."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

from skyrl.backends.skyrl_train.isoexec.ops.moe import moe_batched_experts as B
from skyrl.backends.skyrl_train.isoexec.ops.moe import moe_combine_backward as C
from skyrl.backends.skyrl_train.isoexec.ops.moe import moe_epilogue_kernel as E


def _trainer_epilogue_stats():
    # inlined: the public module keeps the state but not this test accessor
    return dict(B._TRAINER_EPILOGUE_COUNTS)


def _combine_backward_stats():
    # inlined: the public module keeps the state but not this test accessor
    out = dict(C._COUNTS)
    out["decline_reasons"] = dict(C._DECLINES)
    return out


def _torch_epilogue_reference(inter, probs):
    # inlined verbatim from the private module: the production 5-op chain, the bitwise reference
    T = inter.shape[0]
    gate, up = inter.reshape(T, -1).chunk(2, dim=-1)
    h = torch.nn.functional.silu(gate) * up
    h = (h * probs.unsqueeze(-1)).to(h.dtype)
    return h.contiguous()



def test_fused_combine_counts_only_after_forward_and_backward_are_served(monkeypatch):
    tokens, topk, hidden = 5, 3, 7
    rows = torch.arange(tokens * topk).view(tokens, topk)
    permuted = torch.randn(tokens * topk, hidden, requires_grad=True)

    def fake_forward(permuted_tokens, sorted_indices, restore_shape, *, permuted_probs=None, rows=None, out_dtype=None):
        # `out_dtype` is the round fold (SKYRL_ISOEXEC_MOE_COMBINE_FOLD_ROUND); this stand-in only has
        # to accept it, since the counter under test is engagement and not arithmetic.
        del sorted_indices, restore_shape, permuted_probs, out_dtype
        return permuted_tokens.index_select(0, rows.reshape(-1)).view(tokens, topk, hidden).sum(dim=1)

    def fake_backward(
        dout,
        combine_rows,
        num_permuted,
        *,
        out_dtype,
        permuted=None,
        permuted_probs=None,
        sorted_indices=None,
        impl="auto",
    ):
        del permuted, permuted_probs, sorted_indices, impl
        grad = torch.empty((num_permuted, hidden), dtype=out_dtype)
        grad[combine_rows.reshape(-1)] = dout[:, None, :].expand(-1, topk, -1).reshape(-1, hidden)
        return grad, None

    monkeypatch.setattr(C, "fused_fixed_order_combine", fake_forward)
    monkeypatch.setattr(C, "combine_backward", fake_backward)
    before = _combine_backward_stats()

    out = C.fused_combine(permuted, torch.arange(tokens * topk) % tokens, tokens, rows=rows)
    after_forward = _combine_backward_stats()
    assert after_forward["forward_served"] == before["forward_served"] + 1
    assert after_forward["backward_served"] == before["backward_served"]

    out.sum().backward()
    after_backward = _combine_backward_stats()
    assert after_backward["backward_served"] == before["backward_served"] + 1
    assert permuted.grad is not None


def _epilogue_cfg(*, clamp=None, offset=0.0):
    return SimpleNamespace(
        activation_func=torch.nn.functional.silu,
        activation_func_clamp_value=clamp,
        glu_linear_offset=offset,
    )


def test_trainer_epilogue_serves_no_grad_but_preserves_literal_autograd(monkeypatch):
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_FUSED_EPILOGUE", "1")
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_FUSED_EPILOGUE_TRAINER", "1")
    inter = torch.randn(3, 5, 16, dtype=torch.bfloat16)
    probs = torch.randn(3, 5, dtype=torch.float32)
    reference = _torch_epilogue_reference(inter.reshape(-1, 16), probs.reshape(-1)).view(3, 5, 8)
    calls = {"fused": 0}

    def fake_fused(flat_inter, flat_probs):
        calls["fused"] += 1
        return _torch_epilogue_reference(flat_inter, flat_probs)

    monkeypatch.setattr(E, "apply_glu_probs_epilogue", fake_fused)
    before = _trainer_epilogue_stats()
    with torch.no_grad():
        got = B._apply_trainer_epilogue(inter, probs, _epilogue_cfg())
    after_no_grad = _trainer_epilogue_stats()
    assert torch.equal(got, reference)
    assert calls["fused"] == 1
    assert after_no_grad["served"] == before["served"] + 1
    assert after_no_grad["rows"] == before["rows"] + 15

    grad_inter = inter.clone().requires_grad_()
    got_grad = B._apply_trainer_epilogue(grad_inter, probs, _epilogue_cfg())
    got_grad.float().sum().backward()
    after_grad = _trainer_epilogue_stats()
    assert calls["fused"] == 1, "raw Triton epilogue must never own a grad-enabled forward"
    assert after_grad["grad_fallback"] == after_no_grad["grad_fallback"] + 1
    assert grad_inter.grad is not None


def test_trainer_epilogue_refuses_unsupported_math_even_under_no_grad(monkeypatch):
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_FUSED_EPILOGUE", "1")
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_FUSED_EPILOGUE_TRAINER", "1")
    inter = torch.randn(4, 10, dtype=torch.bfloat16)
    probs = torch.randn(4)

    def should_not_run(*args, **kwargs):
        raise AssertionError("unsupported clamp/offset math reached the fused kernel")

    monkeypatch.setattr(E, "apply_glu_probs_epilogue", should_not_run)
    before = _trainer_epilogue_stats()
    with torch.no_grad():
        got = B._apply_trainer_epilogue(inter, probs, _epilogue_cfg(clamp=1.0, offset=0.125))
    after = _trainer_epilogue_stats()
    assert after["math_fallback"] == before["math_fallback"] + 1
    assert got.shape == (4, 5)


def test_trainer_epilogue_defaults_off_even_when_shared_engine_flag_is_on(monkeypatch):
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_FUSED_EPILOGUE", "1")
    monkeypatch.delenv("SKYRL_ISOEXEC_MOE_FUSED_EPILOGUE_TRAINER", raising=False)
    inter = torch.randn(4, 10, dtype=torch.bfloat16)
    probs = torch.randn(4)

    def should_not_run(*args, **kwargs):
        raise AssertionError("default-OFF trainer admission reached the fused kernel")

    monkeypatch.setattr(E, "apply_glu_probs_epilogue", should_not_run)
    with torch.no_grad():
        got = B._apply_trainer_epilogue(inter, probs, _epilogue_cfg())
    assert got.shape == (4, 5)

    source = inspect.getsource(B._batched_experts_gemm)
    assert "if _TRAINER_EPILOGUE_ENABLED:" in source
    assert "inter = _apply_trainer_epilogue(inter, pp[s:e], cfg)" in source
    assert "inter = cfg.activation_func(x_glu) * (x_linear + cfg.glu_linear_offset)" in source


def test_real_batched_expert_branch_moves_trainer_epilogue_served_count(monkeypatch):
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_FUSED_EPILOGUE", "1")
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_FUSED_EPILOGUE_TRAINER", "1")
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_WEIGHT_CACHE", "0")

    cfg = SimpleNamespace(
        fp8=False,
        bias_activation_fusion=False,
        gated_linear_unit=True,
        add_bias_linear=False,
        activation_func=torch.nn.functional.silu,
        activation_func_clamp_value=None,
        glu_linear_offset=0.0,
    )
    experts = []
    for _ in range(3):
        experts.append(
            SimpleNamespace(
                linear_fc1=torch.nn.Linear(8, 12, bias=False, dtype=torch.bfloat16),
                linear_fc2=torch.nn.Linear(6, 8, bias=False, dtype=torch.bfloat16),
            )
        )
    module = SimpleNamespace(config=cfg, num_local_experts=3, local_experts=experts)
    counts = torch.tensor([2, 1, 3])
    x = torch.randn(6, 8, dtype=torch.bfloat16)
    probs = torch.rand(6)

    def fake_fused(flat_inter, flat_probs):
        return _torch_epilogue_reference(flat_inter, flat_probs)

    monkeypatch.setattr(E, "apply_glu_probs_epilogue", fake_fused)
    monkeypatch.setattr(B, "_TRAINER_EPILOGUE_ENABLED", False)
    with torch.no_grad():
        reference, _ = B._batched_experts_forward(module, x, counts, probs)

    before = _trainer_epilogue_stats()
    monkeypatch.setattr(B, "_TRAINER_EPILOGUE_ENABLED", True)
    with torch.no_grad():
        got, _ = B._batched_experts_forward(module, x, counts, probs)
    after = _trainer_epilogue_stats()

    assert torch.equal(got, reference)
    assert after["served"] > before["served"]


def test_leaftree_owner_report_is_execution_evidence(capsys):
    before = dict(B._LEAFTREE_OWNER_COUNTS)
    reported = B._LEAFTREE_OWNER_REPORTED
    try:
        B._LEAFTREE_OWNER_COUNTS.update({"ingemm": 1, "fused": 0, "buffer": 0, "single_leaf": 0})
        B._LEAFTREE_OWNER_REPORTED = 0
        B._report_leaftree_owner()
        line = capsys.readouterr().out
        assert "[ISOEXEC-MOE-LEAFTREE-OWNER]" in line
        assert "served=1 ingemm=1 fused=0 buffer=0 single_leaf=0" in line
    finally:
        B._LEAFTREE_OWNER_COUNTS.update(before)
        B._LEAFTREE_OWNER_REPORTED = reported
