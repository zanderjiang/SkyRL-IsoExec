"""CPU proofs for the packed, atomics-free native GDN convolution VJP."""

from __future__ import annotations

import inspect
import sys
from types import ModuleType

import pytest
import torch

from skyrl.backends.skyrl_train.isoexec.ops.gdn import gdn_ops as G


def _native_conv_backward_stats():
    # inlined: the public module keeps the counters but not this test accessor
    return dict(G._NATIVE_CONV_BWD_COUNTS)



def _eager_reference(x, weight, bias, bounds, activation):
    outputs = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        outputs.append(G.gdn_causal_conv(x[start:stop], weight, bias, activation=activation))
    return torch.cat(outputs, dim=0)


def test_native_conv_channel_last_input_is_a_zero_copy_view_for_production_layout():
    x = torch.randn(19, 7)

    x_dt = G._native_conv_channel_last_input(x)

    assert x_dt.shape == (7, 19)
    assert x_dt.stride() == (1, 7)
    assert x_dt.data_ptr() == x.data_ptr()
    assert x_dt.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    assert torch.equal(x_dt, x.transpose(0, 1))


def test_native_conv_channel_last_input_normalizes_only_an_unusual_source_layout():
    base = torch.randn(13, 18)
    x = base[:, ::2]
    assert x.stride(1) != 1

    x_dt = G._native_conv_channel_last_input(x)

    assert x_dt.shape == (9, 13)
    assert x_dt.stride(0) == 1 and x_dt.stride(1) > 1
    assert torch.equal(x_dt, x.transpose(0, 1))


def test_native_conv_forward_source_does_not_compact_the_transpose():
    source = inspect.getsource(G._GdnNativeConvAutograd.forward)
    helper = inspect.getsource(G._native_conv_channel_last_input)
    assert "_native_conv_channel_last_input(x)" in source
    assert "transpose(0, 1).contiguous()" not in source
    assert "x = x.contiguous()" in helper
    assert "x_dt = x.transpose(0, 1)" in helper
    assert "metadata = causal_conv1d_metadata(cu_seqlens)" in source


def test_native_conv_cached_launch_args_remove_only_stateless_scratch_work():
    source = inspect.getsource(G._GdnNativeConvAutograd.forward)
    assert "metadata = causal_conv1d_metadata(cu_seqlens)" in source
    assert "scratch = x.new_empty" in source
    assert "idx = metadata.cache_indices" in source
    assert "has0 = metadata.has_initial_state" in source
    # Cache-off is the unchanged fallback, including the original zero-initialised scratch.
    assert "scratch = x.new_zeros" in source


@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_conv_vjp_matches_eager_autograd(monkeypatch, activation, with_bias, dtype):
    monkeypatch.setenv("SKYRL_ISOEXEC_GDN_CONV_BWD_ROWS", "5")
    torch.manual_seed(17)
    bounds = [0, 7, 8, 17]
    cu = torch.tensor(bounds, dtype=torch.int32)
    x = torch.randn(bounds[-1], 11, dtype=dtype, requires_grad=True)
    weight = torch.randn(11, 4, dtype=dtype, requires_grad=True)
    bias = torch.randn(11, dtype=dtype, requires_grad=True) if with_bias else None
    dy = torch.randn_like(x)

    y = _eager_reference(x, weight, bias, bounds, activation)
    leaves = (x, weight, bias) if with_bias else (x, weight)
    reference = torch.autograd.grad(y, leaves, dy)
    with torch.no_grad():
        got = G._native_conv_vjp(dy, x.detach(), weight.detach(), bias.detach() if with_bias else None, cu, activation)

    rtol, atol = (3e-2, 3e-2) if dtype == torch.bfloat16 else (2e-5, 2e-5)
    torch.testing.assert_close(got[0], reference[0], rtol=rtol, atol=atol)
    torch.testing.assert_close(got[1], reference[1], rtol=rtol, atol=atol)
    if with_bias:
        torch.testing.assert_close(got[2], reference[2], rtol=rtol, atol=atol)
    else:
        assert got[2] is None


def test_native_conv_vjp_is_repeatable_and_does_not_cross_sequence_boundaries(monkeypatch):
    monkeypatch.setenv("SKYRL_ISOEXEC_GDN_CONV_BWD_ROWS", "3")
    cu = torch.tensor([0, 5, 9], dtype=torch.int32)
    x = torch.arange(9 * 3, dtype=torch.float32).view(9, 3)
    weight = torch.arange(1, 13, dtype=torch.float32).view(3, 4)
    dy = torch.zeros_like(x)
    dy[5] = 1.0  # first output of sequence two must not reach sequence one's final inputs

    with torch.no_grad():
        first = G._native_conv_vjp(dy, x, weight, None, cu, None)
        second = G._native_conv_vjp(dy, x, weight, None, cu, None)
    assert all(torch.equal(a, b) for a, b in zip(first[:2], second[:2]))
    assert torch.count_nonzero(first[0][:5]) == 0


def test_native_conv_custom_function_backward_moves_served_census(monkeypatch):
    leaf_name = "vllm.model_executor.layers.mamba.ops.causal_conv1d"
    names = ["vllm", "vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.mamba"]
    names.append("vllm.model_executor.layers.mamba.ops")
    modules = {name: ModuleType(name) for name in names}
    leaf = ModuleType(leaf_name)

    def fake_causal_conv(x_t, weight, bias, *, query_start_loc, activation, **_):
        bounds = query_start_loc.tolist()
        outputs = [
            G.gdn_causal_conv(x_t.T[start:stop], weight, bias, activation=activation)
            for start, stop in zip(bounds[:-1], bounds[1:])
        ]
        return torch.cat(outputs, dim=0).T.contiguous()

    leaf.causal_conv1d_fn = fake_causal_conv
    modules[leaf_name] = leaf
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    cu = torch.tensor([0, 4, 9], dtype=torch.int32)
    x = torch.randn(9, 6, requires_grad=True)
    weight = torch.randn(6, 4, requires_grad=True)
    bias = torch.randn(6, requires_grad=True)
    before = _native_conv_backward_stats()
    monkeypatch.setattr(G, "_NATIVE_CONV_ANALYTIC_BWD_ENABLED", False)
    out = G.gdn_native_conv(x, weight, bias, cu_seqlens=cu, activation="silu")
    out.sum().backward()
    after_fallback = _native_conv_backward_stats()

    assert after_fallback["served"] == before["served"]
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (x, weight, bias))

    x2 = x.detach().clone().requires_grad_()
    weight2 = weight.detach().clone().requires_grad_()
    bias2 = bias.detach().clone().requires_grad_()
    monkeypatch.setattr(G, "_NATIVE_CONV_ANALYTIC_BWD_ENABLED", True)
    out2 = G.gdn_native_conv(x2, weight2, bias2, cu_seqlens=cu, activation="silu")
    out2.sum().backward()
    after_analytic = _native_conv_backward_stats()

    assert after_analytic["served"] == after_fallback["served"] + 1
    assert after_analytic["tokens"] == after_fallback["tokens"] + 9
    assert after_analytic["taps"] == after_fallback["taps"] + 4
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (x2, weight2, bias2))


def test_analytic_native_conv_branch_has_no_eager_graph_or_sequence_host_loop():
    source = inspect.getsource(G._GdnNativeConvAutograd.backward)
    assert "_native_conv_vjp" in source
    assert "if not _NATIVE_CONV_ANALYTIC_BWD_ENABLED:" in source
    # The three per-chunk bodies were extracted (2026-08-16) so they can be routed through the
    # backward-region compile ledger; the properties this test guards are unchanged, they just
    # live in the helpers now.
    helper = inspect.getsource(G._native_conv_vjp)
    chunks = "".join(
        inspect.getsource(fn) for fn in (G._conv_vjp_dz_chunk, G._conv_vjp_act_chunk, G._conv_vjp_dxdw_chunk)
    )
    whole = helper + chunks
    assert "for tap in range(taps)" in chunks
    assert "torch.enable_grad" not in whole
    assert "torch.autograd.grad" not in whole
    assert ".tolist()" not in whole
    assert ".index_add" not in whole and ".scatter" not in whole
    assert "dxf = torch.zeros((rows, channels)" in chunks
    assert "torch.zeros_like(dz)" not in whole, "dx accumulation must stay row-chunked"
    # The row-chunk loop stays OUTSIDE every compiled region, so compiling cannot defeat the
    # temporary bound the chunking exists to provide.
    assert "for start in range(0, total, chunk_rows)" in helper
    assert "range(0, total, chunk_rows)" not in chunks


def test_native_conv_backward_rows_refuses_nonpositive(monkeypatch):
    monkeypatch.setenv("SKYRL_ISOEXEC_GDN_CONV_BWD_ROWS", "0")
    with pytest.raises(ValueError, match="must be positive"):
        G._native_conv_backward_rows()
