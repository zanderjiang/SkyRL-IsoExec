"""Bitwise parity tests for the native-kernel GDN forward (vLLM's own fused conv + recurrent core).

T1/T2 pin conv continuity and chunked resume; T3/T4 pin that decode is a bitwise continuation of
recurrent prefill and that a chunked resume equals the whole; T5/T6 are informational. Dims are the
35B TP=8 local shard, so what passes here is what the production engine runs.
"""

import os
import sys

import torch

if not torch.cuda.is_available():  # promoted nightly battery: needs one CUDA device
    print("SKIP: no CUDA device")
    raise SystemExit(0)

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

DEV = "cuda"
H, HV, K, V, W = 2, 4, 128, 128, 4
D = 2 * H * K + HV * V  # 1024, the mixed_qkv width
DTYPE = torch.bfloat16


def _mk_inputs(T, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    mixed = torch.randn(T, D, generator=g, device=DEV, dtype=DTYPE) * 0.5
    a = torch.randn(T, HV, generator=g, device=DEV, dtype=DTYPE) * 0.5
    b = torch.randn(T, HV, generator=g, device=DEV, dtype=DTYPE) * 0.5
    return mixed, a, b


def _split_qkv(y):
    """[T, D] post-conv -> q [1,T,H,K], k [1,T,H,K], v [1,T,HV,V] (kernel handles GQA in-kernel)."""
    T = y.shape[0]
    kd = H * K
    q = y[:, :kd].contiguous().view(T, H, K)
    k = y[:, kd : 2 * kd].contiguous().view(T, H, K)
    v = y[:, 2 * kd :].contiguous().view(T, HV, V)
    return q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)


def _conv_weight(seed=7):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randn(D, W, generator=g, device=DEV, dtype=DTYPE) * 0.3
    bias = torch.randn(D, generator=g, device=DEV, dtype=DTYPE) * 0.1
    return w, bias


def _gating_params(seed=11):
    g = torch.Generator(device=DEV).manual_seed(seed)
    A_log = torch.randn(HV, generator=g, device=DEV, dtype=DTYPE) * 0.2
    dt_bias = torch.randn(HV, generator=g, device=DEV, dtype=DTYPE) * 0.2
    return A_log, dt_bias


def _bitwise(name, x, y):
    same = torch.equal(x, y)
    if same:
        print(f"  PASS (bitwise) {name}")
        return True
    d = (x.float() - y.float()).abs()
    n_diff = (x != y).sum().item()
    print(
        f"  FAIL {name}: {n_diff}/{x.numel()} elements differ, " f"max {d.max().item():.3e} mean {d.mean().item():.3e}"
    )
    return False


def test_conv(results):
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )

    w, bias = _conv_weight()
    T = 37
    mixed, _, _ = _mk_inputs(T + 1, seed=1)

    def fresh_states(n=4):
        return torch.zeros(n, D, W - 1, device=DEV, dtype=DTYPE)

    # --- T1: fn(T) + update(1) == fn(T+1) ---
    cs_a = fresh_states()
    qsl = torch.tensor([0, T], dtype=torch.int32, device=DEV)
    idx = torch.tensor([1], dtype=torch.int32, device=DEV)
    has0 = torch.zeros(1, dtype=torch.bool, device=DEV)
    causal_conv1d_fn(
        mixed[:T].transpose(0, 1).contiguous(),
        w,
        bias,
        conv_states=cs_a,
        query_start_loc=qsl,
        cache_indices=idx,
        has_initial_state=has0,
        activation="silu",
    )
    y_dec = causal_conv1d_update(
        mixed[T : T + 1].clone(),
        cs_a,
        w,
        bias,
        "silu",
        conv_state_indices=idx,
    )

    cs_b = fresh_states()
    qsl_b = torch.tensor([0, T + 1], dtype=torch.int32, device=DEV)
    y_b = causal_conv1d_fn(
        mixed[: T + 1].transpose(0, 1).contiguous(),
        w,
        bias,
        conv_states=cs_b,
        query_start_loc=qsl_b,
        cache_indices=idx,
        has_initial_state=has0,
        activation="silu",
    ).transpose(0, 1)
    results.append(_bitwise("T1 conv: prefill-then-update last tok == longer prefill", y_dec[0], y_b[T]))
    results.append(_bitwise("T1 conv: state after update == state after longer prefill", cs_a[1], cs_b[1]))

    # --- T2: chunked prefill resume ---
    T1, T2 = 19, T + 1 - 19
    cs_c = fresh_states()
    y_c1 = causal_conv1d_fn(
        mixed[:T1].transpose(0, 1).contiguous(),
        w,
        bias,
        conv_states=cs_c,
        query_start_loc=torch.tensor([0, T1], dtype=torch.int32, device=DEV),
        cache_indices=idx,
        has_initial_state=has0,
        activation="silu",
    ).transpose(0, 1)
    has1 = torch.ones(1, dtype=torch.bool, device=DEV)
    y_c2 = causal_conv1d_fn(
        mixed[T1 : T + 1].transpose(0, 1).contiguous(),
        w,
        bias,
        conv_states=cs_c,
        query_start_loc=torch.tensor([0, T2], dtype=torch.int32, device=DEV),
        cache_indices=idx,
        has_initial_state=has1,
        activation="silu",
    ).transpose(0, 1)
    y_c = torch.cat([y_c1, y_c2], dim=0)
    results.append(_bitwise("T2 conv: chunk1+chunk2 outputs == whole", y_c, y_b))
    results.append(_bitwise("T2 conv: chunk1+chunk2 final state == whole", cs_c[1], cs_b[1]))
    return y_b  # post-conv activations for the core tests


def test_core(results, y_conv):
    from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )

    A_log, dt_bias = _gating_params()
    T = y_conv.shape[0]
    _, a, b = _mk_inputs(T, seed=2)
    q, k, v = _split_qkv(y_conv)

    def fresh_ssm(n=4):
        return torch.zeros(n, HV, V, K, device=DEV, dtype=torch.float32)

    def grid_idx(row, t):
        """[1, t] per-token state-index grid: col 0 = load slot, col t-1 = final store, 0 = skip.

        The kernel stores at every token whose index is > 0, so all other columns must stay 0.
        """
        g = torch.zeros(1, t, dtype=torch.int32)
        g[0, 0] = row
        g[0, t - 1] = row
        return g.to(DEV)

    # --- one varlen call over the whole sequence ---
    ssm_a = fresh_ssm()
    idx = grid_idx(1, T)
    cu = torch.tensor([0, T], dtype=torch.int32, device=DEV)
    o_a, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=ssm_a,
        inplace_final_state=True,
        cu_seqlens=cu,
        ssm_state_indices=idx,
        use_qk_l2norm_in_kernel=True,
    )

    # --- T3: T sequential single-token calls resuming through the state row ---
    ssm_b = fresh_ssm()
    outs = []
    cu1 = torch.tensor([0, 1], dtype=torch.int32, device=DEV)
    idx1 = torch.tensor([1], dtype=torch.int32, device=DEV)  # decode: 1-D [N], row per seq
    for t in range(T):
        o_t, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a[t : t + 1],
            b=b[t : t + 1],
            dt_bias=dt_bias,
            q=q[:, t : t + 1],
            k=k[:, t : t + 1],
            v=v[:, t : t + 1],
            initial_state=ssm_b,
            inplace_final_state=True,
            cu_seqlens=cu1,
            ssm_state_indices=idx1,
            use_qk_l2norm_in_kernel=True,
        )
        outs.append(o_t)
    o_b = torch.cat(outs, dim=1)
    results.append(_bitwise("T3 core: one varlen prefill call == T sequential decode calls", o_a, o_b))
    results.append(_bitwise("T3 core: final state prefill == decode chain", ssm_a[1], ssm_b[1]))

    # --- T4: two-chunk resume == whole ---
    T1 = 19
    ssm_c = fresh_ssm()
    o_c1, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a[:T1],
        b=b[:T1],
        dt_bias=dt_bias,
        q=q[:, :T1],
        k=k[:, :T1],
        v=v[:, :T1],
        initial_state=ssm_c,
        inplace_final_state=True,
        cu_seqlens=torch.tensor([0, T1], dtype=torch.int32, device=DEV),
        ssm_state_indices=grid_idx(1, T1),
        use_qk_l2norm_in_kernel=True,
    )
    o_c2, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a[T1:],
        b=b[T1:],
        dt_bias=dt_bias,
        q=q[:, T1:],
        k=k[:, T1:],
        v=v[:, T1:],
        initial_state=ssm_c,
        inplace_final_state=True,
        cu_seqlens=torch.tensor([0, T - T1], dtype=torch.int32, device=DEV),
        ssm_state_indices=grid_idx(1, T - T1),
        use_qk_l2norm_in_kernel=True,
    )
    o_c = torch.cat([o_c1, o_c2], dim=1)
    results.append(_bitwise("T4 core: chunk1+chunk2 == whole (outputs)", o_c, o_a))
    results.append(_bitwise("T4 core: chunk1+chunk2 == whole (final state)", ssm_c[1], ssm_a[1]))

    # --- T5: in-kernel l2norm vs standalone l2norm_fwd ---
    from vllm.model_executor.layers.fla.ops.chunk import l2norm_fwd

    qn = l2norm_fwd(q.squeeze(0)).unsqueeze(0)
    kn = l2norm_fwd(k.squeeze(0)).unsqueeze(0)
    ssm_d = fresh_ssm()
    o_d, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=qn,
        k=kn,
        v=v,
        initial_state=ssm_d,
        inplace_final_state=True,
        cu_seqlens=cu,
        ssm_state_indices=idx,
        use_qk_l2norm_in_kernel=False,
    )
    # Informational, not a gate: if the in-kernel l2norm and l2norm_fwd differ, trainer and engine
    # must both use the in-kernel form.
    same = torch.equal(o_a, o_d)
    d5 = (o_a.float() - o_d.float()).abs()
    print(
        f"  INFO T5 in-kernel l2norm vs standalone l2norm_fwd: "
        f"{'BITWISE EQUAL' if same else f'DIFFER max {d5.max().item():.3e} mean {d5.mean().item():.3e}'}"
        f" -> trainer {'may use either' if same else 'MUST use in-kernel l2norm (composition A)'}"
    )
    results.append(True)
    return q, k, v, a, b, A_log, dt_bias, o_a


def test_old_vs_native(results, pack):
    """T6: the current isoexec composition vs the native one; a nonzero diff is expected."""
    from skyrl.backends.skyrl_train.isoexec.ops.gdn.gdn_ops import (
        gdn_gate_and_beta,
        gdn_l2norm,
        gdn_recurrent_kernel,
    )

    q, k, v, a, b, A_log, dt_bias, o_native = pack
    T = q.shape[1]
    # current composition: standalone l2norm + GQA expand + bf16-exp gating + our recurrent kernel
    rep = HV // H
    qz = gdn_l2norm(q.squeeze(0).contiguous()).repeat_interleave(rep, dim=1)
    kz = gdn_l2norm(k.squeeze(0).contiguous()).repeat_interleave(rep, dim=1)
    g, beta = gdn_gate_and_beta(a, b, A_log, dt_bias)
    ssm = torch.zeros(4, HV, V, K, device=DEV, dtype=torch.float32)
    idx_grid = torch.zeros(1, T, dtype=torch.int32)
    idx_grid[0, 0] = 1
    idx_grid[0, T - 1] = 1
    o_old = gdn_recurrent_kernel(
        qz.unsqueeze(0),
        kz.unsqueeze(0),
        v,
        g.unsqueeze(0),
        beta.unsqueeze(0),
        ssm_state=ssm,
        state_indices=idx_grid.to(DEV),
        cu_seqlens=torch.tensor([0, T], dtype=torch.int32, device=DEV),
    )
    d = (o_old.float() - o_native.float()).abs()
    exact = (o_old == o_native).float().mean().item()
    print(
        f"  INFO T6 old-vs-native composition: exact={exact * 100:.1f}% "
        f"max {d.max().item():.3e} mean {d.mean().item():.3e} "
        f"(nonzero EXPECTED: bf16-exp gating + conv/l2norm variants differ; this is why the "
        f"trainer shim must switch compositions with the engine)"
    )
    results.append(True)  # informational, not a gate


def main():
    torch.cuda.init()
    results = []
    print(f"[gdn-native-parity] dims: H={H} HV={HV} K={K} V={V} W={W} D={D} dtype={DTYPE}")
    y_conv = test_conv(results)
    pack = test_core(results, y_conv)
    test_old_vs_native(results, pack)
    n_pass = sum(results)
    print(f"[gdn-native-parity] {n_pass}/{len(results)} PASS")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
