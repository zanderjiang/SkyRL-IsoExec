"""CPU/AOT proof obligations for the isolated exact shared+routed owner composition.

The live TP8 battery proves cross-rank bytes and performance.  These tests pin the part that is
cheaper and stronger to prove before a GPU is free: the two balanced trees are the canonical
left=lower-rank tree, all four bf16 rounding boundaries are present in the generated program, the
compiler did not contract the shared multiply with the final add, and unsupported plumbing refuses
instead of silently returning a different expression.
"""

from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

_real_capability = torch.cuda.get_device_capability
torch.cuda.get_device_capability = lambda _device=None: (9, 0)
try:
    OC = importlib.import_module("skyrl.backends.skyrl_train.isoexec.ops.moe.moe_pik_combine_owner")
finally:
    torch.cuda.get_device_capability = _real_capability

PRE = importlib.import_module("skyrl.backends.skyrl_train.isoexec.ops.moe.moe_preamble_o12")


def _reset_shared_owner_fusion_state():
    # inlined: the public module keeps the state but not this test hook
    OC._SHARED_OWNER_ADMIT.clear()
    OC._SHARED_OWNER_GROUP.clear()
    OC._SHARED_OWNER_PROFILE_PROVISIONED.clear()
    for name in OC._SHARED_OWNER_COUNTS:
        OC._SHARED_OWNER_COUNTS[name] = 0
    OC._SHARED_OWNER_FIRST_REJECT = None


def _compile(world=8, k=8, routed_bf16=True, shared_bf16=True):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    fn = OC._shared_owner_push_kernel(world, k, routed_bf16, shared_bf16)
    while not hasattr(fn, "arg_names"):
        fn = fn.fn
    sig, consts = {}, {}
    for name in fn.arg_names:
        if name.startswith("rin"):
            sig[name] = "*bf16" if routed_bf16 else "*fp32"
        elif name.startswith("sin"):
            sig[name] = "*bf16" if shared_bf16 else "*fp32"
        elif name == "gate_ptr" or (name.startswith("out") and name != "out_base"):
            sig[name] = "*bf16"
        elif name == "rows_ptr":
            sig[name] = "*i32"
        elif name in ("K", "BLOCK"):
            sig[name] = "constexpr"
            consts[name] = {"K": k, "BLOCK": 1024}[name]
        else:
            sig[name] = "i32"
    return triton.compile(
        ASTSource(fn=fn, signature=sig, constexprs=consts),
        target=GPUTarget("cuda", 90, 32),
        options={"enable_fp_fusion": False},
    ).asm


def _fp_ops(ptx: str) -> list[str]:
    ops = []
    for line in ptx.splitlines():
        fields = line.strip().split()
        if not fields:
            continue
        op = fields[1] if fields[0].startswith("@") and len(fields) > 1 else fields[0]
        if op.split(".")[0] in ("add", "mul", "fma", "sub", "cvt") and (".f32" in op or ".bf16" in op or ".f16" in op):
            ops.append(op.rstrip(";"))
    return ops


def test_existing_routed_owner_source_is_not_modified():
    """The experiment must not mutate the current production kernel family."""
    assert OC._shared_owner_core(2, 2, True, True) not in OC._push_src(2, 2, True)
    assert "shared_root_bf" not in OC._push_src(8, 8, True)


def test_generated_source_pins_every_existing_rounding_boundary():
    src = OC._shared_push_src(2, 2, True, True)
    routed_round = src.index("routed_bf = routed_acc.to(tl.bfloat16)")
    shared_round = src.index("shared_root_bf = st0.to(tl.bfloat16)")
    gate_load = src.index("gate_bf = tl.load(gate_ptr + global_t).to(tl.bfloat16)")
    shared_product_round = src.index(
        "shared_bf = (shared_root_bf.to(tl.float32) * gate_bf.to(tl.float32)).to(tl.bfloat16)"
    )
    final_add_round = src.index("result = (routed_bf.to(tl.float32) + shared_bf.to(tl.float32)).to(tl.bfloat16)")
    store = src.index("tl.store(out0 + goff, result")
    assert routed_round < final_add_round < store
    assert shared_round < gate_load < shared_product_round < final_add_round


def test_both_peer_folds_use_the_same_canonical_balanced_tree():
    routed = OC._named_tree_src(8, True, inp="rin", tmp="rt", base="rbase", indent="")
    shared = OC._named_tree_src(8, True, inp="sin", tmp="st", base="sbase", indent="")
    # They are not two hand-written approximations: after renaming inputs/SSA/base, they are the
    # same emitted schedule character for character.
    normalized = routed.replace("rin", "sin").replace("rt", "st").replace("rbase", "sbase")
    assert normalized == shared
    assert routed.count("tl.load(") == 8
    assert routed.count(" = rt") == 7  # seven tree additions over eight leaves


def test_root_rounding_negative_control_has_teeth():
    """A concrete population where moving the shared-root round changes one bf16 bit."""
    words = torch.tensor(
        [-17751, -17670, 14873, -17612, 15207, -17567, 14798, -17682],
        dtype=torch.int16,
    )
    vals = words.view(torch.bfloat16).float()
    gate = torch.tensor([16063], dtype=torch.int16).view(torch.bfloat16).float()[0]
    cur = vals.clone()
    stride = 1
    while stride < 8:
        cur[:: 2 * stride] = cur[:: 2 * stride] + cur[stride :: 2 * stride]
        stride *= 2
    root = cur[0]
    exact = (root.to(torch.bfloat16).float() * gate).to(torch.bfloat16)
    moved = (root * gate).to(torch.bfloat16)
    assert exact.view(torch.int16).item() == -17629
    assert moved.view(torch.int16).item() == -17628


@pytest.mark.parametrize("world,k", [(2, 1), (2, 8), (8, 1), (8, 8)])
def test_sm90_machine_code_has_four_rounds_and_no_fma(world, k):
    asm = _compile(world, k)
    ptx = asm["ptx"]
    ops = _fp_ops(ptx)
    # Frontend IR pins four explicit truncations.  PTX legally selects native bf16 mul/add for the
    # last two; those instructions mean exactly "operate on bf16 operands and round to bf16", so
    # counting only cvt.rn would incorrectly call a preserved round missing.
    assert asm["ttir"].count("arith.truncf") == 4
    assert ops.count("mul.rn.bf16") == 8
    assert ops.count("add.rn.bf16") >= 8
    assert not any(op.startswith("fma") for op in ops), ops


def test_launch_disables_fp_contraction_and_unsupported_transport_refuses(monkeypatch):
    assert "enable_fp_fusion=False" in inspect.getsource(OC._SharedExchange.run)
    monkeypatch.setattr(OC, "_p2p_ok", lambda _group: False)
    routed = torch.empty(16, 4, dtype=torch.bfloat16)
    shared = torch.empty(2, 4, dtype=torch.bfloat16)
    gate = torch.empty(2, dtype=torch.bfloat16)
    rows = torch.empty(2, 8, dtype=torch.int32)
    assert OC._owner_shared_combine(routed, shared, gate, rows, 2, 8, None, True, True) is None


def test_flag_is_default_off_and_forwarded_through_the_colocated_actor():
    from skyrl.backends.skyrl_train.isoexec.core import flags as FLAGS

    matches = [f for f in FLAGS.FLAGS if f.name == "SKYRL_ISOEXEC_MOE_SHARED_OWNER_FUSION"]
    assert len(matches) == 1
    flag = matches[0]
    assert flag.default == "0"
    assert flag.sides == ("engine",)
    assert flag.disposition == FLAGS.DEPLOYMENT
    assert flag.forwarded_by == (FLAGS.TRAIN,)
    assert flag.name in FLAGS.actor_forwarding_tuple(FLAGS.TRAIN)


def test_group_vote_is_required_and_cached(monkeypatch):
    OC._ensure_canonical_pik()
    import pik.allreduce as AR

    _reset_shared_owner_fusion_state()
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_SHARED_OWNER_FUSION", "1")
    monkeypatch.setattr(OC.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(OC, "_group_key", lambda _group: (0, 1, 2, 3, 4, 5, 6, 7))
    monkeypatch.setattr(OC, "_ensure_canonical_pik", lambda: None)
    monkeypatch.setattr(AR, "_capturing", lambda: False)
    votes = []

    def disagree(local, _group, _device):
        votes.append(local)
        return False

    monkeypatch.setattr(AR, "_agree", disagree)
    assert not OC.shared_owner_group_enabled(object(), torch.device("cpu"))
    assert votes == [True]
    # The same process group may own 40 layers. It votes once, then every layer consumes the
    # identical cached install decision without 39 extra collectives.
    assert not OC.shared_owner_group_enabled(object(), torch.device("cpu"))
    assert votes == [True]
    counts = OC.shared_owner_fusion_counts()
    assert counts["group_agreement"] == {"(0, 1, 2, 3, 4, 5, 6, 7)": False}


def test_shared_expert_instance_handoff_rounds_gate_before_payload(monkeypatch):
    hidden = torch.tensor(
        [[[0.5, -0.25, 1.0, -2.0]], [[-0.0, 0.75, -1.5, 2.5]]],
        dtype=torch.bfloat16,
    )
    inter = torch.arange(24, dtype=torch.bfloat16).reshape(2, 1, 12) / 16
    partial = torch.arange(8, dtype=torch.bfloat16).reshape(2, 1, 4) / 8
    dispatcher = SimpleNamespace(_isoexec_shared_owner_payload=None)
    fallback = object()
    shared = SimpleNamespace(
        linear_fc1=lambda _x: (inter, None),
        config=SimpleNamespace(activation_func=F.silu, glu_linear_offset=0.0),
        use_shared_expert_gate=True,
        gate_weight=torch.tensor([[0.25, -0.5, 0.75, 1.0]], dtype=torch.bfloat16),
        _ix_shared_owner_dispatcher=dispatcher,
        _ix_shared_owner_orig_forward=lambda _x: fallback,
    )
    monkeypatch.setattr(PRE, "preamble_o12_enabled", lambda: False)
    monkeypatch.setattr(OC, "shared_fc2_subtree_partial", lambda _linear, _h: (partial, True))
    shared.linear_fc2 = object()

    assert PRE._shared_owner_expert_forward(shared, hidden) is None
    got_partial, got_gate, got_bf16 = dispatcher._isoexec_shared_owner_payload
    expected_gate = torch.sigmoid(F.linear(hidden, shared.gate_weight)).reshape(-1)
    assert got_partial.data_ptr() == partial.data_ptr()
    assert got_gate.dtype == torch.bfloat16
    assert torch.equal(got_gate.view(torch.int16), expected_gate.view(torch.int16))
    assert got_bf16 is True


def test_shared_expert_handoff_fails_closed_before_it_can_drop_shared_output(monkeypatch):
    hidden = torch.zeros(2, 1, 4, dtype=torch.bfloat16)
    inter = torch.zeros(2, 1, 12, dtype=torch.bfloat16)
    fallback = torch.full_like(hidden, 7)
    shared = SimpleNamespace(
        linear_fc1=lambda _x: (inter, None),
        linear_fc2=object(),
        config=SimpleNamespace(activation_func=F.silu, glu_linear_offset=0.0),
        use_shared_expert_gate=False,
        _ix_shared_owner_dispatcher=SimpleNamespace(_isoexec_shared_owner_payload=None),
        _ix_shared_owner_orig_forward=lambda _x: fallback,
    )
    monkeypatch.setattr(PRE, "preamble_o12_enabled", lambda: False)
    monkeypatch.setattr(
        OC,
        "shared_fc2_subtree_partial",
        lambda _linear, _h: (torch.zeros_like(hidden), True),
    )
    assert PRE._shared_owner_expert_forward(shared, hidden) is fallback
    assert shared._ix_shared_owner_dispatcher._isoexec_shared_owner_payload is None


def test_dispatcher_consumes_payload_once_and_keeps_the_existing_fallback():
    src = inspect.getsource(OC.install_moe_pik_owner_combine)
    read = src.index('payload = getattr(self, "_isoexec_shared_owner_payload", None)')
    clear = src.index("self._isoexec_shared_owner_payload = None")
    fused = src.index("out = _shared_owner_dispatch(")
    fallback = src.index("out = _owner_combine(", fused)
    assert read < clear < fused < fallback
    assert "shared_owner_fusion_enabled() and payload is not None" in src


def test_memory_profile_scope_is_nested_and_exception_safe():
    assert not OC.shared_owner_memory_profile_active()
    with OC.shared_owner_memory_profile_scope():
        assert OC.shared_owner_memory_profile_active()
        with OC.shared_owner_memory_profile_scope():
            assert OC.shared_owner_memory_profile_active()
        assert OC.shared_owner_memory_profile_active()
    assert not OC.shared_owner_memory_profile_active()

    with pytest.raises(RuntimeError, match="injected"):
        with OC.shared_owner_memory_profile_scope():
            raise RuntimeError("injected")
    assert not OC.shared_owner_memory_profile_active()


def _profile_dispatch_operands():
    T, H, k = 2, 4, 2
    return (
        torch.arange(T * k * H, dtype=torch.bfloat16).reshape(T * k, H),
        torch.arange(T * H, dtype=torch.bfloat16).reshape(T, H),
        torch.full((T,), 0.5, dtype=torch.bfloat16),
        torch.arange(T * k, dtype=torch.int32).reshape(T, k),
        T,
        k,
        object(),
        True,
        True,
    )


def test_memory_profile_provisions_persistent_pools_but_never_calls_candidate(monkeypatch):
    _reset_shared_owner_fusion_state()
    operands = _profile_dispatch_operands()
    routed, shared, _gate, _rows, T, k, group, _rb, _sb = operands
    key = (2, T, routed.shape[-1], k, str(routed.dtype), str(shared.dtype))
    reference = torch.full((T, routed.shape[-1]), 7, dtype=torch.bfloat16)
    calls = {"reference": 0, "candidate": 0, "provision": 0}
    order = []

    monkeypatch.setattr(OC.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(OC, "_ensure_canonical_pik", lambda: None)
    import pik.allreduce as AR

    monkeypatch.setattr(AR, "_capturing", lambda: False)
    monkeypatch.setattr(AR, "_agree", lambda local, _group, _device: local)

    def provision(*args):
        calls["provision"] += 1
        order.append("provision")
        # Full live geometry, not a toy shape hidden from the memory profile.
        assert args[0].numel() == T * k * routed.shape[-1]
        assert args[1].numel() == T * routed.shape[-1]
        return True, ""

    def ref(*_args):
        calls["reference"] += 1
        order.append("reference")
        return reference

    def candidate(*_args):
        calls["candidate"] += 1
        return torch.zeros_like(reference)

    monkeypatch.setattr(OC, "_shared_owner_provision", provision)
    monkeypatch.setattr(OC, "_shared_owner_reference", ref)
    monkeypatch.setattr(OC, "_owner_shared_combine", candidate)
    with OC.shared_owner_memory_profile_scope():
        assert OC._shared_owner_dispatch(*operands) is reference
        assert OC._shared_owner_dispatch(*operands) is reference

    assert calls == {"reference": 2, "candidate": 0, "provision": 1}
    assert order[:2] == ["reference", "provision"]
    assert key not in OC._SHARED_OWNER_ADMIT
    assert OC._SHARED_OWNER_PROFILE_PROVISIONED[key] is True
    counts = OC.shared_owner_fusion_counts()
    assert counts["profile_provisions"] == 1
    assert counts["profile_deferred"] == 2

    # Leaving vLLM's peak-measurement scope does not silently admit the shape.  The next ordinary
    # eager warmup performs the unchanged live reference/candidate compare and only then admits.
    monkeypatch.setattr(OC, "_owner_shared_combine", lambda *_args: reference.clone())
    got = OC._shared_owner_dispatch(*operands)
    assert torch.equal(got.view(torch.int16), reference.view(torch.int16))
    assert OC._SHARED_OWNER_ADMIT[key] is True
    assert OC.shared_owner_fusion_counts()["admitted"] == 1


def test_memory_profile_provision_refusal_is_group_uniform_and_fail_closed(monkeypatch):
    _reset_shared_owner_fusion_state()
    operands = _profile_dispatch_operands()
    routed, shared, _gate, _rows, T, k, _group, _rb, _sb = operands
    key = (2, T, routed.shape[-1], k, str(routed.dtype), str(shared.dtype))
    reference = torch.full((T, routed.shape[-1]), 9, dtype=torch.bfloat16)

    monkeypatch.setattr(OC.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(OC, "_ensure_canonical_pik", lambda: None)
    import pik.allreduce as AR

    monkeypatch.setattr(AR, "_capturing", lambda: False)
    votes = []
    monkeypatch.setattr(AR, "_agree", lambda local, _group, _device: votes.append(local) or False)
    monkeypatch.setattr(OC, "_shared_owner_provision", lambda *_args: (False, "injected refusal"))
    monkeypatch.setattr(OC, "_shared_owner_reference", lambda *_args: reference)
    monkeypatch.setattr(
        OC,
        "_owner_shared_combine",
        lambda *_args: pytest.fail("candidate must not run after provisioning refusal"),
    )

    with OC.shared_owner_memory_profile_scope():
        assert OC._shared_owner_dispatch(*operands) is reference
    assert votes == [False]
    assert OC._SHARED_OWNER_ADMIT[key] is False
    assert OC.shared_owner_fusion_counts()["fallbacks"] == 1


def test_vllm_worker_scope_patch_is_scoped_and_idempotent():
    from skyrl.backends.skyrl_train.isoexec.runtimes.vllm import vllm_plugin

    seen = []

    class FakeRunner:
        def profile_run(self):
            seen.append(("profile_run", OC.shared_owner_memory_profile_active()))

        def profile_cudagraph_memory(self):
            seen.append(("profile_cudagraph", OC.shared_owner_memory_profile_active()))

    class FakeWorker:
        def __init__(self):
            self.cache_config = SimpleNamespace(kv_cache_memory_bytes=None)
            self.model_runner = FakeRunner()

        def determine_available_memory(self, value):
            seen.append(("determine_entry", OC.shared_owner_memory_profile_active()))
            self.model_runner.profile_run()
            self.model_runner.profile_cudagraph_memory()
            return value

    original = FakeWorker.determine_available_memory
    vllm_plugin._install_shared_owner_memory_profile_scope(FakeWorker)
    installed = FakeWorker.determine_available_memory
    vllm_plugin._install_shared_owner_memory_profile_scope(FakeWorker)
    assert FakeWorker.determine_available_memory is installed
    assert installed._isoexec_shared_owner_profile_original is original
    worker = FakeWorker()
    original_profile_run = worker.model_runner.profile_run
    assert worker.determine_available_memory(123) == 123
    assert seen == [
        ("determine_entry", False),
        ("profile_run", True),
        ("profile_cudagraph", False),
    ]
    assert worker.model_runner.profile_run == original_profile_run
    assert not OC.shared_owner_memory_profile_active()


def test_vllm_worker_scope_refuses_manual_kv_bytes_that_skip_pool_accounting():
    from skyrl.backends.skyrl_train.isoexec.runtimes.vllm import vllm_plugin

    class FakeWorker:
        cache_config = SimpleNamespace(kv_cache_memory_bytes=1 << 30)

        def determine_available_memory(self):
            pytest.fail("manual KV sizing must be refused before upstream profiling")

    vllm_plugin._install_shared_owner_memory_profile_scope(FakeWorker)
    with pytest.raises(RuntimeError, match="bypasses persistent-pool accounting"):
        FakeWorker().determine_available_memory()
    assert not OC.shared_owner_memory_profile_active()
