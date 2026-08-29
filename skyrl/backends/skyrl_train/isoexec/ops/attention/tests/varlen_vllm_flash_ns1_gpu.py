"""Bitwise gate for attention.varlen: vllm_flash_ns1 == varlen_custom at the registered pins.

Runs vLLM's flash_attn_varlen_func(num_splits=1, fa_version=3, causal) and torch's
varlen_attn(num_splits=1, window=(-1,0), FA3) on identical varlen batches (GQA, decode-length
rows, non_contiguous operands) and asserts torch.equal. One GPU, seconds.
"""

import os
import sys

import torch

if not torch.cuda.is_available():
    print("SKIP: no CUDA device")
    raise SystemExit(0)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 7)))  # repo root

import torch.nn.attention.varlen  # noqa: E402,F401
from torch.nn.attention import activate_flash_attention_impl  # noqa: E402
from vllm.vllm_flash_attn import flash_attn_varlen_func  # noqa: E402

DEV = "cuda"
_fails = []
_checks = 0


def _batch(seq_lens, h_q, h_kv, d, seed, non_contig=False):
    g = torch.Generator(device=DEV).manual_seed(seed)
    total = sum(seq_lens)

    def mk(heads):
        if non_contig:
            # channel-sliced view of a wider tensor: same values, non-unit outer stride
            base = torch.randn(total, heads, 2 * d, device=DEV, dtype=torch.bfloat16, generator=g)
            return base[..., :d]
        return torch.randn(total, heads, d, device=DEV, dtype=torch.bfloat16, generator=g)

    q, k, v = mk(h_q), mk(h_kv), mk(h_kv)
    cu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=DEV)
    cu[1:] = torch.cumsum(torch.tensor(seq_lens, device=DEV), 0)
    return q, k, v, cu, max(seq_lens)


def _check(name, seq_lens, h_q, h_kv, d, seed, non_contig=False):
    global _checks
    q, k, v, cu, max_len = _batch(seq_lens, h_q, h_kv, d, seed, non_contig)
    scale = d**-0.5
    ref = torch.nn.attention.varlen.varlen_attn(
        q, k, v, cu, cu, max_len, max_len, scale=scale, num_splits=1, enable_gqa=(h_q != h_kv), window_size=(-1, 0)
    )
    if isinstance(ref, tuple):
        ref = ref[0]
    got = flash_attn_varlen_func(
        q,
        k,
        v,
        max_len,
        cu,
        max_len,
        cu,
        causal=True,
        softmax_scale=scale,
        window_size=[-1, 0],
        num_splits=1,
        fa_version=3,
    )
    if isinstance(got, tuple):
        got = got[0]
    _checks += 1
    if torch.equal(ref, got):
        print(f"  PASS (bitwise) {name}")
    else:
        diff = (ref.float() - got.float()).abs().max().item()
        _fails.append(f"{name}: max|diff|={diff:.3e}")
        print(f"  FAIL {name}: max|diff|={diff:.3e}")


def main():
    activate_flash_attention_impl("FA3")  # load-bearing: the registered pin (see _register.py)
    # Qwen-3.5-shaped GQA at TP8 (h_q=2, h_kv=1, d=256) plus full-rotary and decode-like rows.
    _check("prefill mix, GQA d=256", [2239, 17, 384], 2, 1, 256, 0)
    _check("decode-like rows (len 1)", [1, 1, 1, 977], 2, 1, 256, 1)
    _check("MHA d=128", [513, 64], 4, 4, 128, 2)
    _check("single long seq", [4096], 2, 1, 256, 3)
    _check("non_contiguous operands", [321, 64, 1], 2, 1, 128, 4, non_contig=True)
    if _fails:
        print(f"FAILED {len(_fails)}/{_checks}:")
        for f in _fails:
            print("  - " + f)
        sys.exit(1)
    print(f"ALL {_checks} CHECKS PASS (torch.equal): vllm_flash_ns1 == varlen_custom at num_splits=1/FA3")


if __name__ == "__main__":
    main()
