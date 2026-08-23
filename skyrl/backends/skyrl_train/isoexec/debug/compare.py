"""Offline comparator for debug-mode traces: where and how do the two runtimes diverge.

Usage:
    python -m skyrl.backends.skyrl_train.isoexec.debug.compare TRACE_DIR_A TRACE_DIR_B \\
        [--case-a trainer_score] [--case-b engine_prefill] [--layers N] \\
        [--regions moe.router,gdn.core] [--json out.json]

A and B are trace directories (conventionally trainer and engine); every ``*.jsonl`` inside is
read. Records are grouped by (region, layer, out-index) after optional case filtering, aligned
by order of appearance (step, then call), and compared digest-to-digest. When both sides carry a
k-ladder, a mismatch also yields an approximate divergence magnitude: agreement at k mantissa
bits bounds the relative error near 2**-k (a max-error bound -- one bad element breaks a rung).

Alignment note: the two sides only align index-for-index when they saw the same forwards in the
same order for the compared case pair, so filter to a case pair whose multiplicity matches
(e.g. one trainer scoring pass vs one engine prefill). ``--layers N`` folds the per-region call
ordinal into a layer index ((call-1) % N) for records that carry no layer of their own.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple


def load_dir(path: str) -> List[dict]:
    recs: List[dict] = []
    files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
    if not files:
        raise SystemExit(f"no *.jsonl trace files in {path!r}")
    for fp in files:
        with open(fp) as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[compare] skipping bad line {fp}:{ln}", file=sys.stderr)
                    continue
                r["_file"] = os.path.basename(fp)
                recs.append(r)
    return recs


def _layer_of(r: dict, layers: Optional[int]) -> Optional[int]:
    if r.get("layer") is not None:
        return r["layer"]
    if layers:
        return (r.get("call", 1) - 1) % layers
    return None


def _group(recs: List[dict], case: Optional[str], layers: Optional[int], regions) -> Dict[tuple, List[dict]]:
    out: Dict[tuple, List[dict]] = {}
    for r in recs:
        if case and r.get("case") != case:
            continue
        if regions and r.get("region") not in regions:
            continue
        key = (r["region"], _layer_of(r, layers), r.get("out", 0))
        out.setdefault(key, []).append(r)
    for v in out.values():
        v.sort(key=lambda r: (r.get("step") or 0, r.get("call", 0)))
    return out


def _ladder_verdict(a: dict, b: dict) -> Tuple[Optional[int], str]:
    """(finest agreeing mantissa rung or None, human-readable magnitude)."""
    la, lb = a.get("ladder") or {}, b.get("ladder") or {}
    ks = sorted(
        (int(k[1:]) for k in set(la) & set(lb) if k.startswith("k")),
        reverse=True,
    )
    if not ks:
        return None, "unknown (no shared k-ladder; rerun with SKYRL_ISOEXEC_DEBUG_LADDER=1)"
    agree = None
    for k in ks:  # finest first
        if la[f"k{k}"] == lb[f"k{k}"]:
            agree = k
            break
    if agree is None:
        return None, "exponent-level (>~1 relative; diverges even at k=0)"
    return agree, f"~{2.0 ** -agree:.0e} relative (agrees at k<={agree} mantissa bits)"


def compare(
    recs_a: List[dict],
    recs_b: List[dict],
    *,
    case_a: Optional[str] = None,
    case_b: Optional[str] = None,
    layers: Optional[int] = None,
    regions=None,
) -> dict:
    ga = _group(recs_a, case_a, layers, regions)
    gb = _group(recs_b, case_b, layers, regions)
    per_region: Dict[str, dict] = {}
    only_a = sum(len(v) for k, v in ga.items() if k not in gb)
    only_b = sum(len(v) for k, v in gb.items() if k not in ga)
    for key in sorted(set(ga) & set(gb), key=lambda k: (k[0], k[1] if k[1] is not None else -1, k[2])):
        region, layer, out_idx = key
        va, vb = ga[key], gb[key]
        st = per_region.setdefault(
            region, {"compared": 0, "matched": 0, "mismatched": 0, "unaligned": 0, "first_mismatch": None}
        )
        st["unaligned"] += abs(len(va) - len(vb))
        for i in range(min(len(va), len(vb))):
            ra, rb = va[i], vb[i]
            st["compared"] += 1
            if ra.get("shape") != rb.get("shape") or ra.get("dtype") != rb.get("dtype"):
                st["mismatched"] += 1
                mm = {
                    "kind": "shape/dtype",
                    "layer": layer,
                    "out": out_idx,
                    "index": i,
                    "a": {"shape": ra.get("shape"), "dtype": ra.get("dtype"), "case": ra.get("case")},
                    "b": {"shape": rb.get("shape"), "dtype": rb.get("dtype"), "case": rb.get("case")},
                    "magnitude": "n/a (different shape/dtype)",
                }
            elif ra["digest"] == rb["digest"]:
                st["matched"] += 1
                continue
            else:
                st["mismatched"] += 1
                agree_k, mag = _ladder_verdict(ra, rb)
                mm = {
                    "kind": "value",
                    "layer": layer,
                    "out": out_idx,
                    "index": i,
                    "step": ra.get("step"),
                    "call_a": ra.get("call"),
                    "call_b": rb.get("call"),
                    "case_a": ra.get("case"),
                    "case_b": rb.get("case"),
                    "agree_k": agree_k,
                    "magnitude": mag,
                }
            first = st["first_mismatch"]
            if first is None or _order_key(mm) < _order_key(first):
                st["first_mismatch"] = mm
    firsts = [(r, s["first_mismatch"]) for r, s in per_region.items() if s["first_mismatch"]]
    first_div = None
    if firsts:
        region, mm = min(firsts, key=lambda rm: _order_key(rm[1]))
        first_div = dict(mm, region=region)
    return {
        "regions": per_region,
        "first_divergence": first_div,
        "keys_only_in_a": only_a,
        "keys_only_in_b": only_b,
    }


def _order_key(mm: dict) -> tuple:
    return (
        mm.get("step") or 0,
        mm.get("index", 0),
        mm["layer"] if mm.get("layer") is not None else 1 << 30,
        mm.get("out", 0),
    )


def render_text(rep: dict) -> str:
    lines = ["isoexec debug-trace comparison", "=" * 34, ""]
    if not rep["regions"]:
        lines.append("no overlapping (region, layer, out) keys between the two traces.")
        lines.append("check --case-a/--case-b filters and that both sides actually traced.")
    for region in sorted(rep["regions"]):
        s = rep["regions"][region]
        lines.append(
            f"{region:20s} compared={s['compared']:6d} matched={s['matched']:6d} "
            f"mismatched={s['mismatched']:6d} unaligned={s['unaligned']:5d}"
        )
        if s["first_mismatch"]:
            mm = s["first_mismatch"]
            lines.append(
                f"{'':20s} first mismatch: layer={mm.get('layer')} out={mm.get('out')} "
                f"step={mm.get('step')} pair#{mm.get('index')} -> {mm['magnitude']}"
            )
    lines.append("")
    fd = rep["first_divergence"]
    if fd is None:
        lines.append("NO DIVERGENCE: every aligned pair matched bitwise.")
    else:
        lines.append(
            f"FIRST DIVERGENCE: region={fd['region']} layer={fd.get('layer')} out={fd.get('out')} "
            f"step={fd.get('step')} case=({fd.get('case_a')} vs {fd.get('case_b')})"
        )
        lines.append(f"  magnitude: {fd['magnitude']}")
    if rep["keys_only_in_a"] or rep["keys_only_in_b"]:
        lines.append(
            f"note: {rep['keys_only_in_a']} record(s) under keys only in A, "
            f"{rep['keys_only_in_b']} only in B (not compared)"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dir_a", help="first trace directory (e.g. trainer)")
    p.add_argument("dir_b", help="second trace directory (e.g. engine)")
    p.add_argument("--case-a", default=None, help="only compare A-records of this case")
    p.add_argument("--case-b", default=None, help="only compare B-records of this case")
    p.add_argument("--layers", type=int, default=None, help="layers per forward, to fold call -> layer")
    p.add_argument("--regions", default=None, help="comma-separated region filter")
    p.add_argument("--json", dest="json_out", default=None, help="also write the report as JSON here")
    args = p.parse_args(argv)
    regions = set(args.regions.split(",")) if args.regions else None
    rep = compare(
        load_dir(args.dir_a),
        load_dir(args.dir_b),
        case_a=args.case_a,
        case_b=args.case_b,
        layers=args.layers,
        regions=regions,
    )
    print(render_text(rep))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rep, f, indent=2)
    return 0 if rep["first_divergence"] is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
