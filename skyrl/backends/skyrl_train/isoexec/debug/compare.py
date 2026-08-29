"""Offline comparator for debug-mode traces: where and how do the two runtimes diverge.

Supported invocations (all stdlib-only -- this module imports nothing outside the standard
library so it runs on a laptop with no torch, no CUDA and no TransformerEngine):

    python <repo>/skyrl/backends/skyrl_train/isoexec/debug/compare.py TRACE_A TRACE_B [opts]
    PYTHONPATH=<repo>/skyrl/backends/skyrl_train/isoexec python -m debug.compare TRACE_A TRACE_B
    PYTHONPATH=<repo>/skyrl/backends/skyrl_train/isoexec python -m debug TRACE_A TRACE_B

The fully qualified ``python -m skyrl.backends.skyrl_train.isoexec.debug.compare`` form does NOT
work offline and is not supported: ``-m`` imports every parent package first, and the ``isoexec``
package ``__init__`` eagerly installs runtime guards (``install_no_te_guard`` -> ``import
transformer_engine``), which needs the CUDA/TE stack. The forms above skip that package entirely.

    options: [--case-a trainer_score] [--case-b engine_prefill] [--layers N]
             [--regions moe.router,gdn.core] [--json out.json]

A and B are trace directories (conventionally trainer and engine); every ``*.jsonl`` inside is
read, along with the per-process ``manifest-*.json`` sidecars that say what each side sampled.
Records must carry format version :data:`FORMAT_VERSION`; older traces are refused rather than
mis-read.

What the comparison guarantees:

  * **Rank-aware.** Records are grouped by (region, layer, out-path, RANK), so many processes
    writing one directory align per rank instead of by pid sort order. A rank-set mismatch
    between the two sides is reported as one structural divergence, not as a flood of element
    mismatches.
  * **Key-aligned, not position-aligned.** Within a group the two sides are matched on ``step``
    (else ``call``, else position). A record missing mid-stream therefore shows up as exactly
    one ``absent`` divergence instead of shifting every later pair into a fabricated value
    mismatch. A region or step present on one side and missing on the other IS a divergence and
    does set the exit code.
  * **Causally ordered.** Divergences are ordered by the recorded execution timestamp, so the
    reported FIRST DIVERGENCE is the one that happened first, not the alphabetically first
    contaminated region. Every later divergence on the same rank is marked
    "contaminated (after first divergence)".
  * **Honest magnitude.** A k-ladder mismatch yields a bracket ("between ~2^-10 and 2^-6"): the
    tightest rung that still matches against the first that differs, as a MAX over elements. Only
    the upper end is a bound -- truncation is a step function, so a differing rung can mean a
    straddled boundary rather than a large error. When every rung differs the verdict says the
    ladder does not bound it; it never converts saturation into a claim of large relative error.
  * **Honest coverage.** Sampled-out steps are reported as "not observed", never as clean, and a
    comparison whose two sides sampled disjoint record sets is reported as inconclusive.

Known limitation (recorded, not fixed): digests are position-keyed, so a permutation of the same
values -- a batch-order or token-ordering bug -- is detected but is indistinguishable from a
value change. Both break every ladder rung. ``SKYRL_ISOEXEC_DEBUG_SEGMENTS`` narrows it: a
localized fault differs in a few segments, a permutation or a whole-tensor round-off difference
differs in all of them.

Exit codes: 0 clean, 2 divergence found, 3 comparison inconclusive (disjoint sampling), 1 bad
input (missing or wrong-version traces).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

# Must match trace.FORMAT_VERSION. Duplicated rather than imported: trace.py pulls in torch and
# this module is the one that has to run without it.
FORMAT_VERSION = 3

MAX_LOCATIONS = 10  # per list, in the rendered text; --json carries up to --json-max-per-region


# -- loading -------------------------------------------------------------------------------


def load_dir(path: str) -> List[dict]:
    recs: List[dict] = []
    files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
    if not files:
        raise SystemExit(f"no *.jsonl trace files in {path!r}")
    stale = set()
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
                if r.get("v") != FORMAT_VERSION:
                    stale.add(r.get("v"))
                    continue
                r["_file"] = os.path.basename(fp)
                recs.append(r)
    if stale and not recs:
        raise SystemExit(
            f"{path!r}: trace record format {sorted(stale, key=str)} but this comparator requires "
            f"v{FORMAT_VERSION} (records now carry rank, string out-paths and manifests). "
            "Re-run the trace with the current build."
        )
    if stale:
        print(
            f"[compare] {path}: ignored records of format {sorted(stale, key=str)} (need v{FORMAT_VERSION})",
            file=sys.stderr,
        )
    return recs


def load_manifests(path: str) -> List[dict]:
    """Per-process sidecars written by ``trace.Tracer``. Absent for hand-written traces."""
    out = []
    for fp in sorted(glob.glob(os.path.join(path, "manifest-*.json"))):
        try:
            with open(fp) as f:
                out.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            print(f"[compare] unreadable manifest {fp}", file=sys.stderr)
    return out


# -- grouping and alignment ----------------------------------------------------------------


def _layer_of(r: dict, layers: Optional[int]) -> Optional[int]:
    if r.get("layer") is not None:
        return r["layer"]
    if layers:
        return (r.get("call", 1) - 1) % layers
    return None


def _group(recs: List[dict], case: Optional[str], layers: Optional[int], regions) -> Dict[tuple, List[dict]]:
    """(region, layer, out-path, rank) -> records, in recorded execution order."""
    out: Dict[tuple, List[dict]] = {}
    for r in recs:
        if case and r.get("case") != case:
            continue
        if regions and r.get("region") not in regions:
            continue
        key = (r["region"], _layer_of(r, layers), str(r.get("out", "0")), r.get("rank"))
        out.setdefault(key, []).append(r)
    for v in out.values():
        v.sort(key=lambda r: (r.get("ts") or 0.0, r.get("seq") or 0, r.get("step") or 0, r.get("call", 0)))
    return out


def _align(va: List[dict], vb: List[dict]) -> Tuple[List[Tuple[Optional[dict], Optional[dict]]], str]:
    """Pair up two groups on a shared record key, so a missing record leaves a hole.

    Aligning on position instead would shift every later pair and manufacture value divergences
    out of records whose bits were never compared. ``step`` is preferred (it survives a call
    counter that never fired), ``call`` is the fallback, position is the last resort.
    """
    for field in ("step", "call"):
        ka = [r.get(field) for r in va]
        kb = [r.get(field) for r in vb]
        if all(k is not None for k in ka + kb) and len(set(ka)) == len(ka) and len(set(kb)) == len(kb):
            da, db = dict(zip(ka, va)), dict(zip(kb, vb))
            return [(da.get(k), db.get(k)) for k in sorted(set(ka) | set(kb))], field
    n = max(len(va), len(vb))
    return [(va[i] if i < len(va) else None, vb[i] if i < len(vb) else None) for i in range(n)], "position"


# -- magnitude -----------------------------------------------------------------------------


def _ladder_verdict(a: dict, b: dict) -> Tuple[Optional[int], str]:
    """(finest agreeing mantissa rung or None, honest magnitude bracket).

    The bracket is (first rung that differs, finest rung that still matches) mapped to relative
    error, and it is a MAX over elements. When nothing matches, say so -- do not translate a
    saturated ladder into a magnitude.
    """
    la, lb = a.get("ladder") or {}, b.get("ladder") or {}
    ks = sorted((int(k[1:]) for k in set(la) & set(lb) if k.startswith("k")), reverse=True)
    if not ks:
        return None, "unknown (no shared k-ladder; rerun both sides with SKYRL_ISOEXEC_DEBUG_LADDER=1)"
    agree = next((k for k in ks if la[f"k{k}"] == lb[f"k{k}"]), None)
    if agree is None:
        return None, (
            "not bounded by the ladder (differs at every rung down to k=0) -- either a "
            "sign/exponent-level difference, or a small difference spread over enough elements "
            "that some straddle every truncation boundary; this is NOT a claim of large error"
        )
    finer = min((k for k in ks if k > agree), default=None)
    if finer is None:
        return agree, (
            f"< 2^-{agree} (~{2.0 ** -agree:.0e}) relative, max over elements "
            f"(matches at k={agree}, the finest rung recorded for this dtype)"
        )
    return agree, (
        f"between ~2^-{finer} and 2^-{agree} (~{2.0 ** -finer:.0e} .. ~{2.0 ** -agree:.0e}) relative, "
        "max over elements -- the upper end is a bound, the lower end holds only if no element "
        f"straddles the k={finer} truncation boundary"
    )


def _short_ladder(agree_k: Optional[int], magnitude: str) -> str:
    """One-column form of the ladder verdict, for the per-region table."""
    if magnitude.startswith("unknown"):
        return "magnitude unknown (no k-ladder)"
    if agree_k is None:
        return "not bounded by the ladder"
    return f"<= 2^-{agree_k} relative"


def _segment_note(ra: dict, rb: dict) -> Optional[str]:
    """Row localization from ``SKYRL_ISOEXEC_DEBUG_SEGMENTS``, when both sides recorded it."""
    sa, sb = ra.get("segments"), rb.get("segments")
    if not sa or not sb or len(sa) != len(sb):
        return None
    diff = [i for i, (x, y) in enumerate(zip(sa, sb)) if x != y]
    if not diff:
        return f"all {len(sa)} row segments match (the difference is below segment resolution)"
    if len(diff) == len(sa):
        return (
            f"all {len(sa)} row segments differ -> a whole-tensor difference (round-off, "
            "reduction order, or a permutation), not a fault localized to some rows"
        )
    rows = ra.get("seg_rows") or 0
    span = f", rows {diff[0] * rows}..{(diff[0] + 1) * rows - 1}" if rows else ""
    axa, axb = ra.get("seg_axis"), rb.get("seg_axis")
    # The two sides segment their own first non-unit dim, so [1,T,H,D] and [T,1,C] both slice T.
    ax = f" of dim {axa}" if axa == axb and axa is not None else ""
    return f"{len(diff)} of {len(sa)} row segments{ax} differ (first: segment {diff[0]}{span})"


# -- ordering ------------------------------------------------------------------------------


def _order_key(mm: dict) -> tuple:
    """Causal order: recorded execution time first, then the per-process sequence number, which
    is exact where the wall clock's resolution is not. Records without a ts (hand-written traces)
    sort after, on the old step/index key."""
    ts = mm.get("ts")
    step = mm.get("step")
    return (
        0 if ts is not None else 1,
        ts if ts is not None else 0.0,
        mm.get("seq") or 0,
        step if isinstance(step, int) else 0,
        mm.get("index", 0),
        mm["layer"] if mm.get("layer") is not None else 1 << 30,
        str(mm.get("out", "")),
        str(mm.get("region", "")),
    )


# -- comparison ----------------------------------------------------------------------------


def _side_label(recs: List[dict], man: List[dict]) -> str:
    labels = {m.get("side") for m in man} or {r.get("side") for r in recs}
    labels.discard(None)
    return "/".join(sorted(labels)) if labels else "?"


def _sampling(recs: List[dict], man: List[dict]) -> dict:
    """What this side actually covered: sample rate, whether set_step drove it, steps seen vs
    recorded. Without manifests, fall back to what the records themselves show."""
    seen, recorded = set(), {r.get("step") for r in recs if r.get("step") is not None}
    sample = 1
    step_signal = any(r.get("step") is not None for r in recs)
    for m in man:
        sample = max(sample, int(m.get("sample", 1)))
        step_signal = step_signal or bool(m.get("step_signal"))
        seen.update(m.get("steps_seen") or [])
        recorded.update(m.get("steps_recorded") or [])
    return {
        "sample": sample,
        "step_signal": step_signal,
        "steps_seen": sorted(seen),
        "steps_recorded": sorted(recorded),
        "steps_not_observed": sorted(seen - recorded),
        "ladder": any(m.get("ladder") for m in man),
        "segment_rows": max((int(m.get("segment_rows") or 0) for m in man), default=0),
        "layer_src": sorted({r.get("layer_src") for r in recs if r.get("layer_src")}),
        "capture_skipped": sum(int(m.get("capture_skipped") or 0) for m in man),
        "records": len(recs),
    }


def _layer_src_mismatch(sa: dict, sb: dict) -> Optional[str]:
    """The two sides derived the layer key differently, so its values are not comparable.

    ``layer`` is part of the grouping key, so a side reading real module indices against a side
    counting call order produces key-level ``absent`` divergences on a run that may be perfectly
    clean. Naming the cause is the whole point of this comparator.
    """
    la, lb = set(sa.get("layer_src") or []), set(sb.get("layer_src") or [])
    if not la or not lb or la == lb:
        return None
    return (
        f"side A derived the layer index from {sorted(la)} and side B from {sorted(lb)}. Those "
        "index spaces do not agree (a model with mixed layer types gives sparse module indices "
        "and dense call ordinals), so per-layer keys will not align. Pass the model to "
        "install_debug_hooks(model) on BOTH sides so both read layer_src='module'."
    )


def _disjoint_sampling(sa: dict, sb: dict) -> Optional[str]:
    if sa["sample"] <= 1 and sb["sample"] <= 1:
        return None
    if sa["step_signal"] != sb["step_signal"]:
        keyed, ordinal = ("A", "B") if sa["step_signal"] else ("B", "A")
        return (
            f"side {keyed} samples every Nth step (set_step is wired) while side {ordinal} samples "
            "every Nth region call (set_step is not wired) -- the two sides select disjoint records"
        )
    if sa["steps_recorded"] and sb["steps_recorded"] and not (set(sa["steps_recorded"]) & set(sb["steps_recorded"])):
        return "the two sides recorded disjoint step sets"
    return None


def _loc(r: dict, key: tuple) -> dict:
    region, layer, out_idx, rank = key
    return {
        "region": region,
        "rank": rank,
        "layer": layer,
        "out": out_idx,
        "step": (r or {}).get("step"),
        "call": (r or {}).get("call"),
        "case": (r or {}).get("case"),
    }


def _new_stat() -> dict:
    return {
        "compared": 0,
        "matched": 0,
        "mismatched": 0,
        "absent": 0,
        "unrecordable": 0,
        "unaligned": 0,
        "first_mismatch": None,
    }


def compare(
    recs_a: List[dict],
    recs_b: List[dict],
    *,
    case_a: Optional[str] = None,
    case_b: Optional[str] = None,
    layers: Optional[int] = None,
    regions=None,
    man_a: Optional[List[dict]] = None,
    man_b: Optional[List[dict]] = None,
) -> dict:
    man_a, man_b = man_a or [], man_b or []
    ga = _group(recs_a, case_a, layers, regions)
    gb = _group(recs_b, case_b, layers, regions)
    label_a, label_b = _side_label(recs_a, man_a), _side_label(recs_b, man_b)
    samp_a, samp_b = _sampling(recs_a, man_a), _sampling(recs_b, man_b)

    ranks_a = sorted({r.get("rank") for r in recs_a}, key=str)
    ranks_b = sorted({r.get("rank") for r in recs_b}, key=str)
    rank_only_a, rank_only_b = set(ranks_a) - set(ranks_b), set(ranks_b) - set(ranks_a)
    rank_div = None
    if rank_only_a or rank_only_b:
        rank_div = {
            "kind": "rank_mismatch",
            "region": "(all)",
            "ranks_a": ranks_a,
            "ranks_b": ranks_b,
            "only_in_a": sorted(rank_only_a, key=str),
            "only_in_b": sorted(rank_only_b, key=str),
            "rank_src_a": sorted({r.get("rank_src") for r in recs_a} - {None}),
            "rank_src_b": sorted({r.get("rank_src") for r in recs_b} - {None}),
            "magnitude": "n/a (structural: the two sides traced different rank sets)",
            "magnitude_short": "different rank sets",
        }

    per_region: Dict[str, dict] = {}
    divs: List[dict] = []
    unrecordable: List[dict] = []
    locations: Dict[str, List[dict]] = {"only_in_a": [], "only_in_b": [], "unaligned": []}
    only_a = only_b = 0

    for key in sorted(set(ga) | set(gb), key=lambda k: (k[0], k[1] if k[1] is not None else -1, k[2], str(k[3]))):
        region, layer, out_idx, rank = key
        st = per_region.setdefault(region, _new_stat())
        va, vb = ga.get(key), gb.get(key)
        if va is None or vb is None:
            present, absent_in = (va, "B") if vb is None else (vb, "A")
            n = len(present)
            where = dict(_loc(present[0], key), step=None, call=None, records=n)
            if vb is None:
                only_a += n
                locations["only_in_a"].append(where)
            else:
                only_b += n
                locations["only_in_b"].append(where)
            # A rank that exists on one side only is one structural fact, already reported as
            # rank_mismatch; do not restate it once per key.
            if rank in rank_only_a or rank in rank_only_b:
                continue
            st["absent"] += n
            divs.append(
                dict(
                    _loc(present[0], key),
                    kind="absent",
                    scope="key",
                    records=n,
                    step=None,  # the whole key is missing, not one step of it
                    index=0,
                    ts=present[0].get("ts"),
                    seq=present[0].get("seq"),
                    absent_in=absent_in,
                    present_in="A" if absent_in == "B" else "B",
                    side_absent=label_b if absent_in == "B" else label_a,
                    side_present=label_a if absent_in == "B" else label_b,
                    magnitude=f"n/a (all {n} record(s) of this {region} key absent on side {absent_in})",
                    magnitude_short=f"whole key absent on side {absent_in}",
                )
            )
            continue

        pairs, how = _align(va, vb)
        for i, (ra, rb) in enumerate(pairs):
            if ra is None or rb is None:
                present = ra or rb
                absent_in = "B" if rb is None else "A"
                st["absent"] += 1
                st["unaligned"] += 1
                loc = dict(_loc(present, key), aligned_by=how)
                locations["unaligned"].append(loc)
                divs.append(
                    dict(
                        loc,
                        kind="absent",
                        scope="record",
                        index=i,
                        ts=present.get("ts"),
                        seq=present.get("seq"),
                        absent_in=absent_in,
                        present_in="A" if absent_in == "B" else "B",
                        side_absent=label_b if absent_in == "B" else label_a,
                        side_present=label_a if absent_in == "B" else label_b,
                        magnitude=f"n/a (record absent on side {absent_in})",
                        magnitude_short=f"absent on side {absent_in}",
                    )
                )
                continue

            base = dict(
                region=region,
                rank=rank,
                layer=layer,
                out=out_idx,
                index=i,
                aligned_by=how,
                step=ra.get("step") if ra.get("step") is not None else rb.get("step"),
                ts=ra.get("ts"),
                seq=ra.get("seq"),
                call_a=ra.get("call"),
                call_b=rb.get("call"),
                case_a=ra.get("case"),
                case_b=rb.get("case"),
            )
            ua, ub = ra.get("unrecordable"), rb.get("unrecordable")
            if ua or ub:
                st["unrecordable"] += 1
                unrecordable.append(dict(base, reason_a=ua, reason_b=ub))
                if ua and ub:
                    continue  # both sides equally undigestable: not evidence of divergence
                st["mismatched"] += 1
                mm = dict(
                    base,
                    kind="unrecordable",
                    reason_a=ua,
                    reason_b=ub,
                    magnitude=f"n/a (side {'A' if ua else 'B'} could not digest this output: {ua or ub})",
                    magnitude_short=f"unrecordable on side {'A' if ua else 'B'}",
                )
            else:
                st["compared"] += 1
                if ra.get("shape") != rb.get("shape") or ra.get("dtype") != rb.get("dtype"):
                    st["mismatched"] += 1
                    mm = dict(
                        base,
                        kind="shape/dtype",
                        a={"shape": ra.get("shape"), "dtype": ra.get("dtype")},
                        b={"shape": rb.get("shape"), "dtype": rb.get("dtype")},
                        magnitude=(
                            f"n/a (different shape/dtype: {ra.get('shape')} {ra.get('dtype')} vs "
                            f"{rb.get('shape')} {rb.get('dtype')})"
                        ),
                        magnitude_short=(f"{ra.get('shape')} {ra.get('dtype')} vs {rb.get('shape')} {rb.get('dtype')}"),
                    )
                elif ra.get("digest") == rb.get("digest"):
                    st["matched"] += 1
                    continue
                else:
                    st["mismatched"] += 1
                    agree_k, mag = _ladder_verdict(ra, rb)
                    mm = dict(
                        base,
                        kind="value",
                        agree_k=agree_k,
                        magnitude=mag,
                        magnitude_short=_short_ladder(agree_k, mag),
                        segments=_segment_note(ra, rb),
                    )
            divs.append(mm)

    # Causal ordering: earliest execution timestamp wins, per rank and overall.
    divs.sort(key=_order_key)
    origins: Dict[object, dict] = {}
    for mm in divs:
        origins.setdefault(mm.get("rank"), mm)
    for mm in divs:
        mm["contaminated"] = mm is not origins.get(mm.get("rank"))
    for mm in divs:
        st = per_region[mm["region"]] if mm["region"] in per_region else None
        if st is not None and (st["first_mismatch"] is None or _order_key(mm) < _order_key(st["first_mismatch"])):
            st["first_mismatch"] = mm

    disjoint = _disjoint_sampling(samp_a, samp_b)
    layer_src = _layer_src_mismatch(samp_a, samp_b)
    first_div = rank_div if rank_div is not None else (divs[0] if divs else None)
    if disjoint:
        status = "inconclusive"
    elif first_div is not None:
        status = "divergent"
    else:
        status = "clean"

    return {
        "format_version": FORMAT_VERSION,
        "status": status,
        "sides": {
            "a": dict(samp_a, label=label_a, ranks=ranks_a),
            "b": dict(samp_b, label=label_b, ranks=ranks_b),
        },
        "rank_mismatch": rank_div,
        "disjoint_sampling": disjoint,
        "layer_src_mismatch": layer_src,
        "regions": per_region,
        "first_divergence": first_div,
        "origins": [origins[r] for r in sorted(origins, key=str)],
        "divergences": divs,
        "unrecordable": unrecordable,
        "keys_only_in_a": only_a,
        "keys_only_in_b": only_b,
        "locations": locations,
    }


# -- rendering -----------------------------------------------------------------------------


def _fmt_loc(d: dict) -> str:
    bits = f"{d.get('region')} rank={d.get('rank')} layer={d.get('layer')} step={d.get('step')} out={d.get('out')}"
    if d.get("records"):
        bits += f" ({d['records']} record(s))"
    return bits


def _capped(lines: List[str], items: List[dict], indent: str = "    ") -> None:
    for d in items[:MAX_LOCATIONS]:
        lines.append(f"{indent}{_fmt_loc(d)}")
    if len(items) > MAX_LOCATIONS:
        lines.append(f"{indent}... and {len(items) - MAX_LOCATIONS} more (full list in --json)")


def _side_line(tag: str, s: dict) -> str:
    cov = f"sample=1/{s['sample']} ({'by step' if s['step_signal'] else 'by call ordinal'})"
    extra = f" ladder={'on' if s['ladder'] else 'off'} segments={s['segment_rows'] or 'off'}"
    return f"trace {tag}: side={s['label']} ranks={s['ranks']} records={s['records']} {cov}{extra}"


def render_text(rep: dict) -> str:
    lines = ["isoexec debug-trace comparison", "=" * 34, ""]
    lines.append(_side_line("A", rep["sides"]["a"]))
    lines.append(_side_line("B", rep["sides"]["b"]))
    lines.append("")

    if not rep["regions"]:
        lines.append("no (region, layer, out, rank) keys in either trace after filtering.")
        lines.append("check --case-a/--case-b filters and that both sides actually traced.")
    for region in sorted(rep["regions"]):
        s = rep["regions"][region]
        lines.append(
            f"{region:28s} compared={s['compared']:6d} matched={s['matched']:6d} "
            f"mismatched={s['mismatched']:6d} absent={s['absent']:5d} unaligned={s['unaligned']:5d}"
        )
        mm = s["first_mismatch"]
        if mm:
            tag = "  [contaminated: after the first divergence]" if mm.get("contaminated") else "  [ORIGIN]"
            lines.append(
                f"{'':28s} first {mm['kind']}: rank={mm.get('rank')} layer={mm.get('layer')} "
                f"step={mm.get('step')} out={mm.get('out')} pair#{mm.get('index')} "
                f"-> {mm.get('magnitude_short', mm['magnitude'])}{tag}"
            )
    lines.append("")

    fd = rep["first_divergence"]
    if rep.get("layer_src_mismatch"):
        lines.append("WARNING: the two sides key `layer` differently -- absences below may be an artifact.")
        lines.append(f"  {rep['layer_src_mismatch']}")
    for side in ("a", "b"):
        skipped = rep["sides"][side].get("capture_skipped") or 0
        if skipped:
            lines.append(
                f"WARNING: side {side.upper()} skipped {skipped} record(s) reached under CUDA-graph "
                "capture; that part of the forward is unobserved (run the engine eager)."
            )
    if rep["disjoint_sampling"]:
        lines.append("COMPARISON INCONCLUSIVE: side-disjoint sampling.")
        lines.append(f"  {rep['disjoint_sampling']}")
        lines.append("  The two sides did not observe the same forwards, so nothing below is")
        lines.append("  evidence of agreement or of divergence. Wire set_step on both sides, or")
        lines.append("  re-run with SKYRL_ISOEXEC_DEBUG_SAMPLE=1.")
    elif fd is None:
        lines.append("NO DIVERGENCE: every compared pair matched bitwise.")
    elif fd["kind"] == "rank_mismatch":
        lines.append("STRUCTURAL DIVERGENCE: the two sides traced different rank sets.")
        lines.append(f"  A ranks={fd['ranks_a']} (from {fd['rank_src_a']})")
        lines.append(f"  B ranks={fd['ranks_b']} (from {fd['rank_src_b']})")
        lines.append(f"  only in A: {fd['only_in_a']}   only in B: {fd['only_in_b']}")
        lines.append("  Records from the one-sided ranks were not compared. Fix the topology or")
        lines.append("  filter to the shared ranks before reading anything below as numerics.")
    else:
        case = (
            f"case={fd.get('case')}" if fd["kind"] == "absent" else f"case=({fd.get('case_a')} vs {fd.get('case_b')})"
        )
        lines.append(
            f"FIRST DIVERGENCE: region={fd['region']} rank={fd.get('rank')} layer={fd.get('layer')} "
            f"out={fd.get('out')} step={fd.get('step')} {case}"
        )
        lines.append(f"  kind: {fd['kind']}   (aligned by {fd.get('aligned_by', 'n/a')})")
        lines.append(f"  magnitude: {fd['magnitude']}")
        if fd.get("segments"):
            lines.append(f"  segments: {fd['segments']}")
        if fd["kind"] == "absent":
            lines.append(
                f"  present on side {fd['present_in']} ({fd['side_present']}), "
                f"absent on side {fd['absent_in']} ({fd['side_absent']})"
            )
        lines.append("  ordered by recorded execution time (ts), so this is the causally first one.")

    later = [d for d in rep["divergences"] if d.get("contaminated")]
    if later:
        regions = sorted({d["region"] for d in later})
        lines.append(
            f"  {len(later)} later divergence(s) marked contaminated (after first divergence), "
            f"in: {', '.join(regions)}"
        )
    if len(rep["origins"]) > 1:
        lines.append(f"  independent first divergences on {len(rep['origins'])} ranks:")
        _capped(lines, rep["origins"], indent="    ")

    notes = _notes(rep)
    if notes:
        lines.append("")
        lines += notes
    return "\n".join(lines)


def _notes(rep: dict) -> List[str]:
    lines: List[str] = []
    loc = rep["locations"]
    if rep["keys_only_in_a"] or rep["keys_only_in_b"]:
        lines.append(
            f"note: {rep['keys_only_in_a']} record(s) under keys only in A, "
            f"{rep['keys_only_in_b']} only in B -- reported as 'absent' divergences above:"
        )
        _capped(lines, loc["only_in_a"] + loc["only_in_b"])
    if loc["unaligned"]:
        lines.append(f"note: {len(loc['unaligned'])} record(s) with no counterpart on the other side:")
        _capped(lines, loc["unaligned"])
    if any(d.get("side_absent") == "engine" for d in rep["divergences"] if d["kind"] == "absent"):
        lines.append(
            "hint: engine-side records are missing. A replayed CUDA graph executes no Python, so "
            "graph-captured decode steps produce no records -- re-run the engine eager "
            "(enforce_eager) before reading this as a real absence."
        )
    if rep["unrecordable"]:
        lines.append(f"note: {len(rep['unrecordable'])} output(s) could not be digested:")
        for d in rep["unrecordable"][:MAX_LOCATIONS]:
            lines.append(f"    {_fmt_loc(d)} -> A: {d.get('reason_a') or 'ok'} | B: {d.get('reason_b') or 'ok'}")
        if len(rep["unrecordable"]) > MAX_LOCATIONS:
            lines.append(f"    ... and {len(rep['unrecordable']) - MAX_LOCATIONS} more (full list in --json)")
    for tag in ("a", "b"):
        miss = rep["sides"][tag]["steps_not_observed"]
        if miss:
            lines.append(
                f"note: side {tag.upper()} step(s) {miss} not observed (sampled out at "
                f"1/{rep['sides'][tag]['sample']}) -- a fault on those steps is invisible here, "
                "not absent"
            )
    if any(d["kind"] == "value" and d.get("agree_k") is None for d in rep["divergences"]):
        lines.append(
            "note: digests are position-keyed, so a permutation of the same values (a batch-order "
            "or token-ordering bug) breaks every ladder rung exactly as a value change does and is "
            "reported identically. SKYRL_ISOEXEC_DEBUG_SEGMENTS=<rows> narrows it."
        )
    return lines


DEFAULT_JSON_MAX_PER_REGION = 200


def cap_report(rep: dict, max_per_region: int = DEFAULT_JSON_MAX_PER_REGION) -> dict:
    """Report with its per-record lists capped for serialization; counts stay exact.

    A fully divergent 110k-record run serializes every mismatch: 66MB of JSON, most of it the
    same fact repeated. Cap the divergence list at the first ``max_per_region`` entries PER
    REGION (they are already sorted causally, so those are the earliest, which is what triage
    reads) and cap the flat location/unrecordable lists at the same number overall. ``regions``
    (the counts), ``first_divergence`` and ``origins`` are never trimmed -- they are the verdict.
    ``truncated`` says exactly what was dropped, so a reader is never silently short.
    """
    if not max_per_region or max_per_region <= 0:
        return rep
    out = dict(rep)
    kept: List[dict] = []
    seen: Dict[str, int] = {}
    dropped: Dict[str, int] = {}
    for d in rep.get("divergences", []):
        region = d.get("region")
        n = seen.get(region, 0)
        if n < max_per_region:
            seen[region] = n + 1
            kept.append(d)
        else:
            dropped[region] = dropped.get(region, 0) + 1
    out["divergences"] = kept
    trunc = {
        "max_per_region": max_per_region,
        "divergences_dropped": dropped,
        "divergences_total": len(rep.get("divergences", [])),
    }
    for key in ("unrecordable",):
        items = rep.get(key) or []
        if len(items) > max_per_region:
            out[key] = items[:max_per_region]
            trunc[f"{key}_dropped"] = len(items) - max_per_region
    locs = rep.get("locations") or {}
    new_locs, loc_dropped = {}, {}
    for name, items in locs.items():
        if len(items) > max_per_region:
            new_locs[name] = items[:max_per_region]
            loc_dropped[name] = len(items) - max_per_region
        else:
            new_locs[name] = items
    out["locations"] = new_locs
    if loc_dropped:
        trunc["locations_dropped"] = loc_dropped
    any_dropped = any(k.endswith("_dropped") and trunc[k] for k in list(trunc))
    out["truncated"] = trunc if any_dropped else None
    return out


# -- CLI -----------------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dir_a", help="first trace directory (e.g. trainer)")
    p.add_argument("dir_b", help="second trace directory (e.g. engine)")
    p.add_argument("--case-a", default=None, help="only compare A-records of this case")
    p.add_argument("--case-b", default=None, help="only compare B-records of this case")
    p.add_argument("--layers", type=int, default=None, help="layers per forward, to fold call -> layer")
    p.add_argument("--regions", default=None, help="comma-separated region filter")
    p.add_argument("--json", dest="json_out", default=None, help="also write the report as JSON here")
    p.add_argument(
        "--json-max-per-region",
        type=int,
        default=DEFAULT_JSON_MAX_PER_REGION,
        help=(
            "cap the per-region divergence lists written to --json (default "
            f"{DEFAULT_JSON_MAX_PER_REGION}; 0 = uncapped). Counts and the first divergence are "
            "always complete; the JSON records what it dropped under 'truncated'."
        ),
    )
    args = p.parse_args(argv)
    regions = set(args.regions.split(",")) if args.regions else None
    rep = compare(
        load_dir(args.dir_a),
        load_dir(args.dir_b),
        case_a=args.case_a,
        case_b=args.case_b,
        layers=args.layers,
        regions=regions,
        man_a=load_manifests(args.dir_a),
        man_b=load_manifests(args.dir_b),
    )
    print(render_text(rep))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(cap_report(rep, args.json_max_per_region), f, indent=2)
    return {"clean": 0, "divergent": 2, "inconclusive": 3}[rep["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
