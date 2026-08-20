# Developing an op — instructions for agents

An op is a mathematical function with a pinned rounding schedule, packaged as one folder:
kernels, wrappers, its `_register.py` declarations, and `tests/` — all colocated. The contract
system consumes what you declare here; an undeclared fact is invisible to it and an unproven
claim will be refused or flagged. Work in this order.

## 1. Decide the op boundary

Op boundary == kernel boundary == rounding boundary. If a fused kernel spans what used to be
three ops, the fused thing is the op, and it declares `subsumes=["old.op1", "old.op2"]` so the
adapter derives passthrough for the absorbed names. Classify the arithmetic before writing code:

- **discrete decision** (top-k, argmax, tie-break) → never fuse across it; declare the tie rule
  as a pinned constant.
- **float reduction** (sums, softmax, scans, cross-rank) → order is identity; a new order is a
  composition event for BOTH runtimes, never a one-sided optimization.
- **pointwise** → bit-safe only with matched transcendental provider, FTZ policy, signed-zero
  behavior, and FMA contraction. Integer-only fusions are unconditionally safe.

## 2. Register (`_register.py`)

Declarations only — no behavior in this file. Follow the existing pattern (see `rope/_register.py`):

```python
reg.register_op(OpSpec(name="family.op", sites=[...]))          # ONLY the sites the op has;
    .add_impl(ImplSpec(                                          # absence == "no such site"
        impl_id="...", version=1,
        supported_archs=frozenset({"sm90"}),                     # proofs are arch-scoped
        rounding=RoundingSchedule(
            machine_assertable={...},   # dtypes, pins, block sizes — validate_pins checks
                                        # every manifest pin against these; PER_MODEL = value
                                        # comes from the model profile; OneOf(...) = enumerated
            documentary="internal reduction order, formula variants — feeds identity via
                         the version, not asserted",
        ),
        capabilities={...},             # see §3 — equivalence claims live here
        subsumes=[...],
        hazards=[...],                  # from core/registry.py HAZARDS — the FLOOR your
    ))                                  # tests must EXERCISE, not merely mention
```

Rules the system enforces on you: unknown site/hazard names refuse at registration; selecting
an undeclared site or pinning an undeclared constant refuses at contract build; changing the
rounding schedule (either half) requires a version bump — the version is identity, and a new
`numerical_policy` hash means the gate signature re-freezes with proof.

## 3. Make the equivalence claim honestly

If the op's sites all resolve to ONE impl, you are done — consistency by construction, no claim
needed. Otherwise every asymmetric pairing must carry exactly one of:

- `capabilities={"bitwise_equal_to": "<impl_id on the same op>"}` — byte-equal twin. The referent
  must name a registered impl (the alignment checker fails dangling refs) and your tests must
  prove it with bit-pattern comparison.
- `capabilities={"equivalence_proof": "<gate pointer>"}` — distinct algorithms, exact for a
  structural reason (e.g. split-exactness at any token boundary). Declaring `bitwise_equal_to`
  when the kernels do NOT agree bitwise is a lie that passes construction and fails admission —
  use this form and prove the actual property.
- deployment classification with a neutrality proof — only for entries proven not to move bits,
  and the proof licenses exactly that entry.

Without one of these, the contract refuses to build. That is intended.

## 4. Test (`tests/`, colocated) — the five gate families

Instantiate from `core/contracts.py`. Every impl needs, where applicable:

1. **Site parity** — bitwise across the sites it serves (stateful ops: one call == N sequential
   calls including the state round-trip).
2. **Batch invariance** — a token's result never depends on batchmates, padding, or ragged lanes
   (including NULL-lane inertness for graph replay).
3. **Resume exactness** — split-anywhere == unsplit (chunked prefill, cache resume).
4. **Degenerate shapes** — T=0, non-contiguous, E≠expected, vLLM profiling shapes. Offline gates
   that only see well-formed shapes have crashed engine init before.
5. **Backward** — grads exist for every differentiable input; declare which inputs carry
   param grads.

Non-negotiable test mechanics:

- **Bit patterns, never `torch.equal` or allclose** for equality claims — use
  `contracts.bitwise_equal` (integer views; `torch.equal(+0., -0.)` is True and will lie to you).
- **Hazards must be EXERCISED, not covered** — call `contracts.assert_hazard_exercised(name,
  evidence)` proving the subnormal occurred, the tie actually tied, the NULL lane was present.
  A 130-check gate once passed because well-conditioned inputs never produced a subnormal.
- Guard GPU/world-size requirements with graceful skips; keep single-GPU tests minutes-scale.
- Name tests so claim → test is obvious; update the family's `tests/README.md` claim map.

`core/tests/test_op_gate_alignment.py` statically checks: claim referents resolve, claimed impls
have tests, declared hazards are referenced. Run it plus your family's tests before calling the
op done. Invocation from repo root: `PYTHONPATH=. $VENV/bin/python <test file>` with the pinned
venv and `LD_LIBRARY_PATH=$VENV/lib/python3.12/site-packages/nvidia/cu13/lib` (no pytest; some
paths need `import torch` first).

## 5. What your evidence licenses — and what it doesn't

A passing battery licenses the claim over the domain it exercised: those shapes, those batch
compositions, that topology, that arch. Outside the domain there is no claim. sm90 evidence says
nothing about sm100 — new arch means rerunning the battery, not editing the declaration. The
live step-1 gate remains the final oracle; your offline proofs exist so it confirms instead of
discovers.

## 6. Checklist before handing off

- [ ] `_register.py` declares sites, schedule (both halves), hazards, arch, claims
- [ ] asymmetric pairings carry the correct claim form (§3)
- [ ] `tests/` covers the five families, bit-pattern comparisons, hazards exercised
- [ ] `tests/README.md` claim map updated
- [ ] `test_op_gate_alignment.py` clean; family tests green; core battery untouched-green
- [ ] version bumped iff the rounding schedule changed; expect the signature re-freeze
