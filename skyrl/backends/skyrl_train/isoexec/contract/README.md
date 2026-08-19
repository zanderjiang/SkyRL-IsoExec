# The IsoExec Execution Contract

IsoExec is the execution contract behind SkyRL-IsoExec: one versioned, content-hashed object
declaring the numerical policy that every logprob execution runtime must implement identically for fixed weights,
history, and selected token. Kernels achieve zero KL; the contract makes it verified at
construction, survivable under change, and portable across runtimes.

## The IsoExec contract

```text
ExecutionContract
  schema_version
  model           # family, architectures key, profile ref
  identities      # semantic / numerical_policy / deployment
  cases           # every logprob-producing path, named
  composition     # (region, cases) -> impl@version × arch
  claims          # topology, state, admitted tolerances
```

### `identities` — three hashes, three jobs

| hash | computed over | job |
|---|---|---|
| `semantic` | model ref, logical-op vocabulary, case structure | "same logical model?" — catches green-gate-wrong-model |
| `numerical_policy` | function-half entries: impl ids/versions/arch, constants (bit patterns), routes, artifacts, topology claims | **the signature key** — mismatch refuses before weight sync; any bit-relevant change rotates it |
| `deployment` | deployment-half entries, adapters, ABI, proven-neutral knobs | compared/logged, never signature-keyed — neutral toggles don't fragment the signature table |

Identities are stored for the handshake and recomputed by the validator. The numerical hash
is closed over dependencies: it reaches through impl names to artifact identities, so an
implementation can't change under a stable name.

### `cases` — the closed list of ways logprobs get produced

```text
ExecutionCase
  id
  runtime_role
  ...

ExecutionCase
  id            engine_decode
  runtime_role  engine
  grad_mode     no_grad
  state_mode    continued
  shape_domain  T=1, B<=512
  constraints   cudagraph_capturable, host_free, address_stable
```

Two purposes: **completeness** — an unknown logprob path is a refusal, never "probably like
`trainer_score`" (recompute is named separately because it is a distinct code path that must
run the same composition) — and **the axis of legal asymmetry**: entries key on cases, so
deliberate trainer/engine differences are visible and proof-carrying.

### `composition` — a partition of logical ops, resolved to implementations

Three layers, each owned once: **logical ops** (the stable semantic vocabulary — never
changed by fusion), **regions** (a *partition* of logical ops into implementation units;
fusion is re-partitioning: rotates `numerical_policy`, leaves `semantic` untouched), and
**entries**:

```text
CompositionEntry
  region        # one partition unit — a fused group is one region
  impl          # impl id @ version × arch
  ...

CompositionEntry                                    # one fused unit owning three logical ops
  region        gdn.core + gdn.gating + norms.l2
  cases         all five
  impl          native_fused_sigmoid@1 × sm90
  route         protected

CompositionEntry                                    # EP-invariant combine, engine side
  region        moe.combine
  cases         engine_prefill, engine_decode
  impl          pik_leaf_tree@2 × sm90
  route         composition_defining                # Route B: sealed, no one-sided fallback
  constants     leaves=8, leaf_dtype=fp32           # in the hash
  artifact      sha256:2c8a41d9…
  discharge     equivalence_proof -> gates/ep_invariant_combine
```

Rules, all mechanical: explicit entry for every installed (region, case) — absence means
"no such case", never "default"; asymmetric regions must carry an `EquivalenceProof`
(`bitwise_equal_to` | `equivalence_proof` | entry-scoped `neutrality_proof`); constants are
identity, stored as **bit patterns, never decimal floats**; duplicate region ownership per
case is an error, not last-writer-wins.  refs are deliberately not carried by the
contract until they are content-addressed and CI-resolvable — an unverifiable pointer is
worse than none.

**Routes** determine failure semantics:

- `reference_preserving` (Route A): bit-equal to a named reference over admitted shapes.
  Foreign shape falls back to the reference legally, mid-run. Adopting rotates nothing.
- `composition_defining` (Route B): re-partitioning event. Built once, sealed, same artifact
  loaded by every case on both runtimes. Missing artifact or foreign shape **refuses** — no
  one-sided fallback, no local re-JIT. Adoption rotates the hash: a human-admitted event.
- `protected`: opaque manual provider; the system may not transform it.
- `canonical`: framework-default op, the fallback floor.

### `claims` — what's asserted about the world around the arithmetic

- **Topology, per axis**: `pinned(axis, degree, collective_plan)` — cheap, fragile to
  redeploy — or `invariant(axis, domain, proof)` — expensive once, buys mesh freedom. The
  handshake refuses a deployed topology outside the admitted domain.
- **State obligations**: each state (KV, recurrent, prefix cache, fused buffers, graph
  addresses) declares what invalidates it (weight sync, sleep/wake, storage move) and
  replay safety. Stale derived state after weight sync becomes machine-visible.
- **Admitted tolerances** — residual attribution: every non-bitwise case-pair boundary is
  *named*, bounded, and attributed to responsible regions. An observed diff outside the
  named set is a coverage gap, and alarms. Empty set ⇔ zero KL is a theorem, not a measurement.

## Enforcement — three checkpoints

1. **Build-time** (`contract.validate`, CI + freeze): partition totality and no duplicate
   ownership per case; explicit-entry rule; discharge obligations present; no raw floats;
   route requirements (B ⇒ artifact, A ⇒ reference); stored identities == recomputed;
   unknown function-half fields refuse.
2. **Startup/handshake** (runtimes): load the frozen artifact, install bindings, record
   first-forward resolved fingerprints, assert fingerprint == plan and trainer hash ==
   engine hash before weight sync. Refuse, don't warn.
3. **Live** (adapters): served/fallback/refusal counters per entry, first-backward
   connectivity assert, step-1 signature check. Requested ≠ installed ≠ served — each rung
   observed by the layer that can see it.

## Example — Qwen3.5-35B-A3B

```jsonc
{
  "schema_version": "1",
  "model": {
    "family": "qwen3_5",
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "profile_ref": "profiles/qwen3_5"
  },
  "identities": {
    "semantic":         "17eea127…",
    "numerical_policy": "a511d194…",   // the signature key
    "deployment":       "08469a74…"
  },
  "cases": [
    {
      "id": "trainer_score",
      "runtime_role": "trainer",
      "grad_mode": "no_grad",
      "state_mode": "fresh",
      "shape_domain": "packed_thd"
    },
    {
      "id": "engine_decode",
      "runtime_role": "engine",
      "grad_mode": "no_grad",
      "state_mode": "continued",
      "shape_domain": "T=1,B<=512",
      "constraints": ["cudagraph_capturable", "host_free", "address_stable"]
    },
    // ... trainer_fwd, trainer_recompute, engine_prefill
  ],
  "composition": [Evidence
    {
      // one fused unit owning three logical ops
      "region": ["gdn.core", "gdn.gating", "norms.l2"],
      "cases": ["trainer_fwd", "trainer_recompute", "trainer_score",
                "engine_prefill", "engine_decode"],
      "impl": {"id": "native_fused_sigmoid", "version": 1, "arch": "sm90"},
      "route": "protected"
    },
    {
      // EP-invariant combine, engine side; the trainer side is its own entry
      "region": ["moe.combine"],
      "cases": ["engine_prefill", "engine_decode"],
      "impl": {"id": "pik_leaf_tree", "version": 2, "arch": "sm90"},
      "route": "composition_defining",   // Route B: sealed, no one-sided fallback
      "constants": {"leaves": 8, "leaf_dtype": "fp32"},
      "artifact": "sha256:2c8a41d9…",
      "discharge": {"kind": "equivalence_proof", "ref": "gates/ep_invariant_combine"}
    },
    {
      "region": ["moe.router"],
      "cases": ["trainer_fwd", "trainer_recompute", "trainer_score",
                "engine_prefill", "engine_decode"],
      "impl": {"id": "fp32_router", "version": 1, "arch": "sm90"},
      "route": "protected",              // never fused: routing is a discrete decision
      "constants": {"dtype": "fp32", "score_fn": "sigmoid", "topk_tie_rule": "lowest_index"}
    },
    // ... 18 more: mm.matmul, attention.varlen, gdn.conv fn/update pair,
    //     moe.dispatch, moe.weights, collectives.row_parallel_ar, nccl_pin, ...
  ],
  "claims": {
    "topology": [
      {"axis": "TP", "kind": "invariant", "domain": [1, 4, 8], "proof": "gates/tp_invariance"},
      {"axis": "PP", "kind": "pinned", "degree": 1, "collective_plan": "none"},
      // ... EP_combine, SP invariant; CP pinned (GDN requires CP=1)
    ],
    "state": [
      {"state_id": "prefix_cache", "invalidated_by": ["weight_sync"],
       "replay_safe": true, "ref": "lifecycle/prefix_flush_on_sync"},
      // ... kv_cache, gdn.entry_state, moe.fused_weight_buffer
    ],
    "tolerances": [
      {
        "case_pair": ["engine_decode", "trainer_score"],
        "bounds": {"mean_abs_logp_diff": "1e-7", "max_abs_logp_diff": "5e-6"},
        "attributed_to": ["attention.varlen", "gdn.conv"]
      }
    ]
  }
}
```

The artifact round-trips byte-stably, and any bit-relevant edit — a constant, an impl
version, a re-partition — rotates `numerical_policy`, while entry reordering and
deployment-half changes leave it fixed.
