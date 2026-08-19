"""IsoExec weight-sync lifecycle contract: the sync-path ordering invariants as executable,
observation-only asserts wired at their seams, so a reorder trips a warning instead of silently
reintroducing an OOM or a stale-weights rollout.

Importing this package has no side effects (kept empty on purpose)."""
