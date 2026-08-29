"""Launch-readiness gate: run immediately before a production launch, from the repo root.

    python skyrl/backends/skyrl_train/isoexec/core/tests/preflight_launch.py

Phase 1 runs the complete enforcement battery (every core/tests/test_*.py, each in its own
subprocess, plus the contract leaf suite via unittest). Phase 2 builds the production contract
under the run-script env (examples/train/isoexec/run_qwen35_dapo_isoexec.sh) and prints the four
identity values plus the derived obligation counts. Any failure exits nonzero and prints
PREFLIGHT: BLOCKED (<reason>); a clean run prints PREFLIGHT: READY.

GPU safety: a production run may own this node's GPUs, so every child runs with
CUDA_VISIBLE_DEVICES="" and core/arch.ARCH pointed at the production accelerator tag (sm90).
Nothing here touches CUDA, processes, or Ray.
"""

import os
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve()
TEST_DIR = HERE.parent
REPO = HERE.parents[6]
RUN_SCRIPT = REPO / "examples/train/isoexec/run_qwen35_dapo_isoexec.sh"
LEAF_PKG = "skyrl/backends/skyrl_train/isoexec/contract/tests"
PROD_ARCH = "sm90"


def _child_env():
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""  # never touch the GPUs a live run owns
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for k in list(env):
        if k.startswith("SKYRL_ISOEXEC") or k in ("ISOEXEC_CONTRACT_PATH", "ISOEXEC_CONTRACT_HASH"):
            del env[k]  # the battery judges code defaults, not the caller's shell
    return env


def _patch_arch():
    import skyrl.backends.skyrl_train.isoexec.core.arch as arch_mod

    if arch_mod.ARCH == arch_mod.NON_ACCELERATOR_ARCH:
        arch_mod.ARCH = PROD_ARCH


def _run_file(path):
    import importlib.util
    import traceback

    _patch_arch()
    spec = importlib.util.spec_from_file_location(pathlib.Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [(k, v) for k, v in vars(mod).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            traceback.print_exc()
            print(f"FAIL {name}")
            failed += 1
    print(f"{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


def _run_script_env():
    """The run script's literal SKYRL_ISOEXEC_*/ISOEXEC_* exports; $-bearing values are listed as
    skipped (they name node-local paths or derived vars, none of which enter the contract)."""
    applied, skipped = {}, []
    text = RUN_SCRIPT.read_text()
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("export "):
            continue
        for tok in line[len("export ") :].split():
            if "=" not in tok:
                continue
            k, _, v = tok.partition("=")
            if not (k.startswith("SKYRL_ISOEXEC") or k.startswith("ISOEXEC_")):
                continue
            v = v.strip("\"'")
            if "$" in v:
                skipped.append(k)
            else:
                applied[k] = v
    m = re.search(r"MODEL=\"?\$\{ISOEXEC_MODEL:-([^}\"]+)\}", text)
    model = m.group(1) if m else applied.get("SKYRL_ISOEXEC_MODEL_PATH", "qwen3.5-35b-a3b")
    return applied, sorted(skipped), model


def _build_contract():
    from collections import Counter

    applied, skipped, model = _run_script_env()
    os.environ.update(applied)
    os.environ.pop("ISOEXEC_CONTRACT_PATH", None)  # print-only phase: no artifacts
    _patch_arch()

    from skyrl.backends.skyrl_train.isoexec.core import enforce
    from skyrl.backends.skyrl_train.isoexec.core import process_contract as pc
    from skyrl.backends.skyrl_train.isoexec.core.registry_build import build_registry

    print(f"run-script env: applied {len(applied)} literal export(s); skipped non-literal: {skipped}")
    print(f"model: {model}  arch: {PROD_ARCH} (production tag; this driver is CPU-only)")
    c = pc.get_process_contract(model, arch=PROD_ARCH)
    if c is None:
        print("PREFLIGHT-CONTRACT: build returned None")
        return 1
    ids = c.identities
    print(f"IDENTITY semantic         = {ids.semantic}")
    print(f"IDENTITY numerical_policy = {ids.numerical_policy}")
    print(f"IDENTITY deployment       = {ids.deployment}")
    print(f"IDENTITY composite        = {pc.contract_hash()} (handshake: extensions folded, process-local)")
    print(
        f"contract: {len(c.composition)} composition entries; claims: "
        f"{len(c.claims.topology)} topology, {len(c.claims.state)} state, {len(c.claims.tolerances)} tolerance"
    )
    reg = build_registry(strict=True)
    for side in ("trainer", "engine"):
        plan = enforce.derive_obligation_plan(c, reg, side)
        cnt = Counter((o.phase, o.kind) for o in plan.obligations)
        by_phase = ", ".join(f"{p}:{sum(n for (ph, _), n in cnt.items() if ph == p)}" for p in enforce.PHASES)
        n_exc = sum(1 for o in plan.obligations if enforce.exemption_for(o.obligation_id) is not None)
        print(
            f"obligations[{side}]: total={len(plan.obligations)} ({by_phase}) "
            f"excepted={n_exc} no_served_counter={len(plan.no_served_counter)}"
        )
        for (phase, kind), n in sorted(cnt.items(), key=lambda kv: (enforce.PHASES.index(kv[0][0]), kv[0][1])):
            print(f"  {side} {phase:13s} {kind:18s} x{n}")
    return 0


def main():
    t0 = time.time()
    env = _child_env()
    total = passed = 0
    blocked = None

    files = sorted(p for p in TEST_DIR.glob("test_*.py"))
    for f in files:
        r = subprocess.run(
            [sys.executable, str(HERE), "--run-file", str(f)],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
        )
        m = re.search(r"^(\d+)/(\d+) passed\s*$", r.stdout, re.M)
        n_pass, n_total = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        total += n_total
        passed += n_pass
        status = "ok" if (r.returncode == 0 and m) else "FAIL"
        print(f"[battery] {f.name:45s} {n_pass}/{n_total} {status}")
        if status == "FAIL" and blocked is None:
            blocked = f"{f.name}: {n_total - n_pass or '?'} test failure(s)"
            print(r.stdout[-4000:])
            print(r.stderr[-4000:])

    r = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", LEAF_PKG, "-t", "."],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    m = re.search(r"Ran (\d+) tests", r.stderr)
    n_leaf = int(m.group(1)) if m else 0
    leaf_ok = r.returncode == 0 and n_leaf > 0
    total += n_leaf
    passed += n_leaf if leaf_ok else 0
    print(f"[battery] contract leaf (unittest){'':16s}{n_leaf}/{n_leaf} {'ok' if leaf_ok else 'FAIL'}")
    if not leaf_ok and blocked is None:
        blocked = "contract leaf suite failed"
        print(r.stderr[-4000:])

    if blocked is None:
        print("\n[contract] building the production contract under the run-script env")
        r = subprocess.run(
            [sys.executable, str(HERE), "--build-contract"],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
        )
        for line in r.stdout.splitlines():
            print(f"  {line}")
        if r.returncode != 0:
            blocked = "production contract build failed"
            print(r.stderr[-4000:])

    print(f"\nbattery: {passed}/{total} passed in {time.time() - t0:.0f}s")
    if blocked is None and passed == total and total > 0:
        print("PREFLIGHT: READY")
        return 0
    print(f"PREFLIGHT: BLOCKED ({blocked or 'incomplete battery'})")
    return 1


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--run-file":
        sys.exit(_run_file(sys.argv[2]))
    if len(sys.argv) > 1 and sys.argv[1] == "--build-contract":
        sys.exit(_build_contract())
    sys.exit(main())
