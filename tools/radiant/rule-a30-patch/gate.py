"""Offline gate for the bend-rule prototype (CPU only, no containers, no model usage).

Usage: python gate.py SOLVE_PY RULE TAG
  SOLVE_PY  a solve.py taking (plan, layout[, rule]); e.g. solver/solve.py for the Rule-A30 reference
  RULE      A30 | A | B | none: the rule the layouts are judged by (the task's checker plus the rule)
Runs the solver on the 15 plans (12 held-out + 3 samples), one at a time; a plan fails if no
layout is written, the process takes over 15 s, or the layout fails the checker or the rule.
Layouts go to results/TAG/, the summary to stdout."""
import json, os, subprocess, sys, time  # noqa: E401
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rules  # noqa: E402

solve_py, rule, tag = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3]
PLANS = sorted((rules.TASK / "tests/plans").glob("*.json")) + sorted((rules.TASK / "environment/app/plans").glob("*.json"))
out_dir = HERE / "results" / tag
out_dir.mkdir(parents=True, exist_ok=True)
env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
passed, slowest, lines = 0, 0.0, []
for pp in PLANS:
    lay = out_dir / f"{pp.stem}.json"
    lay.unlink(missing_ok=True)
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-E", "-s", str(solve_py), str(pp), str(lay), rule],
                           capture_output=True, timeout=60, env=env)
        dt = time.time() - t0
        st = [x for x in p.stderr.decode(errors="replace").splitlines() if x.startswith("SOLVE_TIME")]
        inner = float(st[-1].split()[1]) if st else None
        err = p.stderr.decode(errors="replace").strip().splitlines()[-1:] if p.returncode else []
        if dt > 15:
            err = err or [f"over 15 s: process {dt:.1f} s, solver {inner} s"]
    except subprocess.TimeoutExpired:
        dt, inner, err = 60.0, None, ["TIMEOUT (60 s)"]
    slowest = max(slowest, dt)
    if not lay.is_file() or err:
        lines.append(f"{pp.stem}: FAIL no layout {err} [{dt:.1f} s]")
        continue
    plan, layout = json.loads(pp.read_text()), json.loads(lay.read_text())
    rep = rules.evaluate(plan, layout, rule)
    if rep["geometry_problems"]:
        lines.append(f"{pp.stem}: FAIL geometry {rep['geometry_problems'][0]} [{dt:.1f} s]")
        continue
    bad = [f"{r['id']} {k}" for r in rep["rooms"] for k, ok in r["checks"].items() if not ok]
    tight = min(min(r["output_w"] - r["required_w"], r["max_w"] - r["output_w"]) for r in rep["rooms"])
    tsp = min(r["limit_spread"] - r["zone_spread"] for r in rep["rooms"])
    passed += rep["passed"]
    lines.append(f"{pp.stem}: {'PASS' if rep['passed'] else 'FAIL ' + ', '.join(bad)}  short<0.20: "
                 f"{len(rules.pieces(plan, layout)):3}  tightest W margin {tight:6.1f}  tightest spread margin "
                 f"{tsp:.2f} K  [{dt:.1f} s, solver {inner} s]")
print("\n".join(lines))
print(f"{tag} rule {rule}: {passed}/{len(PLANS)} pass, slowest {slowest:.1f} s")
