"""Regenerate the autopsy layouts (CPU only, no containers): GPT-6.1's saved v2.1 solver (the copy that
scored 12/12 at 21:31) and the v2.1 reference, one plan at a time with a 15 s limit and numeric threads
capped at 2. Usage: python regenerate.py [gpt] [ref]   (writes layouts/<name>/<plan>.json)

The v2.1 reference is taken from git (05f9827), since the task directory holds v2."""
import os, subprocess, sys, tempfile, time  # noqa: E401
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TASK = ROOT / "tasks" / "radiant-floor-layout"
PLANS = sorted((TASK / "tests/plans").glob("*.json")) + sorted((TASK / "environment/app/plans").glob("*.json"))


def reference_v21():
    d = Path(tempfile.mkdtemp(prefix="ref-v21-"))
    for src, dst in (("tasks/radiant-floor-layout/solution/solver/layout_solver.py", "layout_solver.py"),
                     ("tasks/radiant-floor-layout/environment/app/check_layout.py", "check_layout.py")):
        (d / dst).write_bytes(subprocess.run(["git", "-C", str(ROOT), "show", f"05f9827:{src}"],
                                             capture_output=True, check=True).stdout)
    (d / "solve.py").write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).resolve().parent))\n"
                                "from layout_solver import main\nmain()\n")
    return d


SOLVERS = {"gpt": lambda: ROOT / "trials/radiant-v2.1/gpt-run1/solver-at-12of12", "ref": reference_v21}
env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
for name in sys.argv[1:] or list(SOLVERS):
    solver = SOLVERS[name]()
    out = HERE / "layouts" / name
    out.mkdir(parents=True, exist_ok=True)
    for plan in PLANS:
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, "-E", "-s", str(solver / "solve.py"), str(plan), str(out / f"{plan.stem}.json")],
                               cwd=tempfile.gettempdir(), env=env, capture_output=True, timeout=15)
            note = f"exit {p.returncode}"
        except subprocess.TimeoutExpired:
            note = "TIMEOUT"
        print(f"{name} {plan.stem}: {note} {time.time() - t0:.1f} s", flush=True)
