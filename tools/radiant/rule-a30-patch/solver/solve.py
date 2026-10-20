"""Rule-A30 reference: python solve.py PLAN.json LAYOUT.json [RULE]  (RULE defaults to A30).
The solver scores candidate layouts with the task's checker plus the rule (rules.py), so it never
returns a layout that breaks the rule. Prints SOLVE_TIME to stderr."""
import time as _t
_T0 = _t.perf_counter()
import json, sys  # noqa: E401,E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import rules  # noqa: E402  (imports the task's tests/check_layout.py)
from scipy.sparse.linalg import splu  # noqa: E402
rules.ck.spsolve = lambda a, b: splu(a.tocsc(), permc_spec="MMD_AT_PLUS_A").solve(b)
import layout_solver  # noqa: E402

rule = sys.argv[3] if len(sys.argv) > 3 else "A30"
plan = json.load(open(sys.argv[1]))
layout, _ = layout_solver.solve(plan, lambda p, lay: rules.evaluate(p, lay, rule), final_check=False)
json.dump(layout, open(sys.argv[2], "w"))
print(f"SOLVE_TIME {_t.perf_counter() - _T0:.2f}", file=sys.stderr)
