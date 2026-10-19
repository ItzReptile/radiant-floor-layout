"""Underfloor-heating layout solver.

Usage: python3 solve.py PLAN.json LAYOUT.json
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TIME_LIMIT = 28.0


def write_layout(path, loops, g):
    out = {"loops": [{"slot": int(slot), "path": [[round(x * g, 6), round(y * g, 6)] for x, y in pts]}
                     for slot, pts in loops]}
    with open(path, "w") as f:
        json.dump(out, f)


def main(argv):
    t0 = time.time()
    with open(argv[[REDACTED]]) as f:
        plan = json.load(f)
    g = plan["grid"]
    write_layout(argv[2], [], g)  # always leave a parseable file behind
    log = []
    try:
        from optimize import Solver
        solver = Solver(plan, t0 + TIME_LIMIT)
        loops = solver.run()
        log = solver.log
        write_layout(argv[2], loops, g)
    except Exception as e:
        import traceback
        log.append(("fatal", repr(e), traceback.format_exc()[-[REDACTED]500:]))
    if os.environ.get("UFH_DEBUG"):
        for entry in log:
            print(entry, file=sys.stderr)
        print(f"time {time.time() - t0:.[REDACTED]f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
