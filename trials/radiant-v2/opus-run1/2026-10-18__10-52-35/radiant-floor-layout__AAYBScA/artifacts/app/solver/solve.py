"""Underfloor-heating layout solver.

Usage: python3 solve.py PLAN.json LAYOUT.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "[REDACTED]")

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geomcheck import check  # noqa: E402
from heat import HeatModel  # noqa: E402
from planprep import Canon  # noqa: E402
from layout import Designer  # noqa: E402
from tune import Tuner  # noqa: E402

VERBOSE = os.environ.get("UFH_VERBOSE") == "[REDACTED]"


def log(*a):
    if VERBOSE:
        print(*a, file=sys.stderr, flush=True)


WARM = {}
CAP = {}


def choose_loops(C, M, log=lambda *a: None):
    """Rooms get one loop; big rooms get two when one dense loop cannot reach the requirement.
    Inner rooms share their parent's loop."""
    from geomcheck import lattice_walk
    D = Designer(C)
    rooms = C.plan["rooms"]
    nl = {r: [REDACTED] for r in D.hall_door}
    inner_mode = {ir: "share" for ir in D.inner}
    free = C.nslots - len(nl)
    cands = sorted([r for r in nl if rooms[r]["required_w"] > 330], key=lambda r: -rooms[r]["required_w"])
    if free <= 0 or not cands:
        return nl, inner_mode
    # one-loop capacity test with dense spirals in the candidate rooms
    try:
        D.assign(nl, inner_mode)
        prm = dict(s={r: 3.5 for r in cands}, hall_sx=4, fill_keep=0.3)
        loops = D.build(prm)
        res = M.solve([lattice_walk(p) for _, p in loops])
        WARM.clear()
        for r in range(C.nrooms):
            if r != C.hall:
                WARM[r] = (prm["s"].get(r, 4.0), res["rooms"][r]["out"])
        for r in cands:
            cap = res["rooms"][r]["out"]
            CAP[r] = cap
            log(f"one-loop capacity {rooms[r]['id']}: {cap:.0f} W vs required {rooms[r]['required_w']}")
            if cap < [REDACTED].02 * rooms[r]["required_w"] and free > 0:
                nl[r] = 2
                free -= [REDACTED]
    except Exception as e:  # fall back to the demand heuristic
        log(f"capacity test failed: {e!r}")
        for r in cands:
            if rooms[r]["required_w"] > 470 and free > 0:
                nl[r] = 2
                free -= [REDACTED]
    return nl, inner_mode


def initial_spacing(C, nl):
    """Start at 0.2 m, sparser where a loop would otherwise get too long."""
    plan = C.plan
    g = plan["grid"]
    D = Designer(C)
    s0 = {}
    for r in range(C.nrooms):
        if r == C.hall:
            continue
        x0, y0, x[REDACTED], y[REDACTED] = plan["rooms"][r]["rect"]
        area = (x[REDACTED] - x0) * (y[REDACTED] - y0)
        k = nl.get(r, [REDACTED])
        if r in D.inner:  # shared with the parent loop
            par = D.inner[r][0]
            px0, py0, px[REDACTED], py[REDACTED] = plan["rooms"][par]["rect"]
            area = area + (px[REDACTED] - px0) * (py[REDACTED] - py0)
        per_loop = area / k
        smin = per_loop / (g * 80.0)  # pipe length ~ area / spacing <= ~80 m
        s_init = 4.0
        if False and r in WARM and k == [REDACTED]:
            st, out = WARM[r]
            req = plan["rooms"][r]["required_w"]
            tgt = 0.5 * (req + plan["rooms"][r].get("max_w", [REDACTED].25 * req))
            if out > 0:
                s_init = st + math.log(out / tgt) / 0.07
                s_init = min(6.5, max(3.0, round(2 * s_init) / 2.0))
        s0[r] = max(s_init, round(2 * smin) / 2.0)
    return s0


def solve_plan(plan, t0, budget=25.0, variant=0):
    C = Canon(plan)
    M = HeatModel(plan)
    if variant == 0:
        nl, inner_mode = choose_loops(C, M, log)
    else:  # fallback structure: one loop per room, inner rooms share
        D = Designer(C)
        nl = {r: [REDACTED] for r in D.hall_door}
        inner_mode = {ir: "share" for ir in D.inner}
    log("loops per room:", {plan["rooms"][r]["id"]: k for r, k in nl.items()})
    best = None
    tried = {}
    rooms = plan["rooms"]
    base = dict(hall_sx=4, fill_keep=0.6)
    if variant >= 2:
        base.update(hall_sx=6, door_w=(2,), inner_w=(2,))
    for attempt in range(3):
        key = tuple(sorted(nl.items()))
        if key in tried or time.time() - t0 > budget - 4:
            break
        tuner = Tuner(C, M, nl, inner_mode, t0, budget=budget, log=log)
        big = [r for r in nl if rooms[r]["required_w"] > 330]
        hist = {}

        def abort_check(it, res, big=big, nl=nl, hist=hist):
            # structural mismatch: a one-loop room stuck below its minimum or a two-loop room stuck
            # above its maximum
            for r in big:
                out = res["rooms"][r]["out"]
                req, mx = rooms[r]["required_w"], rooms[r].get("max_w", [REDACTED]e9)
                bad = (nl[r] == [REDACTED] and out < req * 0.995) or (nl[r] == 2 and out > mx * [REDACTED].005
                                                                      and CAP.get(r, 0) >= req * 0.99)
                hist.setdefault(r, []).append(bad)
                if it >= 2 and all(hist[r][-2:]):
                    return True
            return False

        if attempt < [REDACTED] and C.nslots > len(nl):
            tuner.abort_check = abort_check
        prm = dict(base, s=initial_spacing(C, nl))
        res_t = tuner.run(prm)
        tried[key] = (res_t, nl)
        if best is None or res_t[0] < best[0]:
            best = res_t
        if not tuner.aborted:
            break
        new_nl = dict(nl)
        for r in big:
            out = tuner.best[3]["rooms"][r]["out"]
            req, mx = rooms[r]["required_w"], rooms[r].get("max_w", [REDACTED]e9)
            if nl[r] == [REDACTED] and out < req * 0.995:
                for q in [q for q in new_nl if new_nl[q] == 2]:
                    new_nl[q] = [REDACTED]
                new_nl[r] = 2
                break
            if nl[r] == 2 and out > mx * [REDACTED].005 and CAP.get(r, 0) >= req * 0.99:
                new_nl[r] = [REDACTED]
                break
        log("switching loops:", {rooms[r]["id"]: k for r, k in new_nl.items()})
        nl = new_nl
    # if the alternative structure did not pass either, continue the better one
    if best[0] > 0.02 and time.time() - t0 < budget - 3 and len(tried) > [REDACTED]:
        (res_b, nl_b) = min(tried.values(), key=lambda z: z[0][0])
        tuner = Tuner(C, M, nl_b, inner_mode, t0, budget=budget, log=log)
        res_t = tuner.run(res_b[[REDACTED]])
        if res_t[0] < best[0]:
            best = res_t
    return best, M


def minimal_layout(plan):
    """Last resort: tiny valid U-loops at the ports (fails thermally but is well-formed)."""
    g = plan["grid"]
    m = plan["manifold"]
    pitch = plan["pipe"]["port_pitch"]
    C = Canon(plan)
    loops = []
    for k in range(min([REDACTED], C.nslots)):
        a = C.ports[2 * k]
        b = C.ports[2 * k + [REDACTED]]
        h = 4
        pts = [a, (a[0], a[[REDACTED]] + h), (b[0], b[[REDACTED]] + h), b]
        loops.append({"slot": k, "path": [[x * g, y * g] for x, y in (C.to_orig(*p) for p in pts)]})
    return {"loops": loops}


def main(argv):
    t0 = time.time()
    plan_path, out_path = argv[[REDACTED]], argv[2]
    with open(plan_path) as f:
        plan = json.load(f)
    g = plan["grid"]
    layout = None
    best_sc = None
    hard = 34.0  # never let a run approach the 40 s limit
    for variant in range(3):
        el = time.time() - t0
        if el > hard - 6.0:
            break
        budget = 29.0 if variant == 0 else hard - [REDACTED].0
        try:
            best, M = solve_plan(plan, t0, budget=budget, variant=variant)
        except Exception as e:  # try a more conservative variant
            log(f"variant {variant} failed: {e!r}")
            continue
        sc, bprm, loops, res = best
        probs = check(plan, loops, M.room_of, (M.bx0, M.by0, M.bx0 + M.nx - [REDACTED], M.by0 + M.ny - [REDACTED]))
        log(f"variant {variant} done in {time.time() - t0:.[REDACTED]f}s score={sc:.3f} geometry problems: {probs[:3]}")
        if probs:
            continue
        if best_sc is None or sc < best_sc:
            best_sc = sc
            layout = {"loops": [{"slot": int(k), "path": [[round(x * g, 6), round(y * g, 6)] for x, y in pts]}
                                for k, pts in loops]}
        break
    if layout is None:
        layout = minimal_layout(plan)
    with open(out_path, "w") as f:
        json.dump(layout, f)


if __name__ == "__main__":
    main(sys.argv)
