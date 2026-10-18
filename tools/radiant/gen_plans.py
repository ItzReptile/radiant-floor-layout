"""Generate radiant-floor-layout floor plans, built around the reference solver.

Every plan is a corridor apartment: a hall strip with the manifold, two rooms on one side
(living room with two loops, kitchen with counters and an island) and three on the other
(two bedrooms with wardrobes, a bath with a shower tray), each opening onto the hall through
one doorway. Heat targets come from a per-type W/m2 rule; the reference solver is run against
them, and each room's limits are set around what it achieved: the heat window at about
-10%/+12%, the spread limit at the reference spread + 0.5 K (never above the room type's
default) and, for the hall, + 0.3 K. A plan is kept only when the reference solver, run again
from scratch on the finished plan, passes it, when layouts that pack the hall leaders
against either hall wall fail it (otherwise the hall is too narrow for the hall check to
matter), and when a naive single-pass serpentine fails it (ablation.py has both baselines).
The plan is then rotated/mirrored at random.

Usage: python gen_plans.py OUT_DIR --count N --seed S [--prefix P --start I]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1] / "tasks" / "radiant-floor-layout"
sys.path.insert(0, str(TASK / "environment" / "app"))
sys.path.insert(0, str(TASK / "solution" / "solver"))
import check_layout as ck  # noqa: E402
import layout_solver as ls  # noqa: E402

sys.path.insert(0, str(HERE))
import ablation as ab  # noqa: E402

BASE = json.loads((HERE / "base_plan.json").read_text())
Q = {"living": 34.0, "kitchen": 27.0, "bed": 24.0, "bath": 30.0, "study": 24.0}
# limits set around the reference result (None: keep the fixed per-type limit)
OPTS = {"spread_margin": 0.45, "spread_cap": 3.2, "bath_cap": 4.0, "hall_peak_margin": None,
        "hall_spread_margin": 0.3, "q_scale": 1.0, "hall_margin": 0.15,
        "filters": ("north", "south", "uniform", "serp")}
REJECTED = __import__("collections").Counter()


def r2(v):
    return round(round(v / 0.05) * 0.05, 4)


def pick(rng, lo, hi, step=0.05):
    n = int(round((hi - lo) / step))
    return r2(lo + rng.randint(0, n) * step)


def make_geometry(rng):
    """Canonical frame: hall along x, living + kitchen to the south, bed/bath/bed to the north,
    and a study behind one bedroom that opens only into that bedroom."""
    W = pick(rng, 9.5, 11.0, 0.5)
    hall_h = pick(rng, 1.25, 1.6)
    south_d = pick(rng, 3.4, 4.0)
    north_d = pick(rng, 4.1, 4.6)
    H = r2(south_d + hall_h + north_d)
    hy0, hy1 = south_d, r2(south_d + hall_h)
    rooms, passages, keepouts = [], [], []
    KIND = {"living": (20.0, 0.05, 29.0, 3.0, 0.5), "kitchen": (20.0, 0.01, 29.0, 3.0, 0.5),
            "bed": (20.0, None, 29.0, 3.0, 0.5), "bath": (24.0, 0.01, 33.0, 4.0, 0.3),
            "study": (20.0, 0.05, 29.0, 3.0, 0.5)}

    def room(rid, kind, rect):
        air, fr, mx, sp, zm = KIND[kind]
        rooms.append({"id": rid, "kind": kind, "rect": rect, "air_temp": air,
                      "floor_r": fr if fr is not None else rng.choice([0.05, 0.10]),
                      "max_surface": mx, "max_spread": sp, "zone_margin": zm})

    def door(x0, x1, y, width, centre=None):
        if centre is None:
            centre = pick(rng, x0 + 0.3 + width / 2, x1 - 0.3 - width / 2)
        a = r2(centre - width / 2)
        passages.append([a, y, r2(a + width), y])
        return passages[-1]

    # south side: living (2 loops) with a peninsula, kitchen with counters and an island
    s = pick(rng, 0.50 * W, 0.58 * W)
    living_west = rng.random() < 0.5
    lv = [0.0, 0.0, s, hy0] if living_west else [r2(W - s), 0.0, W, hy0]
    kt = [s, 0.0, W, hy0] if living_west else [0.0, 0.0, r2(W - s), hy0]
    room("living", "living", lv)
    room("kitchen", "kitchen", kt)
    lw = lv[2] - lv[0]
    door(lv[0], lv[2], hy0, 0.8, centre=pick(rng, lv[0] + 0.4 * lw, lv[0] + 0.6 * lw))
    door(kt[0], kt[2], hy0, pick(rng, 0.8, 1.0))
    # living: a peninsula (chimney breast) off the outer wall, 0.80-1.05 m from a side wall
    pw, pd, ch = pick(rng, 0.4, 0.6), pick(rng, 1.0, 1.6), pick(rng, 0.80, 1.05)
    px = r2(lv[0] + ch) if rng.random() < 0.5 else r2(lv[2] - ch - pw)
    keepouts.append([px, 0.0, r2(px + pw), pd])
    # kitchen: counters along the outer wall, an island 0.80-1.05 m from the counters and from
    # one side wall
    keepouts.append([kt[0], 0.0, kt[2], 0.6])
    iw, ih = pick(rng, 1.0, 1.5), pick(rng, 0.6, 0.8)
    iy = r2(0.6 + pick(rng, 0.80, 1.05))
    if iy + ih > hy0 - 0.9:
        return None
    side = pick(rng, 0.80, 1.05)
    ix = r2(kt[0] + side) if rng.random() < 0.5 else r2(kt[2] - side - iw)
    keepouts.append([ix, iy, r2(ix + iw), r2(iy + ih)])
    # north side: two bedrooms and a bath; a study behind the narrower bedroom
    bath_w = pick(rng, 2.0, 2.4)
    rest = W - bath_w
    b1 = pick(rng, 0.45 * rest, 0.55 * rest)
    widths = [b1, r2(rest - b1)]
    order = rng.choice([("bed", "bath", "bed"), ("bath", "bed", "bed"), ("bed", "bed", "bath")])
    parent = 0 if widths[0] <= widths[1] else 1
    x = 0.0
    beds = 0
    for kind in order:
        w = bath_w if kind == "bath" else widths[beds]
        x1 = W if abs(x + w - W) < 0.06 else r2(x + w)
        if kind == "bath":
            rect = [x, hy1, x1, H]
            room("bath", "bath", rect)
            door(x, x1, hy1, pick(rng, 0.7, 0.9))
            tray = pick(rng, 0.9, 1.0)
            if rng.random() < 0.5:
                keepouts.append([x, r2(H - tray), r2(x + tray), H])
            else:
                keepouts.append([r2(x1 - tray), r2(H - tray), x1, H])
        else:
            rid = f"bed{beds + 1}"
            ww = pick(rng, 1.4, 1.8)
            if beds == parent:
                width = x1 - x
                sd = r2(min(max(pick(rng, 4.5, 6.5) / width, 1.6), 2.1))
                ys = r2(H - sd)
                if ys - hy1 < 2.5:
                    return None
                room(rid, "bed", [x, hy1, x1, ys])
                room("study", "study", [x, ys, x1, H])
                door(x, x1, hy1, pick(rng, 0.8, 0.9))
                # wardrobe along a side wall against the study wall; the study door on the other half
                left = rng.random() < 0.5
                if left:
                    keepouts.append([x, r2(ys - ww), r2(x + 0.6), ys])
                    door(r2(x + 0.6 + 0.3), x1, ys, pick(rng, 0.8, 0.9))
                else:
                    keepouts.append([r2(x1 - 0.6), r2(ys - ww), x1, ys])
                    door(x, r2(x1 - 0.6 - 0.3), ys, pick(rng, 0.8, 0.9))
            else:
                room(rid, "bed", [x, hy1, x1, H])
                door(x, x1, hy1, pick(rng, 0.8, 0.9))
                if rng.random() < 0.5:
                    keepouts.append([x, r2(H - 0.6), r2(x + ww), H])
                else:
                    keepouts.append([r2(x1 - ww), r2(H - 0.6), x1, H])
            beds += 1
        x = x1
    hall = {"id": "hall", "kind": "hall", "rect": [0.0, hy0, W, hy1], "air_temp": 18.0, "floor_r": 0.01,
            "max_surface": 29.0, "max_spread": 9.0, "zone_margin": OPTS["hall_margin"]}
    rooms.append(hall)
    # manifold on the hall's north wall, clear of the doors on that wall
    north_doors = [p for p in passages if abs(p[1] - hy1) < 1e-9]
    for _ in range(200):
        # in the middle part of the hall, so that loops leave it in both directions
        mx = pick(rng, 0.3 * W, 0.7 * W - 1.15)
        if all(mx + 1.25 < d[0] - 0.3 or mx - 0.15 > d[2] + 0.3 for d in north_doors):
            break
    else:
        return None
    oy = r2(hy1 - 0.15)
    keepouts.append([r2(mx - 0.05), r2(hy1 - 0.10), r2(mx + 1.15), hy1])
    plan = copy.deepcopy(BASE)
    plan.update({"name": "", "building": [0.0, 0.0, W, H], "rooms": rooms, "passages": passages,
                 "keepouts": keepouts, "manifold": {"origin": [mx, oy], "direction": [1, 0], "ports": 12}})
    return plan


def finish(plan, rng):
    """Solve towards heat targets, then set the limits around the reference result."""
    targets = {}
    for r in plan["rooms"]:
        x0, y0, x1, y1 = r["rect"]
        if r["kind"] != "hall":
            targets[r["id"]] = OPTS["q_scale"] * Q[r["kind"]] * (x1 - x0) * (y1 - y0)
        r["required_w"], r["max_w"] = 0, 10 ** 6
    try:
        layout, rep = ls.solve(plan, ck.evaluate, targets, max_evals=30, time_budget=120, spread_goal=2.5)
    except Exception:
        REJECTED["no layout"] += 1
        return None
    if rep["geometry_problems"]:
        REJECTED["no layout"] += 1
        return None
    for r, res in zip(plan["rooms"], rep["rooms"]):
        if res["max_surface"] > r["max_surface"] - 0.4:
            REJECTED["reference peak"] += 1
            return None
        if r["kind"] != "hall":
            lim = math.ceil((res["zone_spread"] + OPTS["spread_margin"]) * 10) / 10
            if lim > (OPTS["bath_cap"] if r["kind"] == "bath" else OPTS["spread_cap"]):
                REJECTED["reference spread"] += 1
                return None
            r["max_spread"] = lim
        else:
            r["max_spread"] = math.ceil((res["zone_spread"] + OPTS["hall_spread_margin"]) * 10) / 10
            if OPTS["hall_peak_margin"] is not None:
                r["max_surface"] = math.ceil((res["max_surface"] + OPTS["hall_peak_margin"]) * 10) / 10
        q = res["output_w"]
        r["required_w"] = int(math.floor(0.90 * q))
        r["max_w"] = int(math.ceil(1.12 * q))
    # the finished plan must be solvable from scratch by the reference solver
    layout, rep = ls.solve(plan, ck.evaluate)
    if not rep["passed"]:
        REJECTED["reference from scratch"] += 1
        return None
    # every check must matter: drop plans that a baseline missing one idea passes
    canon = ls.transform_plan(plan, ls.canonical_transform(plan))
    for mode in OPTS["filters"]:
        _, brep = ab.solve_mode(copy.deepcopy(canon), mode)
        if brep is not None and not brep["geometry_problems"] and brep["passed"]:
            REJECTED[f"passed by {mode}"] += 1
            return None
    return plan


def d4(plan, swap, fx, fy):
    W, H = plan["building"][2], plan["building"][3]
    if swap:
        W, H = H, W
    t = (swap, fx, fy, W, H)
    q = copy.deepcopy(plan)
    q["building"] = [0.0, 0.0, W, H]
    for r in q["rooms"]:
        r["rect"] = ls._tf_rect(t, r["rect"])
    q["passages"] = [ls._tf_rect(t, p) for p in q["passages"]]
    q["keepouts"] = [ls._tf_rect(t, k) for k in q["keepouts"]]
    m = q["manifold"]
    m["origin"] = list(ls._tf_point(t, *m["origin"]))
    dx, dy = m["direction"]
    if swap:
        dx, dy = dy, dx
    m["direction"] = [-dx if fx else dx, -dy if fy else dy]
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--prefix", default="plan")
    ap.add_argument("--flow", type=float, help="override physics.flow_constant")
    ap.add_argument("--flow-offset", type=float, help="override physics.flow_offset")
    ap.add_argument("--hall-margin", type=float, default=0.15, help="hall zone_margin")
    ap.add_argument("--start", type=int, default=0, help="number of the first plan")
    ap.add_argument("--no-filters", action="store_true", help="keep plans that a baseline passes")
    ap.add_argument("--supply", type=float, help="override physics.supply_temp")
    ap.add_argument("--spread-margin", type=float, default=0.45, help="room spread limit = reference spread + margin")
    ap.add_argument("--hall-peak-margin", type=float, help="hall peak limit = reference peak + margin")
    ap.add_argument("--hall-spread-margin", type=float, default=0.3, help="hall spread limit = reference + margin")
    ap.add_argument("--q-scale", type=float, default=1.0, help="scale the heat targets")
    args = ap.parse_args()
    if args.flow:
        BASE["physics"]["flow_constant"] = args.flow
    if args.flow_offset is not None:
        BASE["physics"]["flow_offset"] = args.flow_offset
    if args.supply:
        BASE["physics"]["supply_temp"] = args.supply
    OPTS.update(spread_margin=args.spread_margin, hall_peak_margin=args.hall_peak_margin,
                hall_spread_margin=args.hall_spread_margin, q_scale=args.q_scale,
                hall_margin=args.hall_margin, filters=() if args.no_filters else OPTS["filters"])
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    made = tries = 0
    while made < args.count:
        tries += 1
        plan = make_geometry(rng)
        if plan is None:
            continue
        plan = finish(plan, rng)
        if plan is None:
            continue
        plan = d4(plan, rng.random() < 0.5, rng.random() < 0.5, rng.random() < 0.5)
        layout, rep = ls.solve(plan, ck.evaluate)
        if not rep["passed"]:
            continue
        for r in plan["rooms"]:
            r.pop("kind", None)
        plan["name"] = f"{args.prefix}-{args.start + made:02d}"
        (args.out / f"{plan['name']}.json").write_text(json.dumps(plan, indent=1) + "\n")
        made += 1
        print(f"{plan['name']}: ok after {tries} tries  W={plan['building'][2]} H={plan['building'][3]}  "
              f"rejected: {dict(REJECTED)}", flush=True)


if __name__ == "__main__":
    main()
