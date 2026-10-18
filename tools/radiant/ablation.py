"""Baselines for radiant-floor-layout, built from the reference solver with one idea removed.

  uniform   series-capable counterflow with one pitch per room (perimeter pitch = core pitch)
  serp      naive single-pass serpentine in every room; the series zone (bedroom + study) keeps
            the reference's counterflow coverage, so this baseline can still reach the study
  north     reference rooms, hall leaders packed at minimum spacing against the manifold wall,
            never re-spread (where shortest Manhattan routes from the ports end up)
  south     the same, packed against the opposite hall wall

Every baseline gets a larger tuning budget than the reference (40 evaluations instead of 12).

Usage:
  python ablation.py ref PLAN...                    the reference solver itself
  python ablation.py run MODE PLAN... [--json OUT]  MODE: uniform | serp | north | south
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1] / "tasks" / "radiant-floor-layout"
sys.path.insert(0, str(TASK / "environment" / "app"))
sys.path.insert(0, str(TASK / "solution" / "solver"))
import check_layout as ck  # noqa: E402
import layout_solver as ls  # noqa: E402

G = 0.05
UNIFORM = [(p, p) for p in (0.10, 0.15, 0.20, 0.25, 0.30)]


def r4(v):
    return round(v, 4)


# ----------------------------------------------------------------- naive serpentine


def point_ok(x, y, zone, plan):
    b = ls.inner_bounds(zone.rect, plan)
    if not (b[0] - 1e-9 <= x <= b[2] + 1e-9 and b[1] - 1e-9 <= y <= b[3] + 1e-9):
        return False
    c = ls.CLEAR
    for k in plan["keepouts"]:
        if k[0] - c + 1e-9 < x < k[2] + c - 1e-9 and k[1] - c + 1e-9 < y < k[3] + c - 1e-9:
            return False
    return True


def serp_fill(zone, plan, max_length):
    """Naive serpentine (points from the left door point to the right one): the supply runs
    along the door row to one side wall, the pipe sweeps the room in columns perpendicular to
    the door wall (stopping at keep-outs and the far wall), and the return comes back along the
    door row from the other side wall. The bedroom + study zone keeps counterflow coverage."""
    if zone.extra:
        return ls.zone_path(zone, plan, max_length)
    p = zone.combo[0]
    away = -1 if zone.side == "S" else 1
    y0 = ls.door_row_y(zone)
    b = ls.inner_bounds(zone.rect, plan)
    lo, hi = b[0], b[2]

    def Y(j):
        return r4(y0 + away * j * p)

    def ok(x, y):
        return point_ok(x, y, zone, plan)

    def reach_of(cx):
        e = 0
        while ok(cx, Y(e + 1)) and ls.box_clear(cx, Y(1), cx, Y(e + 1), plan):
            e += 1
        return e

    d0, d1 = zone.door[0] + 0.05, zone.door[2] - 0.05
    if zone.kind == "west":
        d1 = min(d1, zone.rect[2] - ls.CLEAR)
    if zone.kind == "east":
        d0 = max(d0, zone.rect[0] + ls.CLEAR)
    cands = [r4(d0 + i * G) for i in range(int(round((d1 - d0) / G)) + 1)]
    mid = (zone.door[0] + zone.door[2]) / 2
    sup_port, ret_port = getattr(zone, "port_xs", (None, None))
    if zone.side == "N":
        sup_port, ret_port = ret_port, sup_port

    def jog_ok(x, port):
        return port is None or abs(x - port) < 1e-9 or abs(x - port) >= 0.10 - 1e-9

    def build(cols, reach):
        n = len(cols)
        A, B = cols[0], cols[-1]
        best = None
        for xs in cands:
            if not (abs(xs - A) < 1e-9 or xs - A >= 0.10 - 1e-9) or not jog_ok(xs, sup_port):
                continue
            for xr in cands:
                if xr < xs + 0.10 - 1e-9 or xr > B + 1e-9 or not jog_ok(xr, ret_port):
                    continue
                if not (abs(xr - B) < 1e-9 or B - xr >= 0.10 - 1e-9):
                    continue
                score = abs((xs + xr) / 2 - mid)
                if best is None or score < best[0]:
                    best = (score, xs, xr)
        if best is None:
            return None
        _, xs, xr = best
        pts = [(xs, Y(0)), (A, Y(0))]
        for k in range(0, n, 2):
            d = min(reach[k], reach[k + 1])
            if d < 2 and k > 0:
                return None
            d = max(d, 1)
            while d >= 1 and not ls.box_clear(cols[k], Y(d), cols[k + 1], Y(d), plan):
                d -= 1
            if d < 1:
                return None
            base = Y(0) if k == 0 else Y(1)
            pts += [(cols[k], base), (cols[k], Y(d)), (cols[k + 1], Y(d))]
            pts += [(cols[k + 1], Y(1))] if k + 2 < n else [(cols[k + 1], Y(0))]
        pts.append((xr, Y(0)))
        out = []
        for q in pts:
            if not out or q != out[-1]:
                out.append(q)
        return out if ls.path_length(out) <= max_length else None

    options = []
    for off in range(int(round(p / G))):
        cols = []
        x = lo + off * G
        while x <= hi + 1e-9:
            cols.append(r4(x))
            x += p
        if len(cols) < 2 or not all(ok(cx, Y(0)) for cx in cols):
            continue
        reach = [reach_of(cx) for cx in cols]
        for n in range(len(cols) - len(cols) % 2, 1, -2):
            for s0 in range(0, len(cols) - n + 1):
                options.append((-(cols[s0 + n - 1] - cols[s0]), off, s0, n, cols, reach))
    options.sort(key=lambda o: (o[0], o[1], o[2]))
    for _, off, s0, n, cols, reach in options:
        path = build(cols[s0:s0 + n], reach[s0:s0 + n])
        if path is not None:
            return path
    raise ValueError("serpentine does not fit")


# ----------------------------------------------------------------- runs


MODES = {
    "uniform": dict(combos=UNIFORM, sweep=UNIFORM[1:]),
    "serp": dict(combos=UNIFORM, sweep=UNIFORM[1:], fill=serp_fill),
    "north": dict(lanes="north", respread=False),
    "south": dict(lanes="south", respread=False),
}


def solve_mode(canon, mode, max_evals=40, time_budget=240.0):
    try:
        return ls.solve_canonical(canon, ck.evaluate, max_evals=max_evals, time_budget=time_budget, **MODES[mode])
    except (ValueError, RuntimeError):
        return None, None


def summarize(plan, rep):
    out = []
    for room, res in zip(plan["rooms"], rep["rooms"]):
        bad = [k for k, v in res["checks"].items() if not v]
        out.append({"id": room["id"], "w": round(res["output_w"]), "win": [room["required_w"], room["max_w"]],
                    "peak": round(res["max_surface"], 2), "peak_lim": room["max_surface"],
                    "spread": round(res["zone_spread"], 2), "spread_lim": room["max_spread"], "fail": bad})
    return out


def run_one(args):
    path, mode = args
    plan = json.loads(Path(path).read_text())
    t = ls.canonical_transform(plan)
    canon = ls.transform_plan(plan, t)
    layout, rep = solve_mode(canon, mode)
    if rep is None or layout is None:
        return {"plan": Path(path).stem, "passed": False, "note": "no valid layout"}
    if rep["geometry_problems"]:
        return {"plan": Path(path).stem, "passed": False, "note": "geometry: " + rep["geometry_problems"][0]}
    return {"plan": Path(path).stem, "passed": bool(rep["passed"]), "rooms": summarize(canon, rep),
            "layout": {"loops": [{"slot": lp["slot"], "path": [list(ls._tf_inv_point(t, x, y)) for x, y in lp["path"]]}
                                 for lp in layout["loops"]]}}


def ref_one(path):
    plan = json.loads(Path(path).read_text())
    _, rep = ls.solve(plan, ck.evaluate)
    return Path(path).stem, bool(rep["passed"])


def main():
    cmd = sys.argv[1]
    if cmd == "ref":
        with ProcessPoolExecutor(max_workers=8) as ex:
            res = list(ex.map(ref_one, sys.argv[2:]))
        for name, ok in res:
            print(f"reference {name}: {'PASS' if ok else 'fail'}")
        print(f"reference: {sum(ok for _, ok in res)}/{len(res)} passed")
        return
    mode = sys.argv[2]
    args = sys.argv[3:]
    out = None
    if "--json" in args:
        i = args.index("--json")
        out = Path(args[i + 1])
        args = args[:i] + args[i + 2:]
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(run_one, [(a, mode) for a in args]))
    for r in results:
        fails = "; ".join(f"{q['id']}:{','.join(q['fail'])}" + (f" ({q['spread']}/{q['spread_lim']})" if "max_spread" in q["fail"] else "")
                          for q in r.get("rooms", []) if q["fail"])
        print(f"{mode} {r['plan']}: {'PASS' if r['passed'] else 'fail'} {fails or r.get('note', '')}")
    print(f"{mode}: {sum(r['passed'] for r in results)}/{len(results)} passed")
    if out:
        out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
