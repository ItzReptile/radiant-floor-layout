"""Ablations and a naive baseline for radiant-floor-layout.

Room fill:
  cf    the reference's counterflow spanning-tree coverage
  serp  a naive single-pass serpentine: the supply runs along the door row to one side wall,
        the pipe sweeps the room in columns perpendicular to the door wall (columns stop at
        keep-outs and the far wall), and the return comes back along the door row from the
        other side wall
Hall leaders:
  spread  the reference's lanes, spread over the hall's width
  north   every leader packed at minimum spacing against the manifold wall (where shortest
          Manhattan routes from the ports end up)
  south   every leader packed at minimum spacing against the opposite hall wall

Every variant gets a stronger pitch tuner than the reference: pitches 0.10-0.40, a uniform
sweep, per-room best pick, then greedy per-room coordinate descent against the checker.

Usage:
  python ablation.py table PLAN...                    per-room serpentine vs counterflow table
  python ablation.py ref PLAN...                      the reference solver itself
  python ablation.py run FILL LANES PLAN... [--json OUT]
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1] / "tasks" / "radiant-floor-layout"
sys.path.insert(0, str(TASK / "environment" / "app"))
sys.path.insert(0, str(TASK / "solution" / "solver"))
import check_layout as ck  # noqa: E402
import layout_solver as ls  # noqa: E402

PITCHES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
G = 0.05


def r4(v):
    return round(v, 4)


# ----------------------------------------------------------------- naive serpentine


def zone_bounds(zone, plan):
    x0, y0, x1, y1 = zone.rect
    bx0, by0, bx1, by1 = plan["building"]
    em = plan["edge_margin"]
    lo = bx0 + em if abs(x0 - bx0) < 1e-9 else x0 + ls.CLEAR
    hi = bx1 - em if abs(x1 - bx1) < 1e-9 else x1 - ls.CLEAR
    return ls.snap_up(lo), ls.snap_down(hi)


def serp_layout(zone, plan, max_length):
    """Points from the left door point to the right one (same convention as ls.zone_path)."""
    p = zone.pitch
    away = -1 if zone.side == "S" else 1
    y0 = ls.door_row_y(zone)
    lo, hi = zone_bounds(zone, plan)

    def Y(j):
        return r4(y0 + away * j * p)

    def ok(x, y):
        return ls.point_ok(x, y, zone, plan)

    def reach_of(cx):
        """Rows 1..e of a column away from the door: all points ok, clear of keep-outs."""
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

    def build(cols, reach):
        """Path sweeping the given columns (even count), or None."""
        n = len(cols)
        A, B = cols[0], cols[-1]
        best = None
        def jog_ok(x, port):
            # the leader from the manifold port must not make a jog shorter than a straight piece
            return port is None or abs(x - port) < 1e-9 or abs(x - port) >= 0.10 - 1e-9

        sup_port, ret_port = getattr(zone, "port_xs", (None, None))
        if zone.side == "N":
            sup_port, ret_port = ret_port, sup_port   # the path is reversed for north rooms
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
        length = sum(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(out, out[1:]))
        return out if length <= max_length else None

    # try every column offset and every even window of columns, widest coverage first
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


def serp_place_entry(zone, plan):
    lo, _ = zone_bounds(zone, plan)
    zone.entry_left = max(lo, zone.door[0] + 0.05)
    return True


# ----------------------------------------------------------------- layout with lane choice


def build(plan, zones, hall_idx, fill, lanes):
    hall = plan["rooms"][hall_idx]
    pp = plan["pipe"]["port_pitch"]
    ox, oy = plan["manifold"]["origin"]
    for z in zones:
        if fill == "cf":
            if not ls.place_entry(z, plan):
                raise ValueError("zone does not fit its doorway")
        else:
            serp_place_entry(z, plan)
    ordered = ls.fan_order(plan, zones)
    n = len(ordered)
    y_high = ls.snap_down(oy - 0.15)
    y_low = ls.snap_up(hall["rect"][1] + 0.15)
    ports_x = [r4(ox + 2 * k * pp) for k in range(n)]
    serp_paths = {}
    if fill == "serp":
        # lay the serpentine first, so the leaders are grouped by where the supply really enters
        hall_h = hall["rect"][3] - hall["rect"][1]
        for k, z in enumerate(ordered):
            z.port_xs = (ports_x[k], r4(ports_x[k] + pp))
            est = 2 * abs(z.door[0] - ports_x[k]) + 2 * hall_h + 2.0
            path = serp_layout(z, plan, plan["pipe"]["max_loop_length"] - 0.5 - est)
            z.entry_left = (path if z.side == "S" else path[::-1])[0][0]
            serp_paths[k] = path
    west = [k for k in range(n) if ordered[k].entry_left < ports_x[k]]
    east = [k for k in range(n) if k not in west]
    tracks = {}
    for order in (west, east[::-1]):
        m = len(order)
        if m == 0:
            continue
        if lanes == "spread":
            step = max((y_high - y_low - 0.10) / (m - 1) if m > 1 else 0.0, 0.20)
            for rank, k in enumerate(order):
                tracks[k] = ls.snap_down(y_high - rank * step)
        elif lanes == "north":
            for rank, k in enumerate(order):
                tracks[k] = r4(y_high - rank * 0.20)
        else:
            for rank, k in enumerate(order):
                tracks[k] = r4(y_low + 0.10 + (m - 1 - rank) * 0.20)
    loops = []
    for k, z in enumerate(ordered):
        sx, rx = ports_x[k], r4(ports_x[k] + pp)
        hall_len = abs(oy - tracks[k]) * 2 + abs(z.entry_left - sx) * 2 + 2.0
        budget = plan["pipe"]["max_loop_length"] - 0.5 - hall_len
        path = ls.zone_path(z, plan, budget) if fill == "cf" else serp_paths[k]
        y_row = path[0][1]
        t_upper = tracks[k]
        t_lower = r4(t_upper - 0.10)
        y_s, y_r = (t_upper, t_lower) if k in west else (t_lower, t_upper)
        inner = path if z.side == "S" else path[::-1]
        x_sup, x_ret = inner[0][0], inner[-1][0]
        pts = [(sx, oy), (sx, y_s), (x_sup, y_s), (x_sup, y_row)]
        pts += inner[1:-1]
        pts += [(x_ret, y_row), (x_ret, y_r), (rx, y_r), (rx, oy)]
        loops.append({"slot": k, "path": [[r4(x), r4(y)] for x, y in pts], "zone": z})
    return loops


def layout_of(loops):
    return {"loops": [{"slot": lp["slot"], "path": lp["path"]} for lp in loops]}


def try_eval(plan, zones, hi, fill, lanes):
    try:
        loops = build(plan, zones, hi, fill, lanes)
    except (ValueError, RuntimeError):
        return None, None
    layout = layout_of(loops)
    rep = ck.evaluate(plan, layout)
    if rep["geometry_problems"]:
        return None, rep
    return (loops, layout), rep


def total_score(plan, rep):
    return sum(ls.room_score(room, res) for room, res in zip(plan["rooms"], rep["rooms"]))


def tune(canon, fill, lanes, max_evals=60):
    """Uniform sweep, per-room best pick, then greedy coordinate descent."""
    zones, hi = ls.make_zones(canon)
    rooms = canon["rooms"]
    sweep = {}
    evals = 0
    for p in PITCHES:
        for z in zones:
            z.pitch = p
        got, rep = try_eval(canon, zones, hi, fill, lanes)
        evals += 1
        if got:
            sweep[p] = {r["id"]: r for r in rep["rooms"]}
    if not sweep:
        return None, None, evals
    choice = [min(sweep, key=lambda p: ls.room_score(rooms[z.room_idx], sweep[p][rooms[z.room_idx]["id"]]))
              for z in zones]

    def run(ch):
        for z, p in zip(zones, ch):
            z.pitch = p
        got, rep = try_eval(canon, zones, hi, fill, lanes)
        if not got:
            return 1e18, None, rep
        return total_score(canon, rep), got[1], rep

    best = run(choice)
    evals += 1
    improved = True
    while improved and evals < max_evals and not (best[2] and best[2]["passed"]):
        improved = False
        for zi in range(len(zones)):
            for step in (-1, 1):
                j = PITCHES.index(choice[zi]) + step
                if not 0 <= j < len(PITCHES):
                    continue
                trial = list(choice)
                trial[zi] = PITCHES[j]
                cand = run(trial)
                evals += 1
                if cand[0] < best[0]:
                    best, choice, improved = cand, trial, True
                    if best[2]["passed"]:
                        break
            if best[2] and best[2]["passed"]:
                break
    return best[1], best[2], evals


def summarize(canon, rep):
    out = []
    for room, res in zip(canon["rooms"], rep["rooms"]):
        bad = [k for k, v in res["checks"].items() if not v]
        out.append({"id": room["id"], "w": round(res["output_w"]), "win": [room["required_w"], room["max_w"]],
                    "peak": round(res["max_surface"], 2), "peak_lim": room["max_surface"],
                    "spread": round(res["zone_spread"], 2), "spread_lim": room["max_spread"], "fail": bad})
    return out


def run_one(args):
    path, fill, lanes = args
    plan = json.loads(Path(path).read_text())
    t = ls.canonical_transform(plan)
    canon = ls.transform_plan(plan, t)
    layout, rep, evals = tune(canon, fill, lanes)
    if rep is None or layout is None:
        return {"plan": Path(path).stem, "passed": False, "note": "no valid layout", "evals": evals}
    return {"plan": Path(path).stem, "passed": bool(rep["passed"]), "evals": evals, "rooms": summarize(canon, rep),
            "layout": {"loops": [{"slot": lp["slot"], "path": [list(ls._tf_inv_point(t, x, y)) for x, y in lp["path"]]}
                                 for lp in layout["loops"]]}}


# ----------------------------------------------------------------- per-room table


def table_one(path):
    plan = json.loads(Path(path).read_text())
    for key, val in json.loads(os.environ.get("PHYS", "{}")).items():
        plan["physics"][key] = val
    canon = ls.transform_plan(plan, ls.canonical_transform(plan))
    rooms = canon["rooms"]
    lines = []
    for p in [0.10, 0.15, 0.20, 0.25, 0.30]:
        per = {}
        for fill in ("cf", "serp"):
            zones, hi = ls.make_zones(canon)
            for z in zones:
                z.pitch = p
            got, rep = try_eval(canon, zones, hi, fill, "spread")
            if not got:
                per[fill] = None
                continue
            loops = got[0]
            dts = {}
            for lp, lr in zip(loops, sorted(rep["loops"], key=lambda q: q["slot"])):
                rid = rooms[lp["zone"].room_idx]["id"]
                dts.setdefault(rid, []).append(40.0 - lr["return_temp"])
            per[fill] = (rep, dts)
        for ri, room in enumerate(rooms):
            if room["id"] == "hall":
                continue
            cells = []
            for fill in ("cf", "serp"):
                if per[fill] is None:
                    cells.append(f"{fill:>4}: {'(does not fit)':<46}")
                    continue
                rep, dts = per[fill]
                res = rep["rooms"][ri]
                ok_w = res["checks"]["output_min"] and res["checks"]["output_max"]
                ok_s = res["checks"]["max_spread"]
                ok_all = all(res["checks"].values())
                dt = "/".join(f"{d:.1f}" for d in dts.get(room["id"], []))
                cells.append(f"{fill:>4}: {res['output_w']:5.0f}W {'in ' if ok_w else 'OUT'} dTw {dt:>9} "
                             f"spread {res['zone_spread']:4.2f}{'' if ok_s else '!'} {'PASS' if ok_all else '    '}")
            lines.append(f"{Path(path).stem} p={p:.2f} {room['id']:>8} [{room['required_w']},{room['max_w']}] "
                         f"lim {room['max_spread']}: " + " | ".join(cells))
    return "\n".join(lines)


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
    if cmd == "table":
        with ProcessPoolExecutor(max_workers=8) as ex:
            for text in ex.map(table_one, sys.argv[2:]):
                print(text, flush=True)
        return
    fill, lanes = sys.argv[2], sys.argv[3]
    args = sys.argv[4:]
    out = None
    if "--json" in args:
        i = args.index("--json")
        out = Path(args[i + 1])
        args = args[:i] + args[i + 2:]
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(run_one, [(a, fill, lanes) for a in args]))
    n_pass = 0
    for r in results:
        n_pass += r["passed"]
        fails = "; ".join(f"{q['id']}:{','.join(q['fail'])}" for q in r.get("rooms", []) if q["fail"])
        print(f"{fill}/{lanes} {r['plan']}: {'PASS' if r['passed'] else 'fail'} ({r['evals']} evals) {fails or r.get('note', '')}")
    print(f"{fill}/{lanes}: {n_pass}/{len(results)} passed")
    if out:
        out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
