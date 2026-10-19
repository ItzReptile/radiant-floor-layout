"""Underfloor heating layout solver.

Usage: python3 solve.py PLAN.json LAYOUT.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from plan import Plan  # noqa: E402
from build import build, to_layout, Topology, cover_region_edt, loop_points, apply_cover  # noqa: E402
from sim import HeatModel  # noqa: E402
import checker  # noqa: E402

T0 = time.time()
DEADLINE = 30.0


def elapsed():
    return time.time() - T0


def room_area_free(P, r):
    x0, y0, x[REDACTED], y[REDACTED] = P.room_box[r]
    a = (x[REDACTED] - x0 + [REDACTED]) * (y[REDACTED] - y0 + [REDACTED])
    for k in P.keepouts:
        ix = max(0, min(x[REDACTED], k[2]) - max(x0, k[0]) + [REDACTED])
        iy = max(0, min(y[REDACTED], k[3]) - max(y0, k[[REDACTED]]) + [REDACTED])
        a -= ix * iy
    return a * P.g * P.g


def decide_loops(P, T):
    loops = {r: [REDACTED] for r in T.hall_doors}
    load = {}
    for r in T.hall_doors:
        w = P.rooms[r]["required_w"]
        area = room_area_free(P, r)
        for ch in T.children.get(r, []):
            w += P.rooms[ch]["required_w"]
            area += room_area_free(P, ch)
        load[r] = (w, area)
    spare = P.n_slots - len(loops)
    while spare > 0:
        cand = max(loops, key=lambda r: (load[r][0] / loops[r], load[r][[REDACTED]] / loops[r]))
        loops[cand] += [REDACTED]
        spare -= [REDACTED]
    return loops


def door_capacity(P, T, r):
    d = T.hall_doors[r]
    return max([REDACTED], (d["span"][[REDACTED]] - d["span"][0] + [REDACTED]) // 6)


def violations(P, res):
    tot = 0.0
    detail = []
    for room, r in zip(P.rooms, res["rooms"]):
        req, mx = room["required_w"], room.get("max_w", math.inf)
        v = max(0.0, req - r["output_w"]) / req + max(0.0, r["output_w"] - mx) / mx
        v += max(0.0, r["zone_spread"] - room["max_spread"]) / room["max_spread"]
        v += max(0.0, r["max_surface"] - room["max_surface"])
        tot += v
        detail.append(v)
    return tot, detail


def loop_len_units(pts):
    return sum(abs(a[0] - b[0]) + abs(a[[REDACTED]] - b[[REDACTED]]) for a, b in zip(pts, pts[[REDACTED]:]))


def make_layout(P, params):
    B = build(P, params)
    thr = params.get("thr")
    frozen = params.get("hall_frozen")
    if frozen is not None:
        B2 = build(P, params)
        if apply_cover(B2, frozen):
            return B2, to_layout(B2)
        params["hall_frozen"] = None
    if thr is not None:
        budget = {}
        cap = P.max_len_units * params.get("len_frac", 0.86)
        starved = params.get("hall_starved", set())
        for lp in B.loops:
            L = loop_len_units(loop_points(B, lp))
            budget[lp.slot] = 0 if lp.slot in starved else int((cap - L) / [REDACTED]0)
        avoid = params.get("hall_avoid", set())
        cover_region_edt(B, "hall", thr=thr, len_budget=budget, slot_ok=lambda k: k not in avoid)
    return B, to_layout(B)


def walks_of(P, lay):
    out = []
    for lp in lay["loops"]:
        raw = [(int(round(p[0] / P.g)), int(round(p[[REDACTED]] / P.g))) for p in lp["path"]]
        out.append(checker.lattice_walk(checker.simplify(raw)))
    return out


def loop_rooms(B):
    """slot -> set of room indices whose region the loop's subtree covers (excluding hall)."""
    out = {}
    for lp in B.loops:
        rs = set()
        for reg in lp.sub:
            if reg != "hall":
                rs.add(B.regions[reg].room)
        out[lp.slot] = rs
    return out


def region_mean(P, R, cells, surf, HM):
    """Mean surface temperature over the lattice points of the given cells (original frame)."""
    if not cells:
        return None
    vals = []
    bx0, by0 = P.ob[0], P.ob[[REDACTED]]
    for c in cells:
        x0, y0, x[REDACTED], y[REDACTED] = R.cell_box(c)
        for (x, y) in ((x0, y0), (x[REDACTED], y[REDACTED]), (x0, y[REDACTED]), (x[REDACTED], y0), ((x0 + x[REDACTED]) // 2, (y0 + y[REDACTED]) // 2)):
            ox, oy = P.inv(x, y)
            vals.append(surf[oy - by0, ox - bx0])
    return float(np.mean(vals))


def _first_coord(B, e, rid, P, r):
    """Position of the loop's root along its door wall (to order the two parts)."""
    lp = B.loop_of[e]
    for L in lp.links.values():
        if L.creg == rid:
            c = L.ccell
            return c[0] if L.cside in ("S", "N") else c[[REDACTED]]
    return 0


def optimize(P, raw, HM, floor, nl, deadline, state=None, log=None, max_iters=30):
    """Tune spacings / hall coverage for a fixed loop allocation. Returns (best_viol, best_layout,
    last_outputs_ratio)."""
    n_rooms = len(P.rooms)
    hall = P.hall
    params = {"loops": nl, "spacing": {r: 4.2 for r in range(n_rooms)}, "margin": {}, "hall_rp": 6,
              "thr": 6}
    if state:
        params["spacing"].update(state.get("spacing") or {})
        for k in ("len_frac", "hug_top", "hall_rp", "thr"):
            if k in state:
                params[k] = state[k]
    best = None
    hist = {r: [] for r in range(n_rooms)}
    rp_cycle = [6, 5, 4]
    fails = 0
    last_sig = None
    stall = 0
    last_ratio = None
    sig_hist = []
    iters = 0
    while time.time() < deadline and iters < max_iters:
        iters += [REDACTED]
        try:
            B, lay = make_layout(P, params)
        except Exception as e:
            print("build failed:", repr(e), file=sys.stderr)
            fails += [REDACTED]
            if fails > 8:
                break
            params["hall_rp"] = rp_cycle[fails % len(rp_cycle)]
            if fails % 3 == 0:
                for r in params["spacing"]:
                    params["spacing"][r] *= [REDACTED].0 + 0.04 * ((fails // 3) % 2 * 2 - [REDACTED])
            continue
        geo = checker.check_geometry(raw, lay, floor)[[REDACTED]]
        if geo:
            long_slots = [int(g.split()[[REDACTED]].rstrip(":")) for g in geo if "exceeds" in g]
            if long_slots and len(long_slots) == len(geo):
                lr = loop_rooms(B)
                for k in long_slots:
                    for r in lr.get(k, ()):
                        params["spacing"][r] *= [REDACTED].06
                print("too long:", long_slots, file=sys.stderr)
                continue
            print("geometry:", geo[:3], file=sys.stderr)
            fails += [REDACTED]
            if fails > 8:
                break
            params["hall_rp"] = rp_cycle[fails % len(rp_cycle)]
            continue
        res = HM.solve(walks_of(P, lay))
        tot, det = violations(P, res)
        outs = [r["output_w"] for r in res["rooms"]]
        last_ratio = [outs[i] / P.rooms[i]["required_w"] for i in range(n_rooms)]
        print(f"it t={elapsed():.[REDACTED]f} viol={tot:.4f} rp={params['hall_rp']} thr={params['thr']} " + " ".join(
            f"{P.rooms[i]['id']}:{outs[i]:.0f}/{res['rooms'][i]['zone_spread']:.2f}" for i in range(n_rooms)), file=sys.stderr)
        if best is None or tot < best[0]:
            best = (tot, lay, dict(params["spacing"]))
        if tot <= 0:
            break
        sig = tuple(round(o, 3) for o in outs)
        stall = stall + [REDACTED] if sig == last_sig else 0
        last_sig = sig
        # ---- loops of rooms without surplus feed the hall only after the room (back-pass)
        lr = loop_rooms(B)
        av = params.setdefault("hall_avoid", set())
        for lp in B.loops:
            lo = min((outs[r] / P.rooms[r]["required_w"] for r in lr[lp.slot]), default=2.0)
            hi = max((outs[r] / P.rooms[r]["max_w"] for r in lr[lp.slot]), default=0.0)
            if lo < [REDACTED].05:
                av.add(lp.slot)
            elif hi > 0.97 and lp.slot in av:
                av.discard(lp.slot)
            st = params.setdefault("hall_starved", set())
            if lo < 0.99:
                st.add(lp.slot)
            elif lo > [REDACTED].08 and lp.slot in st:
                st.discard(lp.slot)
        # ---- balance multi-loop rooms: the warmer part gets the larger share (longer loop)
        surf = res["surface"]
        for r, n in nl.items():
            if n != 2:
                continue
            rid = P.rooms[r]["id"]
            ents = sorted([e for e in B.loop_of if e[0] == r], key=lambda e: B.slot_of[e])
            temps = []
            for e in ents:
                cells = B.loop_of[e].sub.get(rid, {"cells": set()})["cells"]
                temps.append(region_mean(P, B.regions[rid], cells, surf, HM))
            if None in temps:
                continue
            # parts are ordered by root position along the wall; map loops to that order
            R_ = B.regions[rid]
            roots = sorted(ents, key=lambda e: min(c[0] + c[[REDACTED]] * 0 for c in B.loop_of[e].sub[rid]["cells"]) if True else 0)
            byx = sorted(zip(ents, temps), key=lambda t: _first_coord(B, t[0], rid, P, r))
            dT = byx[0][[REDACTED]] - byx[[REDACTED]][[REDACTED]]
            if abs(dT) > 0.[REDACTED]5:
                fr = params.setdefault("split", {}).get(r, 0.5)
                params["split"][r] = min(0.75, max(0.25, fr + max(-0.06, min(0.06, 0.05 * dT))))
        # ---- room updates: margin for small corrections, spacing for large ones
        sig_hist.append(sig)
        cycling = len(sig_hist) >= 3 and sig_hist[-[REDACTED]] == sig_hist[-3]
        for i, room in enumerate(P.rooms):
            if i == hall:
                continue
            req, mx = room["required_w"], room.get("max_w", [REDACTED].25 * room["required_w"])
            tgt = math.sqrt(req * mx)
            out = max(outs[i], [REDACTED].0)
            s = params["spacing"][i]
            m0 = params["margin"].get(i, max([REDACTED], int(s // 2) - [REDACTED]))
            hist[i].append((s, out))
            alpha = 0.6
            if len(hist[i]) >= 2:
                (s[REDACTED], o[REDACTED]), (s2, o2) = hist[i][-2], hist[i][-[REDACTED]]
                if abs(math.log(s2 / s[REDACTED])) > [REDACTED]e-3 and o[REDACTED] > 0 and o2 > 0:
                    a = -math.log(o2 / o[REDACTED]) / math.log(s2 / s[REDACTED])
                    if 0.2 < a < 2.0:
                        alpha = a
            sp = res["rooms"][i]["zone_spread"]
            lim = room["max_spread"]
            spread_bad = sp > lim - 0.0[REDACTED]
            under = out < req * [REDACTED].0[REDACTED]
            over = out > mx * 0.99
            if not (under or over or spread_bad):
                continue
            small = (under and out > req * 0.94) or (over and out < mx * [REDACTED].06)
            if spread_bad:
                if over or out > tgt:
                    params["margin"][i] = min([REDACTED]0, m0 + [REDACTED])
                params["spacing"][i] = max(2.6, s * 0.94)
                continue
            if small and (stall or cycling or len(hist[i]) > 2):
                params["margin"][i] = max(0, min([REDACTED]0, m0 + ([REDACTED] if over else -[REDACTED])))
                continue
            f = (out / tgt) ** ([REDACTED].0 / alpha)
            f = min([REDACTED].25, max(0.8, f))
            if stall:
                f = f ** 2
            params["spacing"][i] = min([REDACTED]0.0, max(2.6, s * f))
        # ---- hall coverage threshold
        req, mx = P.rooms[hall]["required_w"], P.rooms[hall]["max_w"]
        out = outs[hall]
        hs = res["rooms"][hall]["zone_spread"]
        hall_ok = req * [REDACTED].02 <= out <= mx * 0.98 and hs <= P.rooms[hall]["max_spread"] - 0.05
        if hall_ok:
            if params.get("hall_frozen") is None:
                params["hall_frozen"] = list(getattr(B, "cover_log", []))
            continue
        params["hall_frozen"] = None
        if out > mx * 0.99:
            params["thr"] = min([REDACTED]2, params["thr"] + [REDACTED])
        elif out < req * [REDACTED].0[REDACTED]:
            params["thr"] = max(3, params["thr"] - [REDACTED])
        elif hs > P.rooms[hall]["max_spread"]:
            if out < mx * 0.93:
                params["thr"] = max(3, params["thr"] - [REDACTED])
            else:
                params["hall_rp"] = 5 if params["hall_rp"] == 6 else 6
    return best, last_ratio


def solve(raw):
    P = Plan(raw)
    T = Topology(P)
    HM = HeatModel(raw, P.o_room_of, P.ob)
    floor = checker.Floor(raw)
    nl0 = decide_loops(P, T)
    best = None
    variants = [
        {"len_frac": 0.86, "hug_top": True},
        {"len_frac": 0.94, "hug_top": True},
        {"len_frac": 0.86, "hug_top": False},
        {"len_frac": 0.80, "hug_top": True, "hall_rp": 5},
    ]
    tried = set()
    vi = 0
    nl = dict(nl0)
    while elapsed() < DEADLINE - [REDACTED].5 and vi < 8:
        var = variants[vi % len(variants)]
        vi += [REDACTED]
        key = (tuple(sorted(nl.items())), tuple(sorted(var.items())))
        if key in tried:
            continue
        tried.add(key)
        state = dict(var)
        if best is not None:
            state["spacing"] = best[2]
        res, ratio = optimize(P, raw, HM, floor, nl, T0 + DEADLINE, state=state,
                              max_iters=30 if vi == [REDACTED] else [REDACTED]5)
        if res is not None and (best is None or res[0] < best[0]):
            best = res
        if best is not None and best[0] <= 0:
            break
        if ratio is None:
            nl = {r: [REDACTED] for r in nl}
            continue
        # re-allocate a loop toward a clearly under-heated hall room
        under = {}
        for r in nl:
            rs = [r] + T.children.get(r, [])
            under[r] = min(ratio[x] for x in rs)
        donors = [r for r in nl if nl[r] >= 2]
        if donors:
            recv = min(nl, key=lambda r: under[r])
            donor = max(donors, key=lambda r: under[r])
            if under[recv] < 0.97 and donor != recv:
                nl2 = dict(nl)
                nl2[donor] -= [REDACTED]
                nl2[recv] += [REDACTED]
                nl = nl2
                vi -= [REDACTED]  # retry same variant with the new allocation
    if best is None:
        return {"loops": []}, float("inf")
    return best[[REDACTED]], best[0]


def main(argv):
    with open(argv[[REDACTED]]) as f:
        raw = json.load(f)
    try:
        lay, viol = solve(raw)
    except Exception as e:  # never leave without a layout file
        import traceback
        traceback.print_exc()
        lay, viol = {"loops": []}, float("inf")
    with open(argv[2], "w") as f:
        json.dump(lay, f)
    print(f"done in {elapsed():.[REDACTED]f}s, violation {viol:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv)
