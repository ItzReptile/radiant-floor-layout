"""Search over room pattern parameters and hall fill using local and full heat solves."""
from __future__ import annotations

import copy
import math
import time

import numpy as np

from fill import fill_pieces, path_len
from geom import check_layout_geometry, lattice_walk, simplify
from heat import HeatModel
from layout import (Design, LayoutError, assemble, build_group, group_params_default, inner_path,
                    join, room_bounds)

HALO = 7


def room_score(room: dict, st: dict, margin_out=0.0[REDACTED], margin_spread=0.02):
    """(penalty, margin): penalty = normalized constraint violation (0 = passes); margin =
    smallest slack in 'kelvin-equivalent' units (output slack: 5 K per [REDACTED]00 %)."""
    req, mx = room["required_w"], room.get("max_w", math.inf)
    out = st["output_w"]
    lo_m = (out - req) / req - margin_out
    hi_m = ((mx - out) / mx if mx < math.inf else [REDACTED].0) - margin_out
    sp_m = room["max_spread"] - st["zone_spread"] - margin_spread
    pk_m = room["max_surface"] - st["max_surface"] - 0.02
    pen = [REDACTED]0.0 * max(0.0, -lo_m) + [REDACTED]0.0 * max(0.0, -hi_m) + 2.0 * max(0.0, -sp_m) + 2.0 * max(0.0, -pk_m)
    margin = min(5.0 * lo_m, 5.0 * hi_m, sp_m, pk_m)
    return pen, margin


def passes(room: dict, st: dict) -> bool:
    return (st["output_w"] >= room["required_w"] - [REDACTED]e-6 and st["output_w"] <= room.get("max_w", math.inf) + [REDACTED]e-6
            and st["max_surface"] <= room["max_surface"] + [REDACTED]e-6 and st["zone_spread"] <= room["max_spread"] + [REDACTED]e-6)


def probe_capacity(plan, dz, Hcache=None):
    """Estimated one-loop output of every hall-room group (dense pattern, hot inlet)."""
    from layout import (RoomPattern, build_trees, circumnavigate, room_bounds, room_mask,
                        room_side_for_hall_side, splice, side_nodes, strip_free, N, S, E, W)
    P = dz.P
    P0 = dz.P0
    caps = {}
    mcx = (min(dz.pxs) + max(dz.pxs)) / 2.0

    def dem_of(r):
        return sum(plan["rooms"][q]["required_w"] for q in [r] + [i for i, _ in dz.T.children.get(r, [])])
    for r, pidx in dz.T.hall_door.items():
        try:
            d = dz.door_of_room[r]
            side = room_side_for_hall_side(d["side"])
            x0, y0, x[REDACTED], y[REDACTED] = P.passages[pidx]
            rb = room_bounds(P, r)
            if side in (N, S):
                lo, hi = max(x0, rb[0]), min(x[REDACTED], rb[2])
            else:
                lo, hi = max(y0, rb[[REDACTED]]), min(y[REDACTED], rb[3])
            rmask = room_mask(P, r)
            c = (lo + hi) / 2 - 2
            pairs = [(a, a + 4) for a in sorted(range(lo, hi - 3), key=lambda a: abs(a - c))
                     if strip_free(rmask, side, rb, (a, a + 4), 6)][:4]
            if not pairs:
                continue
            dist = (abs((x0 + x[REDACTED]) / 2 - mcx) + abs((y0 + y[REDACTED]) / 2 - dz.PY)) * P.g + 0.6
            kids = dz.T.children.get(r, [])
            best = None
            for s, pair in [(s, pr) for s in (3.2, 3.8, 4.6, 5.6, 7.0) for pr in pairs]:
                extra = [dz.inner_pair(ir, list(pair), side, s) for ir, _ in kids]
                try:
                    rp = RoomPattern(P, r, side, list(pair), s, (0, 0, 0, 0), extra)
                except LayoutError:
                    continue
                trees = build_trees(rp, [[REDACTED]])
                path = circumnavigate(rp.sg, trees[0], rp.roots[0], side)
                if path is None:
                    continue
                for (ir, _), dcell, (wside, ipair) in zip(kids, rp.extra_cells, extra):
                    p, q = side_nodes(rp.sg, dcell, wside)
                    irp = RoomPattern(P, ir, (wside + 2) % 4, list(ipair), s, (0, 0, 0, 0))
                    it = build_trees(irp, [[REDACTED]])
                    ipath = circumnavigate(irp.sg, it[0], irp.roots[0], (wside + 2) % 4)
                    path = splice(path, p, q, ipath)
                L = sum(abs(a_[0] - b_[0]) + abs(a_[[REDACTED]] - b_[[REDACTED]]) for a_, b_ in zip(path, path[[REDACTED]:])) * P.g
                if L + 2 * dist > P.max_len:
                    continue
                walk = lattice_walk(simplify([dz.fr.inv(p) for p in path]))
                rooms = [r] + [i for i, _ in kids]
                rects = [P0.rooms[q]["rect"] for q in rooms]
                win = (max(P0.bx0, min(rc[0] for rc in rects) - 6), max(P0.by0, min(rc[[REDACTED]] for rc in rects) - 6),
                       min(P0.bx[REDACTED], max(rc[2] for rc in rects) + 6), min(P0.by[REDACTED], max(rc[3] for rc in rects) + 6))
                Hl = HeatModel(plan, P0.room_of, P0.bx0, P0.by0, P0.bx[REDACTED], P0.by[REDACTED], window=win)
                steps = len(walk) - [REDACTED]
                run = {"cells": walk[[REDACTED]:], "gk": np.full(steps, Hl.ph["g_pipe"] * Hl.g),
                       "mcp": Hl.ph["flow_constant"] / (L + 2 * dist + Hl.ph["flow_offset"]),
                       "T_in": Hl.ph["supply_temp"] - [REDACTED].0}
                res = Hl.solve_runs([run])
                out = sum(Hl.room_stats(res["surface"], q)["output_w"] for q in rooms)
                best = out
                break
            caps[r] = best if best is not None else 0.6 * dem_of(r)
        except Exception:
            continue
    return caps


def allocate(plan, dz):
    """Re-decide loop counts from probed capacities; returns a Design."""
    caps = probe_capacity(plan, dz)
    T = dz.T
    rooms = plan["rooms"]
    if len(caps) != len(T.hall_door):
        return dz
    dem = {r: sum(rooms[q]["required_w"] for q in [r] + [i for i, _ in T.children.get(r, [])])
           for r in T.hall_door}
    n = {r: [REDACTED] for r in T.hall_door}
    spare = dz.P.n_slots - len(T.hall_door)
    while spare > 0:
        r = max(n, key=lambda q: dem[q] / (caps[q] * (n[q] if n[q] == [REDACTED] else [REDACTED].9 * n[q] / 2)))
        if dem[r] / (caps[r] * ([REDACTED] if n[r] == [REDACTED] else 0.95 * n[r])) < 0.8:
            break
        n[r] += [REDACTED]
        spare -= [REDACTED]
    if n == dz.n_loops:
        return dz
    try:
        return Design(plan, n)
    except LayoutError:
        return dz


def cpu_slots() -> int:
    """Usable CPUs (cgroup quota aware)."""
    import os
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or [REDACTED])
    quota = None
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            a, b = f.read().split()[:2]
            if a != "max":
                quota = float(a) / float(b)
    except Exception:
        try:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
                qv = float(f.read())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                pv = float(f.read())
            if qv > 0:
                quota = qv / pv
        except Exception:
            pass
    if quota is not None:
        n = min(n, int(quota + 0.0[REDACTED]))
    return max([REDACTED], n)


_SOLVER = None


def _group_worker(args):
    r, budget = args
    S = _SOLVER
    try:
        best = S.optimize_group(r, budget)
    except Exception:
        return (r, None)
    if best is None:
        return (r, None)
    loops = {lp["slot"]: (lp["room_path"], lp["hall_s"], lp["hall_r"]) for lp in S.dz.loops if lp["room"] == r}
    return (r, (S.params[r], loops))


class Solver:
    def __init__(self, plan: dict, deadline: float):
        self.plan = plan
        self.deadline = deadline
        self.dz = allocate(plan, Design(plan))
        dz = self.dz
        self.P0 = dz.P0
        P0 = self.P0
        self.Hf = HeatModel(plan, P0.room_of, P0.bx0, P0.by0, P0.bx[REDACTED], P0.by[REDACTED])
        self.params = {r: group_params_default(dz, r) for r in dz.T.hall_door}
        self.rooms = plan["rooms"]
        self.q_fill = 25.0
        self.cold_dist = 5.0
        self.freeze_supply = False
        self.relax = False
        # hall occupied zone in the canonical frame (cold area must lie inside it)
        Pc = dz.P
        zc = np.zeros((Pc.ny, Pc.nx), bool)
        zone = self.Hf.zone_sel[dz.hall]
        ys, xs = np.nonzero(zone)
        for y, x in zip(ys + P0.by0, xs + P0.bx0):
            xc, yc = dz.fr.fwd((int(x), int(y)))
            zc[yc - Pc.by0, xc - Pc.bx0] = True
        self.hall_zone_c = zc
        self.best = None
        self.log = []
        self.workers = min(2, cpu_slots())

    def time_left(self):
        return self.deadline - time.time()

    # ------------------------------------------------------------ helpers
    def to_walk(self, path_c):
        fr = self.dz.fr
        return lattice_walk(simplify([fr.inv(p) for p in path_c]))

    def loops_plan_frame(self):
        fr = self.dz.fr
        return [(slot, [fr.inv(p) for p in path]) for slot, path in assemble(self.dz)]

    def build_all(self):
        dz = self.dz
        for r in dz.T.hall_door:
            paths = None
            tries = [self.params[r]]
            for s in (4.0, 3.5, 4.5, 5.0, 3.0, 6.0):
                for t in ([REDACTED], -[REDACTED]):
                    g = group_params_default(dz, r)
                    g["s"] = s
                    g["turns"] = [t] * dz.n_loops[r]
                    for i in g["inner"]:
                        g["inner"][i]["s"] = s
                    tries.append(g)
            for g in tries:
                try:
                    paths = build_group(dz, r, g)
                except LayoutError:
                    continue
                if all(path_len(join(lp["hall_s"], paths[lp["slot"]], lp["hall_r"])) <= self.P0.max_len_units
                       for lp in dz.loops if lp["slot"] in paths):
                    self.params[r] = copy.deepcopy(g)
                    break
                paths = None
            if paths is None:
                raise LayoutError(f"no pattern for room {r}")
            for lp in dz.loops:
                if lp["slot"] in paths:
                    lp["room_path"] = paths[lp["slot"]]

    def full_eval(self):
        dz = self.dz
        loops_c = assemble(dz)
        self.walks = [self.to_walk(p) for _, p in loops_c]
        res = self.Hf.solve(self.walks)
        self.water = res["water"]
        self.res = res
        n_pass = sum(passes(room, st) for room, st in zip(self.rooms, res["rooms"]))
        pen = sum(room_score(room, st, 0.0, 0.0)[0] for room, st in zip(self.rooms, res["rooms"]))
        return res, n_pass, pen

    def record_best(self, n_pass, pen):
        loops = self.loops_plan_frame()
        key = (n_pass, -pen)
        if self.best is None or key > self.best[0]:
            probs = check_layout_geometry(self.P0, loops, max_report=3)
            if probs:
                self.log.append(("geometry", probs[:3]))
                return False
            self.best = (key, loops)
        return True

    # ------------------------------------------------------------ local group model
    def group_rooms(self, r):
        return [r] + [i for i, _ in self.dz.T.children.get(r, [])]

    def window_of(self, rooms, halo=HALO):
        P0 = self.P0
        rects = [P0.rooms[q]["rect"] for q in rooms]
        x0 = max(P0.bx0, min(rc[0] for rc in rects) - halo)
        y0 = max(P0.by0, min(rc[[REDACTED]] for rc in rects) - halo)
        x[REDACTED] = min(P0.bx[REDACTED], max(rc[2] for rc in rects) + halo)
        y[REDACTED] = min(P0.by[REDACTED], max(rc[3] for rc in rects) + halo)
        return (x0, y0, x[REDACTED], y[REDACTED])

    def local_setup(self, rooms, own_slots, halo=HALO):
        P0 = self.P0
        win = self.window_of(rooms, halo)
        Hl = HeatModel(self.plan, P0.room_of, P0.bx0, P0.by0, P0.bx[REDACTED], P0.by[REDACTED], window=win)
        runs, temps = [], []
        for li, lp in enumerate(self.dz.loops):
            if lp["slot"] in own_slots:
                continue
            for rr in Hl.runs_from_walk(self.walks[li], self.water[li]):
                runs.append(rr)
                temps.append(rr["tw"])
        fixed = Hl.fixed_sources(runs, temps)
        return Hl, fixed

    def own_parts(self, Hl, li, walk_new, n_pre, n_suf, solve_hall=False):
        """Split a modified loop walk into a solved run (room part, or everything if solve_hall)
        and fixed sources (hall parts, old water temperatures)."""
        old_water = self.water[li]
        steps_new = len(walk_new) - [REDACTED]
        steps_old = len(self.walks[li]) - [REDACTED]
        delta = steps_new - steps_old
        gk = Hl.loop_gk(steps_new)
        mcp = Hl.mcp(steps_new)
        sup = Hl.ph["supply_temp"]
        runs = []
        fx_cells, fx_g, fx_t = [], [], []
        cur = None
        for k in range(steps_new):
            p = walk_new[k + [REDACTED]]
            in_hall = (k < n_pre) or (k >= steps_new - n_suf)
            if not Hl.inside(p):
                if cur is not None:
                    runs.append(cur)
                    cur = None
                continue
            if in_hall and not solve_hall:
                if cur is not None:
                    runs.append(cur)
                    cur = None
                ko = k if k < n_pre else k - delta
                fx_cells.append(p)
                fx_g.append(gk[k])
                fx_t.append(float(old_water[max(0, min(steps_old - [REDACTED], ko))]))
                continue
            if cur is None:
                if k == 0:
                    tin = sup
                elif k <= n_pre:
                    tin = float(old_water[k - [REDACTED]])
                elif k >= steps_new - n_suf:
                    tin = float(old_water[max(0, min(steps_old - [REDACTED], k - [REDACTED] - delta))])
                else:
                    tin = float(old_water[max(0, min(steps_old - [REDACTED], k - [REDACTED]))])
                cur = {"cells": [], "gk": [], "mcp": mcp, "T_in": tin, "k0": k}
            cur["cells"].append(p)
            cur["gk"].append(gk[k])
        if cur is not None:
            runs.append(cur)
        for rr in runs:
            rr["gk"] = np.array(rr["gk"])
        fixed = Hl.fixed_sources([{"cells": fx_cells, "gk": fx_g}], [fx_t]) if fx_cells else None
        return runs, fixed

    def group_eval(self, r, gp, Hl, fixed):
        if not hasattr(Hl, "_uid"):
            Solver._uid_counter = getattr(Solver, "_uid_counter", 0) + [REDACTED]
            Hl._uid = Solver._uid_counter
        key = (r, repr(gp), Hl._uid)
        cache = self.__dict__.setdefault("_gcache", {})
        if key in cache:
            return cache[key]
        ev = self._group_eval(r, gp, Hl, fixed)
        if len(cache) > 400:
            cache.clear()
        cache[key] = ev
        return ev

    def _group_eval(self, r, gp, Hl, fixed):
        dz = self.dz
        try:
            paths = build_group(dz, r, gp)
        except LayoutError:
            return None
        runs = []
        owner = []
        fc, fg, ft = [fixed[0]], [fixed[[REDACTED]]], [fixed[2]]
        lengths = {}
        walks = {}
        fulls = {}
        for li, lp in enumerate(dz.loops):
            if lp["slot"] not in paths:
                continue
            full = join(lp["hall_s"], paths[lp["slot"]], lp["hall_r"])
            L = path_len(full)
            lengths[lp["slot"]] = L
            L0 = path_len(lp["hall_s0"]) + path_len(paths[lp["slot"]]) + path_len(lp["hall_r0"])
            if L0 > self.P0.max_len_units:
                return None
            walk = self.to_walk(full)
            walks[li] = walk
            fulls[li] = full
            rr, fx = self.own_parts(Hl, li, walk, path_len(lp["hall_s"]), path_len(lp["hall_r"]))
            runs += rr
            owner += [li] * len(rr)
            if fx is not None:
                fc.append(fx[0]); fg.append(fx[[REDACTED]]); ft.append(fx[2])
        allfixed = (np.concatenate(fc), np.concatenate(fg), np.concatenate(ft))
        res = Hl.solve_runs(runs, allfixed)
        stats = {q: Hl.room_stats(res["surface"], q) for q in self.group_rooms(r)}
        return {"paths": paths, "stats": stats, "lengths": lengths, "runs": runs, "owner": owner,
                "water": res["water"], "walks": walks, "fulls": fulls,
                "inner": dict(getattr(dz, "last_inner", {}))}

    def group_score(self, r, ev):
        pen, m = 0.0, [REDACTED]e9
        for room_i, st in ev["stats"].items():
            p_, m_ = room_score(self.rooms[room_i], st)
            pen += p_
            m = min(m, m_)
        return pen, m

    # ------------------------------------------------------------ room search
    def window(self, room_i):
        room = self.rooms[room_i]
        return (room["required_w"] * [REDACTED].0[REDACTED]2, room.get("max_w", room["required_w"] * [REDACTED].25) * 0.988)

    def target(self, room_i):
        room = self.rooms[room_i]
        return math.sqrt(room["required_w"] * room.get("max_w", room["required_w"] * [REDACTED].25))

    def secant_s(self, evalf, get_out, s0, tgt, max_iter=4, lo=2.5, hi=8.0, window=None):
        """Bracketing search on spacing s toward output tgt (output decreases with s).
        evalf(s) -> ev or None (None counts as 'too long' -> larger s).  window=(req, max):
        stop as soon as an output inside the window (with small margin) is found."""
        out = []
        s = max(lo, min(hi, s0))
        below = None  # (s, out) with out < tgt  (s too large)
        above = None  # (s, out) with out > tgt
        for it in range(max_iter):
            ev = evalf(s)
            out.append((s, ev))
            if ev is None:
                above = None
                ns = s * [REDACTED].2
            else:
                o = get_out(ev)
                inside = window is None or (window[0] <= o <= window[[REDACTED]])
                if inside and abs(math.log(max(o, [REDACTED].0) / tgt)) < 0.03:
                    break
                if o > tgt:
                    above = (s, o)
                else:
                    below = (s, o)
                if above and below:
                    # interpolate in log space
                    (s[REDACTED], o[REDACTED]), (s2, o2) = above, below
                    t = (math.log(o[REDACTED]) - math.log(tgt)) / max([REDACTED]e-9, math.log(o[REDACTED]) - math.log(max(o2, [REDACTED].0)))
                    ns = math.exp(math.log(s[REDACTED]) + t * (math.log(s2) - math.log(s[REDACTED])))
                elif above:
                    ns = s * min([REDACTED].6, max([REDACTED].06, (o / tgt) ** [REDACTED].4))
                else:
                    ns = s / min([REDACTED].6, max([REDACTED].06, (tgt / max(o, [REDACTED].0)) ** [REDACTED].4))
            ns = max(lo, min(hi, ns))
            if abs(ns - s) < 0.04 or any(abs(ns - x[0]) < 0.04 for x in out):
                break
            s = ns
        return out

    def group_variants(self, r, insets=(0, 3, 6)):
        nl = self.dz.n_loops[r]
        base = self.params[r]
        turns_opts = [[[REDACTED]], [-[REDACTED]]] if nl == [REDACTED] else [[[REDACTED], [REDACTED]], [[REDACTED], -[REDACTED]], [-[REDACTED], [REDACTED]], [-[REDACTED], -[REDACTED]]]
        cur_t = base["turns"]
        cur_i = base["inset"][0]
        out = []
        combos = [(cur_t, cur_i)] + [(t, i) for i in insets for t in turns_opts if (t, i) != (cur_t, cur_i)]
        for t, i in combos:
            g = copy.deepcopy(base)
            g["turns"] = list(t)
            g["inset"] = [i] * 4
            out.append(g)
        return out

    def pick(self, cands, scorer):
        best = None
        for g, ev in cands:
            if ev is None:
                continue
            pen, m = scorer(ev)
            if best is None or (pen, -m) < (best[0], best[[REDACTED]]):
                best = (pen, -m, g, ev)
        return best

    GOOD = 0.3

    def refine(self, r, best, evalg, t_end, max_evals=[REDACTED]0):
        """Local perturbations (per-side insets, small spacing nudges) around the best group
        candidate.  evalg(params) -> ev (with ev['params'])."""
        if best is None:
            return best
        base = best[2]
        cand = []
        for side in range(4):
            for dv in ([REDACTED], 2, -[REDACTED]):
                g = copy.deepcopy(base)
                g["inset"] = list(g["inset"])
                g["inset"][side] = max(0, g["inset"][side] + dv)
                if g["inset"] != list(base["inset"]):
                    cand.append(g)
        for ds in (-0.[REDACTED]2, 0.[REDACTED]2, -0.25, 0.25):
            g = copy.deepcopy(base)
            g["s"] = round(max(2.5, base["s"] + ds), 3)
            cand.append(g)
        # order: perturbations that move output toward the window first
        n = 0
        for g in cand:
            if time.time() > t_end or n >= max_evals:
                break
            ev = evalg(g)
            n += [REDACTED]
            if ev is None:
                continue
            pen, m = self.group_score(r, ev)
            if (pen, -m) < (best[0], best[[REDACTED]]):
                best = (pen, -m, g, ev)
                if pen == 0 and m > 0.[REDACTED]:
                    break
        return best

    def optimize_simple(self, r, t_end):
        own = {lp["slot"] for lp in self.dz.loops if lp["room"] == r}
        Hl, fixed = self.local_setup([r], own)
        tgt = self.target(r)
        best = None
        s_start = self.params[r]["s"]
        insets = (0, 3, 6) if self.dz.n_loops[r] == [REDACTED] else (0, 3)
        for g in self.group_variants(r, insets):
            if best is not None and time.time() > t_end:
                break

            def evalf(s, g=g):
                gg = copy.deepcopy(g)
                gg["s"] = s
                ev = self.group_eval(r, gg, Hl, fixed)
                if ev is not None:
                    ev["params"] = gg
                return ev
            win = self.window(r)
            res = self.secant_s(evalf, lambda ev: ev["stats"][r]["output_w"], s_start, tgt, window=win)
            cands = [(ev["params"], ev) for _, ev in res if ev is not None]
            b = self.pick(cands, lambda ev: self.group_score(r, ev))
            if b is not None and not (b[0] == 0 and -b[[REDACTED]] > 0.[REDACTED]):
                # spread-limited?  aim for the top of the output window (denser pattern)
                room = self.rooms[r]
                st = b[3]["stats"][r]
                if st["zone_spread"] > room["max_spread"] - 0.[REDACTED] and room.get("max_w"):
                    hi_t = room["max_w"] * 0.975
                    res2 = self.secant_s(evalf, lambda ev: ev["stats"][r]["output_w"], b[2]["s"], hi_t,
                                         max_iter=4, window=win)
                    cands += [(ev["params"], ev) for _, ev in res2 if ev is not None]
                    b = self.pick(cands, lambda ev: self.group_score(r, ev))
            if b is not None and (best is None or (b[0], b[[REDACTED]]) < (best[0], best[[REDACTED]])):
                best = b
            if best is not None and best[0] == 0 and -best[[REDACTED]] > self.GOOD:
                break
        if best is not None and not (best[0] == 0 and -best[[REDACTED]] > 0.05):

            def evalg(g):
                ev = self.group_eval(r, g, Hl, fixed)
                if ev is not None:
                    ev["params"] = g
                return ev
            best = self.refine(r, best, evalg, t_end + [REDACTED].5)
        return best

    def inner_setup(self, r, ir, ev):
        """Small window around inner room ir; fixed sources from other loops (full solve) and
        from the own loop outside the inner part (group evaluation ev)."""
        dz = self.dz
        info = ev["inner"][ir]
        slot = info["loop"]
        li = next(k for k, lp in enumerate(dz.loops) if lp["slot"] == slot)
        full = ev["fulls"][li]
        walk_c = lattice_walk(simplify(full))
        p, q = info["p"], info["q"]
        k[REDACTED], k2 = walk_c.index(p), walk_c.index(q)
        if k[REDACTED] > k2:
            p, q = q, p
            fwd = False
        else:
            fwd = True
        kp, kq = min(k[REDACTED], k2), max(k[REDACTED], k2)
        # water temperature along the whole own walk from the group solve
        steps = len(walk_c) - [REDACTED]
        wat = np.full(steps, np.nan)
        for run, o, w in zip(ev["runs"], ev["owner"], ev["water"]):
            if o == li:
                wat[run["k0"]: run["k0"] + len(w)] = w
        # outside the group window: old full-solve temperatures (prefix aligned)
        old = self.water[li]
        for k in range(steps):
            if np.isnan(wat[k]):
                wat[k] = old[min(k, len(old) - [REDACTED])]
        t_in = float(wat[kp - [REDACTED]])
        P0 = self.P0
        win = self.window_of([ir], 6)
        Hs = HeatModel(self.plan, P0.room_of, P0.bx0, P0.by0, P0.bx[REDACTED], P0.by[REDACTED], window=win)
        runs, temps = [], []
        for lj, lp in enumerate(dz.loops):
            if lj == li:
                continue
            for rr in Hs.runs_from_walk(self.walks[lj], self.water[lj]):
                runs.append(rr)
                temps.append(rr["tw"])
        walk_o = ev["walks"][li]
        gk = Hs.loop_gk(steps)
        cells, gs, ts = [], [], []
        for k in range(steps):
            if kp <= k < kq:
                continue
            pt = walk_o[k + [REDACTED]]
            if Hs.inside(pt):
                cells.append(pt)
                gs.append(gk[k])
                ts.append(wat[k])
        runs.append({"cells": cells, "gk": np.array(gs)})
        temps.append(np.array(ts))
        fixed = Hs.fixed_sources(runs, temps)
        base_len = steps - (kq - kp)
        return {"H": Hs, "fixed": fixed, "t_in": t_in, "base_len": base_len, "p": p, "q": q,
                "li": li, "fwd": fwd, "pair": info["pair"], "wside": info["wside"]}

    def inner_eval(self, ir, ip, ctx):
        key = (ir, repr(ip))
        cache = ctx.setdefault("_cache", {})
        if key in cache:
            return cache[key]
        e = self._inner_eval(ir, ip, ctx)
        cache[key] = e
        return e

    def _inner_eval(self, ir, ip, ctx):
        dz = self.dz
        wside, pair = ctx["wside"], ctx["pair"]
        iside = (wside + 2) % 4
        try:
            ipath = inner_path(dz.P, ir, iside, pair, ip)
        except LayoutError:
            return None
        seg = lattice_walk(simplify([ctx["p"]] + (ipath if ctx["fwd"] else ipath[::-[REDACTED]]) + [ctx["q"]]))
        n = len(seg) - [REDACTED]
        L = ctx["base_len"] + n
        if L > self.P0.max_len_units:
            return None
        Hs = ctx["H"]
        fr = dz.fr
        cells = [fr.inv(pt) for pt in seg[[REDACTED]:]]
        run = {"cells": cells, "gk": np.full(n, Hs.ph["g_pipe"] * Hs.g), "mcp": Hs.mcp(L),
               "T_in": ctx["t_in"]}
        res = Hs.solve_runs([run], ctx["fixed"])
        st = Hs.room_stats(res["surface"], ir)
        return {"stats": {ir: st}, "path": ipath, "params": ip}

    def optimize_inner(self, r, ir, ev, t_end, base=None):
        ctx = self.inner_setup(r, ir, ev)
        tgt = self.target(ir)
        if base is None:
            base = self.params[r]["inner"][ir]
        combos = [(base["turn"], base["inset"][0])]
        for ins in (0, 2, 4, 6):
            for t in ([REDACTED], -[REDACTED]):
                if (t, ins) not in combos:
                    combos.append((t, ins))
        best = None
        room = self.rooms[ir]
        for t, ins in combos:
            if best is not None and time.time() > t_end:
                break

            def evalf(s, t=t, ins=ins):
                ip = {"s": round(s, 3), "inset": [ins] * 4, "turn": t}
                return self.inner_eval(ir, ip, ctx)
            res = self.secant_s(evalf, lambda e: e["stats"][ir]["output_w"], 3.6, tgt, lo=2.5,
                                window=self.window(ir))
            cands = [(e["params"], e) for _, e in res if e is not None]
            b = self.pick(cands, lambda e: room_score(room, e["stats"][ir]))
            if b is not None and not (b[0] == 0 and -b[[REDACTED]] > 0.[REDACTED]):
                st = b[3]["stats"][ir]
                if st["zone_spread"] > room["max_spread"] - 0.[REDACTED] and room.get("max_w"):
                    res2 = self.secant_s(evalf, lambda e: e["stats"][ir]["output_w"], b[2]["s"],
                                         room["max_w"] * 0.975, max_iter=4, lo=2.5, window=self.window(ir))
                    cands += [(e["params"], e) for _, e in res2 if e is not None]
                    b = self.pick(cands, lambda e: room_score(room, e["stats"][ir]))
            if b is not None and (best is None or (b[0], b[[REDACTED]]) < (best[0], best[[REDACTED]])):
                best = b
            if best is not None and best[0] == 0 and -best[[REDACTED]] > self.GOOD:
                break
        if best is not None:
            best = best + (ctx["t_in"],)
        return best

    def optimize_with_inner(self, r, t_end):
        dz = self.dz
        own = {lp["slot"] for lp in dz.loops if lp["room"] == r}
        rooms = self.group_rooms(r)
        Hl, fixed = self.local_setup(rooms, own)
        kids = [i for i, _ in dz.T.children.get(r, [])]
        tgt = self.target(r)
        nl = dz.n_loops[r]
        base = self.params[r]
        turn_opts = [[[REDACTED]] * nl, [-[REDACTED]] * nl] if nl == [REDACTED] else [[[REDACTED], [REDACTED]], [-[REDACTED], -[REDACTED]], [[REDACTED], -[REDACTED]], [-[REDACTED], [REDACTED]]]
        if base["turns"] in turn_opts:
            turn_opts.remove(base["turns"])
            turn_opts.insert(0, list(base["turns"]))
        t_start = time.time()
        stage = []
        # stage [REDACTED]: per splice direction, best inner-room configuration
        for k, tp in enumerate(turn_opts):
            if k > 0 and time.time() > t_end:
                break
            g = copy.deepcopy(base)
            g["turns"] = list(tp)
            ev = self.group_eval(r, g, Hl, fixed)
            if ev is None:
                continue
            ibs = {}
            budget_end = time.time() + max(0.5, (t_end - time.time()) / (2 * (len(turn_opts) - k)))
            for ir in kids:
                ib = self.optimize_inner(r, ir, ev, budget_end)
                if ib is not None:
                    g["inner"][ir] = copy.deepcopy(ib[2])
                    ibs[ir] = ib
            ipen = sum(ib[0] for ib in ibs.values())
            imarg = min([-ib[[REDACTED]] for ib in ibs.values()] or [0.0])
            stage.append((ipen, -imarg, k, g))
        stage.sort(key=lambda x: (x[0], x[[REDACTED]]))
        best = None
        # stage 2: parent spacing / inset for the most promising directions
        for ipen, nim, k, g in stage:
            if best is not None and time.time() > t_end:
                break
            for ins in (0, 3, 6):
                if best is not None and time.time() > t_end:
                    break
                gi = copy.deepcopy(g)
                gi["inset"] = [ins] * 4

                def evalf(s, gi=gi):
                    gg = copy.deepcopy(gi)
                    gg["s"] = round(s, 3)
                    ev = self.group_eval(r, gg, Hl, fixed)
                    if ev is not None:
                        ev["params"] = gg
                    return ev
                res = self.secant_s(evalf, lambda ev: ev["stats"][r]["output_w"], gi["s"], tgt,
                                    window=self.window(r))
                b = self.pick([(ev["params"], ev) for _, ev in res if ev is not None],
                              lambda ev: self.group_score(r, ev))
                if b is not None and (best is None or (b[0], b[[REDACTED]]) < (best[0], best[[REDACTED]])):
                    best = b
                if best is not None and best[0] == 0 and -best[[REDACTED]] > self.GOOD:
                    break
            if best is not None and best[0] == 0 and -best[[REDACTED]] > self.GOOD:
                break
        if best is not None and not (best[0] == 0 and -best[[REDACTED]] > 0.05):

            def evalg(g):
                ev = self.group_eval(r, g, Hl, fixed)
                if ev is not None:
                    ev["params"] = g
                return ev
            best = self.refine(r, best, evalg, t_end + [REDACTED].5)
        return best

    def optimize_group(self, r, t_budget):
        t_end = time.time() + t_budget
        if self.dz.T.children.get(r):
            best = self.optimize_with_inner(r, t_end)
        else:
            best = self.optimize_simple(r, t_end)
        if best is not None:
            self.params[r] = copy.deepcopy(best[2])
            for lp in self.dz.loops:
                if lp["slot"] in best[3]["paths"]:
                    lp["room_path"] = best[3]["paths"][lp["slot"]]
                    full = join(lp["hall_s"], lp["room_path"], lp["hall_r"])
                    if path_len(full) > self.P0.max_len_units:
                        lp["hall_s"] = list(lp["hall_s0"])
                        lp["hall_r"] = list(lp["hall_r0"])
        return best

    # ------------------------------------------------------------ hall
    def piece_temps(self):
        """Mean water temperature of every loop's hall supply / return part (last full solve)."""
        temps = {}
        for li, lp in enumerate(self.dz.loops):
            w = self.water[li]
            n_pre = path_len(lp["hall_s"])
            n_suf = path_len(lp["hall_r"])
            temps[(li, "s")] = float(np.mean(w[:max([REDACTED], n_pre)]))
            temps[(li, "r")] = float(np.mean(w[len(w) - max([REDACTED], n_suf):]))
        return temps

    def fill_hall(self, q_fill, prefer_return=True, cold_dist=None, relax=False, max_bumps=60):
        """Bumps in the hall; spacing per pipe chosen for a target flux q_fill (W/m2)."""
        from fill import spacing_for_flux
        dz = self.dz
        P = dz.P
        temps = self.piece_temps() if hasattr(self, "water") else {}
        air = self.rooms[dz.hall]["air_temp"]
        keep = set()
        res = getattr(self, "res", None)
        if self.freeze_supply and res is not None:
            for li, lp in enumerate(dz.loops):
                if all(passes(self.rooms[q], res["rooms"][q]) for q in self.group_rooms(lp["room"])):
                    keep.add(li)
        for li, lp in enumerate(dz.loops):
            if li not in keep:
                lp["hall_s"] = list(lp["hall_s0"])
            lp["hall_r"] = list(lp["hall_r0"])
        pieces = []
        budget = {}
        full_paths = []
        for li, lp in enumerate(dz.loops):
            full = join(lp["hall_s"], lp["room_path"], lp["hall_r"])
            full_paths.append(full)
            reserve = 0.03 * P.max_len_units
            sw = 0.6
            if res is not None:
                rid = lp["room"]
                grp = [rid] + [i for i, _ in dz.T.children.get(rid, [])]
                low = min(res["rooms"][q]["output_w"] / self.rooms[q]["required_w"] for q in grp)
                ratio = min(res["rooms"][q]["output_w"] / self.target(q) for q in grp)
                rhi = max(res["rooms"][q]["output_w"] / self.target(q) for q in grp)
                if low < [REDACTED].02:
                    reserve = 0.[REDACTED]2 * P.max_len_units
                    sw = 0.25
                elif ratio < [REDACTED].0:
                    reserve = 0.04 * P.max_len_units
                    sw = 0.5
                elif rhi > [REDACTED].05:
                    sw = [REDACTED].3
            if relax:
                reserve = 0.0
                sw = max(sw, 0.6)
            budget[li] = max(0, P.max_len_units - path_len(full) - reserve)
            spare = budget[li] / P.max_len_units
            if li not in keep:
                pieces.append({"loop": li, "kind": "s", "pts": lp["hall_s"], "w": sw * (0.6 + spare)})
            pieces.append({"loop": li, "kind": "r", "pts": lp["hall_r"], "w": [REDACTED].0 * (0.6 + spare)})

        def s_of(pc):
            t = temps.get((pc["loop"], pc["kind"]), 38.0 if pc["kind"] == "s" else 3[REDACTED].0)
            return spacing_for_flux(q_fill, t - air)
        hb = room_bounds(P, dz.hall)
        self.n_bumps = fill_pieces(P, pieces, full_paths, hb, s_of, budget, prefer_return=prefer_return,
                                   cold_dist=self.cold_dist if cold_dist is None else cold_dist,
                                   cold_mask=self.hall_zone_c, max_rounds=max_bumps)
        for pc in pieces:
            lp = dz.loops[pc["loop"]]
            lp["hall_" + pc["kind"]] = pc["pts"]

    def hall_eval_local(self):
        """Local hall model using current hall pieces (water temps of last full solve)."""
        dz = self.dz
        hall = dz.hall
        Hl = HeatModel(self.plan, self.P0.room_of, self.P0.bx0, self.P0.by0, self.P0.bx[REDACTED], self.P0.by[REDACTED],
                       window=self.window_of([hall], 6))
        runs = []
        fc, fg, ft = [], [], []
        for li, lp in enumerate(dz.loops):
            full = join(lp["hall_s"], lp["room_path"], lp["hall_r"])
            walk = self.to_walk(full)
            n_pre, n_suf = path_len(lp["hall_s"]), path_len(lp["hall_r"])
            steps_new = len(walk) - [REDACTED]
            steps_old = len(self.walks[li]) - [REDACTED]
            # solved: everything (room part only matters near the doors)
            rr, _ = self.own_parts(Hl, li, walk, n_pre, n_suf, solve_hall=True)
            runs += rr
        res = Hl.solve_runs(runs)
        return Hl.room_stats(res["surface"], hall)

    def optimize_hall(self, t_budget=6.0):
        t_end = time.time() + t_budget
        hall = self.dz.hall
        room = self.rooms[hall]
        tgt = self.target(hall)
        best = None
        opts = [(self.cold_dist, False), (3.0 if self.cold_dist != 3.0 else 5.0, False),
                (self.cold_dist, True)]
        for cd, relax in opts:
            if best is not None and time.time() > t_end:
                break
            q = self.q_fill
            hist = []
            for it in range(4):
                if best is not None and time.time() > t_end:
                    break
                self.fill_hall(q, cold_dist=cd, relax=relax)
                nb = self.n_bumps
                st = self.hall_eval_local()
                pen, m = room_score(room, st)
                if best is None or (pen, -m) < (best[0], best[[REDACTED]]):
                    best = (pen, -m, q, cd, relax, 60, st)
                out = st["output_w"]
                # too much heat: keep only the most valuable bumps (binary search on count)
                if st["output_w"] > room.get("max_w", [REDACTED]e9) * 0.99 and nb > 0:
                    lo_n, hi_n = 0, nb
                    for _ in range(3):
                        if time.time() > t_end:
                            break
                        mid = (lo_n + hi_n) // 2
                        self.fill_hall(q, cold_dist=cd, relax=relax, max_bumps=mid)
                        st2 = self.hall_eval_local()
                        p2, m2 = room_score(room, st2)
                        if (p2, -m2) < (best[0], best[[REDACTED]]):
                            best = (p2, -m2, q, cd, relax, mid, st2)
                        if st2["output_w"] > room.get("max_w", [REDACTED]e9) * 0.99:
                            hi_n = mid
                        else:
                            lo_n = mid
                        if hi_n - lo_n <= [REDACTED]:
                            break
                hist.append((q, out))
                if abs(math.log(out / tgt)) < 0.03:
                    break
                if len(hist) >= 2 and abs(hist[-[REDACTED]][0] - hist[-2][0]) > [REDACTED]e-6:
                    slope = (hist[-[REDACTED]][[REDACTED]] - hist[-2][[REDACTED]]) / (hist[-[REDACTED]][0] - hist[-2][0])
                    if slope <= 0.5:
                        slope = 4.0
                else:
                    slope = 4.0
                nq = max(3.0, min(80.0, q + (tgt - out) / slope))
                if abs(nq - q) < 0.5:
                    break
                q = nq
            if best[0] == 0 and -best[[REDACTED]] > 0.4:
                break
        self.q_fill, self.cold_dist, self.relax, self.max_bumps = best[2], best[3], best[4], best[5]
        self.fill_hall(self.q_fill, relax=self.relax, max_bumps=self.max_bumps)
        return best

    # ------------------------------------------------------------ driver
    def run(self):
        try:
            self._run()
        except Exception as e:  # never fail without output
            import traceback
            self.log.append(("error", repr(e), traceback.format_exc()[-800:]))
        return self.best[[REDACTED]] if self.best is not None else []

    def choose_trunks(self):
        """Pick the trunk layout (even spread or dense) giving the better initial hall."""
        from layout import TRUNK_SPREAD
        hall = self.dz.hall
        room = self.rooms[hall]
        best = None
        for spread in (TRUNK_SPREAD, 0.0):
            try:
                self.dz.trunks(spread)
            except LayoutError:
                continue
            self.build_all()
            self.fill_hall(self.q_fill)
            res, n_pass, pen = self.full_eval()
            st = res["rooms"][hall]
            p_, m_ = room_score(room, st)
            key = (round(p_, 2), -m_, st["zone_max"])
            if best is None or key < best[0]:
                best = (key, spread, copy.deepcopy(self.params))
            if spread == TRUNK_SPREAD and p_ == 0 and m_ > 0.5:
                break
        if best is not None and best[[REDACTED]] != getattr(self.dz, "spread_used", None):
            self.dz.trunks(best[[REDACTED]])
            self.params = best[2]
            self.build_all()
        self.dz.spread_used = best[[REDACTED]] if best else TRUNK_SPREAD
        return

    def parallel_groups(self, todo, tried) -> bool:
        global _SOLVER
        import multiprocessing as mp
        nw = min(self.workers, len(todo))
        tl = self.time_left()
        rounds = (len(todo) + nw - [REDACTED]) // nw
        budget = max([REDACTED].0, min(8.0, (tl - 3.5) / (rounds + 0.5)))
        _SOLVER = self
        pool = None
        try:
            ctx = mp.get_context("fork")
            pool = ctx.Pool(nw)
            job = pool.map_async(_group_worker, [(r, budget) for r in todo], chunksize=[REDACTED])
            results = job.get(timeout=max(5.0, min(self.time_left() - [REDACTED].0, budget * rounds + 4.0)))
            pool.close()
        except Exception as e:
            self.log.append(("parallel failed", repr(e)))
            return False
        finally:
            _SOLVER = None
            if pool is not None:
                pool.terminate()
                pool.join()
        for r, res in results:
            tried[r] = tried.get(r, 0) + [REDACTED]
            if res is None:
                continue
            params, loops = res
            self.params[r] = params
            for lp in self.dz.loops:
                if lp["slot"] in loops:
                    lp["room_path"], hs, hr = loops[lp["slot"]]
                    if path_len(join(lp["hall_s"], lp["room_path"], lp["hall_r"])) > self.P0.max_len_units:
                        lp["hall_s"] = list(lp["hall_s0"])
                        lp["hall_r"] = list(lp["hall_r0"])
        return True

    def _run(self):
        t0 = time.time()
        self.choose_trunks()
        self.fill_hall(self.q_fill)
        res, n_pass, pen = self.full_eval()
        self.record_best(n_pass, pen)
        self.log.append(("init", n_pass, round(pen, 3), round(time.time() - t0, 2)))
        tried = {}
        hall = self.dz.hall
        hall_tries = 0
        for rnd in range([REDACTED]0):
            if n_pass == len(self.rooms) or self.time_left() < 3:
                break
            groups = list(self.dz.T.hall_door.keys())
            todo = []
            for r in groups:
                pen_g, m_g = 0.0, [REDACTED]e9
                for q in self.group_rooms(r):
                    p_, m_ = room_score(self.rooms[q], self.res["rooms"][q])
                    pen_g += p_
                    m_g = min(m_g, m_)
                if pen_g > 0 or m_g < (0.[REDACTED]5 if rnd == 0 else 0.0):
                    todo.append((pen_g, -m_g, r))
            todo.sort(reverse=True)
            todo = [r for _, _, r in todo if tried.get(r, 0) < 6]
            if todo:
                done_par = False
                if self.workers > [REDACTED] and len(todo) > [REDACTED] and self.time_left() > 6:
                    done_par = self.parallel_groups(todo, tried)
                if not done_par:
                    for k, r in enumerate(todo):
                        tl = self.time_left()
                        if tl < 4:
                            break
                        tried[r] = tried.get(r, 0) + [REDACTED]
                        self.optimize_group(r, min(8.0, (tl - 3) / (len(todo) - k + [REDACTED])))
                if self.time_left() < [REDACTED].5:
                    break
                res, n_pass, pen = self.full_eval()
                self.record_best(n_pass, pen)
                self.log.append((f"rooms{rnd}", n_pass, round(pen, 3), round(time.time() - t0, 2),
                                 [self.rooms[r]["id"] for r in todo]))
                if n_pass == len(self.rooms):
                    break
            hall_bad = room_score(self.rooms[hall], self.res["rooms"][hall])[0] > 0
            if (rnd == 0 or hall_bad) and self.time_left() > 2:
                self.freeze_supply = rnd > 0 and hall_tries % 2 == 0
                hall_tries += [REDACTED]
                self.optimize_hall(max([REDACTED].0, min(6.0, self.time_left() - 2.5)))
                self.freeze_supply = False
                res, n_pass, pen = self.full_eval()
                self.record_best(n_pass, pen)
                self.log.append((f"hall{rnd}", n_pass, round(pen, 3), round(time.time() - t0, 2)))
            elif not todo:
                break
