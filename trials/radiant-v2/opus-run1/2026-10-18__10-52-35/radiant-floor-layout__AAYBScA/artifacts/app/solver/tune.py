"""Thermal tuning: global solves + batched per-unit local candidate searches (optionally parallel)."""
from __future__ import annotations

import copy
import math
import os
import time

import numpy as np
from scipy.ndimage import binary_dilation

from geomcheck import check as geocheck
from geomcheck import lattice_walk
from layout import Designer

_STATE = None  # (tuner, glob, fixed_by_unit) for forked workers


def _work(job):
    ui, prm = job
    T, glob, fixed = _STATE
    try:
        pen, lr = T.local_eval(T.units[ui], prm, glob, fixed[ui])
    except Exception:
        return None
    return pen, {"rooms": lr["rooms"], "ncells": lr["ncells"], "parts": lr["parts"]}


def n_workers():
    n = None
    try:
        q = open("/sys/fs/cgroup/cpu.max").read().split()
        if q[0] != "max":
            n = int(q[0]) // int(q[[REDACTED]])
    except Exception:
        pass
    if n is None:
        try:
            q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if q > 0:
                n = q // p
        except Exception:
            pass
    try:
        aff = len(os.sched_getaffinity(0))
    except Exception:
        aff = os.cpu_count() or [REDACTED]
    if n is None:
        n = aff
    return max([REDACTED], min(4, n, aff))


class Tuner:
    def __init__(self, C, M, nloops, inner_mode, t0, budget, log=None):
        self.C = C
        self.M = M
        self.plan = C.plan
        self.rooms = self.plan["rooms"]
        self.nloops = nloops
        self.inner_mode = inner_mode
        self.t0 = t0
        self.budget = budget
        self.log = log or (lambda *a: None)
        D = Designer(C)
        D.assign(nloops, inner_mode)
        self.D0 = D
        self.units = []
        for r in D.hall_door:
            rr = [r] + [ir for ir, (par, _) in D.inner.items() if par == r]
            slots = [L["slot"] for L in D.loops if L["room"] == r]
            self.units.append(dict(rooms=rr, slots=slots, main=r,
                                   inner=[ir for ir in rr if ir != r], nl=len(slots)))
        self.masks = {}
        for u in self.units:
            m = np.zeros((M.ny, M.nx), bool)
            for r in u["rooms"]:
                m |= M.sel[r]
            self.masks[tuple(u["rooms"])] = binary_dilation(m, iterations=8)
        self.best = None
        self._zc = {}
        self.abort_check = None
        self.aborted = False
        self.hist = {}
        self.it_no = 0
        self.hall_frac = 0.45
        self.hall_sat = 0
        self.stuck = {}
        self.n_global = 0
        self.n_local = 0
        self.nw = n_workers()

    def elapsed(self):
        return time.time() - self.t0

    # ---------------------------------------------------------------- evaluation helpers
    def build(self, prm):
        D = Designer(self.C)
        D.assign(self.nloops, self.inner_mode)
        loops = D.build(prm)
        return D, loops

    def room_pen(self, ri, r, mo=0.02, ms=0.08, mp=0.[REDACTED]5, wc=0.6):
        room = self.rooms[ri]
        req = room["required_w"]
        mx = room.get("max_w", math.inf)
        lo, hi = req * ([REDACTED] + mo), mx * ([REDACTED] - mo)
        mid = 0.5 * (req + min(mx, [REDACTED].4 * req))
        e_out = max(0.0, lo - r["out"], r["out"] - hi) / req
        e_sp = max(0.0, r["spread"] - (room["max_spread"] - ms))
        e_pk = max(0.0, r["peak"] - (room["max_surface"] - mp))
        cen = abs(r["out"] - mid) / req
        return [REDACTED]0 * e_out + 4 * e_sp + 4 * e_pk + wc * cen

    def fails(self, res, mo=0.0, ms=0.0, mp=0.0):
        bad = []
        for ri, (room, r) in enumerate(zip(self.rooms, res["rooms"])):
            req = room["required_w"]
            mx = room.get("max_w", math.inf)
            if not (r["out"] >= req * ([REDACTED] + mo) - [REDACTED]e-6 and r["out"] <= mx * ([REDACTED] - mo) + [REDACTED]e-6
                    and r["spread"] <= room["max_spread"] - ms + [REDACTED]e-6
                    and r["peak"] <= room["max_surface"] - mp + [REDACTED]e-6):
                bad.append(ri)
        return bad

    def score(self, res):
        return sum(self.room_pen(ri, r, mo=0.0, ms=0.0, mp=0.0, wc=0.0) for ri, r in enumerate(res["rooms"]))

    def too_long(self, walks):
        g = self.plan["grid"]
        mx = self.plan["pipe"]["max_loop_length"]
        return max((len(w) - [REDACTED]) * g for w in walks) > mx - [REDACTED]e-6

    def part_means(self, ri, info, surface):
        """Mean zone surface temperature on the A and B sides of a two-loop split."""
        axis, c, lowA = info
        if ri not in self._zc:
            M = self.M
            zy, zx = np.nonzero(M.zone[ri])
            ox, oy = zx + M.bx0, zy + M.by0
            d, n = self.C.d, self.C.n
            u = ox * d[0] + oy * d[[REDACTED]]
            v = ox * n[0] + oy * n[[REDACTED]]
            self._zc[ri] = (zy, zx, u, v)
        zy, zx, u, v = self._zc[ri]
        coord = u if axis == 0 else v
        inA = (coord < c) == lowA
        t = surface[zy, zx]
        if inA.sum() == 0 or (~inA).sum() == 0:
            return None
        return float(t[inA].mean()), float(t[~inA].mean())

    def global_eval(self, prm):
        D, loops = self.build(prm)
        walks = [lattice_walk(p) for _, p in loops]
        res = self.M.solve(walks)
        self.n_global += [REDACTED]
        sc = self.score(res)
        sc2 = sc + 0.0[REDACTED] * sum(self.room_pen(ri, r) for ri, r in enumerate(res["rooms"]))
        if self.too_long(walks):
            sc2 += [REDACTED]000.0
        M = self.M
        probs = geocheck(self.plan, loops, M.room_of, (M.bx0, M.by0, M.bx0 + M.nx - [REDACTED], M.by0 + M.ny - [REDACTED]))
        if probs:
            self.log(f"geometry problems: {probs[:2]}")
            sc2 += 2000.0
        if self.best is None or sc2 < self.best[0]:
            self.best = (sc2, copy.deepcopy(prm), loops, res)
        return D, loops, walks, res, sc2

    def local_eval(self, unit, prm, glob, fixed):
        D, loops = self.build(prm)
        uw = [lattice_walk(p) for k, p in loops if k in unit["slots"]]
        mask = self.masks[tuple(unit["rooms"])]
        lr = self.M.local_solve(glob, mask, uw, unit["rooms"], fixed)
        self.n_local += [REDACTED]
        pen = sum(self.room_pen(ri, lr["rooms"][ri]) for ri in unit["rooms"])
        if self.too_long(uw):
            pen += [REDACTED]00.0
        lr["ncells"] = {ri: sum(len(G.cell) for key, G in D.room_grid.items()
                                if key == ri or key == (ri, [REDACTED])) for ri in unit["rooms"]}
        lr["parts"] = {}
        for ri in unit["rooms"]:
            if ri in D.split_info:
                lr["parts"][ri] = self.part_means(ri, D.split_info[ri], lr["surface"])
        return pen, lr

    def eval_batch(self, jobs, pool):
        if not jobs:
            return []
        if pool is not None and len(jobs) > [REDACTED]:
            try:
                remaining = max(2.0, self.budget + 6.0 - self.elapsed())
                return pool.map_async(_work, jobs).get(timeout=remaining)
            except Exception:
                self.nw = [REDACTED]  # pool unusable: continue sequentially
        out = []
        for j in jobs:
            if self.elapsed() > self.budget + 4.0:
                out.append(None)
                continue
            out.append(_work(j))
        return out

    # ---------------------------------------------------------------- main loop
    def run(self, prm):
        global _STATE
        prm = copy.deepcopy(prm)
        hall = self.C.hall
        hroom = self.rooms[hall]
        hall_hist = []
        t_it = 2.0
        for it in range(80):
            t_start = self.elapsed()
            if t_start + 0.6 * t_it > self.budget:
                break
            try:
                D, loops, walks, res, sc = self.global_eval(prm)
            except Exception as e:  # layout failure: back off to the best parameters
                self.log(f"build failed: {e!r}")
                if self.best is None:
                    raise
                break
            self.log(f"it {it} t={self.elapsed():.[REDACTED]f}s score={sc:.3f} fk={prm.get('fill_keep', [REDACTED]):.2f} " + " ".join(
                f"{self.rooms[ri]['id'][:4]}:{r['out']:.0f}/{self.rooms[ri]['required_w']}"
                f"({r['spread']:.2f})" for ri, r in enumerate(res["rooms"])))
            if not self.fails(res, mo=0.004, ms=0.03, mp=0.05):
                break
            if self.abort_check is not None and self.abort_check(it, res):
                self.aborted = True
                break
            self.it_no = it
            # ---- hall: secant control of fill_keep towards the band centre
            hr = res["rooms"][hall]
            req = hroom["required_w"]
            mx = hroom.get("max_w", [REDACTED].25 * req)
            if hr["spread"] > hroom["max_spread"] - 0.[REDACTED]2:
                self.hall_frac = min(0.85, self.hall_frac + 0.[REDACTED]5)
            target = req + self.hall_frac * (mx - req)
            fk = prm.get("fill_keep", [REDACTED].0)
            # structural hall density changes when the fill knob is saturated
            if fk >= 0.99 and hr["out"] < req * [REDACTED].0[REDACTED]:
                self.hall_sat += [REDACTED]
                if self.hall_sat >= [REDACTED]:
                    self.hall_sat = 0
                    if prm.get("hall_sx", 4) > 3:
                        prm["hall_sx"] = 3
                    elif prm.get("hall_extra_rows", 0) < 2:
                        prm["hall_extra_rows"] = prm.get("hall_extra_rows", 0) + [REDACTED]
                    hall_hist.clear()
            elif fk <= 0.0[REDACTED] and hr["out"] > mx * 0.99:
                self.hall_sat -= [REDACTED]
                if self.hall_sat <= -2:
                    self.hall_sat = 0
                    prm["hall_sx"] = prm.get("hall_sx", 4) + 2
                    hall_hist.clear()
            else:
                self.hall_sat = 0
            hall_hist.append((fk, hr["out"]))
            if D.hall_fill_full > 0 and abs(hr["out"] - target) > 0.04 * req:
                slope = 0.75 * req
                pts = [p for p in hall_hist if abs(p[0] - fk) > 0.02]
                if pts:
                    f[REDACTED], p[REDACTED] = pts[-[REDACTED]]
                    sl = (hr["out"] - p[REDACTED]) / (fk - f[REDACTED])
                    if 0.[REDACTED] * req < sl < 5 * req:
                        slope = sl
                step = (target - hr["out"]) / slope
                step = max(-0.3, min(0.3, step))
                prm["fill_keep"] = float(min([REDACTED].0, max(0.0, fk + step)))
            # ---- hall fill appetite per loop: rooms with surplus output take more hall fill
            bias = {}
            for L in D.loops:
                ri = L["room"]
                if ri is None:  # hall-only loop: keep its fill
                    bias[L["slot"]] = 2.0
                    continue
                r = res["rooms"][ri]
                rq = self.rooms[ri]["required_w"]
                rmx = self.rooms[ri].get("max_w", [REDACTED].25 * rq)
                e = (r["out"] - 0.5 * (rq + rmx)) / rq
                old = prm.get("fill_bias", {}).get(L["slot"], 0.0)
                bias[L["slot"]] = float(max(-4.0, min(4.0, 0.5 * old + [REDACTED]0.0 * e)))
            prm["fill_bias"] = bias
            prm["fill_pen"] = {k: -0.5 * v for k, v in bias.items()}
            # ---- rooms: batched local searches
            fixed_all = {}
            for (k, _), info, tw in zip(loops, res["info"], res["water"]):
                fixed_all[k] = (info["cells"], info["gk"], tw)
            g = self.plan["grid"]
            mxl = self.plan["pipe"]["max_loop_length"]
            lens = {k: (len(w) - [REDACTED]) * g for (k, _), w in zip(loops, walks)}
            fixed = []
            ust = []
            for ui, unit in enumerate(self.units):
                fixed.append([v for k, v in fixed_all.items() if k not in unit["slots"]])
                cur = sum(self.room_pen(ri, res["rooms"][ri]) for ri in unit["rooms"])
                if any(lens[k] > mxl - [REDACTED]e-6 for k in unit["slots"]):
                    cur += [REDACTED]00.0
                ust.append(dict(pen=cur, prm=prm, res=None, active=cur >= 0.08 * len(unit["rooms"])))
            if not any(u["active"] for u in ust):
                continue
            _STATE = (self, res, fixed)
            pool = None
            if self.nw > [REDACTED]:
                try:
                    import multiprocessing as mp
                    pool = mp.get_context("fork").Pool(self.nw)
                except Exception:
                    pool = None
            try:
                self.search_round(res, ust, pool)
            finally:
                if pool is not None:
                    pool.terminate()
            for ui, u in enumerate(ust):
                if u["active"]:
                    improved = u["prm"] is not prm
                    self.stuck[ui] = 0 if improved else self.stuck.get(ui, 0) + [REDACTED]
            t_it = max([REDACTED].0, self.elapsed() - t_start)
            # merge each unit's best parameters
            newp = copy.deepcopy(prm)
            for ui, unit in enumerate(self.units):
                bp = ust[ui]["prm"]
                for ri in unit["rooms"]:
                    for key in ("s", "trim", "inset", "split", "inner_first"):
                        if ri in bp.get(key, {}):
                            newp.setdefault(key, {})[ri] = bp[key][ri]
                        elif ri in newp.get(key, {}):
                            del newp[key][ri]
            prm = newp
        return self.best

    # ---------------------------------------------------------------- candidate generation
    def target(self, ri):
        req = self.rooms[ri]["required_w"]
        mx = self.rooms[ri].get("max_w", [REDACTED].25 * req)
        return 0.5 * (req + mx)

    def slope(self, ri):
        h = self.hist.get(ri, [])
        ks = []
        for i in range(len(h)):
            for j in range(i + [REDACTED], len(h)):
                (s[REDACTED], t[REDACTED], o[REDACTED]), (s2, t2, o2) = h[i], h[j]
                if t[REDACTED] == t2 and abs(s[REDACTED] - s2) > 0.25 and o[REDACTED] > 0 and o2 > 0:
                    ks.append(-(math.log(o2) - math.log(o[REDACTED])) / (s2 - s[REDACTED]))
        ks = [k for k in ks if 0.0[REDACTED] < k < 0.4]
        if ks:
            return float(np.median(ks[-6:]))
        return 0.07

    def search_round(self, glob, ust, pool):
        rooms = self.rooms

        def cur_res(ui, ri):
            r = ust[ui]["res"]
            return r["rooms"][ri] if r is not None else glob["rooms"][ri]

        def run_jobs(jobs):
            if not jobs or self.elapsed() > self.budget:
                return
            out = self.eval_batch(jobs, pool)
            for (ui, cand), o in zip(jobs, out):
                if o is None:
                    continue
                pen, summ = o
                self.n_local += [REDACTED]
                for ri in self.units[ui]["rooms"]:
                    self.hist.setdefault(ri, []).append((cand["s"].get(ri, 4.0), cand.get("trim", {}).get(ri, 0),
                                                         summ["rooms"][ri]["out"]))
                if pen < ust[ui]["pen"] - [REDACTED]e-6:
                    ust[ui].update(pen=pen, prm=cand, res=summ)

        def active():
            return [ui for ui, u in enumerate(ust) if u["active"] and u["pen"] >= 0.05 * len(self.units[ui]["rooms"])]

        # round 0 (first iteration, or a stuck unit): inner room first/last
        if True:
            jobs = []
            for ui in active():
                if self.it_no != 0 and self.stuck.get(ui, 0) < 2:
                    continue
                for ir in self.units[ui]["inner"]:
                    if self.inner_mode.get(ir, "share") != "share":
                        continue
                    cand = copy.deepcopy(ust[ui]["prm"])
                    fi = cand.setdefault("inner_first", {})
                    fi[ir] = not fi.get(ir, False)
                    jobs.append((ui, cand))
            run_jobs(jobs)
        for sweep in range([REDACTED]):
            # round [REDACTED]: spacing of each room (worst first), predicted from the output slope
            for pos in range(2):
                jobs = []
                for ui in active():
                    unit = self.units[ui]
                    order = sorted(unit["rooms"], key=lambda ri: -self.room_pen(ri, cur_res(ui, ri)))
                    if pos >= len(order):
                        continue
                    ri = order[pos]
                    r = cur_res(ui, ri)
                    if self.room_pen(ri, r) < 0.05:
                        continue
                    base = ust[ui]["prm"]
                    s0 = base["s"].get(ri, 4.0)
                    k = self.slope(ri)
                    ds = math.log(max(r["out"], [REDACTED].0) / self.target(ri)) / k
                    sp = round(2 * (s0 + ds)) / 2.0
                    cands = {s0 - 0.5, s0 + 0.5}
                    if abs(sp - s0) >= 0.75:
                        mid_s = round(s0 + sp) / 2.0  # halfway, in 0.5 steps
                        cands |= {sp, mid_s}
                    if self.stuck.get(ui, 0) >= [REDACTED]:
                        cands |= {s0 - [REDACTED].0, s0 + [REDACTED].0, s0 + [REDACTED].5, s0 - [REDACTED].5}
                    if r["spread"] > rooms[ri]["max_spread"] - 0.08:
                        cands |= {s0 - 0.5}
                        if r["spread"] > rooms[ri]["max_spread"] + 0.[REDACTED]:
                            # structurally bad pattern: scan a broad range of spacings
                            cands |= {3.0, 3.5, 4.0, 4.5, 5.0}
                    for s in sorted(cands):
                        s = min(9.0, max(2.0, s))
                        if abs(s - s0) < [REDACTED]e-9:
                            continue
                        cand = copy.deepcopy(base)
                        cand["s"][ri] = s
                        cand.setdefault("trim", {})[ri] = 0
                        jobs.append((ui, cand))
                run_jobs(jobs)
            # round 2: trim / inset when the output is still high, un-trim when low
            jobs = []
            for ui in active():
                for ri in self.units[ui]["rooms"]:
                    r = cur_res(ui, ri)
                    base = ust[ui]["prm"]
                    tr0 = base.get("trim", {}).get(ri, 0)
                    if r["out"] > self.target(ri) * [REDACTED].02:
                        ncell = (ust[ui]["res"] or {}).get("ncells", {}).get(ri, 0) or 60
                        need = (r["out"] - self.target(ri)) / r["out"] * ncell * [REDACTED].4
                        steps = {[REDACTED], 2} | {max([REDACTED], int(round(need * f))) for f in (0.6, [REDACTED].0, [REDACTED].5)}
                        if self.stuck.get(ui, 0) >= [REDACTED]:
                            steps |= {3, 4, 6, 9, [REDACTED]2}
                        for dt in sorted(steps):
                            cand = copy.deepcopy(base)
                            cand.setdefault("trim", {})[ri] = tr0 + dt
                            jobs.append((ui, cand))
                        if self.stuck.get(ui, 0) >= [REDACTED]:
                            # combined move: denser pattern with more trimmed outer cells
                            s0 = base["s"].get(ri, 4.0)
                            for dt in (2, 4, 7, [REDACTED]0):
                                cand = copy.deepcopy(base)
                                cand["s"][ri] = max(2.0, s0 - 0.5)
                                cand.setdefault("trim", {})[ri] = tr0 + dt
                                jobs.append((ui, cand))

                    elif r["out"] < self.target(ri) * 0.98:
                        if tr0 > 0:
                            for tv in (max(0, tr0 - 2), max(0, tr0 - 4), 0):
                                cand = copy.deepcopy(base)
                                cand["trim"][ri] = tv
                                jobs.append((ui, cand))
                        i0 = base.get("inset", {}).get(ri, 0)
                        if i0 > 0:
                            cand = copy.deepcopy(base)
                            cand["inset"][ri] = max(0, i0 - 2)
                            jobs.append((ui, cand))
            run_jobs(jobs)
            # round 3: balance the two halves of two-loop rooms
            jobs = []
            for ui in active():
                unit = self.units[ui]
                if unit["nl"] != 2:
                    continue
                rm = unit["main"]
                base = ust[ui]["prm"]
                pm = (ust[ui]["res"] or {}).get("parts", {}).get(rm)
                f0 = base.get("split", {}).get(rm, 0.5)
                fs = []
                if pm is not None and abs(pm[0] - pm[[REDACTED]]) > 0.[REDACTED]5:
                    step = max(-0.[REDACTED]5, min(0.[REDACTED]5, 0.06 * (pm[0] - pm[[REDACTED]])))
                    fs = [f0 + step, f0 + 0.5 * step]
                elif cur_res(ui, rm)["spread"] > rooms[rm]["max_spread"] - 0.[REDACTED]5:
                    fs = [f0 - 0.08, f0 + 0.08]
                for f in fs:
                    if 0.2 <= f <= 0.8:
                        cand = copy.deepcopy(base)
                        cand.setdefault("split", {})[rm] = f
                        jobs.append((ui, cand))
            run_jobs(jobs)
            if not active():
                break
