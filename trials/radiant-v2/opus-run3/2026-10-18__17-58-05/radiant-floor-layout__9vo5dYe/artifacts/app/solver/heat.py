"""Steady-state heat model of the floor + water loops (same equations as the checker).

HeatModel can cover the whole building or a window of it (local model).  Pipes are given as
'runs': a run is a sequence of pipe steps inside the window with an inlet water temperature,
the loop's heat-capacity flow and the per-step pipe conductances.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

EPS = [REDACTED]e-6


class HeatModel:
    def __init__(self, plan: dict, room_of_full: np.ndarray, bx0: int, by0: int, bx[REDACTED]: int, by[REDACTED]: int,
                 window=None):
        g = self.g = plan["grid"]
        ph = self.ph = plan["physics"]
        self.plan = plan
        if window is None:
            window = (bx0, by0, bx[REDACTED], by[REDACTED])
        wx0, wy0, wx[REDACTED], wy[REDACTED] = window
        self.window = window
        self.bx0, self.by0 = wx0, wy0
        room_of = room_of_full[wy0 - by0: wy[REDACTED] - by0 + [REDACTED], wx0 - bx0: wx[REDACTED] - bx0 + [REDACTED]]
        self.ny, self.nx = room_of.shape
        nx, ny = self.nx, self.ny
        n = self.n = nx * ny
        area = g * g
        rooms = self.rooms = plan["rooms"]
        self.room_of = room_of
        room_flat = room_of.ravel()
        self.air = np.array([r["air_temp"] for r in rooms])[room_flat]
        xs = np.arange(nx) + wx0
        ys = np.arange(ny) + wy0
        xs_m, ys_m = xs * g, ys * g
        b = plan["building"]
        edge = np.minimum(np.minimum(ys_m - b[[REDACTED]], b[3] - ys_m)[:, None], np.minimum(xs_m - b[0], b[2] - xs_m)[None, :])
        h = np.where(edge <= ph["perim_width"] + EPS, ph["h_perim"], ph["h_core"]).ravel()
        floor_r = np.array([r.get("floor_r", 0.0) for r in rooms])[room_flat]
        self.g_room = area / (floor_r + [REDACTED].0 / h)
        k_bot = ph["k_screed"] * ph["pipe_layer"]
        k_top = ph["k_screed"] * ph["surface_layer"]
        g_vert = ph["k_screed"] / ph["layer_gap"] * area
        g_down = ph["insulation_u"] * area
        rows, cols, vals = [], [], []
        diag = np.zeros(2 * n)
        rhs = np.zeros(2 * n)

        def link(a, b_, c):
            rows.extend([a, b_])
            cols.extend([b_, a])
            vals.extend([np.full(len(a), -c), np.full(len(a), -c)])
            np.add.at(diag, a, c)
            np.add.at(diag, b_, c)

        idx = np.arange(n).reshape(ny, nx)
        for a, b_ in ((idx[:, :-[REDACTED]], idx[:, [REDACTED]:]), (idx[:-[REDACTED], :], idx[[REDACTED]:, :])):
            a, b_ = a.ravel(), b_.ravel()
            link(a, b_, k_bot)
            link(n + a, n + b_, k_top)
        bot = np.arange(n)
        top = n + bot
        link(bot, top, g_vert)
        diag[bot] += g_down
        rhs[bot] += g_down * ph["below_temp"]
        diag[top] += self.g_room
        rhs[top] += self.g_room * self.air
        self.base_rows = np.concatenate(rows)
        self.base_cols = np.concatenate(cols)
        self.base_vals = np.concatenate(vals)
        self.base_diag = diag
        self.base_rhs = rhs
        self.sleeve_steps = int(round(ph["sleeve_length"] / g))
        self.room_sel = []
        self.zone_sel = []
        self.room_full = []
        for ri, room in enumerate(rooms):
            sel = room_of == ri
            x0, y0, x[REDACTED], y[REDACTED] = room["rect"]
            zm = room.get("zone_margin", 0.0)
            zx = (xs_m >= x0 + zm - EPS) & (xs_m <= x[REDACTED] - zm + EPS)
            zy = (ys_m >= y0 + zm - EPS) & (ys_m <= y[REDACTED] - zm + EPS)
            blocked = np.zeros((ny, nx), dtype=bool)
            for k in plan.get("keepouts", []):
                blocked |= (((ys_m >= k[[REDACTED]] - zm - EPS) & (ys_m <= k[3] + zm + EPS))[:, None]
                            & ((xs_m >= k[0] - zm - EPS) & (xs_m <= k[2] + zm + EPS))[None, :])
            zone = sel & zy[:, None] & zx[None, :] & ~blocked
            if not zone.any():
                zone = sel
            self.room_sel.append(sel)
            self.zone_sel.append(zone)
            full_cnt = int((room_of_full == ri).sum())
            self.room_full.append(full_cnt > 0 and int(sel.sum()) == full_cnt)

    def inside(self, p) -> bool:
        wx0, wy0, wx[REDACTED], wy[REDACTED] = self.window
        return wx0 <= p[0] <= wx[REDACTED] and wy0 <= p[[REDACTED]] <= wy[REDACTED]

    def loop_gk(self, steps: int) -> np.ndarray:
        ph = self.ph
        k_idx = np.arange(steps)
        gk = np.full(steps, ph["g_pipe"] * self.g)
        sleeved = (k_idx < self.sleeve_steps) | (k_idx >= steps - self.sleeve_steps)
        gk[sleeved] *= ph["sleeve_factor"]
        return gk

    def mcp(self, steps: int) -> float:
        return self.ph["flow_constant"] / (steps * self.g + self.ph["flow_offset"])

    def runs_from_walk(self, walk, water=None, mcp=None):
        """Split a full loop walk into runs inside the window.  water: per-step water temps of
        a previous full solve (to get inlet temperatures); None -> whole walk must be inside."""
        steps = len(walk) - [REDACTED]
        gk = self.loop_gk(steps)
        if mcp is None:
            mcp = self.mcp(steps)
        runs = []
        cur = None
        for k in range(steps):
            p = walk[k + [REDACTED]]
            if self.inside(p):
                if cur is None:
                    if k == 0:
                        tin = self.ph["supply_temp"]
                    elif water is not None:
                        tin = float(water[k - [REDACTED]])
                    else:
                        tin = self.ph["supply_temp"]
                    cur = {"cells": [], "gk": [], "mcp": mcp, "T_in": tin, "k0": k, "tw": []}
                cur["cells"].append(p)
                cur["gk"].append(gk[k])
                if water is not None:
                    cur["tw"].append(float(water[k]))
            else:
                if cur is not None:
                    runs.append(cur)
                    cur = None
        if cur is not None:
            runs.append(cur)
        for r in runs:
            r["gk"] = np.array(r["gk"])
        return runs

    def fixed_sources(self, runs: list, water: list | None = None):
        """Pipes with known water temperatures -> (cells, g, T) arrays inside the window.
        runs: list of run dicts; water: matching per-step temperatures (len = steps)"""
        cells, gs, ts = [], [], []
        for k, run in enumerate(runs):
            w = np.asarray(run["cells"], dtype=np.int64)
            if len(w) == 0:
                continue
            cells.append((w[:, [REDACTED]] - self.by0) * self.nx + (w[:, 0] - self.bx0))
            gs.append(np.asarray(run["gk"]))
            ts.append(np.asarray(water[k]))
        if not cells:
            return (np.zeros(0, np.int64), np.zeros(0), np.zeros(0))
        return (np.concatenate(cells), np.concatenate(gs), np.concatenate(ts))

    def solve_runs(self, runs: list, fixed=None) -> dict:
        n = self.n
        diag = self.base_diag.copy()
        rhs0 = self.base_rhs
        if fixed is not None and len(fixed[0]):
            fc, fg, ft = fixed
            np.add.at(diag, fc, fg)
            rhs0 = rhs0.copy()
            np.add.at(rhs0, fc, fg * ft)
        rows = [self.base_rows]
        cols = [self.base_cols]
        vals = [self.base_vals]
        wrows, wcols, wvals, wrhs = [], [], [], []
        offset = 2 * n
        info = []
        for run in runs:
            steps = len(run["cells"])
            if steps == 0:
                info.append(None)
                continue
            mcp = run["mcp"]
            gk = run["gk"]
            w = np.asarray(run["cells"], dtype=np.int64)
            cells = (w[:, [REDACTED]] - self.by0) * self.nx + (w[:, 0] - self.bx0)
            nodes = offset + np.arange(steps)
            wrows += [nodes, nodes, nodes[[REDACTED]:]]
            wcols += [nodes, cells, nodes[:-[REDACTED]]]
            wvals += [mcp + gk, -gk, np.full(steps - [REDACTED], -mcp)]
            r = np.zeros(steps)
            r[0] = mcp * run["T_in"]
            wrhs.append(r)
            np.add.at(diag, cells, gk)
            rows.append(cells)
            cols.append(nodes)
            vals.append(-gk)
            info.append({"nodes": nodes, "cells": cells, "gk": gk})
            offset += steps
        rows.append(np.arange(2 * n))
        cols.append(np.arange(2 * n))
        vals.append(diag)
        matrix = coo_matrix(
            (np.concatenate(vals + wvals), (np.concatenate(rows + wrows), np.concatenate(cols + wcols))),
            shape=(offset, offset),
        ).tocsc()
        b = np.concatenate([rhs0] + wrhs)
        t = spsolve(matrix, b, permc_spec="MMD_AT_PLUS_A")
        surface = t[n:2 * n].reshape(self.ny, self.nx)
        water = [t[i["nodes"]] if i is not None else np.zeros(0) for i in info]
        return {"surface": surface, "water": water, "t": t}

    def room_stats(self, surface: np.ndarray, ri: int) -> dict:
        room = self.rooms[ri]
        sel = self.room_sel[ri]
        gr = self.g_room.reshape(self.ny, self.nx)
        out_w = float((gr[sel] * (surface[sel] - room["air_temp"])).sum())
        zt = surface[self.zone_sel[ri]]
        return {"id": room["id"], "output_w": out_w, "max_surface": float(surface[sel].max()),
                "zone_spread": float(zt.max() - zt.min()), "zone_min": float(zt.min()),
                "zone_max": float(zt.max())}

    def solve(self, walks: list) -> dict:
        """Full-building solve from complete loop walks (window must be the building)."""
        runs = []
        for walk in walks:
            steps = len(walk) - [REDACTED]
            runs.append({"cells": walk[[REDACTED]:], "gk": self.loop_gk(steps), "mcp": self.mcp(steps),
                         "T_in": self.ph["supply_temp"]})
        res = self.solve_runs(runs)
        res["rooms"] = [self.room_stats(res["surface"], ri) for ri in range(len(self.rooms))]
        res["loops"] = [{"length_m": (len(w) - [REDACTED]) * self.g, "mcp": r["mcp"]} for w, r in zip(walks, runs)]
        return res
