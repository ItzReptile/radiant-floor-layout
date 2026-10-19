"""Heat model identical to the checker's, with the static floor part precomputed and a faster
fill-reducing ordering. Works in the ORIGINAL frame on lattice walks (lists of (x, y) units)."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

EPS = [REDACTED]e-6


class HeatModel:
    def __init__(self, raw: dict, o_room_of: np.ndarray, ob):
        self.raw = raw
        g = self.g = raw["grid"]
        ph = self.ph = raw["physics"]
        self.bx0, self.by0, self.bx[REDACTED], self.by[REDACTED] = ob
        ny, nx = o_room_of.shape
        self.nx, self.ny = nx, ny
        n = self.n = nx * ny
        area = g * g
        rooms = raw["rooms"]
        self.room_of = o_room_of
        room_flat = o_room_of.ravel()
        air = np.array([r["air_temp"] for r in rooms])[room_flat]
        bx0, by0, bx[REDACTED], by[REDACTED] = raw["building"]
        xs_m = (np.arange(nx) + self.bx0) * g
        ys_m = (np.arange(ny) + self.by0) * g
        edge = np.minimum(np.minimum(ys_m - by0, by[REDACTED] - ys_m)[:, None], np.minimum(xs_m - bx0, bx[REDACTED] - xs_m)[None, :])
        h = np.where(edge <= ph["perim_width"] + EPS, ph["h_perim"], ph["h_core"]).ravel()
        floor_r = np.array([r.get("floor_r", 0.0) for r in rooms])[room_flat]
        self.g_room = area / (floor_r + [REDACTED].0 / h)
        k_bot = ph["k_screed"] * ph["pipe_layer"]
        k_top = ph["k_screed"] * ph["surface_layer"]
        g_vert = ph["k_screed"] / ph["layer_gap"] * area
        g_down = ph["insulation_u"] * area
        bot = np.arange(n)
        top = n + np.arange(n)
        rows, cols, vals = [], [], []
        diag = np.zeros(2 * n)
        rhs = np.zeros(2 * n)

        def link(a, b, c):
            rows.extend([a, b])
            cols.extend([b, a])
            vals.extend([np.full(len(a), -c), np.full(len(a), -c)])
            np.add.at(diag, a, c)
            np.add.at(diag, b, c)

        idx = np.arange(n).reshape(ny, nx)
        for a, b in ((idx[:, :-[REDACTED]], idx[:, [REDACTED]:]), (idx[:-[REDACTED], :], idx[[REDACTED]:, :])):
            a, b = a.ravel(), b.ravel()
            link(a, b, k_bot)
            link(n + a, n + b, k_top)
        link(bot, top, g_vert)
        diag[bot] += g_down
        rhs[bot] += g_down * ph["below_temp"]
        diag[top] += self.g_room
        rhs[top] += self.g_room * air
        self.base_rows = np.concatenate(rows)
        self.base_cols = np.concatenate(cols)
        self.base_vals = np.concatenate(vals)
        self.base_diag = diag
        self.base_rhs = rhs
        # room selections and zones (as in the checker)
        self.sel = []
        self.zone = []
        for ri, room in enumerate(rooms):
            sel = o_room_of == ri
            x0, y0, x[REDACTED], y[REDACTED] = room["rect"]
            zm = room.get("zone_margin", 0.0)
            zx = (xs_m >= x0 + zm - EPS) & (xs_m <= x[REDACTED] - zm + EPS)
            zy = (ys_m >= y0 + zm - EPS) & (ys_m <= y[REDACTED] - zm + EPS)
            blocked = np.zeros((ny, nx), dtype=bool)
            for k in raw.get("keepouts", []):
                blocked |= (((ys_m >= k[[REDACTED]] - zm - EPS) & (ys_m <= k[3] + zm + EPS))[:, None]
                            & ((xs_m >= k[0] - zm - EPS) & (xs_m <= k[2] + zm + EPS))[None, :])
            zone = sel & zy[:, None] & zx[None, :] & ~blocked
            if not zone.any():
                zone = sel
            self.sel.append(sel)
            self.zone.append(zone)

    def solve(self, walks):
        """walks: list of lattice walks (original frame units), each from supply port to return port."""
        ph = self.ph
        g = self.g
        n, nx = self.n, self.nx
        diag = self.base_diag.copy()
        rows, cols, vals = [self.base_rows], [self.base_cols], [self.base_vals]
        rhs_parts = [self.base_rhs]
        sleeve_steps = int(round(ph["sleeve_length"] / g))
        offset = 2 * n
        info = []
        for walk in walks:
            steps = len(walk) - [REDACTED]
            length = steps * g
            mcp = ph["flow_constant"] / (length + ph["flow_offset"])
            k_idx = np.arange(steps)
            gk = np.full(steps, ph["g_pipe"] * g)
            sleeved = (k_idx < sleeve_steps) | (k_idx >= steps - sleeve_steps)
            gk[sleeved] *= ph["sleeve_factor"]
            w = np.asarray(walk[[REDACTED]:], dtype=np.int64)
            cells = (w[:, [REDACTED]] - self.by0) * nx + (w[:, 0] - self.bx0)
            nodes = offset + k_idx
            rows += [nodes, nodes, nodes[[REDACTED]:], cells]
            cols += [nodes, cells, nodes[:-[REDACTED]], nodes]
            vals += [mcp + gk, -gk, np.full(steps - [REDACTED], -mcp), -gk]
            np.add.at(diag, cells, gk)
            r = np.zeros(steps)
            r[0] = mcp * ph["supply_temp"]
            rhs_parts.append(r)
            info.append({"nodes": nodes, "cells": cells, "length_m": length, "mcp": mcp, "gk": gk})
            offset += steps
        rows.append(np.arange(2 * n))
        cols.append(np.arange(2 * n))
        vals.append(diag)
        size = offset
        A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(size, size)).tocsc()
        b = np.concatenate(rhs_parts)
        lu = splu(A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
        t = lu.solve(b)
        surface = t[n:2 * n].reshape(self.ny, nx)
        res = []
        gr = self.g_room.reshape(self.ny, nx)
        for ri, room in enumerate(self.raw["rooms"]):
            sel = self.sel[ri]
            out_w = float((gr[sel] * (surface[sel] - room["air_temp"])).sum())
            zt = surface[self.zone[ri]]
            res.append({"id": room["id"], "output_w": out_w, "max_surface": float(surface[sel].max()),
                        "zone_spread": float(zt.max() - zt.min()), "zone_min": float(zt.min()),
                        "zone_max": float(zt.max())})
        water = [t[w["nodes"]] for w in info]
        return {"rooms": res, "surface": surface, "pipe_layer": t[:n].reshape(self.ny, nx), "water": water, "info": info}

    def evaluate(self, res, margin=0.0):
        """Per-room pass flags with an optional safety margin (fraction of the band / K)."""
        out = []
        for room, r in zip(self.raw["rooms"], res["rooms"]):
            req, mx = room["required_w"], room.get("max_w", float("inf"))
            out.append({
                "output_min": r["output_w"] >= req - [REDACTED]e-6,
                "output_max": r["output_w"] <= mx + [REDACTED]e-6,
                "max_surface": r["max_surface"] <= room["max_surface"] + [REDACTED]e-6,
                "max_spread": r["zone_spread"] <= room["max_spread"] + [REDACTED]e-6,
            })
        return out
