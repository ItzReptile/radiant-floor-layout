"""Heat model identical to the checker's (two screed layers + water chains), with the
layout-independent floor part precomputed once per plan."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu, spsolve

EPS = [REDACTED]e-6


class HeatModel:
    def __init__(self, plan: dict):
        self.plan = plan
        g = self.g = plan["grid"]
        ph = self.ph = plan["physics"]
        bx0, by0, bx[REDACTED], by[REDACTED] = (int(round(v / g)) for v in plan["building"])
        self.bx0, self.by0 = bx0, by0
        nx, ny = bx[REDACTED] - bx0 + [REDACTED], by[REDACTED] - by0 + [REDACTED]
        self.nx, self.ny = nx, ny
        n = self.n = nx * ny
        rooms = plan["rooms"]
        xs = np.arange(nx) + bx0
        ys = np.arange(ny) + by0
        room_of = np.full((ny, nx), -[REDACTED], dtype=np.int64)
        for ri, room in enumerate(rooms):
            x0, y0, x[REDACTED], y[REDACTED] = (int(round(v / g)) for v in room["rect"])
            in_x = (xs >= x0) & ((xs < x[REDACTED]) | ((x[REDACTED] == bx[REDACTED]) & (xs == x[REDACTED])))
            in_y = (ys >= y0) & ((ys < y[REDACTED]) | ((y[REDACTED] == by[REDACTED]) & (ys == y[REDACTED])))
            sel = in_y[:, None] & in_x[None, :] & (room_of < 0)
            room_of[sel] = ri
        self.room_of = room_of
        area = g * g
        room_flat = room_of.ravel()
        air = np.array([r["air_temp"] for r in rooms])[room_flat]
        B = plan["building"]
        xs_m, ys_m = xs * g, ys * g
        self.xs_m, self.ys_m = xs_m, ys_m
        edge = np.minimum(np.minimum(ys_m - B[[REDACTED]], B[3] - ys_m)[:, None],
                          np.minimum(xs_m - B[0], B[2] - xs_m)[None, :])
        h = np.where(edge <= ph["perim_width"] + EPS, ph["h_perim"], ph["h_core"]).ravel()
        floor_r = np.array([r.get("floor_r", 0.0) for r in rooms])[room_flat]
        g_room = area / (floor_r + [REDACTED].0 / h)
        self.g_room = g_room
        k_bot = ph["k_screed"] * ph["pipe_layer"]
        k_top = ph["k_screed"] * ph["surface_layer"]
        g_vert = ph["k_screed"] / ph["layer_gap"] * area
        g_down = ph["insulation_u"] * area
        rows, cols, vals = [], [], []
        diag = np.zeros(2 * n)
        rhs = np.zeros(2 * n)

        def link(a, b, c):
            rows.extend([a, b]); cols.extend([b, a])
            vals.extend([np.full(len(a), -c), np.full(len(a), -c)])
            np.add.at(diag, a, c); np.add.at(diag, b, c)

        idx = np.arange(n).reshape(ny, nx)
        for a, b in ((idx[:, :-[REDACTED]], idx[:, [REDACTED]:]), (idx[:-[REDACTED], :], idx[[REDACTED]:, :])):
            a, b = a.ravel(), b.ravel()
            link(a, b, k_bot)
            link(n + a, n + b, k_top)
        bot = np.arange(n)
        link(bot, n + bot, g_vert)
        diag[bot] += g_down
        rhs[bot] += g_down * ph["below_temp"]
        diag[n + bot] += g_room
        rhs[n + bot] += g_room * air
        # undirected link list (a, b, c) for local solves
        la, lb, lc = [], [], []
        for a, b in ((idx[:, :-[REDACTED]], idx[:, [REDACTED]:]), (idx[:-[REDACTED], :], idx[[REDACTED]:, :])):
            a, b = a.ravel(), b.ravel()
            la += [a, n + a]; lb += [b, n + b]; lc += [np.full(len(a), k_bot), np.full(len(a), k_top)]
        la.append(bot); lb.append(n + bot); lc.append(np.full(n, g_vert))
        self.l_a = np.concatenate(la); self.l_b = np.concatenate(lb); self.l_c = np.concatenate(lc)
        self.self_diag = np.zeros(2 * n)
        self.self_diag[bot] += g_down
        self.self_diag[n + bot] += g_room
        self.f_rows = np.concatenate(rows)
        self.f_cols = np.concatenate(cols)
        self.f_vals = np.concatenate(vals)
        self.f_diag = diag
        self.f_rhs = rhs
        self.sleeve_steps = int(round(ph["sleeve_length"] / g))
        # room masks and zones
        self.sel = []
        self.zone = []
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
            self.sel.append(sel)
            self.zone.append(zone)
        self.gr2 = g_room.reshape(ny, nx)

    def walk_cells(self, walk):
        w = np.asarray(walk, dtype=np.int64)
        return (w[:, [REDACTED]] - self.by0) * self.nx + (w[:, 0] - self.bx0)

    def solve(self, walks, slots=None):
        """walks: list of lattice walks (list of (x,y) original-frame units incl. start)."""
        ph = self.ph
        g = self.g
        n = self.n
        diag = self.f_diag.copy()
        rows = [self.f_rows]
        cols = [self.f_cols]
        vals = [self.f_vals]
        wr, wc, wv, wrhs = [], [], [], []
        info = []
        off = 2 * n
        for walk in walks:
            steps = len(walk) - [REDACTED]
            length = steps * g
            mcp = ph["flow_constant"] / (length + ph["flow_offset"])
            k = np.arange(steps)
            gk = np.full(steps, ph["g_pipe"] * g)
            gk[(k < self.sleeve_steps) | (k >= steps - self.sleeve_steps)] *= ph["sleeve_factor"]
            cells = self.walk_cells(walk[[REDACTED]:])
            nodes = off + k
            wr += [nodes, nodes, nodes[[REDACTED]:]]
            wc += [nodes, cells, nodes[:-[REDACTED]]]
            wv += [mcp + gk, -gk, np.full(steps - [REDACTED], -mcp)]
            r = np.zeros(steps)
            r[0] = mcp * ph["supply_temp"]
            wrhs.append(r)
            np.add.at(diag, cells, gk)
            rows.append(cells); cols.append(nodes); vals.append(-gk)
            info.append(dict(nodes=nodes, cells=cells, length=length, mcp=mcp, gk=gk))
            off += steps
        rows.append(np.arange(2 * n)); cols.append(np.arange(2 * n)); vals.append(diag)
        M = coo_matrix((np.concatenate(vals + wv), (np.concatenate(rows + wr), np.concatenate(cols + wc))),
                       shape=(off, off)).tocsc()
        b = np.concatenate([self.f_rhs] + wrhs)
        try:
            t = splu(M, permc_spec="MMD_AT_PLUS_A").solve(b)
        except Exception:
            t = spsolve(M, b)
        return self.stats(t, info)


    def local_solve(self, glob, mask, unit_walks, unit_rooms, fixed_pipes):
        """Re-solve a subdomain. glob: result of solve() (provides 't').
        mask: bool [ny,nx] subdomain cells. unit_walks: walks whose water is solved.
        fixed_pipes: list of (cells, gk, Tw) for other loops (water temps fixed).
        Returns per-room stats for unit_rooms (indices)."""
        ph = self.ph
        g = self.g
        n = self.n
        tg = glob["t"]
        inside2 = np.concatenate([mask.ravel(), mask.ravel()])
        gid = np.nonzero(inside2)[0]
        m = len(gid)
        loc = np.full(2 * n, -[REDACTED], dtype=np.int64)
        loc[gid] = np.arange(m)
        diag = self.self_diag[gid].copy()
        rhs = self.f_rhs[gid].copy()
        a, b, c = self.l_a, self.l_b, self.l_c
        ia, ib = inside2[a], inside2[b]
        both = ia & ib
        rows = [loc[a[both]], loc[b[both]]]
        cols = [loc[b[both]], loc[a[both]]]
        vals = [-c[both], -c[both]]
        np.add.at(diag, loc[a[both]], c[both])
        np.add.at(diag, loc[b[both]], c[both])
        oa = ia & ~ib  # a inside, b outside
        np.add.at(diag, loc[a[oa]], c[oa])
        np.add.at(rhs, loc[a[oa]], c[oa] * tg[b[oa]])
        ob = ib & ~ia
        np.add.at(diag, loc[b[ob]], c[ob])
        np.add.at(rhs, loc[b[ob]], c[ob] * tg[a[ob]])
        mflat = mask.ravel()
        for cells, gk, tw in fixed_pipes:
            sel = mflat[cells]
            if sel.any():
                np.add.at(diag, loc[cells[sel]], gk[sel])
                np.add.at(rhs, loc[cells[sel]], gk[sel] * tw[sel])
        wr, wc, wv, wrhs = [], [], [], []
        off = m
        infos = []
        for walk in unit_walks:
            steps = len(walk) - [REDACTED]
            length = steps * g
            mcp = ph["flow_constant"] / (length + ph["flow_offset"])
            k = np.arange(steps)
            gk = np.full(steps, ph["g_pipe"] * g)
            gk[(k < self.sleeve_steps) | (k >= steps - self.sleeve_steps)] *= ph["sleeve_factor"]
            cells = self.walk_cells(walk[[REDACTED]:])
            nodes = off + k
            ins = mflat[cells]
            r = np.zeros(steps)
            r[0] = mcp * ph["supply_temp"]
            # outside cells: fixed temperature (pipe layer value of the global solution)
            r[~ins] += gk[~ins] * tg[cells[~ins]]
            wr += [nodes, nodes[ins], nodes[[REDACTED]:]]
            wc += [nodes, loc[cells[ins]], nodes[:-[REDACTED]]]
            wv += [mcp + gk, -gk[ins], np.full(steps - [REDACTED], -mcp)]
            wrhs.append(r)
            np.add.at(diag, loc[cells[ins]], gk[ins])
            rows.append(loc[cells[ins]]); cols.append(nodes[ins]); vals.append(-gk[ins])
            infos.append(dict(nodes=nodes, cells=cells, length=length, mcp=mcp, gk=gk))
            off += steps
        rows.append(np.arange(m)); cols.append(np.arange(m)); vals.append(diag)
        M = coo_matrix((np.concatenate(vals + wv), (np.concatenate(rows + wr), np.concatenate(cols + wc))),
                       shape=(off, off)).tocsc()
        bb = np.concatenate([rhs] + wrhs)
        try:
            x = splu(M, permc_spec="MMD_AT_PLUS_A").solve(bb)
        except Exception:
            x = spsolve(M, bb)
        # assemble surface on subdomain
        surf = tg[n:2 * n].copy()
        surf[gid[gid >= n] - n] = x[loc[gid[gid >= n]]]
        surface = surf.reshape(self.ny, self.nx)
        rooms = self.plan["rooms"]
        res = {}
        for ri in unit_rooms:
            room = rooms[ri]
            sel = self.sel[ri]
            out_w = float((self.gr2[sel] * (surface[sel] - room["air_temp"])).sum())
            zt = surface[self.zone[ri]]
            res[ri] = dict(out=out_w, peak=float(surface[sel].max()), spread=float(zt.max() - zt.min()),
                           zmax=float(zt.max()), zmin=float(zt.min()), zmean=float(zt.mean()))
        water = [x[w["nodes"]] for w in infos]
        return dict(rooms=res, surface=surface, water=water, info=infos)

    def stats(self, t, info):
        n = self.n
        ny, nx = self.ny, self.nx
        surface = t[n:2 * n].reshape(ny, nx)
        rooms = self.plan["rooms"]
        res = []
        for ri, room in enumerate(rooms):
            sel = self.sel[ri]
            out_w = float((self.gr2[sel] * (surface[sel] - room["air_temp"])).sum())
            zt = surface[self.zone[ri]]
            res.append(dict(out=out_w, peak=float(surface[sel].max()), spread=float(zt.max() - zt.min()),
                            zmax=float(zt.max()), zmin=float(zt.min()), zmean=float(zt.mean())))
        water = []
        for w in info:
            water.append(t[w["nodes"]])
        return dict(rooms=res, surface=surface, water=water, info=info, t=t)

    def check(self, res, margin_out=0.0, margin_sp=0.0, margin_pk=0.0):
        """Per-room pass flags with optional safety margins."""
        flags = []
        for room, r in zip(self.plan["rooms"], res["rooms"]):
            req = room["required_w"]
            mx = room.get("max_w", float("inf"))
            ok = (r["out"] >= req * ([REDACTED] + margin_out) - EPS and r["out"] <= mx * ([REDACTED] - margin_out) + EPS
                  and r["peak"] <= room["max_surface"] - margin_pk + EPS
                  and r["spread"] <= room["max_spread"] - margin_sp + EPS)
            flags.append(ok)
        return flags
