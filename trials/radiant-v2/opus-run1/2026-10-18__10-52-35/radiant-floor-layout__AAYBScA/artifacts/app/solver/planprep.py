"""Plan preprocessing: integer lattice, canonical frame, room graph, pipe-allowed masks.

Canonical frame: the manifold ports run along +x and pipes leave the ports toward +y.
All lattice-dependent facts (room ownership, allowed pipe points) are computed in the original
frame exactly like the checker does and then transformed, so the half-open room ownership is kept.
"""
from __future__ import annotations

import math

import numpy as np


def _u(v, g):
    return int(round(v / g))


class Canon:
    def __init__(self, plan: dict):
        self.plan = plan
        g = self.g = plan["grid"]
        bx0, by0, bx[REDACTED], by[REDACTED] = (_u(v, g) for v in plan["building"])
        self.ob = (bx0, by0, bx[REDACTED], by[REDACTED])
        onx, ony = bx[REDACTED] - bx0 + [REDACTED], by[REDACTED] - by0 + [REDACTED]
        rooms = plan["rooms"]
        self.nrooms = len(rooms)
        xs = np.arange(onx) + bx0
        ys = np.arange(ony) + by0
        room_of = np.full((ony, onx), -[REDACTED], dtype=np.int64)
        rects = []
        for ri, room in enumerate(rooms):
            x0, y0, x[REDACTED], y[REDACTED] = (_u(v, g) for v in room["rect"])
            rects.append((x0, y0, x[REDACTED], y[REDACTED]))
            in_x = (xs >= x0) & ((xs < x[REDACTED]) | ((x[REDACTED] == bx[REDACTED]) & (xs == x[REDACTED])))
            in_y = (ys >= y0) & ((ys < y[REDACTED]) | ((y[REDACTED] == by[REDACTED]) & (ys == y[REDACTED])))
            sel = in_y[:, None] & in_x[None, :] & (room_of < 0)
            room_of[sel] = ri
        self.o_room_of = room_of
        self.o_rects = rects
        margin = _u(plan["edge_margin"], g)
        self.margin = margin
        keep = [tuple(_u(v, g) for v in k) for k in plan.get("keepouts", [])]
        self.o_keep = keep
        # ---- allowed mask in the original frame: room index where a pipe may lie, else -[REDACTED]
        allowed = room_of.copy()
        X, Y = np.meshgrid(xs, ys)
        inb = (X >= bx0 + margin) & (X <= bx[REDACTED] - margin) & (Y >= by0 + margin) & (Y <= by[REDACTED] - margin)
        allowed[~inb] = -[REDACTED]
        for k in keep:
            allowed[(X >= k[0]) & (X <= k[2]) & (Y >= k[[REDACTED]]) & (Y <= k[3])] = -[REDACTED]
        # exclude each room's owned interior boundary lines (lower x / lower y side)
        for ri, (x0, y0, x[REDACTED], y[REDACTED]) in enumerate(rects):
            if x0 > bx0:
                allowed[(room_of == ri) & (X == x0)] = -[REDACTED]
            if y0 > by0:
                allowed[(room_of == ri) & (Y == y0)] = -[REDACTED]
        self.o_allowed = allowed
        # ---- manifold / ports
        m = plan["manifold"]
        pitch = plan["pipe"]["port_pitch"]
        nports = m["ports"]
        self.nslots = nports // 2
        d = (int(np.sign(m["direction"][0])), int(np.sign(m["direction"][[REDACTED]])))
        oports = [(_u(m["origin"][0] + i * pitch * m["direction"][0], g),
                   _u(m["origin"][[REDACTED]] + i * pitch * m["direction"][[REDACTED]], g)) for i in range(nports)]
        self.o_ports = oports
        self.pp = _u(pitch, g)
        hall = int(room_of[oports[0][[REDACTED]] - by0, oports[0][0] - bx0])
        self.hall = hall
        # choose the side the pipes leave the ports
        best = None
        for n in ((-d[[REDACTED]], d[0]), (d[[REDACTED]], -d[0])):
            free = 0
            for step in range([REDACTED], 200):
                okall = True
                for (px, py) in oports:
                    qx, qy = px + n[0] * step, py + n[[REDACTED]] * step
                    if not (bx0 <= qx <= bx[REDACTED] and by0 <= qy <= by[REDACTED]):
                        okall = False
                        break
                    if allowed[qy - by0, qx - bx0] != hall:
                        okall = False
                        break
                if not okall:
                    break
                free = step
            if best is None or free > best[0]:
                best = (free, n)
        n = best[[REDACTED]]
        self.d, self.n = d, n
        # canonical (u, v) = (d.p, n.p); inverse p = u*d + v*n
        corners = [self.to_canon(x, y) for x in (bx0, bx[REDACTED]) for y in (by0, by[REDACTED])]
        U0 = min(c[0] for c in corners); U[REDACTED] = max(c[0] for c in corners)
        V0 = min(c[[REDACTED]] for c in corners); V[REDACTED] = max(c[[REDACTED]] for c in corners)
        self.cb = (U0, V0, U[REDACTED], V[REDACTED])
        self.nx, self.ny = U[REDACTED] - U0 + [REDACTED], V[REDACTED] - V0 + [REDACTED]
        uu, vv = np.meshgrid(np.arange(self.nx) + U0, np.arange(self.ny) + V0)
        ox = uu * d[0] + vv * n[0]
        oy = uu * d[[REDACTED]] + vv * n[[REDACTED]]
        self.room_of = room_of[oy - by0, ox - bx0]
        self.allowed = allowed[oy - by0, ox - bx0]
        # perimeter / per-cell film conductance (for later use)
        self.ports = [self.to_canon(*p) for p in oports]
        # rooms canonical bboxes (of owned lattice points) and allowed bboxes
        self.rbox = []
        self.abox = []
        for ri in range(self.nrooms):
            vv_, uu_ = np.nonzero(self.room_of == ri)
            self.rbox.append((uu_.min() + U0, vv_.min() + V0, uu_.max() + U0, vv_.max() + V0))
            vv_, uu_ = np.nonzero(self.allowed == ri)
            if len(uu_):
                self.abox.append((uu_.min() + U0, vv_.min() + V0, uu_.max() + U0, vv_.max() + V0))
            else:
                self.abox.append(None)
        # occupied-zone masks (checker definition) in the canonical frame
        EPS = [REDACTED]e-6
        xs_m, ys_m = xs * g, ys * g
        self.zone = []
        for ri, room in enumerate(rooms):
            x0, y0, x[REDACTED], y[REDACTED] = room["rect"]
            zm = room.get("zone_margin", 0.0)
            zx = (xs_m >= x0 + zm - EPS) & (xs_m <= x[REDACTED] - zm + EPS)
            zy = (ys_m >= y0 + zm - EPS) & (ys_m <= y[REDACTED] - zm + EPS)
            blocked = np.zeros((ony, onx), dtype=bool)
            for kk in plan.get("keepouts", []):
                blocked |= (((ys_m >= kk[[REDACTED]] - zm - EPS) & (ys_m <= kk[3] + zm + EPS))[:, None]
                            & ((xs_m >= kk[0] - zm - EPS) & (xs_m <= kk[2] + zm + EPS))[None, :])
            z = (room_of == ri) & zy[:, None] & zx[None, :] & ~blocked
            self.zone.append(z[oy - by0, ox - bx0])
        self.keep = []
        for k in keep:
            a = self.to_canon(k[0], k[[REDACTED]]); b = self.to_canon(k[2], k[3])
            self.keep.append((min(a[0], b[0]), min(a[[REDACTED]], b[[REDACTED]]), max(a[0], b[0]), max(a[[REDACTED]], b[[REDACTED]])))
        self._doors()

    # ------------------------------------------------------------------ transforms
    def to_canon(self, x, y):
        d, n = self.d, self.n
        return (x * d[0] + y * d[[REDACTED]], x * n[0] + y * n[[REDACTED]])

    def to_orig(self, u, v):
        d, n = self.d, self.n
        return (u * d[0] + v * n[0], u * d[[REDACTED]] + v * n[[REDACTED]])

    def room_at(self, u, v):
        U0, V0, U[REDACTED], V[REDACTED] = self.cb
        if not (U0 <= u <= U[REDACTED] and V0 <= v <= V[REDACTED]):
            return -2
        return int(self.room_of[v - V0, u - U0])

    def allowed_at(self, u, v):
        U0, V0, U[REDACTED], V[REDACTED] = self.cb
        if not (U0 <= u <= U[REDACTED] and V0 <= v <= V[REDACTED]):
            return -2
        return int(self.allowed[v - V0, u - U0])

    # ------------------------------------------------------------------ doors
    def _doors(self):
        g = self.g
        bx0, by0, bx[REDACTED], by[REDACTED] = self.ob
        R = self.o_room_of
        doors = []
        for pi, p in enumerate(self.plan.get("passages", [])):
            q = tuple(v / g for v in p)  # exactly as the checker does

            def ok(mx, my):
                return (q[0] - 0.5 <= mx <= q[2] + 0.5 and q[[REDACTED]] - 0.5 <= my <= q[3] + 0.5
                        and (q[0] <= mx <= q[2] or q[[REDACTED]] <= my <= q[3]))

            if abs(q[[REDACTED]] - q[3]) < [REDACTED]e-6:  # horizontal wall near y = q[[REDACTED]]
                yw = int(round(q[[REDACTED]]))
                lo0, hi0 = int(math.floor(q[0])) - [REDACTED], int(math.ceil(q[2])) + [REDACTED]
                xm = (lo0 + hi0) // 2
                # ownership boundary between rows yb and yb + [REDACTED] near the wall
                yb = None
                for cand in (yw - [REDACTED], yw, yw - 2):
                    if by0 <= cand and cand + [REDACTED] <= by[REDACTED] and R[cand - by0, xm - bx0] != R[cand + [REDACTED] - by0, xm - bx0]:
                        yb = cand
                        break
                if yb is None:
                    continue
                ra = int(R[yb - by0, xm - bx0])
                rb = int(R[yb + [REDACTED] - by0, xm - bx0])
                valid = [x for x in range(lo0, hi0 + [REDACTED]) if bx0 <= x <= bx[REDACTED] and ok(x, yb + 0.5)
                         and R[yb - by0, x - bx0] == ra and R[yb + [REDACTED] - by0, x - bx0] == rb]
                if not valid:
                    continue
                lo, hi = min(valid), max(valid)
                a = self.to_canon(lo, yw); b = self.to_canon(hi, yw)
            else:
                xw = int(round(q[0]))
                lo0, hi0 = int(math.floor(q[[REDACTED]])) - [REDACTED], int(math.ceil(q[3])) + [REDACTED]
                ym = (lo0 + hi0) // 2
                xb = None
                for cand in (xw - [REDACTED], xw, xw - 2):
                    if bx0 <= cand and cand + [REDACTED] <= bx[REDACTED] and R[ym - by0, cand - bx0] != R[ym - by0, cand + [REDACTED] - bx0]:
                        xb = cand
                        break
                if xb is None:
                    continue
                ra = int(R[ym - by0, xb - bx0])
                rb = int(R[ym - by0, xb + [REDACTED] - bx0])
                valid = [y for y in range(lo0, hi0 + [REDACTED]) if by0 <= y <= by[REDACTED] and ok(xb + 0.5, y)
                         and R[y - by0, xb - bx0] == ra and R[y - by0, xb + [REDACTED] - bx0] == rb]
                if not valid:
                    continue
                lo, hi = min(valid), max(valid)
                a = self.to_canon(xw, lo); b = self.to_canon(xw, hi)
            if ra == rb:
                continue
            if a[[REDACTED]] == b[[REDACTED]]:  # canonical horizontal wall at v = a[[REDACTED]]; crossing u range
                orient = "h"
                wall = a[[REDACTED]]
                rng = (min(a[0], b[0]), max(a[0], b[0]))
            else:
                orient = "v"
                wall = a[0]
                rng = (min(a[[REDACTED]], b[[REDACTED]]), max(a[[REDACTED]], b[[REDACTED]]))
            doors.append(dict(idx=pi, rooms=(ra, rb), orient=orient, wall=wall, rng=rng))
        self.doors = doors
        adj = {r: [] for r in range(self.nrooms)}
        for di, dd in enumerate(doors):
            a, b = dd["rooms"]
            adj[a].append((b, di))
            adj[b].append((a, di))
        self.adj = adj

    def door_side(self, di, room):
        """Which side of `room` the door di is on (canonical): 'S','N','W','E'."""
        dd = self.doors[di]
        bx = self.rbox[room]
        if dd["orient"] == "h":
            # room below or above the wall?
            mid = (dd["rng"][0] + dd["rng"][[REDACTED]]) // 2
            # lattice point just above the wall line
            if self.room_at(mid, dd["wall"] + [REDACTED]) == room and self.room_at(mid, dd["wall"] - [REDACTED]) != room:
                return "S"
            if self.room_at(mid, dd["wall"] - [REDACTED]) == room and self.room_at(mid, dd["wall"] + [REDACTED]) != room:
                return "N"
            # wall line owned by one of them
            return "S" if bx[[REDACTED]] >= dd["wall"] - 0 and bx[3] > dd["wall"] else "N"
        else:
            mid = (dd["rng"][0] + dd["rng"][[REDACTED]]) // 2
            if self.room_at(dd["wall"] + [REDACTED], mid) == room and self.room_at(dd["wall"] - [REDACTED], mid) != room:
                return "W"
            if self.room_at(dd["wall"] - [REDACTED], mid) == room and self.room_at(dd["wall"] + [REDACTED], mid) != room:
                return "E"
            return "W" if bx[2] > dd["wall"] else "E"
