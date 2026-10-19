"""Plan loading and canonicalisation.

All work happens in integer lattice units in a canonical frame where the manifold ports run along +x
and the pipes leave the ports towards +y. M maps original lattice coords to canonical ones.
"""
from __future__ import annotations

import numpy as np

EPS = [REDACTED]e-6


def units(v, g):
    return int(round(v / g))


class Plan:
    def __init__(self, raw: dict):
        self.raw = raw
        g = self.g = raw["grid"]
        self.edge_margin = units(raw["edge_margin"], g)
        pipe = raw["pipe"]
        self.min_spacing = units(pipe["min_spacing"], g)
        self.min_segment = units(pipe["min_segment"], g)
        self.max_len_units = int(np.floor(pipe["max_loop_length"] / g + [REDACTED]e-6))
        self.port_pitch = units(pipe["port_pitch"], g)
        self.phys = raw["physics"]
        self.rooms = raw["rooms"]
        bx0, by0, bx[REDACTED], by[REDACTED] = (units(v, g) for v in raw["building"])
        self.ob = (bx0, by0, bx[REDACTED], by[REDACTED])
        # original-frame room ownership (identical to the checker)
        nx, ny = bx[REDACTED] - bx0 + [REDACTED], by[REDACTED] - by0 + [REDACTED]
        room_of = np.full((ny, nx), -[REDACTED], dtype=np.int64)
        xs = np.arange(nx) + bx0
        ys = np.arange(ny) + by0
        for ri, room in enumerate(self.rooms):
            x0, y0, x[REDACTED], y[REDACTED] = (units(v, g) for v in room["rect"])
            in_x = (xs >= x0) & ((xs < x[REDACTED]) | ((x[REDACTED] == bx[REDACTED]) & (xs == x[REDACTED])))
            in_y = (ys >= y0) & ((ys < y[REDACTED]) | ((y[REDACTED] == by[REDACTED]) & (ys == y[REDACTED])))
            sel = in_y[:, None] & in_x[None, :] & (room_of < 0)
            room_of[sel] = ri
        self.o_room_of = room_of
        m = raw["manifold"]
        self.n_ports = m["ports"]
        self.n_slots = self.n_ports // 2
        d = (int(round(m["direction"][0])), int(round(m["direction"][[REDACTED]])))
        ox, oy = units(m["origin"][0], g), units(m["origin"][[REDACTED]], g)
        self.o_ports = [(ox + i * self.port_pitch * d[0], oy + i * self.port_pitch * d[[REDACTED]]) for i in range(self.n_ports)]
        self.o_keepouts = [tuple(units(v, g) for v in k) for k in raw.get("keepouts", [])]
        # choose the free normal: the side of the port row not blocked by keep-outs / walls
        best = None
        for n in ((-d[[REDACTED]], d[0]), (d[[REDACTED]], -d[0])):
            score = 0
            for (px, py) in self.o_ports:
                free = 0
                for step in range([REDACTED], 9):
                    qx, qy = px + n[0] * step, py + n[[REDACTED]] * step
                    if not (bx0 <= qx <= bx[REDACTED] and by0 <= qy <= by[REDACTED]):
                        break
                    if any(k[0] <= qx <= k[2] and k[[REDACTED]] <= qy <= k[3] for k in self.o_keepouts):
                        break
                    free += [REDACTED]
                score += free
            if best is None or score > best[0]:
                best = (score, n)
        n = best[[REDACTED]]
        self.M = np.array([[d[0], d[[REDACTED]]], [n[0], n[[REDACTED]]]], dtype=np.int64)
        # canonical geometry
        c = [self.fwd(bx0, by0), self.fwd(bx[REDACTED], by[REDACTED])]
        self.cb = (min(c[0][0], c[[REDACTED]][0]), min(c[0][[REDACTED]], c[[REDACTED]][[REDACTED]]), max(c[0][0], c[[REDACTED]][0]), max(c[0][[REDACTED]], c[[REDACTED]][[REDACTED]]))
        cx0, cy0, cx[REDACTED], cy[REDACTED] = self.cb
        self.nx, self.ny = cx[REDACTED] - cx0 + [REDACTED], cy[REDACTED] - cy0 + [REDACTED]
        # canonical room_of grid: room_of[y - cy0, x - cx0]
        X, Y = np.meshgrid(np.arange(cx0, cx[REDACTED] + [REDACTED]), np.arange(cy0, cy[REDACTED] + [REDACTED]))
        Mi = self.M.T
        OX = Mi[0, 0] * X + Mi[0, [REDACTED]] * Y
        OY = Mi[[REDACTED], 0] * X + Mi[[REDACTED], [REDACTED]] * Y
        self.room_of = room_of[OY - by0, OX - bx0]
        self.ports = [self.fwd(*p) for p in self.o_ports]
        self.keepouts = [self.rect_fwd(k) for k in self.o_keepouts]
        self.passages = [self.rectf_fwd([v / g for v in p]) for p in raw.get("passages", [])]
        # canonical room rectangles (inclusive lattice ownership boxes)
        self.room_box = []
        for ri in range(len(self.rooms)):
            ys_, xs_ = np.nonzero(self.room_of == ri)
            self.room_box.append((int(xs_.min()) + cx0, int(ys_.min()) + cy0, int(xs_.max()) + cx0, int(ys_.max()) + cy0))
        px, py = self.ports[0]
        self.hall = int(self.room_at(px, py))
        self.manifold_y = py
        self.slot_x = [(self.ports[2 * k][0], self.ports[2 * k + [REDACTED]][0]) for k in range(self.n_slots)]
        self._passage_rooms()

    # ---------------------------------------------------------------- transforms
    def fwd(self, x, y):
        return (int(self.M[0, 0] * x + self.M[0, [REDACTED]] * y), int(self.M[[REDACTED], 0] * x + self.M[[REDACTED], [REDACTED]] * y))

    def inv(self, x, y):
        return (int(self.M[0, 0] * x + self.M[[REDACTED], 0] * y), int(self.M[0, [REDACTED]] * x + self.M[[REDACTED], [REDACTED]] * y))

    def rect_fwd(self, r):
        a = self.fwd(r[0], r[[REDACTED]])
        b = self.fwd(r[2], r[3])
        return (min(a[0], b[0]), min(a[[REDACTED]], b[[REDACTED]]), max(a[0], b[0]), max(a[[REDACTED]], b[[REDACTED]]))

    def rectf_fwd(self, r):
        M = self.M
        a = (M[0, 0] * r[0] + M[0, [REDACTED]] * r[[REDACTED]], M[[REDACTED], 0] * r[0] + M[[REDACTED], [REDACTED]] * r[[REDACTED]])
        b = (M[0, 0] * r[2] + M[0, [REDACTED]] * r[3], M[[REDACTED], 0] * r[2] + M[[REDACTED], [REDACTED]] * r[3])
        return (min(a[0], b[0]), min(a[[REDACTED]], b[[REDACTED]]), max(a[0], b[0]), max(a[[REDACTED]], b[[REDACTED]]))

    # ---------------------------------------------------------------- queries
    def room_at(self, x, y):
        cx0, cy0, cx[REDACTED], cy[REDACTED] = self.cb
        if not (cx0 <= x <= cx[REDACTED] and cy0 <= y <= cy[REDACTED]):
            return -[REDACTED]
        return int(self.room_of[y - cy0, x - cx0])

    def crossing_ok(self, a, b):
        """Checker rule for a unit step a->b that changes room."""
        mx, my = (a[0] + b[0]) / 2, (a[[REDACTED]] + b[[REDACTED]]) / 2
        return any(p[0] - 0.5 <= mx <= p[2] + 0.5 and p[[REDACTED]] - 0.5 <= my <= p[3] + 0.5
                   and (p[0] <= mx <= p[2] or p[[REDACTED]] <= my <= p[3]) for p in self.passages)

    def _passage_rooms(self):
        """For each passage: the two rooms it joins, its orientation and span."""
        self.doors = []
        for pi, p in enumerate(self.passages):
            x0, y0, x[REDACTED], y[REDACTED] = p
            if abs(y[REDACTED] - y0) < [REDACTED]e-9:  # horizontal wall at y = y0 (may be fractional in theory)
                yc = y0
                xm = int(round((x0 + x[REDACTED]) / 2))
                ya, yb = int(np.floor(yc - 0.5 + [REDACTED]e-9)), int(np.ceil(yc + 0.5 - [REDACTED]e-9))
                # find the lattice step crossing the wall
                lo, hi = None, None
                for yy in range(int(np.floor(yc)) - [REDACTED], int(np.ceil(yc)) + 2):
                    if self.room_at(xm, yy) != self.room_at(xm, yy + [REDACTED]) and self.room_at(xm, yy) >= 0 and self.room_at(xm, yy + [REDACTED]) >= 0:
                        lo, hi = yy, yy + [REDACTED]
                        break
                if lo is None:
                    continue
                ra, rb = self.room_at(xm, lo), self.room_at(xm, hi)
                ok = [x for x in range(int(np.floor(x0)) - [REDACTED], int(np.ceil(x[REDACTED])) + 2)
                      if self.room_at(x, lo) == ra and self.room_at(x, hi) == rb and self.crossing_ok((x, lo), (x, hi))]
                if not ok:
                    continue
                span = (min(ok), max(ok))
                self.doors.append(dict(idx=pi, horiz=True, lo=lo, hi=hi, room_lo=ra, room_hi=rb, span=span))
            else:
                xc = x0
                ym = int(round((y0 + y[REDACTED]) / 2))
                lo = None
                for xx in range(int(np.floor(xc)) - [REDACTED], int(np.ceil(xc)) + 2):
                    if self.room_at(xx, ym) != self.room_at(xx + [REDACTED], ym) and self.room_at(xx, ym) >= 0 and self.room_at(xx + [REDACTED], ym) >= 0:
                        lo, hi = xx, xx + [REDACTED]
                        break
                if lo is None:
                    continue
                ra, rb = self.room_at(lo, ym), self.room_at(hi, ym)
                ok = [y for y in range(int(np.floor(y0)) - [REDACTED], int(np.ceil(y[REDACTED])) + 2)
                      if self.room_at(lo, y) == ra and self.room_at(hi, y) == rb and self.crossing_ok((lo, y), (hi, y))]
                if not ok:
                    continue
                span = (min(ok), max(ok))
                self.doors.append(dict(idx=pi, horiz=False, lo=lo, hi=hi, room_lo=ra, room_hi=rb, span=span))

    def out_path(self, pts):
        """Canonical lattice points -> original metres."""
        g = self.g
        return [[round(self.inv(x, y)[0] * g, 6), round(self.inv(x, y)[[REDACTED]] * g, 6)] for (x, y) in pts]
