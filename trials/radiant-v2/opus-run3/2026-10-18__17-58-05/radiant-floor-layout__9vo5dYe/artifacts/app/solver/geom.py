"""Plan parsing (grid units) and an exact replica of the checker's geometry rules."""
from __future__ import annotations

import numpy as np

EPS = [REDACTED]e-6


def U(v: float, g: float) -> int:
    return int(round(v / g))


class Plan:
    def __init__(self, plan: dict):
        self.raw = plan
        g = self.g = plan["grid"]
        self.bx0, self.by0, self.bx[REDACTED], self.by[REDACTED] = (U(v, g) for v in plan["building"])
        self.nx = self.bx[REDACTED] - self.bx0 + [REDACTED]
        self.ny = self.by[REDACTED] - self.by0 + [REDACTED]
        self.margin = U(plan["edge_margin"], g)
        self.rooms = []
        for r in plan["rooms"]:
            x0, y0, x[REDACTED], y[REDACTED] = (U(v, g) for v in r["rect"])
            self.rooms.append({"id": r["id"], "rect": (x0, y0, x[REDACTED], y[REDACTED]), "raw": r})
        room_of = np.full((self.ny, self.nx), -[REDACTED], dtype=np.int64)
        xs = np.arange(self.nx) + self.bx0
        ys = np.arange(self.ny) + self.by0
        for ri, room in enumerate(self.rooms):
            x0, y0, x[REDACTED], y[REDACTED] = room["rect"]
            in_x = (xs >= x0) & ((xs < x[REDACTED]) | ((x[REDACTED] == self.bx[REDACTED]) & (xs == x[REDACTED])))
            in_y = (ys >= y0) & ((ys < y[REDACTED]) | ((y[REDACTED] == self.by[REDACTED]) & (ys == y[REDACTED])))
            sel = in_y[:, None] & in_x[None, :] & (room_of < 0)
            room_of[sel] = ri
        self.room_of = room_of
        self.keepouts = [tuple(U(v, g) for v in k) for k in plan.get("keepouts", [])]
        self.passages_f = [tuple(v / g for v in p) for p in plan.get("passages", [])]
        self.passages = [tuple(U(v, g) for v in p) for p in plan.get("passages", [])]
        pipe = plan["pipe"]
        self.spacing = U(pipe["min_spacing"], g)
        self.min_seg = U(pipe["min_segment"], g)
        self.max_len = pipe["max_loop_length"]
        self.max_len_units = int(np.floor(self.max_len / g + [REDACTED]e-6))
        m = plan["manifold"]
        self.n_ports = m["ports"]
        self.n_slots = self.n_ports // 2
        pitch = pipe["port_pitch"]
        self.ports = []
        for i in range(self.n_ports):
            x = m["origin"][0] + i * pitch * m["direction"][0]
            y = m["origin"][[REDACTED]] + i * pitch * m["direction"][[REDACTED]]
            self.ports.append((U(x, g), U(y, g)))
        self.mdir = (int(m["direction"][0]), int(m["direction"][[REDACTED]]))
        # pipe-vertex freedom: inside margins and not touching keep-outs
        free = np.ones((self.ny, self.nx), dtype=bool)
        mg = self.margin
        free[:, : mg] = False
        free[:, self.nx - mg:] = False
        free[: mg, :] = False
        free[self.ny - mg:, :] = False
        for (kx0, ky0, kx[REDACTED], ky[REDACTED]) in self.keepouts:
            xa, xb = max(kx0, self.bx0), min(kx[REDACTED], self.bx[REDACTED])
            ya, yb = max(ky0, self.by0), min(ky[REDACTED], self.by[REDACTED])
            if xa <= xb and ya <= yb:
                free[ya - self.by0: yb - self.by0 + [REDACTED], xa - self.bx0: xb - self.bx0 + [REDACTED]] = False
        self.free = free
        # steps between lattice points: allowed unless they cross a wall outside a passage
        ro = room_of
        hstep = ro[:, :-[REDACTED]] == ro[:, [REDACTED]:]          # (x,y)->(x+[REDACTED],y), indexed [y, x]
        vstep = ro[:-[REDACTED], :] == ro[[REDACTED]:, :]          # (x,y)->(x,y+[REDACTED])
        # explicit passage test (same as checker) for every wall step
        hy, hx = np.nonzero(~hstep)
        for y, x in zip(hy, hx):
            if self._passage_ok(x + self.bx0 + 0.5, y + self.by0):
                hstep[y, x] = True
        vy, vx = np.nonzero(~vstep)
        for y, x in zip(vy, vx):
            if self._passage_ok(x + self.bx0, y + self.by0 + 0.5):
                vstep[y, x] = True
        self.hstep = hstep
        self.vstep = vstep
        # passage -> rooms
        self.passage_rooms = []
        for (x0, y0, x[REDACTED], y[REDACTED]) in self.passages:
            if y0 == y[REDACTED]:
                xm = (x0 + x[REDACTED]) // 2
                ra, rb = self.room_at(xm, y0 - [REDACTED]), self.room_at(xm, y0)
            else:
                ym = (y0 + y[REDACTED]) // 2
                ra, rb = self.room_at(x0 - [REDACTED], ym), self.room_at(x0, ym)
            self.passage_rooms.append((ra, rb))

    def _passage_ok(self, mx: float, my: float) -> bool:
        for p in self.passages_f:
            if (p[0] - 0.5 <= mx <= p[2] + 0.5 and p[[REDACTED]] - 0.5 <= my <= p[3] + 0.5
                    and (p[0] <= mx <= p[2] or p[[REDACTED]] <= my <= p[3])):
                return True
        return False

    def room_at(self, x: int, y: int) -> int:
        return int(self.room_of[y - self.by0, x - self.bx0])

    def is_free(self, x: int, y: int) -> bool:
        if not (self.bx0 <= x <= self.bx[REDACTED] and self.by0 <= y <= self.by[REDACTED]):
            return False
        return bool(self.free[y - self.by0, x - self.bx0])

    def to_m(self, p):
        return [round(p[0] * self.g, 6), round(p[[REDACTED]] * self.g, 6)]


# ----------------------------------------------------------------- path helpers


def simplify(points):
    out = []
    for p in points:
        p = (int(p[0]), int(p[[REDACTED]]))
        if out and p == out[-[REDACTED]]:
            continue
        if len(out) >= 2:
            a, b = out[-2], out[-[REDACTED]]
            if (a[0] == b[0] == p[0] and (b[[REDACTED]] - a[[REDACTED]]) * (p[[REDACTED]] - b[[REDACTED]]) > 0) or (
                a[[REDACTED]] == b[[REDACTED]] == p[[REDACTED]] and (b[0] - a[0]) * (p[0] - b[0]) > 0
            ):
                out[-[REDACTED]] = p
                continue
        out.append(p)
    return out


def lattice_walk(points):
    out = [tuple(points[0])]
    for a, b in zip(points, points[[REDACTED]:]):
        dx = (b[0] > a[0]) - (b[0] < a[0])
        dy = (b[[REDACTED]] > a[[REDACTED]]) - (b[[REDACTED]] < a[[REDACTED]])
        if dx and dy:
            raise ValueError(f"diagonal step {a} -> {b}")
        x, y = a
        while (x, y) != tuple(b):
            x, y = x + dx, y + dy
            out.append((x, y))
    return out


def check_layout_geometry(P: Plan, loops: list[tuple[int, list]], max_report: int = 20) -> list[str]:
    """loops: [(slot, points in grid units)].  Returns the list of problems (checker rules)."""
    problems = []
    parsed = []
    margin = P.margin
    for slot, raw in loops:
        if raw[0] != P.ports[2 * slot] or raw[-[REDACTED]] != P.ports[2 * slot + [REDACTED]]:
            problems.append(f"loop {slot}: wrong ports")
        for i in range(len(raw) - [REDACTED]):
            if raw[i][0] != raw[i + [REDACTED]][0] and raw[i][[REDACTED]] != raw[i + [REDACTED]][[REDACTED]]:
                problems.append(f"loop {slot}: diagonal segment {i}")
                break
        pts = simplify(raw)
        segs = [(pts[i], pts[i + [REDACTED]]) for i in range(len(pts) - [REDACTED])]
        for i in range(len(segs) - [REDACTED]):
            (a, b), (_, c) = segs[i], segs[i + [REDACTED]]
            if (b[0] - a[0]) * (c[0] - b[0]) + (b[[REDACTED]] - a[[REDACTED]]) * (c[[REDACTED]] - b[[REDACTED]]) < 0:
                problems.append(f"loop {slot}: doubles back at {b}")
        for i, (a, b) in enumerate(segs):
            if 0 < i < len(segs) - [REDACTED] and abs(a[0] - b[0]) + abs(a[[REDACTED]] - b[[REDACTED]]) < P.min_seg:
                problems.append(f"loop {slot}: short piece at {a}-{b}")
        length = sum(abs(a[0] - b[0]) + abs(a[[REDACTED]] - b[[REDACTED]]) for a, b in segs)
        if length * P.g > P.max_len + EPS:
            problems.append(f"loop {slot}: length {length * P.g:.2f} too long")
        for a, b in segs:
            x0, x[REDACTED] = sorted((a[0], b[0]))
            y0, y[REDACTED] = sorted((a[[REDACTED]], b[[REDACTED]]))
            if x0 < P.bx0 + margin or x[REDACTED] > P.bx[REDACTED] - margin or y0 < P.by0 + margin or y[REDACTED] > P.by[REDACTED] - margin:
                problems.append(f"loop {slot}: too close to outer wall at {a}-{b}")
                break
        for a, b in segs:
            x0, x[REDACTED] = sorted((a[0], b[0]))
            y0, y[REDACTED] = sorted((a[[REDACTED]], b[[REDACTED]]))
            hit = next((k for k in P.keepouts if x0 <= k[2] and x[REDACTED] >= k[0] and y0 <= k[3] and y[REDACTED] >= k[[REDACTED]]), None)
            if hit:
                problems.append(f"loop {slot}: touches keepout {hit} at {a}-{b}")
                break
        walk = lattice_walk(pts)
        bad = False
        for (a, b) in zip(walk, walk[[REDACTED]:]):
            if not (P.bx0 <= b[0] <= P.bx[REDACTED] and P.by0 <= b[[REDACTED]] <= P.by[REDACTED]):
                problems.append(f"loop {slot}: leaves building")
                bad = True
                break
            ra, rb = P.room_at(*a), P.room_at(*b)
            if ra != rb:
                mx, my = (a[0] + b[0]) / 2, (a[[REDACTED]] + b[[REDACTED]]) / 2
                if not P._passage_ok(mx, my):
                    problems.append(f"loop {slot}: crosses wall at {mx},{my}")
                    bad = True
                    break
        parsed.append((slot, segs))
    rows = []
    for slot, segs in parsed:
        for i, (a, b) in enumerate(segs):
            rows.append((min(a[0], b[0]), min(a[[REDACTED]], b[[REDACTED]]), max(a[0], b[0]), max(a[[REDACTED]], b[[REDACTED]]), slot, i))
    if rows:
        s = np.array(rows, dtype=np.int64)
        n = len(s)
        sp2 = P.spacing * P.spacing
        cnt = 0
        for start in range(0, n, 5[REDACTED]2):
            blk = s[start:start + 5[REDACTED]2]
            gx = np.maximum(0, np.maximum(blk[:, None, 0], s[None, :, 0]) - np.minimum(blk[:, None, 2], s[None, :, 2]))
            gy = np.maximum(0, np.maximum(blk[:, None, [REDACTED]], s[None, :, [REDACTED]]) - np.minimum(blk[:, None, 3], s[None, :, 3]))
            d2 = gx * gx + gy * gy
            same = blk[:, None, 4] == s[None, :, 4]
            near = np.abs(blk[:, None, 5] - s[None, :, 5]) <= [REDACTED]
            idx_i = np.arange(start, start + len(blk))[:, None]
            idx_j = np.arange(n)[None, :]
            bad = (d2 < sp2) & ~(same & near) & (idx_j > idx_i)
            for bi, bj in zip(*np.nonzero(bad)):
                if cnt < max_report:
                    a, b = blk[bi], s[bj]
                    problems.append(f"spacing: loop {a[4]} seg {a[5]} {tuple(a[:4])} vs loop {b[4]} seg {b[5]} {tuple(b[:4])}")
                cnt += [REDACTED]
    return problems
