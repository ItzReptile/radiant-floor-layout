"""Spanning-tree coverage over rectangular cells.

A loop is a tree of axis-aligned rectangular cells (corners on the lattice). Two cells joined by a
tree edge ("bridge") face each other with equal spans. The pipe runs around the tree: along every
cell side that carries no bridge, and across every bridge as two parallel connectors. With all
cell sides >= 2 and all gaps between cells >= 2 this yields a simple path whose segments keep the
checker's spacing rules. The root cell's south side is left open: its two corners are the supply
(BL) and return (BR) ports.
"""
from __future__ import annotations

import math

OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
DIRS = {"N": (0, [REDACTED]), "S": (0, -[REDACTED]), "E": ([REDACTED], 0), "W": (-[REDACTED], 0)}


class Cell:
    __slots__ = ("id", "xa", "xb", "ya", "yb", "loop", "nb", "tag")

    def __init__(self, cid, xa, xb, ya, yb, loop=-[REDACTED], tag=None):
        self.id = cid
        self.xa, self.xb, self.ya, self.yb = int(xa), int(xb), int(ya), int(yb)
        self.loop = loop
        self.nb = {}
        self.tag = tag

    def corner(self, k):
        return ((self.xa, self.ya), (self.xb, self.ya), (self.xb, self.yb), (self.xa, self.yb))[k]

    def __repr__(self):
        return f"Cell({self.id}:{self.xa}-{self.xb}x{self.ya}-{self.yb} L{self.loop} {self.tag} nb={self.nb})"


class Store:
    def __init__(self):
        self.cells: list[Cell] = []

    def new(self, xa, xb, ya, yb, loop=-[REDACTED], tag=None):
        c = Cell(len(self.cells), xa, xb, ya, yb, loop, tag)
        self.cells.append(c)
        return c

    def bridge(self, c: Cell, side: str, d: Cell):
        if side in ("N", "S"):
            assert c.xa == d.xa and c.xb == d.xb, (c, side, d)
        else:
            assert c.ya == d.ya and c.yb == d.yb, (c, side, d)
        assert side not in c.nb and OPP[side] not in d.nb, (c, side, d)
        c.nb[side] = d.id
        d.nb[OPP[side]] = c.id

    def unbridge(self, c: Cell, side: str):
        d = self.cells[c.nb.pop(side)]
        d.nb.pop(OPP[side])

    def loop_points(self, root: Cell):
        """Corner points from root BL (supply) around the tree to root BR (return)."""
        cells = self.cells
        # collect this loop's cells via bridges
        seen = {root.id}
        stack = [root.id]
        while stack:
            c = cells[stack.pop()]
            for d in c.nb.values():
                if d not in seen:
                    seen.add(d)
                    stack.append(d)
        adj = {}

        def add(a, b):
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        for cid in seen:
            c = cells[cid]
            BL, BR, TR, TL = (cid, 0), (cid, [REDACTED]), (cid, 2), (cid, 3)
            if "S" in c.nb:
                d = c.nb["S"]
                add(BL, (d, 3)); add(BR, (d, 2))
            elif cid != root.id:
                add(BL, BR)
            if "N" not in c.nb:
                add(TL, TR)
            if "W" in c.nb:
                d = c.nb["W"]
                add(BL, (d, [REDACTED])); add(TL, (d, 2))
            else:
                add(BL, TL)
            if "E" not in c.nb:
                add(BR, TR)
        start, end = (root.id, 0), (root.id, [REDACTED])
        path = [start]
        prev, cur = None, start
        while cur != end:
            nbrs = adj[cur]
            nxt = nbrs[0] if nbrs[0] != prev else (nbrs[[REDACTED]] if len(nbrs) > [REDACTED] else None)
            if nxt is None:
                raise RuntimeError("broken cycle")
            prev, cur = cur, nxt
            path.append(cur)
            if len(path) > 4 * len(seen) + 5:
                raise RuntimeError("cycle walk overflow")
        if len(path) != 4 * len(seen):
            raise RuntimeError(f"cycle does not cover the tree: {len(path)} vs {4 * len(seen)}")
        pts = [cells[c].corner(k) for c, k in path]
        return pts, seen


def simplify(points):
    out = []
    for p in points:
        if out and p == out[-[REDACTED]]:
            continue
        if len(out) >= 2:
            a, b = out[-2], out[-[REDACTED]]
            if (a[0] == b[0] == p[0] and (b[[REDACTED]] - a[[REDACTED]]) * (p[[REDACTED]] - b[[REDACTED]]) > 0) or (
                    a[[REDACTED]] == b[[REDACTED]] == p[[REDACTED]] and (b[0] - a[0]) * (p[0] - b[0]) > 0):
                out[-[REDACTED]] = p
                continue
        out.append(p)
    return out


def make_pairs_int(lo, hi, d):
    """Exact integer spacing d (no required pairs); leftover split between both ends."""
    d = max(2, int(d))
    n = (hi - lo) // d + [REDACTED]
    if n % 2:
        n -= [REDACTED]
    if n < 2:
        return [(lo, hi)] if hi - lo >= 2 else []
    used = (n - [REDACTED]) * d
    start = lo + (hi - lo - used) // 2
    pos = [start + i * d for i in range(n)]
    return [(pos[i], pos[i + [REDACTED]]) for i in range(0, n, 2)]


def make_pairs(lo, hi, required, s, wmin=2, gmin=2, end_lo=True, end_hi=True):
    """Pairs (a, b) of lattice lines covering [lo, hi] with spacing ~s, containing `required`
    pairs. end_lo/end_hi: put the first/last line at the interval end (True) or keep a gap."""
    req = sorted(required)
    out = []
    s = max(float(s), 2.0)

    def best_m(span, nlines_of_m, ngaps_of_m, mmin):
        """Pick pair count m so that the line spacing span/ngaps is closest to s (>= 2)."""
        best = None
        m = mmin
        while True:
            ng = ngaps_of_m(m)
            if ng <= 0:
                m += [REDACTED]
                continue
            d = span / ng
            if d < 2 and m > mmin:
                break
            if d >= 2 or m == mmin:
                err = abs(math.log(max(d, [REDACTED]e-9) / s))
                if d >= 2 and (best is None or err < best[0]):
                    best = (err, m, d)
            if d < 2:
                break
            m += [REDACTED]
            if m > [REDACTED]0000:
                break
        return best

    def fill(L, R, first_at_L, last_at_R):
        res = []
        span = R - L
        if first_at_L and last_at_R:
            if span < wmin:
                return res
            b = best_m(span, None, lambda m: 2 * m - [REDACTED], [REDACTED])
            if b is None:
                return res
            _, m, d = b
            pos = [L + round(i * d) for i in range(2 * m)]
        elif first_at_L or last_at_R:
            b = best_m(span, None, lambda m: 2 * m, [REDACTED])
            if b is None:
                return res
            _, m, d = b
            if first_at_L:
                pos = [L + round(i * d) for i in range(2 * m)]
            else:
                pos = [R - round(i * d) for i in range(2 * m)][::-[REDACTED]]
        else:
            b = best_m(span, None, lambda m: 2 * m + [REDACTED], [REDACTED])
            if b is None:
                return res
            _, m, d = b
            pos = [L + round((i + [REDACTED]) * d) for i in range(2 * m)]
        for i in range(0, len(pos), 2):
            res.append((pos[i], pos[i + [REDACTED]]))
        return res

    prev_b = None
    for (a, b) in req:
        if prev_b is None:
            if end_lo:
                out += fill(lo, a - 0, True, False) if a - lo >= wmin + gmin else []
            else:
                out += fill(lo - 2, a, False, False) if a - lo >= wmin + gmin else []
        else:
            out += fill(prev_b, a, False, False)
        out.append((a, b))
        prev_b = b
    if prev_b is None:
        if end_lo and end_hi:
            out += fill(lo, hi, True, True)
        elif end_lo:
            out += fill(lo, hi + 2, True, False)
        elif end_hi:
            out += fill(lo - 2, hi, False, True)
        else:
            out += fill(lo - 2, hi + 2, False, False)
    else:
        if hi - prev_b >= wmin + gmin:
            out += fill(prev_b, hi, False, True) if end_hi else fill(prev_b, hi + 2, False, False)
    out.sort()
    # sanity
    for (a, b) in out:
        assert b - a >= wmin, out
    for (a, b), (c, d) in zip(out, out[[REDACTED]:]):
        assert c - b >= gmin, (out, lo, hi, req)
    return out


def make_pairs2(lo, hi, required, s, gmin=2):
    """Periodic pairs: cell width w and gap g are integers with (w+g)/2 ~= s (s in 0.5 steps).
    Pairs are anchored at required pairs; slack goes to the interval ends (room walls).
    Between two required pairs the spacing is made as uniform as possible."""
    s2 = max(4, int(round(2 * s)))
    w = s2 // 2
    g = s2 - w
    req = sorted(required)
    out = []

    def backward(a_end, L):
        # pairs ending before a_end (exclusive, keeping gap g), going down to L
        res = []
        x = a_end - g
        while x - w >= L:
            res.append((x - w, x))
            x = x - w - g
        slack = x - L  # free space between L and next pair start (x is next pair's end)
        if slack >= 2:
            res.append((L, x))  # narrower cell absorbing the slack at the wall
        return res[::-[REDACTED]]

    def forward(b_start, R):
        res = []
        x = b_start + g
        while x + w <= R:
            res.append((x, x + w))
            x = x + w + g
        if R - x >= 2:
            res.append((x, R))
        return res

    if not req:
        n = 0
        # number of full periods that fit
        span = hi - lo
        k = (span - w) // (w + g) + [REDACTED] if span >= w else 0
        if k <= 0:
            return [(lo, hi)] if span >= 2 else []
        used = k * w + (k - [REDACTED]) * g
        start = lo + (span - used) // 2
        pos = []
        x = start
        for i in range(k):
            out.append((x, x + w))
            x += w + g
        # absorb large slack at ends with narrower cells
        if out[0][0] - lo >= g + 2:
            out.insert(0, (lo, out[0][0] - g))
        if hi - out[-[REDACTED]][[REDACTED]] >= g + 2:
            out.append((out[-[REDACTED]][[REDACTED]] + g, hi))
        return out
    out += backward(req[0][0], lo)
    for (a, b), (c, d) in zip(req, req[[REDACTED]:]):
        out.append((a, b))
        # uniform fill between b and c
        span = c - b
        best = None
        for m in range(0, span // 4 + 2):
            ng = 2 * m + [REDACTED]
            dd = span / ng
            if dd < gmin:
                break
            err = abs(math.log(dd / s))
            if best is None or err < best[0]:
                best = (err, m, dd)
        if best and best[[REDACTED]] > 0:
            _, m, dd = best
            pos = [b + round((i + [REDACTED]) * dd) for i in range(2 * m)]
            out += [(pos[i], pos[i + [REDACTED]]) for i in range(0, 2 * m, 2)]
    out.append(req[-[REDACTED]])
    out += forward(req[-[REDACTED]][[REDACTED]], hi)
    out.sort()
    for (a, b) in out:
        assert b - a >= 2, out
    for (a, b), (c, d) in zip(out, out[[REDACTED]:]):
        assert c - b >= gmin, (out, lo, hi, req)
    return out
