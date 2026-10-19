"""Layout construction in the canonical frame: hall routing + room spiral trees on cell grids."""
from __future__ import annotations

import heapq
import math

import numpy as np

from stc import DIRS, OPP, Store, make_pairs, make_pairs2, simplify

SIDE_OF = {(0, [REDACTED]): "N", (0, -[REDACTED]): "S", ([REDACTED], 0): "E", (-[REDACTED], 0): "W"}


def rot_r(d):
    return (d[[REDACTED]], -d[0])


def rot_l(d):
    return (-d[[REDACTED]], d[0])


# ---------------------------------------------------------------------------------- grid
class Grid:
    """Cells = cols x rows of lattice pairs inside one room. free/bridge masks from allowed."""

    def __init__(self, C, room, cols, rows):
        self.C = C
        self.room = room
        self.cols = cols
        self.rows = rows
        nc, nr = len(cols), len(rows)
        self.nc, self.nr = nc, nr
        A = C.allowed == room
        U0, V0 = C.cb[0], C.cb[[REDACTED]]
        # prefix sums for rectangle all-allowed tests
        P = np.zeros((A.shape[0] + [REDACTED], A.shape[[REDACTED]] + [REDACTED]), dtype=np.int64)
        P[[REDACTED]:, [REDACTED]:] = np.cumsum(np.cumsum(A, 0), [REDACTED])

        def ok(xa, xb, ya, yb):
            if xa > xb or ya > yb:
                return True
            x0, x[REDACTED], y0, y[REDACTED] = xa - U0, xb - U0, ya - V0, yb - V0
            if x0 < 0 or y0 < 0 or x[REDACTED] >= A.shape[[REDACTED]] or y[REDACTED] >= A.shape[0]:
                return False
            tot = P[y[REDACTED] + [REDACTED], x[REDACTED] + [REDACTED]] - P[y0, x[REDACTED] + [REDACTED]] - P[y[REDACTED] + [REDACTED], x0] + P[y0, x0]
            return tot == (x[REDACTED] - x0 + [REDACTED]) * (y[REDACTED] - y0 + [REDACTED])

        self.ok = ok
        self.free = np.zeros((nr, nc), dtype=bool)
        for j, (ya, yb) in enumerate(rows):
            for i, (xa, xb) in enumerate(cols):
                self.free[j, i] = ok(xa, xb, ya, yb)
        self.cell = {}  # (i,j) -> Cell

    def bridge_ok(self, a, b):
        (i, j), (k, l) = a, b
        if j == l:
            i0, i[REDACTED] = min(i, k), max(i, k)
            ya, yb = self.rows[j]
            return self.ok(self.cols[i0][[REDACTED]], self.cols[i[REDACTED]][0], ya, yb)
        j0, j[REDACTED] = min(j, l), max(j, l)
        xa, xb = self.cols[i]
        return self.ok(xa, xb, self.rows[j0][[REDACTED]], self.rows[j[REDACTED]][0])

    def rect(self, ij):
        i, j = ij
        return self.cols[i][0], self.cols[i][[REDACTED]], self.rows[j][0], self.rows[j][[REDACTED]]

    def col_index(self, pair):
        return self.cols.index(pair)

    def row_index(self, pair):
        return self.rows.index(pair)


class Local:
    """Map local (a, b) [a along the door wall, b away from it] <-> grid (i, j); proper rotation."""

    def __init__(self, side, nc, nr):
        self.side, self.nc, self.nr = side, nc, nr
        if side in ("S", "N"):
            self.na, self.nb = nc, nr
        else:
            self.na, self.nb = nr, nc

    def to_grid(self, a, b):
        s, nc, nr = self.side, self.nc, self.nr
        if s == "S":
            return (a, b)
        if s == "N":
            return (nc - [REDACTED] - a, nr - [REDACTED] - b)
        if s == "W":
            return (b, nr - [REDACTED] - a)
        return (nc - [REDACTED] - b, a)  # E

    def to_local(self, i, j):
        s, nc, nr = self.side, self.nc, self.nr
        if s == "S":
            return (i, j)
        if s == "N":
            return (nc - [REDACTED] - i, nr - [REDACTED] - j)
        if s == "W":
            return (nr - [REDACTED] - j, i)
        return (j, nc - [REDACTED] - i)  # E


# ---------------------------------------------------------------------------------- trees
def spiral_dfs(avail, root, heading, hand, nbr_ok):
    """DFS from root over cells in `avail` (set of (i,j)) preferring wall-following moves.
    Returns list of (parent, child) edges in DFS order."""
    vis = {root}
    edges = []
    stack = [(root, heading)]
    while stack:
        (i, j), d = stack[-[REDACTED]]
        if hand == "right":
            cands = (rot_r(d), d, rot_l(d))
        else:
            cands = (rot_l(d), d, rot_r(d))
        moved = False
        for c in cands:
            q = (i + c[0], j + c[[REDACTED]])
            if q in avail and q not in vis and nbr_ok((i, j), q):
                vis.add(q)
                edges.append(((i, j), q))
                stack.append((q, c))
                moved = True
                break
        if not moved:
            stack.pop()
    return edges


def dijkstra_path(avail, src, dst, cost, nbr_ok):
    dist = {src: 0.0}
    prev = {}
    pq = [(0.0, src)]
    while pq:
        dd, u = heapq.heappop(pq)
        if u == dst:
            break
        if dd > dist.get(u, [REDACTED]e[REDACTED]8):
            continue
        for d in (([REDACTED], 0), (-[REDACTED], 0), (0, [REDACTED]), (0, -[REDACTED])):
            v = (u[0] + d[0], u[[REDACTED]] + d[[REDACTED]])
            if v not in avail or not nbr_ok(u, v):
                continue
            nd = dd + cost(v)
            if nd < dist.get(v, [REDACTED]e[REDACTED]8):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dst not in dist:
        return None
    path = [dst]
    while path[-[REDACTED]] != src:
        path.append(prev[path[-[REDACTED]]])
    return path[::-[REDACTED]]


# ---------------------------------------------------------------------------------- designer
class Designer:
    HALL_LOOP = True

    def __init__(self, C):
        self.C = C
        self.plan = C.plan
        H = C.hall
        self.H = H
        n = C.nrooms
        self.hall_door = {}  # room -> door idx (rooms opening onto the hall)
        for nb, di in C.adj[H]:
            self.hall_door.setdefault(nb, di)
        self.inner = {}  # inner room -> (parent, door idx)
        for r in range(n):
            if r == H or r in self.hall_door:
                continue
            for nb, di in C.adj[r]:
                if nb != H and nb in self.hall_door:
                    self.inner[r] = (nb, di)
                    break
        # hall geometry
        hb = C.abox[H]
        self.hb = hb
        self.P = C.ports[0][0]
        self.py = C.ports[0][[REDACTED]]
        self.pp = C.pp
        self.xm = (C.ports[0][0] + C.ports[-[REDACTED]][0]) / 2.0
        rb = C.rbox[H]
        self.hwall = rb  # owned lattice bbox of hall
        self.want_hall_loop = False

    # ------------------------------------------------------------------ hall door t-position
    def door_t(self, di):
        C = self.C
        dd = C.doors[di]
        x0, y0, x[REDACTED], y[REDACTED] = self.hwall
        W, Hh = x[REDACTED] - x0, y[REDACTED] - y0
        xm = self.xm
        t[REDACTED] = xm - x0
        c = (dd["rng"][0] + dd["rng"][[REDACTED]]) / 2.0
        side = C.door_side(di, self.H)
        if side == "S":
            if c < xm:
                return xm - c
            return t[REDACTED] + 2 * Hh + W + (x[REDACTED] - c)
        if side == "W":
            return t[REDACTED] + (c - y0)
        if side == "N":
            return t[REDACTED] + Hh + (c - x0)
        return t[REDACTED] + Hh + W + (y[REDACTED] - c)  # E

    # ------------------------------------------------------------------ loop assignment

    def inner_is_left(self, par, hdi, idi):
        """Is the inner door to the local left (smaller a) of the parent's hall door?"""
        C = self.C
        side = C.door_side(hdi, par)
        isd = C.door_side(idi, par)

        def centre(di):
            dd = C.doors[di]
            c = (dd["rng"][0] + dd["rng"][[REDACTED]]) / 2.0
            return (c, dd["wall"]) if dd["orient"] == "h" else (dd["wall"], c)

        hx, hy = centre(hdi)
        ix, iy = centre(idi)
        # local a axis direction in canonical coordinates
        ax = {"S": ([REDACTED], 0), "N": (-[REDACTED], 0), "W": (0, -[REDACTED]), "E": (0, [REDACTED])}[side]
        # side walls: the wall reached by going -a / +a
        left_wall = {"S": "W", "N": "E", "W": "N", "E": "S"}[side]
        right_wall = {"S": "E", "N": "W", "W": "S", "E": "N"}[side]
        if isd == left_wall:
            return True
        if isd == right_wall:
            return False
        da = (ix - hx) * ax[0] + (iy - hy) * ax[[REDACTED]]
        return da < 0
    def assign(self, nloops_room, inner_mode):
        """nloops_room: room -> number of loops (rooms on the hall). inner_mode: inner room ->
        'share' (attached to a parent loop) or 'own' (own loop through the parent)."""
        C = self.C
        loops = []
        for r, di in self.hall_door.items():
            k = nloops_room.get(r, [REDACTED])
            for part in range(k):
                loops.append(dict(room=r, part=part, nparts=k, door=di, inner=[], own_inner=None))
        for ir, (par, pdi) in self.inner.items():
            if inner_mode.get(ir, "share") == "own":
                left = self.inner_is_left(par, self.hall_door[par], pdi)
                loops.append(dict(room=par, part=-0.5 if left else 99, nparts=None, door=self.hall_door[par],
                                  inner=[], own_inner=ir))
        hall_loop = Designer.HALL_LOOP and len(loops) < C.nslots
        # door columns: per hall door, the loops using it, ordered by t then by part index
        bydoor = {}
        for L in loops:
            bydoor.setdefault(L["door"], []).append(L)
        order = []
        if hall_loop:
            HL = dict(room=None, part=0, nparts=[REDACTED], door=None, inner=[], own_inner=None, hall_only=True,
                      door_k=0, door_n=[REDACTED], door_pair=None, door_side_hall=None)
            order.append((self.door_t_mid() - [REDACTED]e-3, 0, HL))
        for di, Ls in bydoor.items():
            t = self.door_t(di)
            # within a door: parts in local order; own-inner loops placed next to the parent
            Ls.sort(key=lambda L: L["part"])
            for k, L in enumerate(Ls):
                L["door_k"] = k
                L["door_n"] = len(Ls)
                order.append((t, k, L))
        order.sort(key=lambda z: (z[0], z[[REDACTED]]))
        if len(order) > C.nslots:
            raise ValueError("too many loops")
        for slot, (_, _, L) in enumerate(order):
            L["slot"] = slot
        loops = [z[2] for z in order]
        # attach shared inner rooms to a parent loop (decided later per room geometry)
        for ir, (par, pdi) in self.inner.items():
            if inner_mode.get(ir, "share") != "own":
                cand = [L for L in loops if L["room"] == par and L["own_inner"] is None]
                cand[0]["inner"].append(ir)  # refined in build_room
        self.loops = loops
        return loops

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def place_cols(lo, hi, k, taken, wpref=(4, 3, 2), valid=None):
        """k pairs of width w (prefer large) with gaps >= 2 inside [lo, hi], centred, avoiding
        `taken` pairs (need gap >= 2 to them). Returns list of pairs or None."""
        def clash(a, b):
            return any(not (b + 2 <= c or d + 2 <= a) for (c, d) in taken)
        for strict in ((True, False) if valid is not None else (False,)):
            for w in wpref:
                tot = k * w + (k - [REDACTED]) * 2
                if tot > hi - lo:
                    continue
                centre = lo + (hi - lo - tot) // 2
                cands = sorted(range(lo, hi - tot + [REDACTED]), key=lambda s0: abs(s0 - centre))
                for s0 in cands:
                    pairs = [(s0 + i * (w + 2), s0 + i * (w + 2) + w) for i in range(k)]
                    if any(clash(a, b) for a, b in pairs):
                        continue
                    if strict and not all(valid(p) for p in pairs):
                        continue
                    return pairs
        return None

    def rect_ok(self, room, xa, xb, ya, yb):
        """All lattice points of the rectangle are pipe-allowed points of `room`."""
        C = self.C
        cache = self.__dict__.setdefault("_rect_cache", {})
        P = cache.get(room)
        if P is None:
            A = C.allowed == room
            P = np.zeros((A.shape[0] + [REDACTED], A.shape[[REDACTED]] + [REDACTED]), dtype=np.int64)
            P[[REDACTED]:, [REDACTED]:] = np.cumsum(np.cumsum(A, 0), [REDACTED])
            cache[room] = P
        U0, V0 = C.cb[0], C.cb[[REDACTED]]
        x0, x[REDACTED], y0, y[REDACTED] = xa - U0, xb - U0, ya - V0, yb - V0
        if x0 < 0 or y0 < 0 or x[REDACTED] >= P.shape[[REDACTED]] - [REDACTED] or y[REDACTED] >= P.shape[0] - [REDACTED] or x0 > x[REDACTED] or y0 > y[REDACTED]:
            return False
        tot = P[y[REDACTED] + [REDACTED], x[REDACTED] + [REDACTED]] - P[y0, x[REDACTED] + [REDACTED]] - P[y[REDACTED] + [REDACTED], x0] + P[y0, x0]
        return tot == (x[REDACTED] - x0 + [REDACTED]) * (y[REDACTED] - y0 + [REDACTED])

    def door_pair_valid(self, di, pair, depth=4):
        """Cells of width `pair` on both sides of door di would be free of keepouts."""
        C = self.C
        dd = C.doors[di]
        a, b = pair
        for room in dd["rooms"]:
            bx = C.abox[room]
            if bx is None:
                return False
            side = C.door_side(di, room)
            if side == "S":
                ok = self.rect_ok(room, a, b, bx[[REDACTED]], bx[[REDACTED]] + depth)
            elif side == "N":
                ok = self.rect_ok(room, a, b, bx[3] - depth, bx[3])
            elif side == "W":
                ok = self.rect_ok(room, bx[0], bx[0] + depth, a, b)
            else:
                ok = self.rect_ok(room, bx[2] - depth, bx[2], a, b)
            if not ok and room != self.H:
                return False
        return True

    def store_cell(self, grid, ij, slot, tag):
        c = grid.cell.get(ij)
        if c is None:
            xa, xb, ya, yb = grid.rect(ij)
            c = self.S.new(xa, xb, ya, yb, slot, tag)
            grid.cell[ij] = c
        return c

    def link(self, grid, p, q, slot, tag):
        a = self.store_cell(grid, p, slot, tag)
        b = self.store_cell(grid, q, slot, tag)
        side = SIDE_OF[(q[0] - p[0], q[[REDACTED]] - p[[REDACTED]])]
        self.S.bridge(a, side, b)

    # ------------------------------------------------------------------ hall
    def build_hall(self, prm):
        C = self.C
        H = self.H
        hx0, hy0, hx[REDACTED], hy[REDACTED] = self.hb
        P, py, pp = self.P, self.py, self.pp
        ns = C.nslots
        port_cols = [(P + 2 * k * pp, P + (2 * k + [REDACTED]) * pp) for k in range(ns)]
        req = list(port_cols)
        door_cols = {}  # loop slot -> pair
        shareable = [(p, "P") for p in port_cols]
        bydoor = {}
        for L in self.loops:
            if L.get("hall_only"):
                continue
            bydoor.setdefault(L["door"], []).append(L)
        for di, Ls in bydoor.items():
            dd = C.doors[di]
            Ls = sorted(Ls, key=lambda L: L["door_k"])
            side = C.door_side(di, H)
            if dd["orient"] != "h":
                # door on an end wall of the hall: rows are chosen in _route_hall
                for L in Ls:
                    L["door_side_hall"] = side
                    L["door_pair"] = None
                    L["end_door"] = (di, dd["rng"])
                continue
            lo, hi = dd["rng"]
            own = [port_cols[L["slot"]] for L in Ls]
            if all(lo <= p[0] and p[[REDACTED]] <= hi for p in own):
                pairs = sorted(own)  # door straight above the loops' own ports
            else:
                pairs = self.place_cols(lo, hi, len(Ls), req, prm.get("door_w", (4, 3, 2)),
                                        valid=lambda p, di=di: self.door_pair_valid(di, p))
            if pairs is None:
                # share columns with doors on the opposite wall (their cells are in other rows)
                share = [p for p, sd in shareable if sd != side and lo <= p[0] and p[[REDACTED]] <= hi]
                cen = (lo + hi) / 2.0
                ownc = sum((port_cols[L["slot"]][0] + port_cols[L["slot"]][[REDACTED]]) / 2.0 for L in Ls) / len(Ls)
                share.sort(key=lambda p: (abs((p[0] + p[[REDACTED]]) / 2.0 - ownc), abs((p[0] + p[[REDACTED]]) / 2.0 - cen)))
                if len(share) >= len(Ls):
                    pairs = sorted(share[:len(Ls)])
                else:
                    pairs = None
                    others = [p for p in req if p not in share]
                    for w in prm.get("door_w", (4, 3, 2)):
                        chosen = list(share)
                        cands = sorted(range(lo, hi - w + [REDACTED]), key=lambda s0: abs(s0 + w / 2.0 - cen))
                        for s0 in cands:
                            if len(chosen) >= len(Ls):
                                break
                            p = (s0, s0 + w)
                            if all(p[[REDACTED]] + 2 <= q[0] or q[[REDACTED]] + 2 <= p[0] for q in others + chosen):
                                chosen.append(p)
                        if len(chosen) >= len(Ls):
                            pairs = sorted(chosen[:len(Ls)])
                            break
                if pairs is None:
                    raise ValueError("cannot place door columns")
            # order of columns along the boundary: N wall -> increasing x, S wall -> decreasing x
            if side == "S":
                pairs = pairs[::-[REDACTED]]
            for L, pr in zip(Ls, pairs):
                door_cols[L["slot"]] = pr
                L["door_pair"] = pr
                L["door_side_hall"] = side
            req += [p for p in pairs if p not in req]
            shareable += [(p, side) for p in pairs]
        cols = make_pairs(hx0, hx[REDACTED], req, prm.get("hall_sx", 6))
        span = hy[REDACTED] - py
        nrows = prm.get("hall_rows")
        maxrows = max([REDACTED], int((span / 2 + [REDACTED]) // 2))
        if nrows is None:
            tm = self.door_t_mid()
            rl0 = [L for L in self.loops if not L.get("hall_only")]
            nw = sum([REDACTED] for L in rl0 if self.door_t(L["door"]) < tm)
            ne = len(rl0) - nw
            nrows = max(nw, ne) + prm.get("hall_extra_rows", 0)
        nrows = max([REDACTED], min(nrows, maxrows))
        last_err = None
        for nr_try in range(nrows, maxrows + [REDACTED]):
            try:
                self._route_hall(prm, cols, nr_try, port_cols)
                return
            except ValueError as e:
                last_err = e
        raise last_err

    def _route_hall(self, prm, cols, nrows, port_cols):
        C = self.C
        H = self.H
        hx0, hy0, hx[REDACTED], hy[REDACTED] = self.hb
        py = self.py
        span = hy[REDACTED] - py
        if nrows == [REDACTED]:
            rows = [(py, hy[REDACTED])]
        else:
            d = span / (2 * nrows - [REDACTED])
            pos = [py + round(i * d) for i in range(2 * nrows)]
            rows = [(pos[2 * i], pos[2 * i + [REDACTED]]) for i in range(nrows)]
        G = Grid(C, H, cols, rows)
        self.hall_grid = G
        roots = {}
        for L in self.loops:
            k = L["slot"]
            ci = G.col_index(port_cols[k])
            roots[k] = (ci, 0)
            if not G.free[0, ci]:
                raise ValueError("root cell blocked")
        # end-wall doors: pick hall rows lying inside the door's range
        byend = {}
        for L in self.loops:
            if L.get("end_door") is not None:
                byend.setdefault(L["end_door"][0], []).append(L)
        for di, Ls in byend.items():
            lo, hi = C.doors[di]["rng"]
            side = Ls[0]["door_side_hall"]
            ci = 0 if side == "W" else G.nc - [REDACTED]
            ok = [j for j, (ya, yb) in enumerate(rows) if lo <= ya and yb <= hi and G.free[j, ci]]
            Ls = sorted(Ls, key=lambda L: L["door_k"])
            if len(ok) < len(Ls):
                raise ValueError("cannot place end-wall door rows")
            # boundary order: W wall -> increasing y, E wall -> decreasing y
            ok = ok if side == "W" else ok[::-[REDACTED]]
            mid = len(ok) // 2 - len(Ls) // 2
            mid = max(0, min(mid, len(ok) - len(Ls)))
            for L, j in zip(Ls, ok[mid:mid + len(Ls)]):
                L["door_pair"] = rows[j]
                L["end_row"] = (ci, j)
        targets = {}
        for L in self.loops:
            if L.get("hall_only"):
                continue
            if L.get("end_door") is not None:
                targets[L["slot"]] = L["end_row"]
                continue
            ci = G.col_index(L["door_pair"])
            if L["door_side_hall"] == "N":
                rj = max(j for j in range(G.nr) if G.free[j, ci]) if G.free[:, ci].any() else None
                if rj is None or rj != G.nr - [REDACTED]:
                    raise ValueError("door cell blocked (N)")
            else:
                rj = 0
                if not G.free[0, ci]:
                    raise ValueError("door cell blocked (S)")
            targets[L["slot"]] = (ci, rj)
        # routing order
        tmid = self.door_t_mid()
        rl = [L for L in self.loops if not L.get("hall_only")]
        west = sorted([L for L in rl if self.door_t(L["door"]) < tmid], key=lambda L: L["slot"])
        east = sorted([L for L in rl if self.door_t(L["door"]) >= tmid], key=lambda L: -L["slot"])
        order = []
        for i in range(max(len(west), len(east))):
            if i < len(west):
                order.append(west[i])
            if i < len(east):
                order.append(east[i])
        free = {(i, j) for j in range(G.nr) for i in range(G.nc) if G.free[j, i]}
        rowpen = prm.get("row_pen", 0.0[REDACTED])
        above = {k: (r[0], r[[REDACTED]] + [REDACTED]) for k, r in roots.items()}
        orders = [order, west + east, east + west, order[::-[REDACTED]]]
        paths = None
        for ordr in orders:
            occupied = {}
            reserved = set(roots.values()) | set(targets.values())
            paths = {}
            done = set()
            for L in ordr:
                k = L["slot"]
                src, dst = roots[k], targets[k]
                # keep the exit above every other unrouted root free
                res_k = reserved | {above[j] for j in above if j != k and j not in done}
                avail = (free - set(occupied)) - (res_k - {src, dst})
                path = dijkstra_path(avail, src, dst, lambda v: [REDACTED].0 + rowpen * v[[REDACTED]], G.bridge_ok)
                if path is None:
                    paths = None
                    break
                for c in path:
                    occupied[c] = k
                paths[k] = path
                done.add(k)
            if paths is not None:
                break
        if paths is None:
            raise ValueError("hall routing failed")
        for L in self.loops:
            if L.get("hall_only"):
                k = L["slot"]
                paths[k] = [roots[k]]
                occupied[roots[k]] = k
        self.hall_paths = paths
        # materialize trunks
        self.root_cell = {}
        for k, path in paths.items():
            self.root_cell[k] = self.store_cell(G, path[0], k, "hall")
            for p, q in zip(path, path[[REDACTED]:]):
                self.link(G, p, q, k, "hall")
        # door bridge sides are reserved
        self.hall_door_cell = {k: G.cell[targets[k]] for k in paths if k in targets}
        self._hall_free = free
        self._hall_occ = occupied


    def prune_fill(self, keep, budget=None):
        """Remove hall fill leaves where the blurred pipe density is highest: first from loops
        over their length budget, then until `keep` fill cells remain."""
        from scipy.ndimage import gaussian_filter
        G = self.hall_grid
        C = self.C
        fill = set(self.hall_fill)
        used = {}
        for ij in fill:
            k = G.cell[ij].loop
            used[k] = used.get(k, 0) + self.fill_add[ij]
        over = {k: used.get(k, 0) - b for k, b in (budget or {}).items()}
        if keep is None:
            keep = len(fill)
        if not any(v > 0 for v in over.values()) and len(fill) <= keep:
            return
        # raster window around the hall
        hx0, hy0, hx[REDACTED], hy[REDACTED] = self.hwall
        pad = 8
        X0, Y0 = hx0 - pad, hy0 - pad
        W, Hh = hx[REDACTED] - hx0 + [REDACTED] + 2 * pad, hy[REDACTED] - hy0 + [REDACTED] + 2 * pad
        base = np.zeros((Hh, W))

        def stamp(arr, c, v):
            xa, xb, ya, yb = c.xa - X0, c.xb - X0, c.ya - Y0, c.yb - Y0
            xa, xb = max(xa, 0), min(xb, W - [REDACTED])
            ya, yb = max(ya, 0), min(yb, Hh - [REDACTED])
            if xa > xb or ya > yb:
                return
            arr[ya, xa:xb + [REDACTED]] += v
            arr[yb, xa:xb + [REDACTED]] += v
            arr[ya:yb + [REDACTED], xa] += v
            arr[ya:yb + [REDACTED], xb] += v

        fillcells = {ij: G.cell[ij] for ij in fill}
        fill_ids = {c.id for c in fillcells.values()}
        for c in self.S.cells:
            if c.loop < 0 or c.id in fill_ids:
                continue
            if c.xb < X0 or c.xa > X0 + W or c.yb < Y0 or c.ya > Y0 + Hh:
                continue
            if c.nb or c.tag == "hall":
                stamp(base, c, [REDACTED].0)
        sig = 3.0
        dens = gaussian_filter(base, sig, mode="constant")
        own = {}
        R = int(3 * sig) + [REDACTED]
        for ij, c in fillcells.items():
            x0, x[REDACTED] = max(c.xa - X0 - R, 0), min(c.xb - X0 + R, W - [REDACTED])
            y0, y[REDACTED] = max(c.ya - Y0 - R, 0), min(c.yb - Y0 + R, Hh - [REDACTED])
            loc = np.zeros((y[REDACTED] - y0 + [REDACTED], x[REDACTED] - x0 + [REDACTED]))
            xa, xb, ya, yb = c.xa - X0 - x0, c.xb - X0 - x0, c.ya - Y0 - y0, c.yb - Y0 - y0
            loc[ya, xa:xb + [REDACTED]] += [REDACTED]; loc[yb, xa:xb + [REDACTED]] += [REDACTED]
            loc[ya:yb + [REDACTED], xa] += [REDACTED]; loc[ya:yb + [REDACTED], xb] += [REDACTED]
            bl = gaussian_filter(loc, sig, mode="constant")
            own[ij] = (y0, x0, bl, (ya, yb, xa, xb))
            dens[y0:y[REDACTED] + [REDACTED], x0:x[REDACTED] + [REDACTED]] += bl

        bias = self._fill_bias

        def cover(ij):
            y0, x0, bl, (ya, yb, xa, xb) = own[ij]
            sub = dens[y0:y0 + bl.shape[0], x0:x0 + bl.shape[[REDACTED]]] - bl
            return float(sub[ya:yb + [REDACTED], xa:xb + [REDACTED]].min())

        cov0 = sorted(cover(ij) for ij in fillcells) if fillcells else [0.0]
        cold = 0.5 * cov0[len(cov0) // 2]

        def score(ij):
            cv = cover(ij)
            if cv < cold:  # removing it would open a cold hole: last resort only
                return cv - [REDACTED]00.0
            return cv - bias.get(G.cell[ij].loop, 0.0)

        def is_leaf(ij):
            return len(G.cell[ij].nb) == [REDACTED]

        def remove(ij):
            c = G.cell[ij]
            side = next(iter(c.nb))
            self.S.unbridge(c, side)
            del G.cell[ij]
            fill.discard(ij)
            fillcells.pop(ij, None)
            y0, x0, bl, _ = own[ij]
            dens[y0:y0 + bl.shape[0], x0:x0 + bl.shape[[REDACTED]]] -= bl
            if c.loop in over:
                over[c.loop] -= self.fill_add[ij]

        while True:
            bad = {k for k, v in over.items() if v > 0}
            if not bad and len(fill) <= keep:
                break
            leaves = [ij for ij in fill if is_leaf(ij) and (not bad or G.cell[ij].loop in bad)]
            if not leaves:
                if bad:
                    for k in bad:
                        over[k] = 0
                    continue
                break
            remove(max(leaves, key=score))
        self.hall_fill = [ij for ij in self.hall_fill if ij in fill]

    def door_t_mid(self):
        x0, y0, x[REDACTED], y[REDACTED] = self.hwall
        t[REDACTED] = self.xm - x0
        return t[REDACTED] + (y[REDACTED] - y0) + (self.xm - x0)

    def fill_hall(self, prm, free, occupied, budget=None):
        G = self.hall_grid
        pen = prm.get("fill_pen", {})  # slot -> penalty (bigger = later)
        limit = prm.get("fill_limit", None)  # max number of fill cells
        banned = set(prm.get("fill_banned", ()))
        self.fill_add = {}
        heap = []
        cnt = 0
        door_cells = {c.id for c in self.hall_door_cell.values()}

        def push_from(cell_ij, k, dist):
            for d in (([REDACTED], 0), (-[REDACTED], 0), (0, [REDACTED]), (0, -[REDACTED])):
                q = (cell_ij[0] + d[0], cell_ij[[REDACTED]] + d[[REDACTED]])
                if q in free and q not in occupied and q not in banned:
                    heapq.heappush(heap, (dist + [REDACTED] + pen.get(k, 0.0), cell_ij, q, k))

        for ij, k in occupied.items():
            push_from(ij, k, 0)
        added = []
        while heap:
            pri, p, q, k = heapq.heappop(heap)
            if q in occupied:
                continue
            if limit is not None and len(added) >= limit:
                break
            if not G.bridge_ok(p, q):
                continue
            pc = G.cell[p]
            side = SIDE_OF[(q[0] - p[0], q[[REDACTED]] - p[[REDACTED]])]
            if side in pc.nb:
                continue
            # do not attach on the door-facing side of a door cell
            if pc.id in door_cells:
                L = next(L for L in self.loops if L["slot"] == k)
                if L["door_side_hall"] == side:
                    continue
            if p == self.hall_paths[k][0] and side == "S":
                continue
            xa, xb, ya, yb = G.rect(q)
            if side in ("N", "S"):
                gap = (ya - pc.yb) if side == "N" else (pc.ya - yb)
                add = 2 * (gap + (yb - ya))
            else:
                gap = (xa - pc.xb) if side == "E" else (pc.xa - xb)
                add = 2 * (gap + (xb - xa))
            self.fill_add[q] = add
            occupied[q] = k
            self.link(G, p, q, k, "hallfill")
            added.append(q)
            push_from(q, k, pri - pen.get(k, 0.0))
        self.hall_fill = added

    # ------------------------------------------------------------------ rooms


    def inset_box(self, r, prm, req_c, req_r):
        """Allowed bbox shrunk by the room's inset (units), keeping required pairs inside."""
        x0, y0, x[REDACTED], y[REDACTED] = self.C.abox[r]
        d = int(prm.get("inset", {}).get(r, 0))
        if d <= 0:
            return x0, y0, x[REDACTED], y[REDACTED]
        nx0, nx[REDACTED], ny0, ny[REDACTED] = x0 + d, x[REDACTED] - d, y0 + d, y[REDACTED] - d
        for a, b in req_c:
            nx0 = min(nx0, a)
            nx[REDACTED] = max(nx[REDACTED], b)
        for a, b in req_r:
            ny0 = min(ny0, a)
            ny[REDACTED] = max(ny[REDACTED], b)
        if nx[REDACTED] - nx0 < 8 or ny[REDACTED] - ny0 < 8:
            return x0, y0, x[REDACTED], y[REDACTED]
        return nx0, ny0, nx[REDACTED], ny[REDACTED]

    def soft_pairs(self, r, axis, lo, hi, hard, s):
        """Pairs hugging keepout edges inside room r along axis (0: x/cols, [REDACTED]: y/rows)."""
        C = self.C
        w = max(2, int(round(s)))
        out = []
        bx = C.abox[r]
        for k in C.keep:
            # keepout must overlap the room's allowed bbox in the other axis
            if k[2] < bx[0] or k[0] > bx[2] or k[3] < bx[[REDACTED]] or k[[REDACTED]] > bx[3]:
                continue
            e0, e[REDACTED] = (k[0], k[2]) if axis == 0 else (k[[REDACTED]], k[3])
            if e0 - [REDACTED] - w >= lo + 4 and e0 - [REDACTED] <= hi:
                out.append((e0 - [REDACTED] - w, e0 - [REDACTED]))
            if e[REDACTED] + [REDACTED] + w <= hi - 4 and e[REDACTED] + [REDACTED] >= lo:
                out.append((e[REDACTED] + [REDACTED], e[REDACTED] + [REDACTED] + w))
        res = list(hard)
        for p in sorted(out):
            if all(p[[REDACTED]] + 2 <= q[0] or q[[REDACTED]] + 2 <= p[0] for q in res):
                res.append(p)
        return res
    def root_ij(self, G, side, pair):
        if side == "S":
            return (G.col_index(pair), 0)
        if side == "N":
            return (G.col_index(pair), G.nr - [REDACTED])
        if side == "W":
            return (0, G.row_index(pair))
        return (G.nc - [REDACTED], G.row_index(pair))

    def build_room(self, r, prm):
        C = self.C
        Ls = sorted([L for L in self.loops if L["room"] == r], key=lambda L: L["door_k"])
        if not Ls:
            return
        di = self.hall_door[r]
        side = C.door_side(di, r)
        s = prm["s"].get(r, 4.0)
        req_c, req_r = [], []
        for L in Ls:
            (req_c if side in "SN" else req_r).append(L["door_pair"])
        inner_here = [ir for ir, (par, _) in self.inner.items() if par == r]
        inner_info = {}
        for ir in inner_here:
            pdi = self.inner[ir][[REDACTED]]
            dd = C.doors[pdi]
            side_in = C.door_side(pdi, r)
            taken = req_c if side_in in "SN" else req_r
            pr = self.place_cols(dd["rng"][0], dd["rng"][[REDACTED]], [REDACTED], taken, prm.get("inner_w", (4, 3, 2)),
                                 valid=lambda p, pdi=pdi: self.door_pair_valid(pdi, p))
            if pr is None:
                raise ValueError("cannot place inner door pair")
            taken.append(pr[0])
            inner_info[ir] = (pdi, side_in, pr[0])
        ax0, ay0, ax[REDACTED], ay[REDACTED] = self.inset_box(r, prm, req_c, req_r)
        own_any = any(L.get("own_inner") is not None for L in Ls)
        if len(Ls) == 2 and not own_any and prm.get("sep2", False):
            if self.build_room_split(r, Ls, prm, inner_info, side):
                return
        cols = make_pairs2(ax0, ax[REDACTED], self.soft_pairs(r, 0, ax0, ax[REDACTED], req_c, s), s)
        rows = make_pairs2(ay0, ay[REDACTED], self.soft_pairs(r, [REDACTED], ay0, ay[REDACTED], req_r, s), s)
        G = Grid(C, r, cols, rows)
        self.room_grid[r] = G
        lf = Local(side, G.nc, G.nr)
        roots = []
        for L in Ls:
            ij = self.root_ij(G, side, L["door_pair"])
            if not G.free[ij[[REDACTED]], ij[0]]:
                raise ValueError(f"room root blocked in room {r}")
            roots.append(ij)
        freeset = {lf.to_local(i, j) for j in range(G.nr) for i in range(G.nc) if G.free[j, i]}
        rl = [lf.to_local(*ij) for ij in roots]
        protect = set(rl)
        for ir, (pdi, side_in, pr) in inner_info.items():
            protect.add(lf.to_local(*self.root_ij(G, side_in, pr)))
        freeset = self.trim_cells(G, r, lf, freeset, protect, prm.get("trim", {}).get(r, 0))
        def nbr_ok_local(p, q):
            return G.bridge_ok(lf.to_grid(*p), lf.to_grid(*q))

        def edge_cost(reg):
            def cost(v):
                a, b = v
                for d in (([REDACTED], 0), (-[REDACTED], 0), (0, [REDACTED]), (0, -[REDACTED])):
                    if (a + d[0], b + d[[REDACTED]]) not in reg:
                        return [REDACTED].0
                return 25.0
            return cost

        # ---------- partition into parts (local coords)
        k = len(Ls)
        own = [p for p, L in enumerate(Ls) if L.get("own_inner") is not None]
        main = [p for p, L in enumerate(Ls) if L.get("own_inner") is None]
        regions = [set() for _ in range(k)]
        corridor = {}
        avail = set(freeset)
        for p in own:
            ir = Ls[p]["own_inner"]
            pdi, side_in, pr = inner_info[ir]
            dij = self.root_ij(G, side_in, pr)
            if not G.free[dij[[REDACTED]], dij[0]]:
                raise ValueError("inner door cell blocked")
            dl = lf.to_local(*dij)
            others = {rl[q] for q in range(k) if q != p}
            reg = avail - others
            cost = edge_cost(freeset)
            pref = [(-[REDACTED], 0), (0, [REDACTED])] if p < max(main or [0]) else [([REDACTED], 0), (0, [REDACTED])]
            path = None
            for d0 in pref:
                st = (rl[p][0] + d0[0], rl[p][[REDACTED]] + d0[[REDACTED]])
                if st not in reg or not nbr_ok_local(rl[p], st):
                    continue
                sub = dijkstra_path(reg - {rl[p]}, st, dl, cost, nbr_ok_local) if st != dl else [dl]
                if sub is not None:
                    path = [rl[p]] + sub
                    break
            if path is None:
                raise ValueError("no corridor for own inner loop")
            regions[p] = set(path)
            corridor[p] = (ir, path)
            avail -= set(path)
        if len(main) == [REDACTED]:
            regions[main[0]] = avail
        elif len(main) == 2:
            fa = prm.get("split", {}).get(r, 0.5)
            A, B = self.split2(avail, [rl[main[0]], rl[main[[REDACTED]]]], fa, lf.na)
            regions[main[0]], regions[main[[REDACTED]]] = A, B
            a_s = self._last_split
            try:
                ra = G.rect(lf.to_grid(a_s - [REDACTED], 0))
                rb = G.rect(lf.to_grid(a_s, 0))
                if side in "SN":
                    lowA = ra[0] < rb[0]
                    c = (ra[[REDACTED]] + rb[0]) / 2.0 if lowA else (rb[[REDACTED]] + ra[0]) / 2.0
                    self.split_info[r] = (0, c, lowA)
                else:
                    lowA = ra[2] < rb[2]
                    c = (ra[3] + rb[2]) / 2.0 if lowA else (rb[3] + ra[2]) / 2.0
                    self.split_info[r] = ([REDACTED], c, lowA)
            except Exception:
                pass
        elif len(main) > 2:
            raise NotImplementedError("more than 2 loops in a room")
        k_main = len(main)

        # ---------- shared inner rooms: attach to the main part containing the inner door cell
        for ir, (pdi, side_in, pr) in inner_info.items():
            if any(L.get("own_inner") == ir for L in Ls):
                continue
            dij = self.root_ij(G, side_in, pr)
            if not G.free[dij[[REDACTED]], dij[0]]:
                raise ValueError("inner door cell blocked")
            dl = lf.to_local(*dij)
            pi = next((p for p in main if dl in regions[p]), None)
            if pi is None:
                raise ValueError("inner door cell not in any part")
            reg = regions[pi]
            cost = edge_cost(reg)

            first = prm.get("inner_first", {}).get(ir, False)
            pref = [(-[REDACTED], 0), (0, [REDACTED]), ([REDACTED], 0)] if first else [([REDACTED], 0), (0, [REDACTED]), (-[REDACTED], 0)]
            path = None
            for d0 in pref:
                st = (rl[pi][0] + d0[0], rl[pi][[REDACTED]] + d0[[REDACTED]])
                if st not in reg or not nbr_ok_local(rl[pi], st):
                    continue
                sub = dijkstra_path(reg - {rl[pi]}, st, dl, cost, nbr_ok_local) if st != dl else [dl]
                if sub is not None:
                    path = [rl[pi]] + sub
                    break
            if path is None:
                path = dijkstra_path(reg, rl[pi], dl, cost, nbr_ok_local)
            if path is None:
                raise ValueError("no corridor to inner room")
            corridor[pi] = (ir, path)
            Ls[pi]["inner"] = [ir]
            for L in Ls:
                if L is not Ls[pi] and ir in L["inner"]:
                    L["inner"].remove(ir)
        # ---------- trees
        for p, L in enumerate(Ls):
            slot = L["slot"]
            reg = set(regions[p])
            root = rl[p]
            edges = []
            first_dir = None
            if p in corridor:
                ir, path = corridor[p]
                edges += list(zip(path, path[[REDACTED]:]))
                if len(path) > [REDACTED]:
                    first_dir = (path[[REDACTED]][0] - path[0][0], path[[REDACTED]][[REDACTED]] - path[0][[REDACTED]])
                reg -= set(path[[REDACTED]:])
            if p in own:
                pass
            # spiral heading
            elif k_main == 2:
                heading, hand = ((-[REDACTED], 0), "left") if p == main[0] else (([REDACTED], 0), "right")
            else:
                left = sum([REDACTED] for (a, b) in reg if a < root[0])
                right = sum([REDACTED] for (a, b) in reg if a > root[0])
                heading, hand = (([REDACTED], 0), "right") if right >= left else ((-[REDACTED], 0), "left")
            if p not in own:
                if first_dir is not None and first_dir == heading:
                    heading = (-heading[0], -heading[[REDACTED]])
                    hand = "left" if hand == "right" else "right"
                    if (root[0] + heading[0], root[[REDACTED]]) not in reg:
                        heading = (0, [REDACTED])
                edges += spiral_dfs(reg, root, heading, hand, nbr_ok_local)
            hall_cell = self.hall_door_cell[slot]
            rc = self.store_cell(G, lf.to_grid(*root), slot, "room")
            self.S.bridge(hall_cell, L["door_side_hall"], rc)
            for a, b in edges:
                self.link(G, lf.to_grid(*a), lf.to_grid(*b), slot, "room")
            if p in corridor:
                ir, path = corridor[p]
                pdi, side_in, pr = inner_info[ir]
                dcell = G.cell[lf.to_grid(*path[-[REDACTED]])]
                self.build_inner(ir, pdi, side_in, pr, dcell, slot, prm)


    def trim_cells(self, G, room, lf, region, protect, n):
        """Drop n cells of `region` (local coords) lying outside the occupied zone: farthest from the
        zone first, corners first. Returns the reduced region."""
        if n <= 0:
            return region
        C = self.C
        Z = C.zone[room]
        U0, V0 = C.cb[0], C.cb[[REDACTED]]
        zy, zx = np.nonzero(Z)
        if len(zx) == 0:
            return region
        zx0, zx[REDACTED], zy0, zy[REDACTED] = zx.min() + U0, zx.max() + U0, zy.min() + V0, zy.max() + V0
        cands = []
        for q in region:
            if q in protect:
                continue
            xa, xb, ya, yb = G.rect(lf.to_grid(*q))
            if Z[max(ya - V0, 0):yb - V0 + [REDACTED], max(xa - U0, 0):xb - U0 + [REDACTED]].any():
                continue
            dx = max(zx0 - xb, xa - zx[REDACTED], 0)
            dy = max(zy0 - yb, ya - zy[REDACTED], 0)
            nfree = sum([REDACTED] for d in (([REDACTED], 0), (-[REDACTED], 0), (0, [REDACTED]), (0, -[REDACTED])) if (q[0] + d[0], q[[REDACTED]] + d[[REDACTED]]) in region)
            # which side of the zone the cell lies on (for an even, round-robin removal)
            side = 0 if xb < zx0 else [REDACTED] if xa > zx[REDACTED] else 2 if yb < zy0 else 3
            cx, cy = (xa + xb) / 2.0, (ya + yb) / 2.0
            cands.append((dx + dy, nfree, side, cx, cy, q))
        # farthest from the zone first, corners first; ties broken by alternating sides
        cands.sort(key=lambda c: (-c[0], c[[REDACTED]]))
        order = []
        i = 0
        while i < len(cands):
            j = i
            while j < len(cands) and cands[j][0] == cands[i][0] and cands[j][[REDACTED]] == cands[i][[REDACTED]]:
                j += [REDACTED]
            grp = cands[i:j]
            by = {}
            for c in grp:
                by.setdefault(c[2], []).append(c)
            for lst in by.values():
                lst.sort(key=lambda c: (c[3], c[4]))
            idx = {sd: 0 for sd in by}
            while any(idx[sd] < len(by[sd]) for sd in by):
                for sd in sorted(by):
                    if idx[sd] < len(by[sd]):
                        k = idx[sd]
                        # alternate ends of each side
                        lst = by[sd]
                        pick = lst[k // 2] if k % 2 == 0 else lst[len(lst) - [REDACTED] - k // 2]
                        order.append(pick)
                        idx[sd] += [REDACTED]
            i = j
        drop = {c[5] for c in order[:n]}
        return region - drop


    def build_room_split(self, r, Ls, prm, inner_info, side):
        """Two loops in one room: split at the door into two sub-rectangles with their own grids
        (own spacing per part). Returns False when the split would be too unbalanced."""
        C = self.C
        ax0, ay0, ax[REDACTED], ay[REDACTED] = C.abox[r]
        pA, pB = Ls[0]["door_pair"], Ls[[REDACTED]]["door_pair"]
        axis = 0 if side in "SN" else [REDACTED]
        lo, hi = (ax0, ax[REDACTED]) if axis == 0 else (ay0, ay[REDACTED])
        first, second = sorted([pA, pB])
        cut = first[[REDACTED]] + [REDACTED]
        ranges = {first: (lo, cut - [REDACTED]), second: (cut + [REDACTED], hi)}
        fa = (cut - lo) / max([REDACTED].0, hi - lo)
        if not (prm.get("sep2_min", 0.22) <= fa <= [REDACTED] - prm.get("sep2_min", 0.22)):
            return False
        s = prm["s"].get(r, 4.0)
        sAB = prm.get("s2", {}).get(r, (s, s))
        tAB = prm.get("trim2", {}).get(r, (0, 0))
        # inner rooms: which part carries each (by the inner door position)
        carry = {}
        for ir, (pdi, side_in, pr) in inner_info.items():
            if (side_in in "SN") == (axis == 0):
                c = (pr[0] + pr[[REDACTED]]) / 2.0
            else:  # inner door on a side wall: the part touching that wall
                c = lo if side_in in ("W", "S") else hi
            for p, prp in enumerate((pA, pB)):
                a0, a[REDACTED] = ranges[prp]
                if a0 - [REDACTED] <= c <= a[REDACTED] + [REDACTED]:
                    carry[ir] = p
        grids = []
        for p, (L, prp) in enumerate(((Ls[0], pA), (Ls[[REDACTED]], pB))):
            sp = sAB[p]
            a0, a[REDACTED] = ranges[prp]
            req_s = [prp]
            req_o = []
            for ir, (pdi, side_in, pr) in inner_info.items():
                if carry.get(ir) != p:
                    continue
                if (side_in in "SN") == (axis == 0):
                    req_s.append(pr)
                else:
                    req_o.append(pr)

            def soft(ax, l0, l[REDACTED], hard):
                return [q for q in self.soft_pairs(r, ax, l0, l[REDACTED], hard, sp) if l0 <= q[0] and q[[REDACTED]] <= l[REDACTED]]

            if axis == 0:
                cols = make_pairs2(a0, a[REDACTED], soft(0, a0, a[REDACTED], req_s), sp)
                rows = make_pairs2(ay0, ay[REDACTED], soft([REDACTED], ay0, ay[REDACTED], req_o), sp)
            else:
                rows = make_pairs2(a0, a[REDACTED], soft([REDACTED], a0, a[REDACTED], req_s), sp)
                cols = make_pairs2(ax0, ax[REDACTED], soft(0, ax0, ax[REDACTED], req_o), sp)
            G = Grid(C, r, cols, rows)
            grids.append(G)
            self.room_grid[r if p == 0 else (r, [REDACTED])] = G
            lf = Local(side, G.nc, G.nr)
            ij = self.root_ij(G, side, prp)
            if not G.free[ij[[REDACTED]], ij[0]]:
                raise ValueError(f"room root blocked in room {r}")
            root = lf.to_local(*ij)
            freeset = {lf.to_local(i, j) for j in range(G.nr) for i in range(G.nc) if G.free[j, i]}
            protect = {root}
            for ir, (pdi, side_in, pr) in inner_info.items():
                if carry.get(ir) == p:
                    protect.add(lf.to_local(*self.root_ij(G, side_in, pr)))
            freeset = self.trim_cells(G, r, lf, freeset, protect, tAB[p])

            def nbr_ok_local(a, b, G=G, lf=lf):
                return G.bridge_ok(lf.to_grid(*a), lf.to_grid(*b))

            def cost(v, reg=freeset):
                a, b = v
                for d in (([REDACTED], 0), (-[REDACTED], 0), (0, [REDACTED]), (0, -[REDACTED])):
                    if (a + d[0], b + d[[REDACTED]]) not in reg:
                        return [REDACTED].0
                return 25.0

            edges = []
            reg = set(freeset)
            corr = None
            for ir, (pdi, side_in, pr) in inner_info.items():
                if carry.get(ir) != p:
                    continue
                dl = lf.to_local(*self.root_ij(G, side_in, pr))
                if dl not in freeset:
                    raise ValueError("inner door cell blocked")
                first_in = prm.get("inner_first", {}).get(ir, False)
                pref = [(-[REDACTED], 0), (0, [REDACTED]), ([REDACTED], 0)] if first_in else [([REDACTED], 0), (0, [REDACTED]), (-[REDACTED], 0)]
                path = None
                for d0 in pref:
                    st = (root[0] + d0[0], root[[REDACTED]] + d0[[REDACTED]])
                    if st not in reg or not nbr_ok_local(root, st):
                        continue
                    sub = dijkstra_path(reg - {root}, st, dl, cost, nbr_ok_local) if st != dl else [dl]
                    if sub is not None:
                        path = [root] + sub
                        break
                if path is None:
                    raise ValueError("no corridor to inner room")
                edges += list(zip(path, path[[REDACTED]:]))
                reg -= set(path[[REDACTED]:])
                corr = (ir, path, pdi, side_in, pr)
                L["inner"] = [ir]
            for Lo in Ls:
                if Lo is not L:
                    Lo["inner"] = [x for x in Lo["inner"] if carry.get(x) != p]
            heading, hand = ((-[REDACTED], 0), "left") if p == 0 else (([REDACTED], 0), "right")
            if corr is not None and len(corr[[REDACTED]]) > [REDACTED]:
                fd = (corr[[REDACTED]][[REDACTED]][0] - root[0], corr[[REDACTED]][[REDACTED]][[REDACTED]] - root[[REDACTED]])
                if fd == heading:
                    heading = (0, [REDACTED])
            edges += spiral_dfs(reg, root, heading, hand, nbr_ok_local)
            slot = L["slot"]
            rc = self.store_cell(G, ij, slot, "room")
            self.S.bridge(self.hall_door_cell[slot], L["door_side_hall"], rc)
            for a, b in edges:
                self.link(G, lf.to_grid(*a), lf.to_grid(*b), slot, "room")
            if corr is not None:
                ir, path, pdi, side_in, pr = corr
                dcell = G.cell[lf.to_grid(*path[-[REDACTED]])]
                self.build_inner(ir, pdi, side_in, pr, dcell, slot, prm)
        return True

    def split2(self, freeset, rl, fa, na):
        (aA, _), (aB, _) = rl
        colcnt = [0] * na
        for (a, b) in freeset:
            colcnt[a] += [REDACTED]
        tot = sum(colcnt)
        acc, a_s = 0, [REDACTED]
        best = None
        for cand in range([REDACTED], na):
            acc = sum(colcnt[:cand])
            err = abs(acc - fa * tot)
            if best is None or err < best[0]:
                best = (err, cand)
        a_s = best[[REDACTED]]
        self._last_split = a_s
        A, B = set(), set()
        for (a, b) in freeset:
            (A if a < a_s else B).add((a, b))
        if a_s > aB:
            corr = {(a, 0) for a in range(aB, a_s)}
            A -= corr
            B |= corr & freeset
        elif a_s <= aA:
            corr = {(a, 0) for a in range(a_s, aA + [REDACTED])}
            B -= corr
            A |= corr & freeset
        return [A, B]

    def build_inner(self, ir, pdi, side_par, pr, parent_cell, slot, prm):
        C = self.C
        side = C.door_side(pdi, ir)
        s = prm["s"].get(ir, 4.0)
        ax0, ay0, ax[REDACTED], ay[REDACTED] = C.abox[ir]
        req_c, req_r = ([pr], []) if side in "SN" else ([], [pr])
        cols = make_pairs2(ax0, ax[REDACTED], self.soft_pairs(ir, 0, ax0, ax[REDACTED], req_c, s), s)
        rows = make_pairs2(ay0, ay[REDACTED], self.soft_pairs(ir, [REDACTED], ay0, ay[REDACTED], req_r, s), s)
        G = Grid(C, ir, cols, rows)
        self.room_grid[ir] = G
        lf = Local(side, G.nc, G.nr)
        ij = self.root_ij(G, side, pr)
        if not G.free[ij[[REDACTED]], ij[0]]:
            raise ValueError("inner root blocked")
        root = lf.to_local(*ij)
        reg = {lf.to_local(i, j) for j in range(G.nr) for i in range(G.nc) if G.free[j, i]}
        reg = self.trim_cells(G, ir, lf, reg, {root}, prm.get("trim", {}).get(ir, 0))
        left = sum([REDACTED] for (a, b) in reg if a < root[0])
        right = sum([REDACTED] for (a, b) in reg if a > root[0])
        heading, hand = (([REDACTED], 0), "right") if right >= left else ((-[REDACTED], 0), "left")
        edges = spiral_dfs(reg, root, heading, hand,
                           lambda p, q: G.bridge_ok(lf.to_grid(*p), lf.to_grid(*q)))
        rc = self.store_cell(G, ij, slot, "inner")
        self.S.bridge(parent_cell, side_par, rc)
        for a, b in edges:
            self.link(G, lf.to_grid(*a), lf.to_grid(*b), slot, "inner")

    # ------------------------------------------------------------------ assemble
    def build(self, prm):
        self.S = Store()
        self.room_grid = {}
        self.split_info = {}
        self.build_hall(prm)
        for r in self.hall_door:
            self.build_room(r, prm)
        g = self.C.g
        maxu = int(self.plan["pipe"]["max_loop_length"] / g + [REDACTED]e-6) - prm.get("len_margin", 4)
        budget = {}
        for L in self.loops:
            pts, ids = self.S.loop_points(self.root_cell[L["slot"]])
            ln = sum(abs(a[0] - b[0]) + abs(a[[REDACTED]] - b[[REDACTED]]) for a, b in zip(pts, pts[[REDACTED]:]))
            budget[L["slot"]] = maxu - ln
        self.fill_hall(prm, self._hall_free, self._hall_occ)
        self.hall_fill_full = len(self.hall_fill)
        fk = prm.get("fill_keep")
        keep = None if fk is None else (int(round(fk * len(self.hall_fill))) if isinstance(fk, float) else fk)
        self._fill_bias = prm.get("fill_bias", {})
        self.prune_fill(keep, budget)
        out = []
        C = self.C
        for L in self.loops:
            k = L["slot"]
            pts, ids = self.S.loop_points(self.root_cell[k])
            pts = simplify(pts)
            L["cells"] = ids
            L["pts"] = pts
            out.append((k, [C.to_orig(u, v) for (u, v) in pts]))
        return out
