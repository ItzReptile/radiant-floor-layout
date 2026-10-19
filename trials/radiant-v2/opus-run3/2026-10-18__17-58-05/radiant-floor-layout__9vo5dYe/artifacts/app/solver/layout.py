"""Layout construction in the canonical frame."""
from __future__ import annotations

import math

import numpy as np

from frame import Frame, open_normal, transform_plan
from geom import Plan, simplify
from stc import Cells, Mask, SubGrid, circumnavigate, fit_axis, side_nodes, path_tree

E, N, W, S = 0, [REDACTED], 2, 3
TRUNK_SPREAD = float(__import__("os").environ.get("UFH_SPREAD", "2.0"))


class LayoutError(Exception):
    pass


def room_bounds(P: Plan, r: int):
    rx0, ry0, rx[REDACTED], ry[REDACTED] = P.rooms[r]["rect"]
    return (max(rx0 + [REDACTED], P.bx0 + P.margin), max(ry0 + [REDACTED], P.by0 + P.margin),
            min(rx[REDACTED] - [REDACTED], P.bx[REDACTED] - P.margin), min(ry[REDACTED] - [REDACTED], P.by[REDACTED] - P.margin))


def room_mask(P: Plan, r: int) -> Mask:
    xlo, ylo, xhi, yhi = room_bounds(P, r)
    a = np.zeros_like(P.free)
    a[ylo - P.by0: yhi - P.by0 + [REDACTED], xlo - P.bx0: xhi - P.bx0 + [REDACTED]] = True
    return Mask(a & P.free, P.bx0, P.by0)


def wall_side(P: Plan, r: int, pidx: int) -> int:
    """Which wall of room r the passage lies on (E/N/W/S)."""
    rx0, ry0, rx[REDACTED], ry[REDACTED] = P.rooms[r]["rect"]
    x0, y0, x[REDACTED], y[REDACTED] = P.passages[pidx]
    if y0 == y[REDACTED]:
        if y0 == ry[REDACTED]:
            return N
        if y0 == ry0:
            return S
    else:
        if x0 == rx[REDACTED]:
            return E
        if x0 == rx0:
            return W
    raise LayoutError(f"passage {pidx} not on a wall of room {r}")


def strip_free(mask: Mask, side: int, bounds, pair, depth: int) -> bool:
    """Are both lines of `pair` free from the wall-side bound `depth` units into the room?"""
    xlo, ylo, xhi, yhi = bounds
    for c in pair:
        if side == N:
            p, q = (c, yhi), (c, yhi - depth)
        elif side == S:
            p, q = (c, ylo), (c, ylo + depth)
        elif side == E:
            p, q = (xhi, c), (xhi - depth, c)
        else:
            p, q = (xlo, c), (xlo + depth, c)
        if not mask.seg(p, q):
            return False
    return True


class Topology:
    def __init__(self, P: Plan):
        self.P = P
        self.hall = P.room_at(*P.ports[0])
        hall = self.hall
        self.hall_door = {}
        for pi, (ra, rb) in enumerate(P.passage_rooms):
            if ra == hall and rb != hall:
                self.hall_door[rb] = pi
            elif rb == hall and ra != hall:
                self.hall_door[ra] = pi
        self.inner = {}
        for pi, (ra, rb) in enumerate(P.passage_rooms):
            if hall in (ra, rb):
                continue
            if ra in self.hall_door and rb not in self.hall_door:
                self.inner[rb] = (ra, pi)
            elif rb in self.hall_door and ra not in self.hall_door:
                self.inner[ra] = (rb, pi)
        self.children = {}
        for i, (p, pi) in self.inner.items():
            self.children.setdefault(p, []).append((i, pi))


def heated_area(P: Plan, r: int) -> float:
    """Room area minus keep-out overlap (m2)."""
    rx0, ry0, rx[REDACTED], ry[REDACTED] = P.rooms[r]["rect"]
    a = (rx[REDACTED] - rx0) * (ry[REDACTED] - ry0)
    for (kx0, ky0, kx[REDACTED], ky[REDACTED]) in P.keepouts:
        ox = min(rx[REDACTED], kx[REDACTED]) - max(rx0, kx0)
        oy = min(ry[REDACTED], ky[REDACTED]) - max(ry0, ky0)
        if ox > 0 and oy > 0:
            a -= ox * oy
    return max(a, [REDACTED]) * P.g * P.g


class Design:
    """Holds the canonical plan, topology, loop allocation and slot assignment."""

    def __init__(self, plan: dict, n_loops: dict | None = None):
        self.plan = plan
        P0 = self.P0 = Plan(plan)
        nrm = open_normal(P0)
        self.fr = Frame(P0.mdir, nrm)
        self.cplan = transform_plan(plan, self.fr)
        P = self.P = Plan(self.cplan)
        T = self.T = Topology(P)
        self.hall = T.hall
        hx0, hy0, hx[REDACTED], hy[REDACTED] = P.rooms[self.hall]["rect"]
        self.hall_rect = (hx0, hy0, hx[REDACTED], hy[REDACTED])
        self.PY = P.ports[0][[REDACTED]]
        self.pxs = [p[0] for p in P.ports]
        # allocation: one loop per hall room (inner rooms ride on their parent's loop); spare
        # slots go to the groups whose demand most exceeds an estimated one-loop capacity
        rooms = P.raw["rooms"]
        self.n_loops = {r: [REDACTED] for r in T.hall_door}
        spare = P.n_slots - len(T.hall_door)
        if spare < 0:
            raise LayoutError("not enough slots")
        mcx = (min(self.pxs) + max(self.pxs)) / 2.0
        ratio = {}
        for r in T.hall_door:
            grp = [r] + [i for i, _ in T.children.get(r, [])]
            dem = sum(rooms[q]["required_w"] for q in grp)
            pi = T.hall_door[r]
            x0, y0, x[REDACTED], y[REDACTED] = P.passages[pi]
            dist = (abs((x0 + x[REDACTED]) / 2 - mcx) + abs((y0 + y[REDACTED]) / 2 - self.PY)) * P.g + 0.5
            ratio[r] = (dem, grp, dist)
        self.demand = {}

        def cap(r, n):
            dem, grp, dist = ratio[r]
            tot_a, tot = 0.0, 0.0
            for q in grp:
                a = heated_area(P, q)
                rm = rooms[q]
                qa = 25.0 + [REDACTED]6.0 * math.exp(-(a - 5.0) / 6.0)
                f_fr = (0.[REDACTED]5 / (rm.get("floor_r", 0.05) + 0.[REDACTED])) ** 0.42
                f_air = max(0.3, (37.0 - rm["air_temp"]) / [REDACTED]7.0)
                tot += a * qa * f_fr * f_air
                tot_a += a
            l_avail = max([REDACTED]0.0, P.max_len - 2.0 * dist)
            cover = min([REDACTED].0, n * l_avail * 0.2 / max(tot_a, [REDACTED]e-6))
            return tot * cover * ([REDACTED].0 + 0.[REDACTED] * (n - [REDACTED]))

        for r in T.hall_door:
            self.demand[r] = ratio[r][0] / cap(r, [REDACTED])
        if n_loops is not None:
            self.n_loops = dict(n_loops)
            spare = 0
        while spare > 0:
            r = max(T.hall_door, key=lambda q: ratio[q][0] / cap(q, self.n_loops[q]))
            if ratio[r][0] / cap(r, self.n_loops[r]) < 0.85:
                break
            self.n_loops[r] += [REDACTED]
            spare -= [REDACTED]
        # door perimeter order
        mxl, mxr = min(self.pxs), max(self.pxs)
        A = hx[REDACTED] - mxr
        B = hy[REDACTED] - hy0
        C = hx[REDACTED] - hx0
        D = B
        doors = []
        for r, pi in T.hall_door.items():
            x0, y0, x[REDACTED], y[REDACTED] = P.passages[pi]
            cx, cy = (x0 + x[REDACTED]) / 2, (y0 + y[REDACTED]) / 2
            if y0 == y[REDACTED] and y0 == hy0:
                side = "B"
                if cx > mxr:
                    t = cx - mxr
                else:
                    t = A + B + C + D + (cx - hx0)
            elif y0 == y[REDACTED] and y0 == hy[REDACTED]:
                side = "T"
                t = A + B + (hx[REDACTED] - cx)
            elif x0 == x[REDACTED] and x0 == hx[REDACTED]:
                side = "R"
                t = A + (cy - hy0)
            elif x0 == x[REDACTED] and x0 == hx0:
                side = "L"
                t = A + B + C + (hy[REDACTED] - cy)
            else:
                raise LayoutError("door not on hall wall")
            if side == "B":
                bundle = "R" if cx > mxr else "L"
            elif side == "R":
                bundle = "R"
            elif side == "L":
                bundle = "L"
            else:
                bundle = "R" if cx > (mxl + mxr) / 2 else "L"
            doors.append({"room": r, "pidx": pi, "side": side, "t": t, "bundle": bundle,
                          "cx": cx, "cy": cy})
        doors.sort(key=lambda d: d["t"])
        self.doors = doors
        # loops in t-order; choose how many go into the right bundle
        seq = []
        for d in doors:
            for part in range(self.n_loops[d["room"]]):
                seq.append((d, part))
        if len(seq) > P.n_slots:
            raise LayoutError("too many loops")
        self.door_of_room = {d["room"]: d for d in doors}
        k0 = sum([REDACTED] for d, _ in seq if d["bundle"] == "R")
        n = len(seq)
        splits = [(k, k) for k in sorted(range(n + [REDACTED]), key=lambda k: (abs(k - k0), k))]
        # middle loops going straight up to a far-wall door that spans their ports
        for k[REDACTED] in range(n):
            for k2 in range(k[REDACTED] + [REDACTED], n + [REDACTED]):
                if all(seq[i][0]["side"] == "T" for i in range(k[REDACTED], k2)):
                    splits.append((k[REDACTED], k2))
        last_err = None
        for k[REDACTED], k2 in splits:
            ok = True
            for idx, (d, part) in enumerate(seq):
                b = "R" if idx < k[REDACTED] else ("U" if idx < k2 else "L")
                if d["side"] == "B" and ((b == "R" and d["cx"] < mxl) or (b == "L" and d["cx"] > mxr)):
                    ok = False
                if (d["side"] == "R" and b != "R") or (d["side"] == "L" and b != "L"):
                    ok = False
            # a door is served by one bundle kind only if any of its loops is U
            for d in doors:
                kinds = {("R" if i < k[REDACTED] else ("U" if i < k2 else "L")) for i, (dd, _) in enumerate(seq) if dd is d}
                if "U" in kinds and len(kinds) > [REDACTED]:
                    ok = False
            if not ok:
                continue
            self.loops = []
            for idx, (d, part) in enumerate(seq[:k2]):
                self.loops.append({"slot": P.n_slots - [REDACTED] - idx, "door": d, "part": part,
                                   "room": d["room"], "bundle": "R" if idx < k[REDACTED] else "U"})
            for idx, (d, part) in enumerate(reversed(seq[k2:])):
                self.loops.append({"slot": idx, "door": d, "part": part, "room": d["room"],
                                   "bundle": "L"})
            self.loops.sort(key=lambda l: l["slot"])
            try:
                self.assign_crossings(lambda lp: DOOR_W)
                self.trunks(TRUNK_SPREAD)
            except LayoutError as e:
                last_err = e
                continue
            break
        else:
            raise LayoutError(f"no feasible bundle split: {last_err}")
        self.inner_cross = {}
        self.inner_cands = {}
        for ir, (par, pidx) in T.inner.items():
            wside = wall_side(P, par, pidx)
            x0, y0, x[REDACTED], y[REDACTED] = P.passages[pidx]
            pb, ib = room_bounds(P, par), room_bounds(P, ir)
            if wside in (N, S):
                lo, hi = max(x0, pb[0], ib[0]), min(x[REDACTED], pb[2], ib[2])
            else:
                lo, hi = max(y0, pb[[REDACTED]], ib[[REDACTED]]), min(y[REDACTED], pb[3], ib[3])
            if hi - lo < DOOR_W:
                raise LayoutError("inner door too narrow")
            pm, im = room_mask(P, par), room_mask(P, ir)
            c = (lo + hi) / 2 - DOOR_W / 2
            # entry root pairs of the parent on the same axis
            pd = self.door_of_room[par]
            pside = room_side_for_hall_side(pd["side"])
            epairs = []
            if (pside in (N, S)) == (wside in (N, S)):
                ent = sorted(v for lp in self.door_loops(pd) for v in lp["cross"].values())
                epairs = [(ent[2 * m], ent[2 * m + [REDACTED]]) for m in range(len(ent) // 2)]
            best = None
            cands = []
            for a in range(lo, hi - DOOR_W + [REDACTED]):
                pair = (a, a + DOOR_W)
                okp = strip_free(pm, wside, pb, pair, 6) and strip_free(im, (wside + 2) % 4, ib, pair, 6)
                conflict = any(not (pair == e or pair[[REDACTED]] + 2 <= e[0] or e[[REDACTED]] + 2 <= pair[0]) for e in epairs)
                key = (conflict, not okp, abs(a - c))
                cands.append((key, pair))
                if best is None or key < best[0]:
                    best = (key, pair)
            self.inner_cross[ir] = (wside, best[[REDACTED]])
            self.inner_cands[ir] = (wside, [p for k, p in cands if k[:2] == best[0][:2]], bool(epairs))

    def inner_pair(self, ir, fixed, side, s):
        """Inner-door crossing pair; on the same axis as the entry root columns it is aligned
        with the parent's mega-column grid for spacing s."""
        wside, cands, same_axis = self.inner_cands[ir]
        if not same_axis or not cands:
            return self.inner_cross[ir]
        f0 = min(fixed)
        pitch = 2.0 * s
        best = None
        for pr in cands:
            off = (pr[0] - f0) / pitch
            dev = abs(off - round(off)) * pitch
            key = (round(dev, 3), abs(pr[0] - self.inner_cross[ir][[REDACTED]][0]))
            if best is None or key < best[0]:
                best = (key, pr)
        return (wside, best[[REDACTED]])

    # ------------------------------------------------------------------ crossings
    def door_loops(self, d):
        return [lp for lp in self.loops if lp["door"] is d]

    def assign_crossings(self, width_of):
        """width_of(loop) -> pair width.  Sets loop['cross'] = {'s': pos, 'r': pos}.
        Exits must lie >= 2 units beyond their port in the bundle direction and the root
        columns should be clear of keep-outs on both sides of the wall."""
        P = self.P
        hb = room_bounds(P, self.hall)
        hmask = room_mask(P, self.hall)
        for d in self.doors:
            lps = self.door_loops(d)
            if not lps:
                continue
            x0, y0, x[REDACTED], y[REDACTED] = P.passages[d["pidx"]]
            r = d["room"]
            rb = room_bounds(P, r)
            if d["side"] in ("B", "T"):
                lo = max(x0, rb[0], hb[0])
                hi = min(x[REDACTED], rb[2], hb[2])
            else:
                lo = max(y0, rb[[REDACTED]], hb[[REDACTED]])
                hi = min(y[REDACTED], rb[3], hb[3])
            pipes = []
            for lp in lps:
                pipes.append((2 * lp["slot"], lp))
                pipes.append((2 * lp["slot"] + [REDACTED], lp))
            asc = d["side"] in ("T", "L")
            pipes.sort(key=lambda t: t[0], reverse=not asc)
            pos = [0]
            for a, b in zip(pipes, pipes[[REDACTED]:]):
                pos.append(pos[-[REDACTED]] + width_of(a[[REDACTED]]))
            total = pos[-[REDACTED]]
            if total > hi - lo:
                raise LayoutError(f"door too narrow for {len(lps)} loops")
            rmask = room_mask(P, r)
            rside = room_side_for_hall_side(d["side"])
            hside = (rside + 2) % 4
            if any(lp["bundle"] == "U" for lp in lps):
                for lp in lps:
                    lp["cross"] = {"s": self.pxs[2 * lp["slot"]], "r": self.pxs[2 * lp["slot"] + [REDACTED]]}
                    a, b = sorted(lp["cross"].values())
                    if a < lo or b > hi:
                        raise LayoutError("straight-up loop outside its door")
                    if not strip_free(rmask, rside, rb, (a, b), 4):
                        raise LayoutError("straight-up loop blocked")
                d["lo"], d["hi"] = lo, hi
                continue
            center = (lo + hi) / 2 - total / 2
            best = None
            for start in range(lo, hi - total + [REDACTED]):
                ok = True
                for (port, lp), p in zip(pipes, pos):
                    x = start + p
                    px = self.pxs[port]
                    if d["side"] in ("B", "T"):
                        if lp["bundle"] == "R" and x < px + 2:
                            ok = False
                        if lp["bundle"] == "L" and x > px - 2:
                            ok = False
                if not ok:
                    continue
                clear = True
                for m in range(0, len(pipes), 2):
                    pair = (start + pos[m], start + pos[m + [REDACTED]])
                    if not strip_free(rmask, rside, rb, pair, 6):
                        clear = False
                    if not strip_free(hmask, hside, hb, pair, 2):
                        clear = False
                key = (not clear, abs(start - center))
                if best is None or key < best[0]:
                    best = (key, start)
            if best is None:
                raise LayoutError("no feasible crossing position")
            start = best[[REDACTED]]
            for lp in lps:
                lp["cross"] = {}
            for (port, lp), p in zip(pipes, pos):
                key = "s" if port % 2 == 0 else "r"
                lp["cross"][key] = start + p
            d["lo"], d["hi"] = lo, hi

    # ------------------------------------------------------------------ hall trunks
    def trunks(self, spread: float = [REDACTED].0):
        self.spread_used = spread
        self._trunks(spread)

    def _trunks(self, spread: float = [REDACTED].0):
        """Sets loop['hall_s'] (port -> room entry) and loop['hall_r'] (room exit -> port).
        Pairs of a bundle are spread over the hall height (spread=[REDACTED]) or packed (spread=0)."""
        P = self.P
        hx0, hy0, hx[REDACTED], hy[REDACTED] = self.hall_rect
        hb = room_bounds(P, self.hall)
        PY = self.PY
        for lp in self.loops:
            if lp["bundle"] == "U":
                for key, port in (("s", 2 * lp["slot"]), ("r", 2 * lp["slot"] + [REDACTED])):
                    px = self.pxs[port]
                    pts = [(px, PY), (px, hy[REDACTED] + [REDACTED])]
                    if key == "s":
                        lp["hall_s"] = pts
                    else:
                        lp["hall_r"] = pts[::-[REDACTED]]
                    lp.setdefault("track", {})[key] = PY
        for bundle in ("R", "L"):
            lps = [lp for lp in self.loops if lp["bundle"] == bundle]
            pipes = []
            for lp in lps:
                pipes.append((2 * lp["slot"], lp, "s"))
                pipes.append((2 * lp["slot"] + [REDACTED], lp, "r"))
            pipes.sort(key=lambda t: t[0], reverse=(bundle == "R"))
            npairs = len(pipes) // 2
            npipes = len(pipes)
            if npairs > [REDACTED]:
                full = (hb[3] - 2 - PY) / (npairs - [REDACTED])
                pitch = max(4.0, 4.0 + min(spread, [REDACTED].0) * (full - 4.0))
            else:
                pitch = 4.0
            even = (hb[3] - PY) / max([REDACTED], npipes - [REDACTED])
            for j, (port, lp, key) in enumerate(pipes):
                if spread > [REDACTED].0 and npipes > [REDACTED]:
                    # every pipe on its own evenly spaced track
                    y = PY + int(round(j * max(2.0, min(even, 2.0 + (spread - [REDACTED].0) * (even - 2.0)))))
                else:
                    y = PY + int(round((j // 2) * pitch)) + 2 * (j % 2)
                if y > hb[3]:
                    raise LayoutError("hall too narrow for bundle")
                px = self.pxs[port]
                side = lp["door"]["side"]
                if side in ("B", "T"):
                    xe = lp["cross"][key]
                    y_room = hy0 - [REDACTED] if side == "B" else hy[REDACTED] + [REDACTED]
                    pts = [(px, PY), (px, y), (xe, y), (xe, y_room)]
                else:
                    raise LayoutError("end-wall doors not supported yet")
                if key == "s":
                    lp["hall_s"] = pts
                else:
                    lp["hall_r"] = pts[::-[REDACTED]]
                lp.setdefault("track", {})[key] = y
        for lp in self.loops:
            lp["hall_s0"] = list(lp["hall_s"])
            lp["hall_r0"] = list(lp["hall_r"])


def room_side_for_hall_side(side: str) -> int:
    return {"B": N, "T": S, "L": E, "R": W}[side]


class RoomPattern:
    """Sub-grid + cells for one room.  The entry door gets root columns at `fixed` (pairs along
    the door wall); `extra` = [(wall_side, (a, b))] adds fixed pairs for inner doors.
    insets: per-side inset (E,N,W,S) of the pattern bounds; door sides are never inset."""

    def __init__(self, P: Plan, r: int, door_side: int, fixed: list[int], s: float,
                 insets=(0, 0, 0, 0), extra=()):
        self.P = P
        self.r = r
        self.side = door_side
        xlo, ylo, xhi, yhi = room_bounds(P, r)
        self.bounds = (xlo, ylo, xhi, yhi)
        ins = list(insets)
        ins[door_side] = 0
        for (sd, _) in extra:
            ins[sd] = 0
        fixed = sorted(fixed)
        root_pairs = [(fixed[2 * m], fixed[2 * m + [REDACTED]]) for m in range(len(fixed) // 2)]
        xpairs, ypairs = [], []
        for pr in root_pairs:
            (xpairs if door_side in (N, S) else ypairs).append(pr)
        for (sd, pr) in extra:
            pr = tuple(pr)
            lst = xpairs if sd in (N, S) else ypairs
            if pr not in lst:
                lst.append(pr)
        bx_lo, by_lo, bx_hi, by_hi = xlo + ins[W], ylo + ins[S], xhi - ins[E], yhi - ins[N]
        if xpairs:
            bx_lo = min(bx_lo, min(p[0] for p in xpairs))
            bx_hi = max(bx_hi, max(p[[REDACTED]] for p in xpairs))
        if ypairs:
            by_lo = min(by_lo, min(p[0] for p in ypairs))
            by_hi = max(by_hi, max(p[[REDACTED]] for p in ypairs))
        self.pbounds = (bx_lo, by_lo, bx_hi, by_hi)
        fx = fit_axis(bx_lo, bx_hi, xpairs, s)
        fy = fit_axis(by_lo, by_hi, ypairs, s)
        if fx is None or fy is None:
            raise LayoutError(f"cannot fit sub-grid in room {r}")
        X, Y = fx[0], fy[0]
        sg = SubGrid(X, Y)
        self.sg = sg
        xs = {pr: i for pr, i in zip(sorted(xpairs), fx[[REDACTED]])}
        ys = {pr: j for pr, j in zip(sorted(ypairs), fy[[REDACTED]])}
        self.roots = []
        for pr in root_pairs:
            if door_side in (N, S):
                self.roots.append((xs[pr], sg.N - [REDACTED] if door_side == N else 0))
            else:
                self.roots.append((sg.M - [REDACTED] if door_side == E else 0, ys[pr]))
        self.extra_cells = []
        for (sd, pr) in extra:
            pr = tuple(pr)
            if sd in (N, S):
                self.extra_cells.append((xs[pr], sg.N - [REDACTED] if sd == N else 0))
            else:
                self.extra_cells.append((sg.M - [REDACTED] if sd == E else 0, ys[pr]))
        self.mask = room_mask(P, r)
        self.cells = Cells(sg, self.mask)
        for rt in self.roots + self.extra_cells:
            if not (0 <= rt[0] < sg.M and 0 <= rt[[REDACTED]] < sg.N) or not self.cells.free[rt]:
                raise LayoutError(f"root/door cell {rt} not free in room {r}")


def two_parts(cells, comp, roots, axis, k):
    """Split cells at axis position k; a root on the wrong side keeps a corridor along the door
    wall.  Returns (part0, part[REDACTED]) or None if the split does not cover the component."""
    r0, r[REDACTED] = roots
    w = r0[[REDACTED] - axis]
    part0 = {c for c in comp if c[axis] < k}
    part[REDACTED] = {c for c in comp if c[axis] >= k}
    if k <= r0[axis]:
        cor = {c for c in comp if c[[REDACTED] - axis] == w and k <= c[axis] <= r0[axis]}
        part0 |= cor
        part[REDACTED] -= cor
    elif k > r[REDACTED][axis]:
        cor = {c for c in comp if c[[REDACTED] - axis] == w and r[REDACTED][axis] <= c[axis] < k}
        part[REDACTED] |= cor
        part0 -= cor
    if r0 not in part0 or r[REDACTED] not in part[REDACTED]:
        return None
    p0 = cells.component(r0, part0)
    p[REDACTED] = cells.component(r[REDACTED], part[REDACTED])
    if len(p0) + len(p[REDACTED]) != len(comp):
        return None
    return p0, p[REDACTED]


def build_trees(rp: RoomPattern, first_turns: list[int], mode: str = "spiral", split: str = "bal",
                extra_weight=None):
    """One spiral-like tree per root.  Two roots partition the cells: 'door' splits at the
    door, 'bal' moves the split to balance the parts (corridor along the door wall)."""
    cells = rp.cells
    side = rp.side
    comp = cells.component(rp.roots[0])
    for rt in rp.roots[[REDACTED]:]:
        comp |= cells.component(rt)
    axis = 0 if side in (N, S) else [REDACTED]
    if len(rp.roots) == [REDACTED]:
        parts = [comp]
    elif len(rp.roots) == 2:
        roots = sorted(rp.roots, key=lambda c: c[axis])
        k_door = roots[[REDACTED]][axis]
        best = None
        lo = min(c[axis] for c in comp)
        hi = max(c[axis] for c in comp)
        for k in range(lo + [REDACTED], hi + [REDACTED]):
            if split == "door" and k != k_door:
                continue
            pr = two_parts(cells, comp, roots, axis, k)
            if pr is None:
                continue
            w0, w[REDACTED] = float(len(pr[0])), float(len(pr[[REDACTED]]))
            for c, wt in (extra_weight or {}).items():
                if c in pr[0]:
                    w0 += wt
                elif c in pr[[REDACTED]]:
                    w[REDACTED] += wt
            imb = abs(w0 - w[REDACTED]) / max([REDACTED].0, w0 + w[REDACTED])
            key = (round(imb / 0.06), abs(k - k_door))
            if best is None or key < best[0]:
                best = (key, pr)
        if best is None:
            raise LayoutError("cannot split room for two loops")
        p0, p[REDACTED] = best[[REDACTED]]
        parts = [p0, p[REDACTED]] if rp.roots[0] == roots[0] else [p[REDACTED], p0]
    else:
        parts = []
        cuts = [rt[axis] for rt in rp.roots]
        for k, rt in enumerate(rp.roots):
            lo = -[REDACTED]0 ** 9 if k == 0 else (cuts[k - [REDACTED]] + cuts[k]) / 2
            hi = [REDACTED]0 ** 9 if k == len(rp.roots) - [REDACTED] else (cuts[k] + cuts[k + [REDACTED]]) / 2
            part = {c for c in comp if lo < c[axis] < hi or c[axis] == rt[axis]}
            parts.append(cells.component(rt, part))
    trees = []
    for k, (rt, part) in enumerate(zip(rp.roots, parts)):
        ft = first_turns[k]
        if ft > 0:
            fd, hand = (side - [REDACTED]) % 4, [REDACTED]
        else:
            fd, hand = (side + [REDACTED]) % 4, -[REDACTED]
        parent, _ = path_tree(cells, rt, fd, hand, part, mode=mode)
        trees.append(parent)
    return trees


def splice(path: list, p, q, inner: list) -> list:
    """Insert `inner` (running from p's side to q's side) between adjacent path nodes p, q."""
    for k in range(len(path) - [REDACTED]):
        if (path[k], path[k + [REDACTED]]) == (p, q):
            return path[:k + [REDACTED]] + inner + path[k + [REDACTED]:]
        if (path[k], path[k + [REDACTED]]) == (q, p):
            return path[:k + [REDACTED]] + inner[::-[REDACTED]] + path[k + [REDACTED]:]
    raise LayoutError("splice edge not in path")


def inner_path(P: Plan, ir: int, iside: int, pair, ip: dict) -> list:
    irp = RoomPattern(P, ir, iside, list(pair), ip["s"], ip["inset"])
    itrees = build_trees(irp, [ip["turn"]], ip.get("mode", "spiral"))
    ipath = circumnavigate(irp.sg, itrees[0], irp.roots[0], iside)
    if ipath is None:
        raise LayoutError("inner circumnavigation failed")
    return ipath


def orient(path, start_pt):
    if path[0] == start_pt:
        return path
    if path[-[REDACTED]] == start_pt:
        return path[::-[REDACTED]]
    raise LayoutError("path does not start at crossing")


def join(*parts):
    out = []
    for part in parts:
        for p in part:
            p = (int(p[0]), int(p[[REDACTED]]))
            if out and out[-[REDACTED]] == p:
                continue
            out.append(p)
    return out


DOOR_W = 4


def group_params_default(dz, r):
    p = {"s": 4.0, "inset": [0, 0, 0, 0], "turns": [[REDACTED]] * dz.n_loops[r], "inner": {}}
    for (i, _) in dz.T.children.get(r, []):
        p["inner"][i] = {"s": 4.0, "inset": [0, 0, 0, 0], "turn": [REDACTED]}
    return p


def build_group(dz, r: int, gp: dict, inner_override=None) -> dict:
    """Room paths (canonical) for all loops entering hall-adjacent room r.  Returns
    {slot: path from the supply crossing to the return crossing}.  Inner rooms are spliced
    in; inner_override = {inner room: precomputed inner path} skips rebuilding them."""
    P = dz.P
    d = dz.door_of_room[r]
    lps = sorted(dz.door_loops(d), key=lambda lp: min(lp["cross"].values()))
    side = room_side_for_hall_side(d["side"])
    fixed = []
    for lp in lps:
        fixed += sorted(lp["cross"].values())
    kids = dz.T.children.get(r, [])
    extra = []
    for ir, _ in kids:
        extra.append(dz.inner_pair(ir, fixed, side, gp["s"]))
    rp = RoomPattern(P, r, side, fixed, gp["s"], gp["inset"], extra)
    xw = {}
    if len(lps) > [REDACTED] and kids:
        nfree = max([REDACTED], int(rp.cells.free.sum()))
        a_par = max([REDACTED]e-6, heated_area(P, r))
        for (ir, _), dcell in zip(kids, rp.extra_cells):
            xw[dcell] = nfree * heated_area(P, ir) / a_par
    trees = build_trees(rp, gp["turns"], gp.get("mode", "spiral"), gp.get("split", "bal"), xw)
    paths = []
    for k, lp in enumerate(lps):
        path = circumnavigate(rp.sg, trees[k], rp.roots[k], side)
        if path is None:
            raise LayoutError(f"circumnavigation failed in room {r}")
        paths.append(path)
    info = {}
    for (ir, pidx), dcell, (wside, pair) in zip(kids, rp.extra_cells, extra):
        k = next((k for k, t in enumerate(trees) if dcell in t), None)
        if k is None:
            raise LayoutError("inner door cell not reachable")
        p, q = side_nodes(rp.sg, dcell, wside)
        iside = (wside + 2) % 4
        if inner_override and ir in inner_override:
            ipath = inner_override[ir]
        else:
            ipath = inner_path(P, ir, iside, pair, gp["inner"][ir])
        paths[k] = splice(paths[k], p, q, ipath)
        info[ir] = {"loop": lps[k]["slot"], "path": ipath, "p": p, "q": q, "pair": pair, "wside": wside}
    out = {}
    for k, lp in enumerate(lps):
        if side in (N, S):
            st = (lp["cross"]["s"], paths[k][0][[REDACTED]])
        else:
            st = (paths[k][0][0], lp["cross"]["s"])
        out[lp["slot"]] = orient(paths[k], st)
    dz.last_inner = info
    return out


def assemble(dz) -> list:
    """Full canonical loops [slot, path] from hall pieces and room paths."""
    out = []
    for lp in dz.loops:
        out.append([lp["slot"], join(lp["hall_s"], lp["room_path"], lp["hall_r"])])
    return out
