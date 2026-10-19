"""Fill free floor area with detours (bumps) branching off existing straight pipe segments."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from stc import Cells, Mask, SubGrid, circumnavigate, fit_lines, lines_from, spiral_tree
from geom import lattice_walk, simplify

E, N, W, S = 0, [REDACTED], 2, 3


def region_mask(P, bounds) -> np.ndarray:
    xlo, ylo, xhi, yhi = bounds
    a = np.zeros((P.ny, P.nx), bool)
    a[ylo - P.by0: yhi - P.by0 + [REDACTED], xlo - P.bx0: xhi - P.bx0 + [REDACTED]] = True
    return a & P.free


def try_bump(P, free: np.ndarray, seg_a, seg_b, normal: int, s: float, offset: int, w: int,
             max_cells=None, comp_cache=None):
    """Attempt a bump on segment seg_a->seg_b (axis aligned) toward side `normal` (E/N/W/S).
    offset: distance along the segment from seg_a to the first stem.  Returns
    (n_cells, insert_points) or None.  insert_points go between seg_a and seg_b."""
    ax, ay = seg_a
    bx, by = seg_b
    horiz = ay == by
    L = abs(bx - ax) + abs(by - ay)
    if offset < 2 or offset + w > L - 2:
        return None
    t = ((bx > ax) - (bx < ax), (by > ay) - (by < ay))
    p[REDACTED] = (ax + t[0] * offset, ay + t[[REDACTED]] * offset)
    p2 = (ax + t[0] * (offset + w), ay + t[[REDACTED]] * (offset + w))
    nv = {E: ([REDACTED], 0), N: (0, [REDACTED]), W: (-[REDACTED], 0), S: (0, -[REDACTED])}[normal]
    if horiz and nv[0] != 0 or (not horiz) and nv[[REDACTED]] != 0:
        return None
    r[REDACTED] = (p[REDACTED][0] + 2 * nv[0], p[REDACTED][[REDACTED]] + 2 * nv[[REDACTED]])
    r2 = (p2[0] + 2 * nv[0], p2[[REDACTED]] + 2 * nv[[REDACTED]])
    fy0, fx0 = P.by0, P.bx0

    def ok(p):
        x, y = p[0] - fx0, p[[REDACTED]] - fy0
        return 0 <= x < free.shape[[REDACTED]] and 0 <= y < free.shape[0] and free[y, x]

    if not (ok(r[REDACTED]) and ok(r2)):
        return None
    # connected free component containing r[REDACTED]
    if comp_cache is not None:
        lab, objs = comp_cache
    else:
        lab, nlab = ndimage.label(free)
        objs = ndimage.find_objects(lab)
    lid = lab[r[REDACTED][[REDACTED]] - fy0, r[REDACTED][0] - fx0]
    if lid == 0:
        return None
    sl = objs[lid - [REDACTED]]
    ylo, yhi = sl[0].start + fy0, sl[0].stop - [REDACTED] + fy0
    xlo, xhi = sl[[REDACTED]].start + fx0, sl[[REDACTED]].stop - [REDACTED] + fx0
    comp = lab == lid
    mask = Mask(comp, fx0, fy0)
    a_, b_ = sorted((r[REDACTED], r2))
    if horiz:
        X = lines_from(a_[0], xlo, s)[::-[REDACTED]] + [a_[0], b_[0]] + lines_from(b_[0], xhi, s)
        if normal == N:
            Y = fit_lines(r[REDACTED][[REDACTED]], yhi, s)
            if not Y or Y[0] != r[REDACTED][[REDACTED]]:
                return None
        else:
            Y = fit_lines(ylo, r[REDACTED][[REDACTED]], s)
            if not Y or Y[-[REDACTED]] != r[REDACTED][[REDACTED]]:
                return None
        sg = SubGrid(X, Y)
        ir = len(lines_from(a_[0], xlo, s)) // 2
        jr = 0 if normal == N else sg.N - [REDACTED]
        door = S if normal == N else N
    else:
        Y = lines_from(a_[[REDACTED]], ylo, s)[::-[REDACTED]] + [a_[[REDACTED]], b_[[REDACTED]]] + lines_from(b_[[REDACTED]], yhi, s)
        if normal == E:
            X = fit_lines(r[REDACTED][0], xhi, s)
            if not X or X[0] != r[REDACTED][0]:
                return None
        else:
            X = fit_lines(xlo, r[REDACTED][0], s)
            if not X or X[-[REDACTED]] != r[REDACTED][0]:
                return None
        sg = SubGrid(X, Y)
        jr = len(lines_from(a_[[REDACTED]], ylo, s)) // 2
        ir = 0 if normal == E else sg.M - [REDACTED]
        door = W if normal == E else E
    if sg.M == 0 or sg.N == 0:
        return None
    cells = Cells(sg, mask)
    root = (ir, jr)
    if not (0 <= ir < sg.M and 0 <= jr < sg.N) or not cells.free[root]:
        return None
    first = (door + [REDACTED]) % 4
    parent, _ = spiral_tree(cells, root, first, -[REDACTED])
    if max_cells is not None and len(parent) > max_cells:
        parent = dict(list(parent.items())[:max_cells])
    path = circumnavigate(sg, parent, root, door)
    if path is None:
        return None
    # orient: path starts near p[REDACTED] (stem [REDACTED])
    if path[0] != r[REDACTED]:
        path = path[::-[REDACTED]]
    if path[0] != r[REDACTED] or path[-[REDACTED]] != r2:
        return None
    rects = []
    hs = int(s // 2)
    for (i, j) in parent:
        rects.append((sg.X[2 * i] - hs, sg.Y[2 * j] - hs, sg.X[2 * i + [REDACTED]] + hs, sg.Y[2 * j + [REDACTED]] + hs))
    return len(parent), [p[REDACTED]] + path + [p2], rects


def insert_bump(path: list, k: int, ins: list) -> list:
    """Insert points `ins` between path[k] and path[k+[REDACTED]]."""
    return path[:k + [REDACTED]] + ins + path[k + [REDACTED]:]


def path_len(path) -> int:
    return sum(abs(a[0] - b[0]) + abs(a[[REDACTED]] - b[[REDACTED]]) for a, b in zip(path, path[[REDACTED]:]))


def dilate_points(mask: np.ndarray, P, pts):
    for x, y in pts:
        yy, xx = y - P.by0, x - P.bx0
        mask[max(0, yy - [REDACTED]): yy + 2, max(0, xx - [REDACTED]): xx + 2] = True


PHI = [(2, 3.76), (3, 3.[REDACTED]3), (4, 2.6), (5, 2.[REDACTED]9), (6, [REDACTED].86), (8, [REDACTED].4), ([REDACTED]0, [REDACTED].[REDACTED]), ([REDACTED]2, 0.9), ([REDACTED]6, 0.68)]


def spacing_for_flux(q: float, dT: float, lo: float = 3.0, hi: float = [REDACTED]4.0) -> float:
    """Pipe spacing (grid units) giving ~q W/m2 for water dT above the air (log-log interp)."""
    import math
    if dT <= 0.5:
        return lo
    phi = q / dT
    pts = [(math.log(s), math.log(f)) for s, f in PHI]
    lf = math.log(max(phi, [REDACTED]e-6))
    if lf >= pts[0][[REDACTED]]:
        return lo
    if lf <= pts[-[REDACTED]][[REDACTED]]:
        return hi
    for (s[REDACTED], f[REDACTED]), (s2, f2) in zip(pts, pts[[REDACTED]:]):
        if f2 <= lf <= f[REDACTED]:
            t = (lf - f[REDACTED]) / (f2 - f[REDACTED])
            return max(lo, min(hi, math.exp(s[REDACTED] + t * (s2 - s[REDACTED]))))
    return hi


def fill_pieces(P, pieces: list, full_paths: list, bounds, s, budget: dict,
                prefer_return: bool = True, max_rounds: int = 60, width=None,
                supply_weight: float = 0.6, cold_dist: float = 5.0, min_cold: int = 40,
                cold_mask=None):
    """Greedy bump filling of cold free area inside `bounds`.
    pieces: [{'loop': li, 'kind': 's'|'r', 'pts': [...]}] (modified in place).
    full_paths: current full paths of all loops (for the occupancy).
    s: spacing or function(piece) -> spacing.  Bumps are scored by the number of 'cold'
    lattice points (>= cold_dist from every pipe, inside cold_mask) their cells cover.
    budget: {li: remaining length units}.  Returns number of bumps added."""
    xlo, ylo, xhi, yhi = bounds
    s_of = s if callable(s) else (lambda pc, s=s: s)
    occ = np.zeros((P.ny, P.nx), bool)
    for path in full_paths:
        for x, y in lattice_walk(simplify(path)):
            occ[y - P.by0, x - P.bx0] = True
    blocked = ndimage.binary_dilation(occ, structure=np.ones((3, 3), bool))
    region = region_mask(P, bounds)
    target = cold_mask if cold_mask is not None else region
    added = 0
    for _ in range(max_rounds):
        free = region & ~blocked
        lab, nlab = ndimage.label(free)
        if nlab == 0:
            break
        objs = ndimage.find_objects(lab)
        cache = (lab, objs)
        dist = ndimage.distance_transform_edt(~occ)
        cold = target & (dist >= cold_dist)
        if cold.sum() < min_cold:
            break
        integ = np.zeros((P.ny + [REDACTED], P.nx + [REDACTED]), np.int64)
        integ[[REDACTED]:, [REDACTED]:] = np.cumsum(np.cumsum(cold, 0), [REDACTED])

        def count(rc):
            x0 = max(rc[0] - P.bx0, 0)
            y0 = max(rc[[REDACTED]] - P.by0, 0)
            x[REDACTED] = min(rc[2] - P.bx0 + [REDACTED], P.nx)
            y[REDACTED] = min(rc[3] - P.by0 + [REDACTED], P.ny)
            if x[REDACTED] <= x0 or y[REDACTED] <= y0:
                return 0
            return int(integ[y[REDACTED], x[REDACTED]] - integ[y0, x[REDACTED]] - integ[y[REDACTED], x0] + integ[y0, x0])
        best = None
        for pi_, pc in enumerate(pieces):
            path = pc["pts"]
            weight = pc.get("w", [REDACTED].0 if (pc["kind"] == "r" or not prefer_return) else supply_weight)
            sp = s_of(pc)
            w = max(2, int(round(sp))) if width is None else width
            for k in range(len(path) - [REDACTED]):
                a, b = path[k], path[k + [REDACTED]]
                sx0, sx[REDACTED] = sorted((a[0], b[0]))
                sy0, sy[REDACTED] = sorted((a[[REDACTED]], b[[REDACTED]]))
                if sx[REDACTED] < xlo or sx0 > xhi or sy[REDACTED] < ylo or sy0 > yhi:
                    continue
                seglen = abs(a[0] - b[0]) + abs(a[[REDACTED]] - b[[REDACTED]])
                if seglen < w + 4:
                    continue
                ta = (b[0] > a[0]) - (b[0] < a[0]), (b[[REDACTED]] > a[[REDACTED]]) - (b[[REDACTED]] < a[[REDACTED]])
                inside = [o for o in range(2, seglen - w - [REDACTED])
                          if xlo <= a[0] + ta[0] * o <= xhi and ylo <= a[[REDACTED]] + ta[[REDACTED]] * o <= yhi
                          and xlo <= a[0] + ta[0] * (o + w) <= xhi and ylo <= a[[REDACTED]] + ta[[REDACTED]] * (o + w) <= yhi]
                if not inside:
                    continue
                horiz = a[[REDACTED]] == b[[REDACTED]]
                normals = (N, S) if horiz else (E, W)
                nstep = max([REDACTED], len(inside) // 6)
                offs = sorted(set(inside[::nstep] + [inside[-[REDACTED]]]))
                for nrm in normals:
                    for off in offs:
                        res = try_bump(P, free, a, b, nrm, sp, off, w, comp_cache=cache)
                        if res is None:
                            continue
                        ncells, ins, rects = res
                        extra = path_len([a] + ins + [b]) - seglen
                        if extra > budget.get(pc["loop"], 0):
                            continue
                        cov = sum(count(rc) for rc in rects)
                        score = cov * weight
                        if cov >= min_cold and (best is None or score > best[0]):
                            best = (score, pi_, k, ins, extra)
        if best is None:
            break
        _, pi_, k, ins, extra = best
        pc = pieces[pi_]
        pc["pts"] = insert_bump(pc["pts"], k, ins)
        budget[pc["loop"]] = budget.get(pc["loop"], 0) - extra
        pts = lattice_walk(simplify([pc["pts"][k]] + ins + [pc["pts"][k + len(ins) + [REDACTED]]]))
        for x, y in pts:
            occ[y - P.by0, x - P.bx0] = True
        dilate_points(blocked, P, pts)
        added += [REDACTED]
    return added
