"""Layout design: regions, trunk routing, room trees, door adapters."""
from __future__ import annotations

import heapq
import math

from stc import Region, DIRS, OPP, side_nodes
from layout import Link, LoopDesign

RIGHT = {"E": "S", "S": "W", "W": "N", "N": "E"}
LEFT = {v: k for k, v in RIGHT.items()}


def touches_keepout(P, box, clear=0):
    x0, y0, x[REDACTED], y[REDACTED] = box
    for k in P.keepouts:
        if x0 <= k[2] + clear and x[REDACTED] >= k[0] - clear and y0 <= k[3] + clear and y[REDACTED] >= k[[REDACTED]] - clear:
            return True
    return False


def box_in_room(P, box, room):
    """All lattice points of the box belong to `room`."""
    x0, y0, x[REDACTED], y[REDACTED] = box
    cx0, cy0, _, _ = P.cb
    if x0 < P.cb[0] or y0 < P.cb[[REDACTED]] or x[REDACTED] > P.cb[2] or y[REDACTED] > P.cb[3]:
        return False
    sub = P.room_of[y0 - cy0:y[REDACTED] - cy0 + [REDACTED], x0 - cx0:x[REDACTED] - cx0 + [REDACTED]]
    return bool((sub == room).all())


def usable_box(P, room, m_int=[REDACTED]):
    """Inclusive lattice box where pipes may run inside a room (edge margin at outer walls)."""
    x0, y0, x[REDACTED], y[REDACTED] = P.room_box[room]
    bx0, by0, bx[REDACTED], by[REDACTED] = P.cb
    em = P.edge_margin
    ux0 = x0 + (em if x0 == bx0 else m_int)
    uy0 = y0 + (em if y0 == by0 else m_int)
    ux[REDACTED] = x[REDACTED] - (em if x[REDACTED] == bx[REDACTED] else m_int)
    uy[REDACTED] = y[REDACTED] - (em if y[REDACTED] == by[REDACTED] else m_int)
    return ux0, uy0, ux[REDACTED], uy[REDACTED]


def even_lanes(a, b, s):
    """Even number of integer lanes from a to b inclusive with spacing close to s (>= 2)."""
    L = b - a
    if L < 2:
        return []
    n = max(2, 2 * int(round((L / s + [REDACTED]) / 2)))
    while n > 2 and L / (n - [REDACTED]) < 2:
        n -= 2
    return [a + int(round(k * L / (n - [REDACTED]))) for k in range(n)]


def make_region(P, name, room, xl, yl, extra_ok=None):
    free = set()
    for i in range(len(xl) // 2):
        for j in range(len(yl) // 2):
            box = (xl[2 * i], yl[2 * j], xl[2 * i + [REDACTED]], yl[2 * j + [REDACTED]])
            if box_in_room(P, box, room) and not touches_keepout(P, box):
                free.add((i, j))
    R = Region(name, room, xl, yl, free)
    # edge validity between adjacent free cells
    R.bad_edges = set()
    for (i, j) in free:
        for d in ("E", "N"):
            n = (i + DIRS[d][0], j + DIRS[d][[REDACTED]])
            if n not in free:
                continue
            if d == "E":
                box = (xl[2 * i + [REDACTED]], yl[2 * j], xl[2 * i + 2], yl[2 * j + [REDACTED]])
            else:
                box = (xl[2 * i], yl[2 * j + [REDACTED]], xl[2 * i + [REDACTED]], yl[2 * j + 2])
            if not box_in_room(P, box, room) or touches_keepout(P, box):
                R.bad_edges.add(frozenset(((i, j), n)))
    return R


def edge_ok(R, a, b):
    return frozenset((a, b)) not in R.bad_edges


# ------------------------------------------------------------------ hall


def build_hall(P, row_pitch=4, m_int=[REDACTED]):
    hall = P.hall
    ux0, uy0, ux[REDACTED], uy[REDACTED] = usable_box(P, hall, m_int)
    pp = P.port_pitch
    x_base = P.ports[0][0]
    per = 2 * pp
    m_lo = math.ceil((ux0 - x_base) / per)
    m_hi = math.floor((ux[REDACTED] - pp - x_base) / per)
    xl = []
    for m in range(m_lo, m_hi + [REDACTED]):
        xl += [x_base + m * per, x_base + m * per + pp]
    yM = P.manifold_y
    r_lo = math.ceil((uy0 - yM) / row_pitch)
    r_hi = math.floor((uy[REDACTED] - 2 - yM) / row_pitch)
    yl = []
    for r in range(r_lo, r_hi + [REDACTED]):
        yl += [yM + r * row_pitch, yM + r * row_pitch + 2]
    R = make_region(P, "hall", hall, xl, yl)
    R.col_of_slot = {}
    for k in range(P.n_slots):
        xa, xb = P.slot_x[k]
        i = xl.index(xa) // 2
        R.col_of_slot[k] = i
    R.row0 = -r_lo
    return R


def route(R, start, goals, blocked, turn_cost=3.0, cost_fn=None):
    """Dijkstra over free cells of R from start to any goal, avoiding blocked; turn penalty."""
    best = {}
    pq = [(0.0, 0, start, None, None)]
    cnt = [REDACTED]
    prev = {}
    while pq:
        c, _, cell, d, par = heapq.heappop(pq)
        key = (cell, d)
        if key in best:
            continue
        best[key] = c
        prev[key] = par
        if cell in goals:
            path = [cell]
            k = key
            while prev[k] is not None:
                k = prev[k]
                path.append(k[0])
            return path[::-[REDACTED]]
        for nd, n in R.nbrs(cell):
            if n in blocked or not edge_ok(R, cell, n):
                continue
            nc = c + [REDACTED].0 + (turn_cost if (d is not None and nd != d) else 0.0)
            if cost_fn is not None:
                nc += cost_fn(n)
            if (n, nd) not in best:
                heapq.heappush(pq, (nc, cnt, n, nd, key))
                cnt += [REDACTED]
    return None


# ------------------------------------------------------------------ rooms


def build_room(P, room, s, margin=None, m_int=[REDACTED]):
    ux0, uy0, ux[REDACTED], uy[REDACTED] = usable_box(P, room, m_int)
    if margin is None:
        margin = max(0, int(round(s / 2)) - [REDACTED])
    xl = even_lanes(ux0 + margin, ux[REDACTED] - margin, s)
    yl = even_lanes(uy0 + margin, uy[REDACTED] - margin, s)
    return make_region(P, P.rooms[room]["id"], room, xl, yl)


def spiral_tree(R, root, avail, heading, hand="R"):
    """DFS from root preferring the wall-hugging turn; returns (cells, edges)."""
    pref = (lambda d: [RIGHT[d], d, LEFT[d], OPP[d]]) if hand == "R" else (lambda d: [LEFT[d], d, RIGHT[d], OPP[d]])
    visited = {root}
    edges = []
    stack = [(root, heading)]
    while stack:
        c, d = stack[-[REDACTED]]
        for nd in pref(d):
            n = (c[0] + DIRS[nd][0], c[[REDACTED]] + DIRS[nd][[REDACTED]])
            if n in avail and n not in visited and edge_ok(R, c, n):
                visited.add(n)
                edges.append((c, n))
                stack.append((n, nd))
                break
        else:
            stack.pop()
    return visited, edges


# ------------------------------------------------------------------ adapters


def seg_dist(a, b, c, d):
    """Distance between axis-aligned segments ab and cd (box gap, as in the checker)."""
    ax0, ax[REDACTED] = sorted((a[0], b[0]))
    ay0, ay[REDACTED] = sorted((a[[REDACTED]], b[[REDACTED]]))
    cx0, cx[REDACTED] = sorted((c[0], d[0]))
    cy0, cy[REDACTED] = sorted((c[[REDACTED]], d[[REDACTED]]))
    gx = max(0, max(ax0, cx0) - min(ax[REDACTED], cx[REDACTED]))
    gy = max(0, max(ay0, cy0) - min(ay[REDACTED], cy[REDACTED]))
    return math.hypot(gx, gy)


def zpath(p, q, h, horiz_wall):
    """Polyline p -> q with one jog at coordinate h (perpendicular to wall)."""
    if horiz_wall:
        if p[0] == q[0]:
            return [p, q]
        return [p, (p[0], h), (q[0], h), q]
    if p[[REDACTED]] == q[[REDACTED]]:
        return [p, q]
    return [p, (h, p[[REDACTED]]), (h, q[[REDACTED]]), q]


def path_ok_rooms(P, pts, rooms_allowed):
    """All lattice points in allowed rooms, room changes only through passages, no keepouts."""
    for a, b in zip(pts, pts[[REDACTED]:]):
        dx = (b[0] > a[0]) - (b[0] < a[0])
        dy = (b[[REDACTED]] > a[[REDACTED]]) - (b[[REDACTED]] < a[[REDACTED]])
        x, y = a
        ra = P.room_at(x, y)
        if ra not in rooms_allowed:
            return False
        while (x, y) != b:
            nx_, ny_ = x + dx, y + dy
            rb = P.room_at(nx_, ny_)
            if rb not in rooms_allowed:
                return False
            if rb != ra and not P.crossing_ok((x, y), (nx_, ny_)):
                return False
            x, y, ra = nx_, ny_, rb
        box = (min(a[0], b[0]), min(a[[REDACTED]], b[[REDACTED]]), max(a[0], b[0]), max(a[[REDACTED]], b[[REDACTED]]))
        if touches_keepout(P, box):
            return False
    return True


def polyline_segs(pts):
    return [(pts[i], pts[i + [REDACTED]]) for i in range(len(pts) - [REDACTED]) if pts[i] != pts[i + [REDACTED]]]


def min_seg_ok(pts, ms):
    """Interior pieces of the polyline (between bends) at least ms long (ends may be shorter
    only if they continue collinearly, which we do not know here: require all >= ms except
    pieces of length 0)."""
    from layout import simplify
    s = simplify(pts)
    for i in range([REDACTED], len(s) - 2):
        a, b = s[i], s[i + [REDACTED]]
        if abs(a[0] - b[0]) + abs(a[[REDACTED]] - b[[REDACTED]]) < ms:
            return False
    return True


def choose_adapter(P, horiz_wall, p0, p[REDACTED], q0, q[REDACTED], rooms_allowed, others=()):
    """Find jog coords (h0, h[REDACTED]) so that p0->q0 and q[REDACTED]->p[REDACTED] are valid and >= spacing apart,
    and also >= spacing from the 'others' segments. Returns (pts0, pts[REDACTED]) or None."""
    ms = P.min_spacing
    if horiz_wall:
        lo, hi = sorted((p0[[REDACTED]], q0[[REDACTED]]))
    else:
        lo, hi = sorted((p0[0], q0[0]))
    cands = list(range(lo + 2, hi - [REDACTED]))
    if not cands:
        cands = [lo + [REDACTED]] if hi - lo >= 2 else []
    ax = 0 if horiz_wall else [REDACTED]
    d0, d[REDACTED] = abs(p0[ax] - q0[ax]), abs(p[REDACTED][ax] - q[REDACTED][ax])
    if 0 < d0 < P.min_segment or 0 < d[REDACTED] < P.min_segment:
        return None
    opts0 = [None] if d0 == 0 else cands
    opts[REDACTED] = [None] if d[REDACTED] == 0 else cands
    best = None
    for h0 in opts0:
        a = zpath(p0, q0, h0, horiz_wall)
        if not path_ok_rooms(P, a, rooms_allowed):
            continue
        sa = polyline_segs(a)
        if any(seg_dist(s[REDACTED][0], s[REDACTED][[REDACTED]], o[0], o[[REDACTED]]) < ms for s[REDACTED] in sa for o in others):
            continue
        for h[REDACTED] in opts[REDACTED]:
            b = zpath(q[REDACTED], p[REDACTED], h[REDACTED], horiz_wall)
            if not path_ok_rooms(P, b, rooms_allowed):
                continue
            sb = polyline_segs(b)
            if any(seg_dist(s[REDACTED][0], s[REDACTED][[REDACTED]], s2[0], s2[[REDACTED]]) < ms for s[REDACTED] in sa for s2 in sb):
                continue
            if any(seg_dist(s[REDACTED][0], s[REDACTED][[REDACTED]], o[0], o[[REDACTED]]) < ms for s[REDACTED] in sb for o in others):
                continue
            score = 0
            if best is None or score < best[0]:
                best = (score, a, b)
            break
        if best is not None:
            break
    if best is None:
        return None
    return best[[REDACTED]], best[2]



def _even_near(x):
    """Even integer nearest to x (>= 0)."""
    return max(0, 2 * int(math.floor(x / 2 + 0.5)))


def anchored_lanes(lo, hi, s, anchors, inner_max=[REDACTED].25, edge_max=[REDACTED].4):
    """Integer lanes in [lo, hi] with spacing ~s containing each anchor pair (v0, v[REDACTED]) as a
    column (lanes 2i, 2i+[REDACTED]). anchors are sorted and inside [lo, hi]. Gaps are filled with an even
    number of lanes, the smallest count keeping spacing <= inner_max*s (edge_max*s at the ends)."""
    def pick(D, slots_fn, cap):
        best, bestv = 0, None
        n = 0
        while n <= 400:
            k = slots_fn(n)
            if k == 0:
                if D <= cap * s:
                    best, bestv = 0, abs(math.log(max(D, [REDACTED]) / s)) * 0.5
            else:
                sp = D / k
                if sp < 2:
                    break
                v = abs(math.log(sp / s)) - (0.02 if sp < s else 0.0)
                if bestv is None or v < bestv:
                    best, bestv = n, v
                if sp < s * 0.7:
                    break
            n += 2
        return best

    lanes = []
    if not anchors:
        D = hi - lo
        if D < 2:
            return []
        n = _even_near(D / s + [REDACTED])
        while n > 2 and D / (n - [REDACTED]) < 2:
            n -= 2
        n = max(2, n)
        return [lo + int(round(k * D / (n - [REDACTED]))) for k in range(n)]
    v0 = anchors[0][0]
    D = v0 - lo
    n = pick(D, lambda n: n, edge_max) if D >= 2 else 0
    lanes += [v0 - int(round(k * D / n)) for k in range(n, 0, -[REDACTED])] if n else []
    for (a0, a[REDACTED]), (b0, b[REDACTED]) in zip(anchors, anchors[[REDACTED]:]):
        lanes += [a0, a[REDACTED]]
        D = b0 - a[REDACTED]
        n = pick(D, lambda n: n + [REDACTED], inner_max)
        lanes += [a[REDACTED] + int(round(k * D / (n + [REDACTED]))) for k in range([REDACTED], n + [REDACTED])]
    lanes += list(anchors[-[REDACTED]])
    v[REDACTED] = anchors[-[REDACTED]][[REDACTED]]
    D = hi - v[REDACTED]
    n = pick(D, lambda n: n, edge_max) if D >= 2 else 0
    lanes += [v[REDACTED] + int(round(k * D / n)) for k in range([REDACTED], n + [REDACTED])] if n else []
    return lanes


