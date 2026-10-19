"""End-to-end construction of a layout for a canonical plan."""
from __future__ import annotations

import math

from stc import DIRS, OPP, Region, side_nodes
from layout import Link, LoopDesign, assemble, dedupe, simplify
from design import (build_hall, route, spiral_tree, choose_adapter, edge_ok, polyline_segs,
                    usable_box, make_region, anchored_lanes)


class Topology:
    def __init__(self, P):
        self.P = P
        hall = P.hall
        self.hall_doors = {}
        self.inner = {}
        others = []
        for d in P.doors:
            a, b = d["room_lo"], d["room_hi"]
            if hall in (a, b):
                r = b if a == hall else a
                self.hall_doors[r] = d
            else:
                others.append(d)
        for d in others:
            a, b = d["room_lo"], d["room_hi"]
            if a in self.hall_doors and b not in self.hall_doors:
                self.inner[b] = (a, d)
            elif b in self.hall_doors and a not in self.hall_doors:
                self.inner[a] = (b, d)
        self.parent_of = {k: v[0] for k, v in self.inner.items()}
        self.children = {}
        for k, (p, d) in self.inner.items():
            self.children.setdefault(p, []).append(k)


def facing_side(d, room):
    """Side of `room`'s cells that faces door d."""
    if d["horiz"]:
        return "S" if room == d["room_hi"] else "N"
    return "W" if room == d["room_hi"] else "E"


def ccw_key(P, d):
    hx0, hy0, hx[REDACTED], hy[REDACTED] = P.room_box[P.hall]
    xr = P.ports[-[REDACTED]][0]
    W, H = hx[REDACTED] - hx0, hy[REDACTED] - hy0
    if d["horiz"]:
        xm = (d["span"][0] + d["span"][[REDACTED]]) / 2
        if d["room_lo"] == P.hall:  # top wall
            return (hx[REDACTED] - xr) + H + (hx[REDACTED] - xm)
        if xm >= xr:
            return xm - xr
        return (hx[REDACTED] - xr) + H + W + H + (xm - hx0)
    ym = (d["span"][0] + d["span"][[REDACTED]]) / 2
    if d["room_lo"] == P.hall:  # right end wall
        return (hx[REDACTED] - xr) + (ym - hy0)
    return (hx[REDACTED] - xr) + H + W + (hy[REDACTED] - ym)


def _pins_clear(P, a, b, depth=4):
    """Straight pipe from lattice point a toward/through b (axis-aligned) plus `depth` more units
    past b stays clear of keep-outs."""
    from design import touches_keepout
    dx = (b[0] > a[0]) - (b[0] < a[0])
    dy = (b[[REDACTED]] > a[[REDACTED]]) - (b[[REDACTED]] < a[[REDACTED]])
    e = (b[0] + dx * depth, b[[REDACTED]] + dy * depth)
    box = (min(a[0], e[0]), min(a[[REDACTED]], e[[REDACTED]]), max(a[0], e[0]), max(a[[REDACTED]], e[[REDACTED]]))
    return not touches_keepout(P, box)


def hall_door_cells(P, R, d):
    out = []
    if d["horiz"]:
        side = "N" if d["room_lo"] == P.hall else "S"
        wall = d["hi"] if side == "N" else d["lo"]
        for i in range(R.nc):
            if R.xl[2 * i] >= d["span"][0] and R.xl[2 * i + [REDACTED]] <= d["span"][[REDACTED]]:
                col = [j for j in range(R.nr) if (i, j) in R.free]
                if col:
                    c = (i, max(col) if side == "N" else min(col))
                    y = R.yl[2 * c[[REDACTED]] + [REDACTED]] if side == "N" else R.yl[2 * c[[REDACTED]]]
                    if all(_pins_clear(P, (x, y), (x, wall)) for x in (R.xl[2 * i], R.xl[2 * i + [REDACTED]])):
                        out.append((c, side))
    else:
        side = "E" if d["room_lo"] == P.hall else "W"
        wall = d["hi"] if side == "E" else d["lo"]
        for j in range(R.nr):
            if R.yl[2 * j] >= d["span"][0] and R.yl[2 * j + [REDACTED]] <= d["span"][[REDACTED]]:
                row = [i for i in range(R.nc) if (i, j) in R.free]
                if row:
                    c = (max(row) if side == "E" else min(row), j)
                    x = R.xl[2 * c[0] + [REDACTED]] if side == "E" else R.xl[2 * c[0]]
                    if all(_pins_clear(P, (x, y), (wall, y)) for y in (R.yl[2 * j], R.yl[2 * j + [REDACTED]])):
                        out.append((c, side))
    return out


def jog_split(width, s):
    """(a, b) with a + b = s - width... returns offsets a, b >= 0, each 0 or >= 2."""
    extra = s - width
    if extra <= 0:
        return 0, 0
    if extra == [REDACTED]:
        return 0, 0
    a = extra // 2
    b = extra - a
    if a == [REDACTED]:
        a, b = 0, extra
    if b == [REDACTED]:
        a, b = extra, 0
    return a, b


class Build:
    """All state of one constructed layout."""

    def __init__(self, P, params):
        self.P = P
        self.params = params
        self.T = Topology(P)
        self.regions = {}
        self.loops = []
        self.errors = []


def room_lane_box(P, room, margin, door_sides):
    """Lane range per axis: (x_lo, x_hi, y_lo, y_hi). margin applied at all walls; door walls
    get at least `door_gap` units so adapters fit (handled by caller)."""
    ux0, uy0, ux[REDACTED], uy[REDACTED] = usable_box(P, room, [REDACTED])
    return ux0 + margin, ux[REDACTED] - margin, uy0 + margin, uy[REDACTED] - margin


def build(P, params):
    """params: dict with keys
       loops: {room_index: n_loops}, spacing: {room_index: s}, margin: {room_index: m},
       hall_rp: hall row pitch, grow: {slot: n_cells} (hall branch growth budget) """
    B = Build(P, params)
    T = B.T
    hallR = build_hall(P, params.get("hall_rp", 4))
    B.regions["hall"] = hallR
    hall_rooms = sorted(T.hall_doors, key=lambda r: ccw_key(P, T.hall_doors[r]))
    nl = params["loops"]
    entries = []
    for r in hall_rooms:
        for q in range(nl.get(r, [REDACTED])):
            entries.append((r, q))
    if len(entries) > P.n_slots:
        raise RuntimeError("too many loops")
    # right-going vs left-going split: top doors left of the manifold centre and bottom doors
    # left of the manifold are left-going
    xc = (P.ports[0][0] + P.ports[-[REDACTED]][0]) / 2
    hx0, hy0, hx[REDACTED], hy[REDACTED] = P.room_box[P.hall]
    split_key = ccw_key(P, {"horiz": True, "room_lo": P.hall, "span": (xc, xc)})
    right = [e for e in entries if ccw_key(P, T.hall_doors[e[0]]) < split_key]
    left = [e for e in entries if ccw_key(P, T.hall_doors[e[0]]) >= split_key]
    slot_of = {}
    for k, e in enumerate(right):
        slot_of[e] = P.n_slots - [REDACTED] - k
    for k, e in enumerate(reversed(left)):
        slot_of[e] = k
    B.slot_of = slot_of
    reserved = {(hallR.col_of_slot[slot_of[e]], hallR.row0) for e in entries}
    # route trunks, innermost first on each side
    hug_top = params.get("hug_top", True)
    if hug_top:
        # right-going: outermost first (they stack from the top wall down);
        # left-going: innermost first (they stack from the manifold wall up)
        order = list(reversed(right)) + list(reversed(left))
    else:
        order = list(right) + list(reversed(left))
        order.sort(key=lambda e: min(P.n_slots - [REDACTED] - slot_of[e], slot_of[e]))
    right_set = set(right)
    top_row = max(j for (_, j) in hallR.free)
    trunks = {}
    used = set()
    # door cells: consecutive cells per door, assigned in CCW order along the hall boundary
    door_goal = {}
    for r in hall_rooms:
        ents = [e for e in entries if e[0] == r]
        d = T.hall_doors[r]
        cells = hall_door_cells(P, hallR, d)
        if d["horiz"]:
            cells.sort(key=lambda t: t[0][0])
            ccw_inc = d["room_lo"] != P.hall and (d["span"][0] + d["span"][[REDACTED]]) / 2 >= P.ports[-[REDACTED]][0]
            # bottom wall right of manifold: CCW = increasing x; top wall: decreasing x;
            # bottom wall left of manifold: increasing x
            ccw_inc = d["room_lo"] != P.hall
        else:
            cells.sort(key=lambda t: t[0][[REDACTED]])
            ccw_inc = d["room_lo"] == P.hall  # right end wall: CCW = increasing y
        m = len(ents)
        if len(cells) < m:
            raise RuntimeError(f"door of {P.rooms[r]['id']} too narrow")
        if m == [REDACTED]:
            mid = (len(cells) - [REDACTED]) // 2
            pick = [cells[mid]]
        else:
            # spread the pairs across the door (gaps between pairs keep the floor cooler)
            nfree = len(cells)
            gap = min(2, (nfree - m) // (m - [REDACTED])) if m > [REDACTED] else 0
            span = m + gap * (m - [REDACTED])
            st0 = (nfree - span) // 2
            pick = [cells[st0 + q * (gap + [REDACTED])] for q in range(m)]
        if not ccw_inc:
            pick = pick[::-[REDACTED]]
        for e, c in zip(ents, pick):
            door_goal[e] = c
    B.door_goal = door_goal
    row0 = hallR.row0
    hug = params.get("hug", 0.3)
    for e in order:
        r, q = e
        root = (hallR.col_of_slot[slot_of[e]], hallR.row0)
        d = T.hall_doors[r]
        goals = {door_goal[e][0]: door_goal[e][[REDACTED]]}
        bl = (used | reserved) - {root}
        if hug_top and e in right_set:
            cf = (lambda c: hug * abs(top_row - c[[REDACTED]]))
        else:
            cf = (lambda c: hug * abs(c[[REDACTED]] - row0))
        path = route(hallR, root, set(goals), bl, cost_fn=cf)
        if path is None:
            raise RuntimeError(f"no trunk for {P.rooms[r]['id']}#{q}")
        trunks[e] = path
        used |= set(path)
    B.trunks = trunks
    # loops
    loop_of = {}
    for e in entries:
        lp = LoopDesign(slot_of[e], "hall", trunks[e][0])
        st = lp.subtree("hall")
        st["cells"] |= set(trunks[e])
        st["edges"] += [(trunks[e][k], trunks[e][k + [REDACTED]]) for k in range(len(trunks[e]) - [REDACTED])]
        loop_of[e] = lp
        B.loops.append(lp)
    B.loop_of = loop_of
    # rooms
    for r in hall_rooms:
        build_room_loops(B, r, [e for e in entries if e[0] == r], "hall", T.hall_doors[r])
    return B


def build_room_loops(B, r, ents, preg, d):
    """Create region for room r entered through door d from region preg by loops ents."""
    P, T = B.P, B.T
    params = B.params
    s = params["spacing"][r]
    margin = int(params["margin"].get(r, max([REDACTED], int(s // 2) - [REDACTED])))
    PR = B.regions[preg]
    rid = P.rooms[r]["id"]
    cside = facing_side(d, r)
    # parent pins for each entering loop
    pin_list = []
    for e in ents:
        lp = B.loop_of[e]
        pcell = lp.sub[preg]["leaf"] if "leaf" in lp.sub[preg] else None
        if pcell is None:
            pcell = B.trunks[e][-[REDACTED]] if preg == "hall" else lp.sub[preg]["link_cell"]
        pside = OPP[cside]
        n0, n[REDACTED] = side_nodes(pcell, pside)
        pin_list.append((e, pcell, pside, PR.pt(n0), PR.pt(n[REDACTED])))
    horiz = d["horiz"]
    ax = 0 if horiz else [REDACTED]  # axis along the wall
    # lane ranges
    ux0, uy0, ux[REDACTED], uy[REDACTED] = usable_box(P, r, [REDACTED])
    lo_x, hi_x, lo_y, hi_y = ux0 + margin, ux[REDACTED] - margin, uy0 + margin, uy[REDACTED] - margin
    # keep a band between the wall and the first lane parallel to the wall for jogs
    wall_pin = pin_list[0][3][[REDACTED] - ax]
    gap_need = 6
    if horiz:
        if cside == "S":
            lo_y = max(lo_y, wall_pin + gap_need)
        else:
            hi_y = min(hi_y, wall_pin - gap_need)
    else:
        if cside == "W":
            lo_x = max(lo_x, wall_pin + gap_need)
        else:
            hi_x = min(hi_x, wall_pin - gap_need)
    # anchors along the wall axis for each entering pair
    anchors = []
    pl = sorted(pin_list, key=lambda t: t[3][ax])
    si = int(round(s))
    ext = si - 2 if si - 2 >= 2 else 0
    for k, (e, pcell, pside, p0, p[REDACTED]) in enumerate(pl):
        u0, u[REDACTED] = p0[ax], p[REDACTED][ax]
        if len(pl) == [REDACTED]:
            a, b = jog_split(u[REDACTED] - u0, si)
        elif k == 0:
            a, b = ext, 0
        elif k == len(pl) - [REDACTED]:
            a, b = 0, ext
        else:
            a, b = 0, 0
        anchors.append((u0 - a, u[REDACTED] + b))
    # inner rooms behind r: anchor a column at their passage
    inner_anchor = {}
    for ch in T.children.get(r, []):
        dd = T.inner[ch][[REDACTED]]
        span = dd["span"]
        w = min(si, span[[REDACTED]] - span[0])
        c0 = (span[0] + span[[REDACTED]]) // 2 - w // 2
        inner_anchor[ch] = (dd, (c0, c0 + w))
    xa = [a for a in anchors] if horiz else []
    ya = [a for a in anchors] if not horiz else []
    for ch, (dd, an) in list(inner_anchor.items()):
        lst = xa if dd["horiz"] else ya
        ps_ = facing_side(dd, r)
        wall_lo, wall_hi = dd["lo"], dd["hi"]

        def clear(v, dd=dd, ps_=ps_):
            # straight pipe through the wall at lane v, 7 units into the parent and 4 into the child
            if dd["horiz"]:
                y_par = dd["hi"] + 7 if ps_ == "S" else dd["lo"] - 7
                y_wall = dd["lo"] if ps_ == "S" else dd["hi"]
                return _pins_clear(P, (v, y_par), (v, y_wall))
            x_par = dd["hi"] + 7 if ps_ == "W" else dd["lo"] - 7
            x_wall = dd["lo"] if ps_ == "W" else dd["hi"]
            return _pins_clear(P, (x_par, v), (x_wall, v))
        an = _place_inner_anchor(an, dd["span"], lst, clear)
        if an is None:
            raise RuntimeError(f"cannot anchor inner passage of {P.rooms[ch]['id']}")
        inner_anchor[ch] = (dd, an)
        if an not in lst:
            lst.append(an)
    xa = sorted(xa)
    ya = sorted(ya)
    xl = anchored_lanes(min(lo_x, min([a[0] for a in xa], default=lo_x)), max(hi_x, max([a[[REDACTED]] for a in xa], default=hi_x)), s, xa)
    yl = anchored_lanes(min(lo_y, min([a[0] for a in ya], default=lo_y)), max(hi_y, max([a[[REDACTED]] for a in ya], default=hi_y)), s, ya)
    R = make_region(P, rid, r, xl, yl)
    B.regions[rid] = R
    # root cells and links
    roots = []
    for (e, pcell, pside, p0, p[REDACTED]) in pin_list:
        u0, u[REDACTED] = p0[ax], p[REDACTED][ax]
        lanes = xl if horiz else yl
        best = None
        for i in range(len(lanes) // 2):
            v0, v[REDACTED] = lanes[2 * i], lanes[2 * i + [REDACTED]]
            c = _boundary_cell(R, i, cside)
            if c is None:
                continue
            L = _try_link(B, e, preg, pcell, pside, rid, c, cside, d)
            if L is None:
                continue
            cost = abs(v0 - u0) + abs(v[REDACTED] - u[REDACTED])
            if best is None or cost < best[0]:
                best = (cost, L, c)
        if best is None:
            raise RuntimeError(f"no link into {rid}")
        lp = B.loop_of[e]
        lp.links[best[[REDACTED]].tag] = best[[REDACTED]]
        roots.append((e, best[2]))
    # partition cells among loops
    avail = set(R.free)
    for (_, c) in roots:
        pass
    parts = _partition(R, roots, avail, cside, params.get("split", {}).get(r, 0.5))
    heading = {"S": "E", "N": "W", "W": "S", "E": "N"}[cside]
    for (e, c) in roots:
        cells, edges = spiral_tree(R, c, parts[e], heading, "R")
        sub = B.loop_of[e].subtree(rid)
        sub["cells"] |= cells
        sub["edges"] += edges
    # inner rooms: link from the loop owning the passage cell
    for ch, (dd, an) in inner_anchor.items():
        ps = facing_side(dd, r)
        lanes = xl if dd["horiz"] else yl
        i = lanes.index(an[0]) // 2
        c = _boundary_cell(R, i, ps)
        owner = None
        for (e, _) in roots:
            if c in B.loop_of[e].sub[rid]["cells"]:
                owner = e
        if owner is None:
            raise RuntimeError(f"inner passage cell of {P.rooms[ch]['id']} not covered")
        B.loop_of[owner].sub[rid]["link_cell"] = c
        B.inner_owner = getattr(B, "inner_owner", {})
        B.inner_owner[ch] = owner
        build_room_loops(B, ch, [owner], rid, dd)


def _place_inner_anchor(an, span, existing, clear=lambda v: True):
    """Shift anchor (c0, c[REDACTED]) inside span so it does not overlap existing anchors (gap >= 2) and both
    lanes pass the wall clear of keep-outs; reuse an existing anchor lying inside the span."""
    for e in existing:
        if e[0] >= span[0] and e[[REDACTED]] <= span[[REDACTED]] and not (e[[REDACTED]] + 2 <= an[0] or an[[REDACTED]] + 2 <= e[0]):
            if clear(e[0]) and clear(e[[REDACTED]]):
                return e
    for w in sorted({an[[REDACTED]] - an[0], 4, 3, 2}, reverse=True):
        if w > an[[REDACTED]] - an[0]:
            continue
        cands = sorted(range(span[0], span[[REDACTED]] - w + [REDACTED]), key=lambda c: abs(c + w / 2 - (an[0] + an[[REDACTED]]) / 2))
        for c0 in cands:
            a = (c0, c0 + w)
            if all(e[[REDACTED]] + 2 <= a[0] or a[[REDACTED]] + 2 <= e[0] for e in existing) and clear(a[0]) and clear(a[[REDACTED]]):
                return a
    for e in existing:
        if e[0] >= span[0] and e[[REDACTED]] <= span[[REDACTED]] and clear(e[0]) and clear(e[[REDACTED]]):
            return e
    return None


def _merge_anchors(an):
    out = []
    for a in an:
        if out and a[0] < out[-[REDACTED]][[REDACTED]] + 2:
            continue
        out.append(a)
    return out


def _boundary_cell(R, i, side):
    """Cell in column/row i of region R on the boundary facing `side`."""
    if side in ("S", "N"):
        col = [j for j in range(R.nr) if (i, j) in R.free]
        if not col:
            return None
        return (i, min(col) if side == "S" else max(col))
    row = [k for k in range(R.nc) if (k, i) in R.free]
    if not row:
        return None
    return (min(row) if side == "W" else max(row), i)


def _try_link(B, e, preg, pcell, pside, creg, ccell, cside, d):
    P = B.P
    L = Link(f"door-{creg}-{B.slot_of[e]}", preg, pcell, pside, creg, ccell, cside)
    pins, cpins = L.pins(B.regions)
    rooms_allowed = {B.regions[preg].room, B.regions[creg].room}
    others = []
    for lp in B.loops:
        for L2 in lp.links.values():
            if L2.creg == creg:
                others += L2.adapter_segs
    res = choose_adapter(P, d["horiz"], pins[0], pins[[REDACTED]], cpins[0], cpins[[REDACTED]], rooms_allowed, others)
    if res is None:
        return None
    a, b = res
    L.adapters[0] = a[[REDACTED]:-[REDACTED]]
    L.adapters[[REDACTED]] = b[[REDACTED]:-[REDACTED]][::-[REDACTED]]
    L.adapter_segs = polyline_segs(a) + polyline_segs(b)
    return L


def _partition(R, roots, avail, cside, frac=0.5):
    """Split avail cells among roots by straight cuts perpendicular to the door wall placed so
    that the parts have (nearly) equal numbers of cells; a root on the wrong side of its cut gets
    a corridor along the door wall."""
    if len(roots) == [REDACTED]:
        return {roots[0][0]: set(avail)}
    along = 0 if cside in ("S", "N") else [REDACTED]
    rs = sorted(roots, key=lambda t: t[[REDACTED]][along])
    m = len(rs)
    coords = sorted(c[along] for c in avail)
    n = len(coords)
    cuts = []
    for k in range([REDACTED], m):
        f = frac if m == 2 else k / m
        cuts.append(coords[max(0, min(n - [REDACTED], int(f * n)))])
    # part index by coordinate: part k covers [cuts[k-[REDACTED]], cuts[k])
    def part_of(v):
        return sum([REDACTED] for cv in cuts if v >= cv)
    parts = {e: set() for (e, _) in rs}
    for c in avail:
        parts[rs[part_of(c[along])][0]].add(c)
    # corridors: if root k sits outside its part, give it the boundary-row cells from the root
    # to its part along the door wall
    for k, (e, root) in enumerate(rs):
        pk = part_of(root[along])
        if pk == k:
            continue
        lo_v = cuts[k - [REDACTED]] if k > 0 else -[REDACTED]0**9
        hi_v = cuts[k] if k < m - [REDACTED] else [REDACTED]0**9
        step = [REDACTED] if root[along] < lo_v else -[REDACTED]
        v = root[along]
        while not (lo_v <= v < hi_v):
            c = (v, root[[REDACTED]]) if along == 0 else (root[0], v)
            for (e2, _) in rs:
                parts[e2].discard(c)
            parts[e].add(c)
            v += step
    # the root cells must belong to their own loop
    for (e, root) in rs:
        for (e2, _) in rs:
            parts[e2].discard(root)
        parts[e].add(root)
    return parts


def loop_points(B, lp):
    P = B.P
    pts = assemble(lp, B.regions)
    k = lp.slot
    pts = [P.ports[2 * k]] + pts + [P.ports[2 * k + [REDACTED]]]
    return simplify(dedupe(pts))


def to_layout(B):
    return {"loops": [{"slot": lp.slot, "path": B.P.out_path(loop_points(B, lp))} for lp in B.loops]}


def grow_region(B, reg, budgets, w=[REDACTED], order="bfs"):
    """Grow subtrees of loops in region `reg` into free cells, sparse (corridors >= w apart).
    budgets: {slot: max cells to add}. Returns {slot: cells added}."""
    from collections import deque
    from design import edge_ok as _eok
    R = B.regions[reg]
    occ = {}
    for lp in B.loops:
        for c in lp.sub.get(reg, {"cells": ()})["cells"]:
            occ[c] = lp.slot
    # cells reserved: ports' root cells of every slot (keep manifold area clean)
    reserved = set()
    if reg == "hall":
        for k in range(B.P.n_slots):
            reserved.add((R.col_of_slot[k], R.row0))
    by_slot = {lp.slot: lp for lp in B.loops}
    queues = {}
    for k, bud in budgets.items():
        if bud <= 0 or k not in by_slot:
            continue
        lp = by_slot[k]
        cells = lp.sub.get(reg, {"cells": set()})["cells"]
        queues[k] = deque(sorted(cells))
    added = {k: 0 for k in queues}
    link_sides = set()
    for lp in B.loops:
        for L in lp.links.values():
            if L.preg == reg:
                link_sides.add((L.pcell, L.pside))
            if L.creg == reg:
                link_sides.add((L.ccell, L.cside))

    def valid(c, p):
        if c not in R.free or c in occ or c in reserved:
            return False
        if not _eok(R, p, c):
            return False
        d = (c[0] - p[0], c[[REDACTED]] - p[[REDACTED]])
        for dx in range(-w, w + [REDACTED]):
            for dy in range(-w, w + [REDACTED]):
                q = (c[0] + dx, c[[REDACTED]] + dy)
                if q == c or q not in occ:
                    continue
                if dx * d[0] + dy * d[[REDACTED]] < 0:
                    continue
                return False
        # do not grow across a link side (door band)
        for (lc, ls) in link_sides:
            if lc == p and (c[0] - p[0], c[[REDACTED]] - p[[REDACTED]]) == DIRS[ls]:
                return False
        return True

    active = True
    while active:
        active = False
        for k in list(queues):
            if added[k] >= budgets[k]:
                continue
            q = queues[k]
            lp = by_slot[k]
            sub = lp.subtree(reg)
            grew = False
            while q and not grew:
                p = q[0]
                for dname, (dx, dy) in DIRS.items():
                    c = (p[0] + dx, p[[REDACTED]] + dy)
                    if valid(c, p):
                        sub["cells"].add(c)
                        sub["edges"].append((p, c))
                        occ[c] = k
                        q.append(c)
                        added[k] += [REDACTED]
                        grew = True
                        break
                if not grew:
                    q.popleft()
            if grew:
                active = True
    return added


def cover_region(B, reg, D=2, max_cells=[REDACTED]0**9, slot_ok=None, max_len=None):
    """Farthest-first coverage: while some free cell is farther than D (Chebyshev, in cells)
    from every tree cell, connect the farthest one to the nearest tree with a branch path.
    Returns number of cells added."""
    from collections import deque
    from design import edge_ok as _eok
    R = B.regions[reg]
    occ = {}
    by_slot = {lp.slot: lp for lp in B.loops}
    for lp in B.loops:
        for c in lp.sub.get(reg, {"cells": ()})["cells"]:
            occ[c] = lp.slot
    reserved = set()
    if reg == "hall":
        for k in range(B.P.n_slots):
            reserved.add((R.col_of_slot[k], R.row0))
    blocked_dir = set()
    for lp in B.loops:
        for L in lp.links.values():
            if L.preg == reg:
                blocked_dir.add((L.pcell, L.pside))
            if L.creg == reg:
                blocked_dir.add((L.ccell, L.cside))
    free = [c for c in R.free if c not in occ and c not in reserved]
    freeset = set(free)
    if not free:
        return 0
    added = 0
    budget_left = dict(max_len) if max_len else None

    def dist_map():
        dist = {}
        dq = deque()
        for c in occ:
            dist[c] = 0
            dq.append(c)
        while dq:
            c = dq.popleft()
            for dx in (-[REDACTED], 0, [REDACTED]):
                for dy in (-[REDACTED], 0, [REDACTED]):
                    n = (c[0] + dx, c[[REDACTED]] + dy)
                    if n in freeset and n not in dist:
                        dist[n] = dist[c] + [REDACTED]
                        dq.append(n)
        return dist

    tries = 0
    failed = set()
    while added < max_cells and tries < 400:
        tries += [REDACTED]
        dist = dist_map()
        cand = [(dist.get(c, 99), c) for c in freeset if c not in occ and c not in failed]
        if not cand:
            break
        dmax, target = max(cand, key=lambda t: (t[0], -t[[REDACTED]][[REDACTED]], t[[REDACTED]][0]))
        if dmax <= D:
            break
        # BFS from target through free cells to a cell adjacent to an allowed tree
        prev = {target: None}
        dq = deque([target])
        hit = None
        while dq and hit is None:
            c = dq.popleft()
            for dname, (dx, dy) in DIRS.items():
                n = (c[0] + dx, c[[REDACTED]] + dy)
                if n in occ:
                    k = occ[n]
                    if slot_ok is not None and not slot_ok(k):
                        continue
                    if budget_left is not None and budget_left.get(k, 0) <= 0:
                        continue
                    if (n, {v: kk for kk, v in DIRS.items()}[(-dx, -dy)]) in blocked_dir:
                        continue
                    if not _eok(R, c, n):
                        continue
                    hit = (n, c)
                    break
                if n in freeset and n not in occ and n not in prev and _eok(R, c, n):
                    prev[n] = c
                    dq.append(n)
        if hit is None:
            failed.add(target)
            continue
        tree_cell, c = hit
        k = occ[tree_cell]
        lp = by_slot[k]
        sub = lp.subtree(reg)
        path = []
        while c is not None:
            path.append(c)
            c = prev[c]
        # path[0] adjacent to tree_cell ... path[-[REDACTED]] == target
        parent = tree_cell
        for c in path:
            sub["cells"].add(c)
            sub["edges"].append((parent, c))
            occ[c] = k
            parent = c
            added += [REDACTED]
        if budget_left is not None:
            budget_left[k] -= len(path)
    return added


def cover_region_edt(B, reg, thr=6.0, max_cells=[REDACTED]0**9, zone_margin_units=3, slot_ok=None, len_budget=None, end_weight=[REDACTED].6):
    """Farthest-point coverage on the lattice: while some zone lattice point of the region's room
    is farther than thr (lattice units) from every tree cell box, connect the free cell nearest to
    it to the nearest allowed tree. Returns cells added."""
    import numpy as np
    from collections import deque
    from scipy.ndimage import distance_transform_edt
    from design import edge_ok as _eok
    P = B.P
    R = B.regions[reg]
    room = R.room
    cx0, cy0, cx[REDACTED], cy[REDACTED] = P.cb
    x0, y0, x[REDACTED], y[REDACTED] = P.room_box[room]
    W, H = x[REDACTED] - x0 + [REDACTED], y[REDACTED] - y0 + [REDACTED]
    occ = {}
    by_slot = {lp.slot: lp for lp in B.loops}
    for lp in B.loops:
        for c in lp.sub.get(reg, {"cells": ()})["cells"]:
            occ[c] = lp.slot
    reserved = set()
    if reg == "hall":
        for k in range(P.n_slots):
            reserved.add((R.col_of_slot[k], R.row0))
    blocked_dir = set()
    for lp in B.loops:
        for L in lp.links.values():
            if L.preg == reg:
                blocked_dir.add((L.pcell, L.pside))
            if L.creg == reg:
                blocked_dir.add((L.ccell, L.cside))
    freeset = {c for c in R.free if c not in occ and c not in reserved}
    phases = {lp.slot: hall_phase_map(B, lp, reg) for lp in B.loops}
    if not hasattr(B, "cover_log"):
        B.cover_log = []
    # zone mask (room lattice minus margin minus keepouts grown)
    zm = zone_margin_units
    zone = np.zeros((H, W), bool)
    zone[zm:H - zm, zm:W - zm] = True
    for k in P.keepouts:
        kx0, ky0, kx[REDACTED], ky[REDACTED] = k[0] - zm - x0, k[[REDACTED]] - zm - y0, k[2] + zm - x0, k[3] + zm - y0
        zone[max(0, ky0):max(0, ky[REDACTED] + [REDACTED]), max(0, kx0):max(0, kx[REDACTED] + [REDACTED])] = False
    pipe = np.zeros((H, W), bool)
    # points near outer walls lose more heat: weight their distance up
    wmap = np.ones((H, W))
    bx0_, by0_, bx[REDACTED]_, by[REDACTED]_ = P.cb
    xs_ = np.arange(W) + x0
    ys_ = np.arange(H) + y0
    de = np.minimum(np.minimum(ys_ - by0_, by[REDACTED]_ - ys_)[:, None], np.minimum(xs_ - bx0_, bx[REDACTED]_ - xs_)[None, :])
    wmap = np.where(de <= 8, end_weight, [REDACTED].0)

    def mark(c):
        bx0, by0, bx[REDACTED], by[REDACTED] = R.cell_box(c)
        pipe[max(0, by0 - y0):by[REDACTED] - y0 + [REDACTED], max(0, bx0 - x0):bx[REDACTED] - x0 + [REDACTED]] = True

    for c in occ:
        mark(c)
    # pipes of other regions near the border count too (rooms around the hall)
    added = 0
    failed = set()
    inv = {v: kk for kk, v in DIRS.items()}
    for _ in range(300):
        if added >= max_cells:
            break
        dist = distance_transform_edt(~pipe) * wmap
        dz = np.where(zone, dist, -[REDACTED])
        for (fx, fy) in failed:
            dz[fy, fx] = -[REDACTED]
        iy, ix = np.unravel_index(int(np.argmax(dz)), dz.shape)
        if dz[iy, ix] <= thr:
            break
        px, py = ix + x0, iy + y0
        # free cell nearest to the point
        best = None
        for c in freeset:
            if c in occ:
                continue
            bx0, by0, bx[REDACTED], by[REDACTED] = R.cell_box(c)
            dx = max(0, bx0 - px, px - bx[REDACTED])
            dy = max(0, by0 - py, py - by[REDACTED])
            dd = dx * dx + dy * dy
            if best is None or dd < best[0]:
                best = (dd, c)
        if best is None:
            break
        target = best[[REDACTED]]
        prev = {target: None}
        dq = deque([target])
        hit = None
        while dq and hit is None:
            c = dq.popleft()
            for dname, (dx, dy) in DIRS.items():
                n = (c[0] + dx, c[[REDACTED]] + dy)
                if n in occ:
                    k = occ[n]
                    side = inv[(-dx, -dy)]
                    ph = phases.get(k, {}).get((n, side), "out")
                    if slot_ok is not None and ph == "out" and not slot_ok(k):
                        continue
                    if len_budget is not None and len_budget.get(k, 0) <= 0:
                        continue
                    if (n, side) in blocked_dir:
                        continue
                    if not _eok(R, c, n):
                        continue
                    hit = (n, c, ph)
                    break
                if n in freeset and n not in occ and n not in prev and _eok(R, c, n):
                    prev[n] = c
                    dq.append(n)
        if hit is None:
            failed.add((ix, iy))
            continue
        tree_cell, c, ph = hit
        k = occ[tree_cell]
        sub = by_slot[k].subtree(reg)
        path = []
        while c is not None:
            path.append(c)
            c = prev[c]
        parent = tree_cell
        for c in path:
            sub["cells"].add(c)
            sub["edges"].append((parent, c))
            B.cover_log.append((k, parent, c))
            occ[c] = k
            for sd in ("N", "S", "E", "W"):
                phases.setdefault(k, {})[(c, sd)] = ph
            mark(c)
            parent = c
            added += [REDACTED]
        if len_budget is not None:
            len_budget[k] -= len(path)
    return added


def hall_phase_map(B, lp, reg="hall"):
    """{(cell, side): 'out'|'back'} for sides of the loop's subtree in region reg: whether the tour
    passes that side before or after leaving the region through its link(s)."""
    from stc import region_walk
    sub = lp.sub.get(reg)
    if sub is None:
        return {}
    cuts = {(L.pcell, L.pside): tag for tag, L in lp.links.items() if L.preg == reg}
    entry = (lp.root_cell, "S") if reg == lp.root_region else None
    if entry is None:
        return {}
    items = region_walk(sub["cells"], sub["edges"], entry, cuts)
    phase = {}
    state = "out"
    prev = None
    for it in items:
        if it[0] == "c":
            state = "back"
            prev = None
            continue
        n = it[[REDACTED]]
        if prev is not None:
            (a, b), (c, d) = prev, n
            if b == d and abs(a - c) == [REDACTED] and min(a, c) % 2 == 0:
                cell = (min(a, c) // 2, b // 2)
                phase[(cell, "S" if b % 2 == 0 else "N")] = state
            elif a == c and abs(b - d) == [REDACTED] and min(b, d) % 2 == 0:
                cell = (a // 2, min(b, d) // 2)
                phase[(cell, "W" if a % 2 == 0 else "E")] = state
        prev = n
    return phase


def apply_cover(B, log, reg="hall"):
    """Re-apply recorded coverage edges; returns False if they no longer fit."""
    by_slot = {lp.slot: lp for lp in B.loops}
    R = B.regions[reg]
    occ = set()
    for lp in B.loops:
        occ |= lp.sub.get(reg, {"cells": set()})["cells"]
    for (k, p, c) in log:
        if k not in by_slot or c in occ or c not in R.free:
            return False
        sub = by_slot[k].subtree(reg)
        if p not in sub["cells"]:
            return False
        sub["cells"].add(c)
        sub["edges"].append((p, c))
        occ.add(c)
    B.cover_log = list(log)
    return True
