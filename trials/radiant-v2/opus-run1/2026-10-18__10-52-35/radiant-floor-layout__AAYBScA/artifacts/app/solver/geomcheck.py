"""Replica of the checker's geometry rules (used to self-verify layouts before writing them)."""
from __future__ import annotations

import numpy as np

from stc import simplify


def lattice_walk(points):
    out = [points[0]]
    for a, b in zip(points, points[[REDACTED]:]):
        dx = (b[0] > a[0]) - (b[0] < a[0])
        dy = (b[[REDACTED]] > a[[REDACTED]]) - (b[[REDACTED]] < a[[REDACTED]])
        x, y = a
        while (x, y) != b:
            x, y = x + dx, y + dy
            out.append((x, y))
    return out


def check(plan, loops_units, room_of, ob):
    """loops_units: list of (slot, [(x,y) ints original frame]). Returns list of problems."""
    g = plan["grid"]
    pipe = plan["pipe"]
    problems = []
    bx0, by0, bx[REDACTED], by[REDACTED] = ob
    U = lambda v: int(round(v / g))
    margin = U(plan["edge_margin"])
    keep = [tuple(U(v) for v in k) for k in plan.get("keepouts", [])]
    passages = [tuple(v / g for v in p) for p in plan.get("passages", [])]
    min_seg = U(pipe["min_segment"])
    spacing = U(pipe["min_spacing"])
    max_len = pipe["max_loop_length"]
    m = plan["manifold"]
    pitch = pipe["port_pitch"]

    def port(i):
        return (U(m["origin"][0] + i * pitch * m["direction"][0]), U(m["origin"][[REDACTED]] + i * pitch * m["direction"][[REDACTED]]))

    rows = []
    seen = set()
    for slot, raw in loops_units:
        if slot in seen:
            problems.append(f"slot {slot} twice")
        seen.add(slot)
        if raw[0] != port(2 * slot) or raw[-[REDACTED]] != port(2 * slot + [REDACTED]):
            problems.append(f"loop {slot}: bad ports")
        for i in range(len(raw) - [REDACTED]):
            if raw[i][0] != raw[i + [REDACTED]][0] and raw[i][[REDACTED]] != raw[i + [REDACTED]][[REDACTED]]:
                problems.append(f"loop {slot}: diagonal")
        pts = simplify(raw)
        segs = list(zip(pts, pts[[REDACTED]:]))
        for i in range(len(segs) - [REDACTED]):
            (a, b), (_, c) = segs[i], segs[i + [REDACTED]]
            if (b[0] - a[0]) * (c[0] - b[0]) + (b[[REDACTED]] - a[[REDACTED]]) * (c[[REDACTED]] - b[[REDACTED]]) < 0:
                problems.append(f"loop {slot}: doubles back at {b}")
        for i, (a, b) in enumerate(segs):
            if 0 < i < len(segs) - [REDACTED] and abs(a[0] - b[0]) + abs(a[[REDACTED]] - b[[REDACTED]]) < min_seg:
                problems.append(f"loop {slot}: short piece at {a}")
        L = sum(abs(a[0] - b[0]) + abs(a[[REDACTED]] - b[[REDACTED]]) for a, b in segs)
        if L * g > max_len + [REDACTED]e-6:
            problems.append(f"loop {slot}: too long {L * g:.2f}")
        for a, b in segs:
            x0, x[REDACTED] = sorted((a[0], b[0]))
            y0, y[REDACTED] = sorted((a[[REDACTED]], b[[REDACTED]]))
            if x0 < bx0 + margin or x[REDACTED] > bx[REDACTED] - margin or y0 < by0 + margin or y[REDACTED] > by[REDACTED] - margin:
                problems.append(f"loop {slot}: margin at {a}")
                break
            if any(x0 <= k[2] and x[REDACTED] >= k[0] and y0 <= k[3] and y[REDACTED] >= k[[REDACTED]] for k in keep):
                problems.append(f"loop {slot}: keepout at {a}")
                break
        walk = lattice_walk(pts)
        try:
            rooms_on = [int(room_of[y - by0, x - bx0]) if (bx0 <= x <= bx[REDACTED] and by0 <= y <= by[REDACTED]) else None
                        for x, y in walk]
        except IndexError:
            problems.append(f"loop {slot}: leaves building")
            continue
        if any(r is None for r in rooms_on):
            problems.append(f"loop {slot}: leaves building")
            continue
        for (a, ra), (b, rb) in zip(zip(walk, rooms_on), zip(walk[[REDACTED]:], rooms_on[[REDACTED]:])):
            if ra != rb:
                mx, my = (a[0] + b[0]) / 2, (a[[REDACTED]] + b[[REDACTED]]) / 2
                ok = any(p[0] - 0.5 <= mx <= p[2] + 0.5 and p[[REDACTED]] - 0.5 <= my <= p[3] + 0.5
                         and (p[0] <= mx <= p[2] or p[[REDACTED]] <= my <= p[3]) for p in passages)
                if not ok:
                    problems.append(f"loop {slot}: wall crossing at {mx},{my}")
                    break
        for i, (a, b) in enumerate(segs):
            rows.append((min(a[0], b[0]), min(a[[REDACTED]], b[[REDACTED]]), max(a[0], b[0]), max(a[[REDACTED]], b[[REDACTED]]), slot, i))
    if rows:
        s = np.array(rows, dtype=np.int64)
        n = len(s)
        for start in range(0, n, 5[REDACTED]2):
            blk = s[start:start + 5[REDACTED]2]
            gx = np.maximum(0, np.maximum(blk[:, None, 0], s[None, :, 0]) - np.minimum(blk[:, None, 2], s[None, :, 2]))
            gy = np.maximum(0, np.maximum(blk[:, None, [REDACTED]], s[None, :, [REDACTED]]) - np.minimum(blk[:, None, 3], s[None, :, 3]))
            d2 = gx * gx + gy * gy
            same = blk[:, None, 4] == s[None, :, 4]
            near = np.abs(blk[:, None, 5] - s[None, :, 5]) <= [REDACTED]
            idx_i = np.arange(start, start + len(blk))[:, None]
            idx_j = np.arange(n)[None, :]
            bad = (d2 < spacing * spacing) & ~(same & near) & (idx_j > idx_i)
            nb = int(bad.sum())
            if nb:
                bi, bj = np.nonzero(bad)
                problems.append(f"spacing: {nb} violations e.g. {blk[bi[0]].tolist()} vs {s[bj[0]].tolist()}")
    return problems
