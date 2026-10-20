"""Candidate bend rules for radiant-floor-layout (offline prototype; the task itself is unchanged).

A "piece" is a straight run between two bends (interior segments only, like the checker's min_segment).
Rule A:   a piece shorter than 0.20 m must be a U-turn end-cap: both bends turn the same way (the path comes
          back parallel), both neighbouring pieces are at least 0.40 m, and it is not in the hall.
Rule A30: Rule A with 0.30 m neighbouring pieces (the variant the reference passes 15/15 under).
Rule B:   no piece shorter than 0.20 m anywhere (min_segment 0.20).
Rule none: the task's checker unchanged.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # keep the task's tests/ free of __pycache__

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "tasks" / "radiant-floor-layout"
if "check_layout" not in sys.modules:
    sys.path.insert(0, str(TASK / "tests"))
import check_layout as ck  # noqa: E402

SHORT = 0.20
NEIGHBOUR = {"A": 0.40, "A30": 0.30}


def pieces(plan, layout, floor=None):
    """Every interior piece shorter than 0.20 m: (slot, length, kind, prev_len, next_len, room_id, x, y)."""
    g = plan["grid"]
    floor = floor or ck.Floor(plan)
    out = []
    for lp in layout["loops"]:
        pts = ck.simplify([(round(x / g), round(y / g)) for x, y in lp["path"]])
        lens = [(abs(a[0] - b[0]) + abs(a[1] - b[1])) * g for a, b in zip(pts, pts[1:])]
        for i in range(1, len(pts) - 2):
            if lens[i] >= SHORT - 1e-9:
                continue
            a, b, c, d = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
            u = ((b[0] > a[0]) - (b[0] < a[0]), (b[1] > a[1]) - (b[1] < a[1]))
            w = ((d[0] > c[0]) - (d[0] < c[0]), (d[1] > c[1]) - (d[1] < c[1]))
            kind = "uturn" if (u[0] == -w[0] and u[1] == -w[1]) else "step"
            mx, my = (b[0] + c[0]) / 2, (b[1] + c[1]) / 2
            ri = int(floor.room_of[int(round(my)) - floor.by0, int(round(mx)) - floor.bx0])
            out.append((lp["slot"], round(lens[i], 3), kind, round(lens[i - 1], 3), round(lens[i + 1], 3),
                        plan["rooms"][ri]["id"], round(mx * g, 3), round(my * g, 3)))
    return out


def violations(plan, layout, rule, floor=None):
    if rule == "none":
        return []
    v = []
    for p in pieces(plan, layout, floor):
        _, _, kind, before, after, room, _, _ = p
        if rule == "B" or room == "hall" or kind != "uturn" or min(before, after) < NEIGHBOUR[rule] - 1e-9:
            v.append(p)
    return v


def evaluate(plan, layout, rule):
    """The checker's evaluate, with the rule's violations added as geometry problems."""
    rep = ck.evaluate(plan, layout)
    if rule != "none" and not rep["geometry_problems"]:
        bad = violations(plan, layout, rule)
        if bad:
            rep["geometry_problems"] = [f"rule {rule}: loop {s} {k} piece {L} m in {r} at {x},{y} "
                                        f"(neighbours {a}, {b})" for s, L, k, a, b, r, x, y in bad[:5]]
            rep["passed"] = False
    return rep
