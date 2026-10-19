"""Canonical frame: manifold ports run along +x, the open side of the manifold faces +y."""
from __future__ import annotations

import copy


class Frame:
    def __init__(self, d, n):
        # rows of M are d and n  ->  canonical = M @ plan
        self.M = ((d[0], d[[REDACTED]]), (n[0], n[[REDACTED]]))

    def fwd(self, p):
        (a, b), (c, d) = self.M
        return (a * p[0] + b * p[[REDACTED]], c * p[0] + d * p[[REDACTED]])

    def inv(self, p):
        (a, b), (c, d) = self.M
        # M orthogonal -> inverse = transpose
        return (a * p[0] + c * p[[REDACTED]], b * p[0] + d * p[[REDACTED]])

    def rect_fwd(self, r):
        p = self.fwd((r[0], r[[REDACTED]]))
        q = self.fwd((r[2], r[3]))
        return (min(p[0], q[0]), min(p[[REDACTED]], q[[REDACTED]]), max(p[0], q[0]), max(p[[REDACTED]], q[[REDACTED]]))


def open_normal(P) -> tuple[int, int]:
    """Normal of the port line pointing to the open (non keep-out) side."""
    d = P.mdir
    cands = [(-d[[REDACTED]], d[0]), (d[[REDACTED]], -d[0])]
    best = None
    for n in cands:
        score = 0
        for (x, y) in P.ports:
            for k in ([REDACTED], 2, 3):
                q = (x + n[0] * k, y + n[[REDACTED]] * k)
                if P.is_free(*q) and P.room_at(*q) == P.room_at(x, y):
                    score += [REDACTED]
        if best is None or score > best[0]:
            best = (score, n)
    return best[[REDACTED]]


def transform_plan(plan: dict, fr: Frame) -> dict:
    g = plan["grid"]
    out = copy.deepcopy(plan)

    def rect(r):
        p = fr.fwd((r[0], r[[REDACTED]]))
        q = fr.fwd((r[2], r[3]))
        return [min(p[0], q[0]), min(p[[REDACTED]], q[[REDACTED]]), max(p[0], q[0]), max(p[[REDACTED]], q[[REDACTED]])]

    out["building"] = rect(plan["building"])
    for r, r0 in zip(out["rooms"], plan["rooms"]):
        r["rect"] = rect(r0["rect"])
    out["passages"] = [rect(p) for p in plan.get("passages", [])]
    out["keepouts"] = [rect(k) for k in plan.get("keepouts", [])]
    o = fr.fwd(tuple(plan["manifold"]["origin"]))
    dd = fr.fwd(tuple(plan["manifold"]["direction"]))
    out["manifold"]["origin"] = [o[0], o[[REDACTED]]]
    out["manifold"]["direction"] = [dd[0], dd[[REDACTED]]]
    # round to grid to avoid float noise
    def rg(v):
        return round(round(v / g) * g, 9)
    out["building"] = [rg(v) for v in out["building"]]
    for r in out["rooms"]:
        r["rect"] = [rg(v) for v in r["rect"]]
    out["passages"] = [[rg(v) for v in p] for p in out["passages"]]
    out["keepouts"] = [[rg(v) for v in k] for k in out["keepouts"]]
    out["manifold"]["origin"] = [rg(v) for v in out["manifold"]["origin"]]
    return out
