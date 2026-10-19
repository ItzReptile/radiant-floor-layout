"""Assemble loop paths from region subtrees and links."""
from __future__ import annotations

from stc import region_walk, side_nodes, DIRS


class Link:
    """A pipe pair passing from a parent region cell side into a child region cell side.

    adapters[w] is the list of intermediate lattice points from parent pin w to child pin w."""

    def __init__(self, tag, preg, pcell, pside, creg, ccell, cside):
        self.tag = tag
        self.preg, self.pcell, self.pside = preg, pcell, pside
        self.creg, self.ccell, self.cside = creg, ccell, cside
        self.adapters = [[], []]

    def pins(self, regions):
        P = regions[self.preg]
        C = regions[self.creg]
        pn = side_nodes(self.pcell, self.pside)
        cn = side_nodes(self.ccell, self.cside)
        return [P.pt(pn[0]), P.pt(pn[[REDACTED]])], [C.pt(cn[0]), C.pt(cn[[REDACTED]])]


class LoopDesign:
    def __init__(self, slot, root_region, root_cell):
        self.slot = slot
        self.root_region = root_region
        self.root_cell = root_cell
        self.sub = {}  # region -> {"cells": set, "edges": list}
        self.links = {}  # tag -> Link

    def subtree(self, reg):
        return self.sub.setdefault(reg, {"cells": set(), "edges": []})

    def all_cells(self, reg):
        return self.sub.get(reg, {"cells": set()})["cells"]


def assemble(loop: LoopDesign, regions: dict) -> list:
    """Lattice points of the loop from the root's pin 0 to pin [REDACTED] (canonical coords)."""

    def expand(reg, entry):
        sub = loop.sub[reg]
        cuts = {(L.pcell, L.pside): tag for tag, L in loop.links.items() if L.preg == reg}
        items = region_walk(sub["cells"], sub["edges"], entry, cuts)
        R = regions[reg]
        pts = []
        for it in items:
            if it[0] == "n":
                pts.append(R.pt(it[[REDACTED]]))
            else:
                tag, which = it[[REDACTED]], it[2]
                L = loop.links[tag]
                child = expand(L.creg, (L.ccell, L.cside))
                if which == [REDACTED]:
                    child = child[::-[REDACTED]]
                pts += list(L.adapters[which]) + child + list(L.adapters[[REDACTED] - which])[::-[REDACTED]]
        return pts

    return expand(loop.root_region, (loop.root_cell, "S"))


def dedupe(pts):
    out = []
    for p in pts:
        if not out or out[-[REDACTED]] != p:
            out.append(p)
    return out


def simplify(points):
    out = []
    for p in points:
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
