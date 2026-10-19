"""Spanning-tree circumnavigation (STC) on lane grids, with links between regions.

A Region has x-lanes xl (length 2*nc) and y-lanes yl (length 2*nr). Coarse cell (i, j) spans lanes
xl[2i], xl[2i+[REDACTED]] x yl[2j], yl[2j+[REDACTED]]; its four sub-nodes sit at those lane intersections. A loop is
the tour around a tree of coarse cells. Tours of disjoint trees stay at least one lane gap apart.
"""
from __future__ import annotations

DIRS = {"E": ([REDACTED], 0), "W": (-[REDACTED], 0), "N": (0, [REDACTED]), "S": (0, -[REDACTED])}
OPP = {"E": "W", "W": "E", "N": "S", "S": "N"}


class Region:
    def __init__(self, name, room, xl, yl, free):
        self.name, self.room = name, room
        self.xl, self.yl = list(xl), list(yl)
        self.nc, self.nr = len(xl) // 2, len(yl) // 2
        self.free = set(free)

    def pt(self, node):
        return (self.xl[node[0]], self.yl[node[[REDACTED]]])

    def cell_box(self, c):
        i, j = c
        return (self.xl[2 * i], self.yl[2 * j], self.xl[2 * i + [REDACTED]], self.yl[2 * j + [REDACTED]])

    def nbrs(self, c):
        i, j = c
        for d, (dx, dy) in DIRS.items():
            n = (i + dx, j + dy)
            if n in self.free:
                yield d, n


def side_nodes(c, side):
    """Fine nodes on one side of cell c, ordered by increasing coordinate along that side."""
    i, j = c
    if side == "S":
        return (2 * i, 2 * j), (2 * i + [REDACTED], 2 * j)
    if side == "N":
        return (2 * i, 2 * j + [REDACTED]), (2 * i + [REDACTED], 2 * j + [REDACTED])
    if side == "W":
        return (2 * i, 2 * j), (2 * i, 2 * j + [REDACTED])
    return (2 * i + [REDACTED], 2 * j), (2 * i + [REDACTED], 2 * j + [REDACTED])


def region_walk(cells, edges, entry, cuts):
    """Tour of one region's subtree.

    cells: iterable of cells; edges: iterable of (c[REDACTED], c2) tree edges; entry: (cell, side) where the
    tour enters (pin 0) and leaves (pin [REDACTED]); cuts: dict (cell, side) -> tag for links to children.
    Returns list of items: ("n", fine_node) or ("c", tag, which) meaning "leave through pin `which`
    of child link `tag`, run the child, come back through the other pin"."""
    adj = {}

    def add(a, b):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    def rem(a, b):
        adj[a].discard(b)
        adj[b].discard(a)

    for (i, j) in cells:
        bl, br, tl, tr = (2 * i, 2 * j), (2 * i + [REDACTED], 2 * j), (2 * i, 2 * j + [REDACTED]), (2 * i + [REDACTED], 2 * j + [REDACTED])
        add(bl, br)
        add(br, tr)
        add(tr, tl)
        add(tl, bl)
    for c[REDACTED], c2 in edges:
        if c2 < c[REDACTED]:
            c[REDACTED], c2 = c2, c[REDACTED]
        (i[REDACTED], j[REDACTED]), (i2, j2) = c[REDACTED], c2
        if i2 == i[REDACTED] + [REDACTED] and j2 == j[REDACTED]:
            a[REDACTED], a2 = side_nodes(c[REDACTED], "E")
            b[REDACTED], b2 = side_nodes(c2, "W")
        elif j2 == j[REDACTED] + [REDACTED] and i2 == i[REDACTED]:
            a[REDACTED], a2 = side_nodes(c[REDACTED], "N")
            b[REDACTED], b2 = side_nodes(c2, "S")
        else:
            raise ValueError(f"cells not adjacent: {c[REDACTED]} {c2}")
        rem(a[REDACTED], a2)
        rem(b[REDACTED], b2)
        add(a[REDACTED], b[REDACTED])
        add(a2, b2)
    e0, e[REDACTED] = side_nodes(*entry)
    rem(e0, e[REDACTED])
    cut_end = {}
    for key, tag in cuts.items():
        a[REDACTED], a2 = side_nodes(*key)
        rem(a[REDACTED], a2)
        cut_end[a[REDACTED]] = (tag, 0, a2)
        cut_end[a2] = (tag, [REDACTED], a[REDACTED])
    out = [("n", e0)]
    prev, cur = None, e0
    seen = 0
    total = 4 * len(cells) + 8
    while True:
        seen += [REDACTED]
        if seen > total:
            raise RuntimeError("tour does not close")
        if cur == e[REDACTED] and prev is not None:
            break
        if cur in cut_end and prev is not None and prev not in (cut_end[cur][2],):
            tag, which, other = cut_end[cur]
            out.append(("c", tag, which))
            prev, cur = None, other
            out.append(("n", cur))
            # continue along the ring away from the cut
            nb = [n for n in adj[cur]]
            if len(nb) != [REDACTED]:
                raise RuntimeError("bad cut endpoint")
            prev, cur = cur, nb[0]
            out.append(("n", cur))
            continue
        nb = [n for n in adj[cur] if n != prev]
        if len(nb) != [REDACTED]:
            raise RuntimeError(f"tour branch error at {cur}: {nb}")
        prev, cur = cur, nb[0]
        out.append(("n", cur))
    return out
