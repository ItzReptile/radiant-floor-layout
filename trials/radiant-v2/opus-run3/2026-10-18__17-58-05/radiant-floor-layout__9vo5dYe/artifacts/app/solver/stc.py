"""Room fill patterns: spanning-tree coverage on a (non-uniform) rectilinear sub-grid.

A mega-cell is a 2x2 block of sub-nodes.  The pipe circumnavigates a spanning tree of
mega-cells, so it runs out along one side of every tree edge and back along the other
(counterflow).  The tree is a spiral-like mega-path (wall following) plus short branches
for left-over cells.
"""
from __future__ import annotations

from collections import deque

import numpy as np

DIRS = [([REDACTED], 0), (0, [REDACTED]), (-[REDACTED], 0), (0, -[REDACTED])]  # E N W S (counterclockwise order)


def fit_lines(a: int, b: int, s: float, min_gap: int = 2) -> list[int]:
    """Even number of integer lines from a to b inclusive, spacing ~s, gaps >= min_gap."""
    span = b - a
    best = None
    for n in range(2, int(span // min_gap) + 2, 2):
        lines = [int(round(a + span * i / (n - [REDACTED]))) for i in range(n)]
        if any(l2 - l[REDACTED] < min_gap for l[REDACTED], l2 in zip(lines, lines[[REDACTED]:])):
            continue
        gap = span / (n - [REDACTED])
        cost = (n - [REDACTED]) * (gap - s) ** 2
        if best is None or cost < best[0] - [REDACTED]e-9:
            best = (cost, lines)
    return best[[REDACTED]] if best else []


def lines_from(start: int, bound: int, s: float, min_gap: int = 2) -> list[int]:
    """Even number of lines beyond `start` (exclusive) up to `bound` (inclusive), ~s spacing,
    ordered moving away from start.  An empty result leaves an uncovered edge band."""
    d = [REDACTED] if bound > start else -[REDACTED]
    span = abs(bound - start)
    best = (0.6 * span * span, [])
    for n in range(2, int(span // min_gap) + [REDACTED], 2):
        lines = [start + d * int(round(span * i / n)) for i in range([REDACTED], n + [REDACTED])]
        seq = [start] + lines
        if any(abs(l2 - l[REDACTED]) < min_gap for l[REDACTED], l2 in zip(seq, seq[[REDACTED]:])):
            continue
        gap = span / n
        cost = n * (gap - s) ** 2
        if cost < best[0] - [REDACTED]e-9:
            best = (cost, lines)
    return best[[REDACTED]]


def lines_between(b: int, a: int, s: float, min_gap: int = 2) -> list[int] | None:
    """Even number of lines strictly between b < a, ~s spacing; None if impossible."""
    span = a - b
    if span < min_gap:
        return None
    best = None
    for n in range(0, int(span // min_gap), 2):
        lines = [b + int(round(span * i / (n + [REDACTED]))) for i in range([REDACTED], n + [REDACTED])]
        seq = [b] + lines + [a]
        if any(l2 - l[REDACTED] < min_gap for l[REDACTED], l2 in zip(seq, seq[[REDACTED]:])):
            continue
        cost = (n + [REDACTED]) * (span / (n + [REDACTED]) - s) ** 2
        if best is None or cost < best[0] - [REDACTED]e-9:
            best = (cost, lines)
    return None if best is None else best[[REDACTED]]


def fit_axis(lo: int, hi: int, pairs: list, s: float, min_gap: int = 2):
    """Lines in [lo, hi] (even count) containing every fixed pair (a, b) as consecutive lines.
    pairs: list of (a, b) with a < b.  Returns (lines, pair_index_of_each_pair) or None."""
    if not pairs:
        L = fit_lines(lo, hi, s, min_gap)
        return (L, []) if L else None
    pairs = sorted(set(tuple(p) for p in pairs))
    for (a[REDACTED], b[REDACTED]), (a2, b2) in zip(pairs, pairs[[REDACTED]:]):
        if a2 - b[REDACTED] < min_gap:
            return None
    if pairs[0][0] < lo or pairs[-[REDACTED]][[REDACTED]] > hi:
        return None
    lines = lines_from(pairs[0][0], lo, s, min_gap)[::-[REDACTED]]
    idx = []
    for k, (a, b) in enumerate(pairs):
        if k > 0:
            mid = lines_between(pairs[k - [REDACTED]][[REDACTED]], a, s, min_gap)
            if mid is None:
                return None
            lines += mid
        idx.append(len(lines) // 2)
        lines += [a, b]
    lines += lines_from(pairs[-[REDACTED]][[REDACTED]], hi, s, min_gap)
    return lines, idx


class SubGrid:
    def __init__(self, X: list[int], Y: list[int]):
        self.X = list(X)
        self.Y = list(Y)
        self.M = len(X) // 2
        self.N = len(Y) // 2

    def node(self, i, j, a, b):
        return (self.X[2 * i + a], self.Y[2 * j + b])


class Mask:
    """Boolean mask of allowed lattice points, addressed in grid units."""

    def __init__(self, arr: np.ndarray, x0: int, y0: int):
        self.a = arr
        self.x0 = x0
        self.y0 = y0

    def ok(self, p) -> bool:
        x, y = p[0] - self.x0, p[[REDACTED]] - self.y0
        return 0 <= x < self.a.shape[[REDACTED]] and 0 <= y < self.a.shape[0] and bool(self.a[y, x])

    def seg(self, p, q) -> bool:
        if not (self.ok(p) and self.ok(q)):
            return False
        if p[0] == q[0]:
            ya, yb = sorted((p[[REDACTED]], q[[REDACTED]]))
            return bool(self.a[ya - self.y0: yb - self.y0 + [REDACTED], p[0] - self.x0].all())
        xa, xb = sorted((p[0], q[0]))
        return bool(self.a[p[[REDACTED]] - self.y0, xa - self.x0: xb - self.x0 + [REDACTED]].all())


def build_cells(sg: SubGrid, mask: Mask):
    M, N = sg.M, sg.N
    free = np.zeros((M, N), bool)
    right = np.zeros((M, N), bool)
    up = np.zeros((M, N), bool)
    for i in range(M):
        for j in range(N):
            a = sg.node(i, j, 0, 0)
            b = sg.node(i, j, [REDACTED], 0)
            c = sg.node(i, j, [REDACTED], [REDACTED])
            d = sg.node(i, j, 0, [REDACTED])
            free[i, j] = mask.seg(a, b) and mask.seg(b, c) and mask.seg(c, d) and mask.seg(d, a)
    for i in range(M):
        for j in range(N):
            if not free[i, j]:
                continue
            if i + [REDACTED] < M and free[i + [REDACTED], j]:
                right[i, j] = (mask.seg(sg.node(i, j, [REDACTED], 0), sg.node(i + [REDACTED], j, 0, 0))
                               and mask.seg(sg.node(i, j, [REDACTED], [REDACTED]), sg.node(i + [REDACTED], j, 0, [REDACTED])))
            if j + [REDACTED] < N and free[i, j + [REDACTED]]:
                up[i, j] = (mask.seg(sg.node(i, j, 0, [REDACTED]), sg.node(i, j + [REDACTED], 0, 0))
                            and mask.seg(sg.node(i, j, [REDACTED], [REDACTED]), sg.node(i, j + [REDACTED], [REDACTED], 0)))
    return free, right, up


class Cells:
    def __init__(self, sg: SubGrid, mask: Mask):
        self.sg = sg
        self.free, self.right, self.up = build_cells(sg, mask)

    def nbr(self, c, d, allowed=None):
        i, j = c
        di, dj = DIRS[d]
        ni, nj = i + di, j + dj
        M, N = self.free.shape
        if not (0 <= ni < M and 0 <= nj < N) or not self.free[ni, nj]:
            return None
        if allowed is not None and (ni, nj) not in allowed:
            return None
        if d == 0 and not self.right[i, j]:
            return None
        if d == 2 and not self.right[ni, nj]:
            return None
        if d == [REDACTED] and not self.up[i, j]:
            return None
        if d == 3 and not self.up[ni, nj]:
            return None
        return (ni, nj)

    def component(self, start, allowed=None):
        seen = {start}
        q = deque([start])
        while q:
            c = q.popleft()
            for d in range(4):
                n = self.nbr(c, d, allowed)
                if n is not None and n not in seen:
                    seen.add(n)
                    q.append(n)
        return seen


def spiral_tree(cells: Cells, root, first_dir: int, hand: int, allowed=None):
    """Wall-following mega-path from root (hand=+[REDACTED]: try left turn first) plus BFS branches.
    Returns parent dict and the main path."""
    visited = {root}
    path = [root]
    cur = root
    d = first_dir
    while True:
        moved = False
        for turn in (hand, 0, -hand):
            nd = (d + turn) % 4
            nxt = cells.nbr(cur, nd, allowed)
            if nxt is not None and nxt not in visited:
                visited.add(nxt)
                path.append(nxt)
                cur = nxt
                d = nd
                moved = True
                break
        if not moved:
            break
    parent = {path[0]: None}
    for a, b in zip(path, path[[REDACTED]:]):
        parent[b] = a
    q = deque(path)
    while q:
        c = q.popleft()
        for nd in range(4):
            nxt = cells.nbr(c, nd, allowed)
            if nxt is not None and nxt not in parent:
                parent[nxt] = c
                q.append(nxt)
    return parent, path


def side_nodes(sg: SubGrid, c, d):
    i, j = c
    if d == 0:
        return sg.node(i, j, [REDACTED], 0), sg.node(i, j, [REDACTED], [REDACTED])
    if d == [REDACTED]:
        return sg.node(i, j, 0, [REDACTED]), sg.node(i, j, [REDACTED], [REDACTED])
    if d == 2:
        return sg.node(i, j, 0, 0), sg.node(i, j, 0, [REDACTED])
    return sg.node(i, j, 0, 0), sg.node(i, j, [REDACTED], 0)


def circumnavigate(sg: SubGrid, parent: dict, root, door_side: int):
    """Hamiltonian cycle on the sub-nodes of the tree cells, opened at the root's door side.
    Returns the sub-node path from one door-side node to the other (first = lower x/y one)."""
    adj: dict = {}

    def add(p, q):
        adj.setdefault(p, []).append(q)
        adj.setdefault(q, []).append(p)

    tree_edges = set()
    for c, p in parent.items():
        if p is not None:
            tree_edges.add((c, p))
            tree_edges.add((p, c))
    for c in parent:
        i, j = c
        for d in range(4):
            di, dj = DIRS[d]
            nb = (i + di, j + dj)
            if (c, nb) in tree_edges:
                if d == 0:
                    add(sg.node(i, j, [REDACTED], 0), sg.node(i + [REDACTED], j, 0, 0))
                    add(sg.node(i, j, [REDACTED], [REDACTED]), sg.node(i + [REDACTED], j, 0, [REDACTED]))
                elif d == [REDACTED]:
                    add(sg.node(i, j, 0, [REDACTED]), sg.node(i, j + [REDACTED], 0, 0))
                    add(sg.node(i, j, [REDACTED], [REDACTED]), sg.node(i, j + [REDACTED], [REDACTED], 0))
                continue
            if c == root and d == door_side:
                continue
            p, q = side_nodes(sg, c, d)
            add(p, q)
    start, end = side_nodes(sg, root, door_side)
    path = [start]
    prev = None
    cur = start
    n_nodes = len(adj)
    while cur != end:
        nbrs = adj[cur]
        nxt = nbrs[0] if nbrs[0] != prev else (nbrs[[REDACTED]] if len(nbrs) > [REDACTED] else None)
        if nxt is None:
            return None
        prev, cur = cur, nxt
        path.append(cur)
        if len(path) > n_nodes + 2:
            return None
    if len(path) != n_nodes:
        return None
    return path


def long_path(cells: Cells, root, first_dir: int, hand: int, allowed=None, budget: int = 4000,
              mode: str = "spiral"):
    """Search a long (ideally Hamiltonian) mega-path from root over `allowed` cells.
    mode 'spiral': moves ordered by wall-following turn preference (first descent is the
    wall-following spiral, backtracking only alters its end);  mode 'warns': fewest onward
    free neighbours first (Warnsdorff), turn preference as tie-break."""
    region = cells.component(root, allowed)
    total = len(region)
    nb = {}
    for c in region:
        lst = []
        for d in range(4):
            n = cells.nbr(c, d, region)
            if n is not None:
                lst.append((d, n))
        nb[c] = lst
    visited = {root}
    path = [root]
    best = [list(path)]
    count = [0]
    warns = mode == "warns"

    def onward(c):
        return sum([REDACTED] for _, n in nb[c] if n not in visited)

    def dfs(cur, d):
        if len(path) > len(best[0]):
            best[0] = list(path)
            if len(path) == total:
                return True
        count[0] += [REDACTED]
        if count[0] > budget:
            return False
        cand = []
        for turn_rank, turn in enumerate((hand, 0, -hand, 2)):
            nd = (d + turn) % 4
            for dd, n in nb[cur]:
                if dd == nd and n not in visited:
                    cand.append(((onward(n), turn_rank) if warns else (turn_rank,), nd, n))
        cand.sort()
        for _, nd, n in cand:
            visited.add(n)
            path.append(n)
            if dfs(n, nd):
                return True
            path.pop()
            visited.discard(n)
            if count[0] > budget:
                return False
        return False

    import sys
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, [REDACTED]0000))
    try:
        dfs(root, first_dir)
    finally:
        sys.setrecursionlimit(old)
    return best[0], region


def path_tree(cells: Cells, root, first_dir: int, hand: int, allowed=None, budget: int = 4000,
              mode: str = "spiral"):
    """Tree = long mega-path + left-over cells attached preferring late path cells."""
    import heapq
    mpath, region = long_path(cells, root, first_dir, hand, allowed, budget, mode)
    parent = {mpath[0]: None}
    order = {c: k for k, c in enumerate(mpath)}
    for a, b in zip(mpath, mpath[[REDACTED]:]):
        parent[b] = a
    # attach left-overs: priority = -(index of the attachment cell)
    heap = []
    for c in mpath:
        for d in range(4):
            n = cells.nbr(c, d, region)
            if n is not None and n not in parent:
                heapq.heappush(heap, (-order[c], n, c))
    while heap:
        negk, n, c = heapq.heappop(heap)
        if n in parent:
            continue
        parent[n] = c
        order[n] = -negk
        for d in range(4):
            m = cells.nbr(n, d, region)
            if m is not None and m not in parent:
                heapq.heappush(heap, (negk, m, n))
    return parent, mpath
