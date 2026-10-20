from __future__ import annotations
import math
from collections import defaultdict, deque
import numpy as np
from scipy.ndimage import binary_erosion


def norm_rect(rect, origin, d, normal, g, step):
    pts = [((x / g - origin[0]) * d[0] + (y / g - origin[1]) * d[1],
            (x / g - origin[0]) * normal[0] + (y / g - origin[1]) * normal[1])
           for x in (rect[0], rect[2]) for y in (rect[1], rect[3])]
    return (round(min(p[0] for p in pts) / step,8), round(min(p[1] for p in pts) / step,8),
            round(max(p[0] for p in pts) / step,8), round(max(p[1] for p in pts) / step,8))


class Domain:
    def __init__(self, plan):
        self.plan = plan
        self.g = plan['grid']
        self.step = round(max(plan['pipe']['min_spacing'], plan['pipe']['min_segment']) / self.g)
        self.unit = self.g * self.step
        self.origin = tuple(round(v / self.g) for v in plan['manifold']['origin'])
        self.d = tuple(plan['manifold']['direction'])
        hall = next((i for i, r in enumerate(plan['rooms']) if r['id'] == 'hall'), None)
        if hall is None:
            ox, oy = plan['manifold']['origin']
            hall = next(i for i, r in enumerate(plan['rooms']) if r['rect'][0] <= ox <= r['rect'][2] and r['rect'][1] <= oy <= r['rect'][3])
        self.hall = hall
        n = (-self.d[1], self.d[0])
        hr = plan['rooms'][hall]['rect']
        cx, cy = (hr[0] + hr[2]) / 2 / self.g, (hr[1] + hr[3]) / 2 / self.g
        if (cx - self.origin[0]) * n[0] + (cy - self.origin[1]) * n[1] < 0:
            n = (-n[0], -n[1])
        self.normal = n
        def rect(r): return norm_rect(r, self.origin, self.d, n, self.g, self.step)
        self.rooms = [rect(r['rect']) for r in plan['rooms']]
        self.building = rect(plan['building'])
        self.keepouts = [rect(r) for r in plan.get('keepouts', [])]
        self.passages = [rect(r) for r in plan.get('passages', [])]
        b = self.building
        self.x0 = math.floor(b[0]) - 1
        self.y0 = math.floor(b[1]) - 1
        self.x1 = math.ceil(b[2]) + 1
        self.y1 = math.ceil(b[3]) + 1
        self.xs = np.arange(self.x0, self.x1 + 1)
        self.ys = np.arange(self.y0, self.y1 + 1)
        x, y = self.xs[None, :], self.ys[:, None]
        m = plan['edge_margin'] / self.unit
        self.allowed = (x >= b[0] + m - 1e-7) & (x <= b[2] - m + 1e-7) & (y >= b[1] + m - 1e-7) & (y <= b[3] - m + 1e-7)
        for k in self.keepouts:
            self.allowed &= ~((x >= k[0] - 1e-7) & (x <= k[2] + 1e-7) & (y >= k[1] - 1e-7) & (y <= k[3] + 1e-7))

    def room_mask(self, ri, inset=0):
        r = self.rooms[ri]
        x, y = self.xs[None, :], self.ys[:, None]
        mask = self.allowed & (x > r[0] + inset + 1e-7) & (x < r[2] - inset - 1e-7) & (y > r[1] + inset + 1e-7) & (y < r[3] - inset - 1e-7)
        return mask

    def is_allowed(self, p):
        x, y = p
        if not self.x0 <= x <= self.x1 or not self.y0 <= y <= self.y1:
            return False
        return self.allowed[y - self.y0, x - self.x0]

    def output(self, p):
        x, y = p[0] * self.step, p[1] * self.step
        return [round((self.origin[0] + x * self.d[0] + y * self.normal[0]) * self.g, 8),
                round((self.origin[1] + x * self.d[1] + y * self.normal[1]) * self.g, 8)]


def clean_diagonals(mask):
    mask = mask.copy()
    for _ in range(8):
        a, b, c, d = mask[:-1,:-1], mask[:-1,1:], mask[1:,:-1], mask[1:,1:]
        bad1 = a & d & ~b & ~c
        bad2 = b & c & ~a & ~d
        if not bad1.any() and not bad2.any(): break
        yy, xx = np.nonzero(bad1)
        mask[yy+1,xx+1] = False
        yy, xx = np.nonzero(bad2)
        mask[yy+1,xx] = False
    return mask


def boundary_cycles(mask, x0, y0):
    mask = clean_diagonals(mask)
    pad = np.pad(mask, 1)
    succ = {}
    for side, neighbor in enumerate((pad[:-2,1:-1], pad[1:-1,2:], pad[2:,1:-1], pad[1:-1,:-2])):
        yy, xx = np.nonzero(mask & ~neighbor)
        for y, x in zip(yy.tolist(), xx.tolist()):
            x += x0; y += y0
            if side == 0: a, b = (x,y),(x+1,y)
            elif side == 1: a,b = (x+1,y),(x+1,y+1)
            elif side == 2: a,b = (x+1,y+1),(x,y+1)
            else: a,b = (x,y+1),(x,y)
            if a in succ:
                raise ValueError('diagonal boundary')
            succ[a] = b
    cycles = []
    while succ:
        start = next(iter(succ)); a = start; path = []
        while True:
            path.append(a); a = succ.pop(a)
            if a == start: break
        cycles.append(path)
    return cycles


def line(a, b):
    dx = (b[0] > a[0]) - (b[0] < a[0]); dy = (b[1] > a[1]) - (b[1] < a[1])
    out = [a]
    while a != b:
        a = a[0]+dx,a[1]+dy; out.append(a)
    return out


class Cycles:
    def __init__(self, cycles):
        self.adj = defaultdict(set)
        self.ids = {}
        self.cuts=defaultdict(list)
        self.protected=set()
        self.parent = list(range(len(cycles)))
        for i, pts in enumerate(cycles):
            for a,b in zip(pts, pts[1:]+pts[:1]):
                self.adj[a].add(b); self.adj[b].add(a); self.ids[a] = i

    def root(self,i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]; i = self.parent[i]
        return i

    def add_line(self,a,b,i):
        pts = line(a,b)
        for a,b in zip(pts,pts[1:]):
            self.adj[a].add(b); self.adj[b].add(a)
            self.ids[a] = self.ids[b] = i

    def remove_edge(self,a,b):
        self.adj[a].remove(b); self.adj[b].remove(a)

    def merge(self, a,b,c,d):
        ida,idc=self.ids[a],self.ids[c]
        self.cuts[ida].append(((a[0]+b[0])/2,(a[1]+b[1])/2))
        self.cuts[idc].append(((c[0]+d[0])/2,(c[1]+d[1])/2))
        i,j = self.root(ida), self.root(idc)
        for u,v in ((a,b),(c,d)):
            cut=line(u,v)
            for p,q in zip(cut,cut[1:]):self.remove_edge(p,q)
            for p in cut[1:-1]:
                del self.adj[p];del self.ids[p]
        self.add_line(a,c,i); self.add_line(b,d,i)
        self.parent[j] = i

    def merge_all(self, allowed, width=1, balanced=False, endpoint_allowed=False):
        remain = max(0,len({self.root(i) for i in range(len(self.parent))})-1)
        while remain:
            vertical, horizontal = defaultdict(list),defaultdict(list)
            for a,ns in self.adj.items():
                for b in ns:
                    if a < b:
                        if tuple(sorted((a,b))) in self.protected:continue
                        if endpoint_allowed and (not allowed(a) or not allowed(b)):continue
                        if a[0] == b[0]:
                            if all((a[0],a[1]+j+1) in self.adj.get((a[0],a[1]+j),()) and ((a[0],a[1]+j),(a[0],a[1]+j+1)) not in self.protected and (not endpoint_allowed or allowed((a[0],a[1]+j+1))) for j in range(width)):
                                vertical[a[1]].append(a[0])
                        else:
                            if all((a[0]+j+1,a[1]) in self.adj.get((a[0]+j,a[1]),()) and ((a[0]+j,a[1]),(a[0]+j+1,a[1])) not in self.protected and (not endpoint_allowed or allowed((a[0]+j+1,a[1]))) for j in range(width)):
                                horizontal[a[0]].append(a[1])
            candidates = []
            for axis, groups in enumerate((vertical,horizontal)):
                for coord, vs in groups.items():
                    vs.sort()
                    for u,v in zip(vs,vs[1:]):
                        if axis == 0: a,b,c,d = (u,coord),(u,coord+width),(v,coord),(v,coord+width)
                        else: a,b,c,d = (coord,u),(coord+width,u),(coord,v),(coord+width,v)
                        if self.root(self.ids[a]) == self.root(self.ids[c]): continue
                        balance=0.
                        for p,q in ((a,b),(c,d)):
                            mid=((p[0]+q[0])/2,(p[1]+q[1])/2)
                            cuts=self.cuts.get(self.ids[p],())
                            if balanced==1 and cuts:balance+=min(abs(mid[0]-z[0])+abs(mid[1]-z[1]) for z in cuts)
                        priority=v-u
                        mid=((a[0]+c[0])/2,(a[1]+b[1])/2)
                        anchor=0.
                        if balanced==2:anchor=mid[0]-mid[1]
                        elif balanced==3:anchor=-mid[0]-mid[1]
                        elif balanced==4:anchor=-mid[0]+mid[1]
                        candidates.append((priority,-balance,anchor,a,b,c,d))
            candidates.sort()
            found = False
            for dist,_,_,a,b,c,d in candidates:
                p1, p2 = line(a,c),line(b,d)
                inner = p1[1:-1]+p2[1:-1]
                if any(p in self.adj or not allowed(p) for p in inner): continue
                self.merge(a,b,c,d); remain -= 1; found = True; break
            if not found:
                if width>1:
                    width-=1;continue
                if not self.corridor_merge(allowed):break
                remain-=1
        return remain

    def corridor_merge(self,allowed):
        # A two-rail corridor can also meet a convex corner through an adjacent cell.
        if not self.adj:return False
        xs=[p[0] for p in self.adj];ys=[p[1] for p in self.adj]
        empty=set();seeds={};corners={};regular=set()
        for y in range(min(ys)-8,max(ys)+8):
            for x in range(min(xs)-8,max(xs)+8):
                cs=((x,y),(x+1,y),(x+1,y+1),(x,y+1))
                if not all(allowed(p) for p in cs):continue
                used=[p for p in cs if p in self.adj]
                if not used:empty.add((x,y));continue
                if len(used)==1:
                    corners[x,y]=self.root(self.ids[used[0]]);continue
                if len(used)!=2:continue
                a,b=used
                if b not in self.adj[a] or tuple(sorted((a,b))) in self.protected:continue
                cell=(x,y);seeds[cell]=self.root(self.ids[a]);regular.add(cell)
        repairs={}
        for cell,root in corners.items():
            x,y=cell
            choices=[p for p in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)) if p in regular and seeds[p]==root]
            if choices:seeds[cell]=root;repairs[cell]=choices
        owner=dict(seeds);prev={p:None for p in seeds};queue=deque(seeds)
        while queue:
            cell=queue.popleft();root=owner[cell];x,y=cell
            for other in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                if other in owner:
                    if owner[other]==root:continue
                    chain=[];a=cell
                    while a is not None:chain.append(a);a=prev[a]
                    chain.reverse();b=other
                    while b is not None:chain.append(b);b=prev[b]
                    for extra_a in repairs.get(chain[0],(None,)):
                        for extra_b in repairs.get(chain[-1],(None,)):
                            cells=set(chain)
                            if extra_a is not None:cells.add(extra_a)
                            if extra_b is not None:cells.add(extra_b)
                            lx=min(p[0] for p in cells);ly=min(p[1] for p in cells)
                            hx=max(p[0] for p in cells);hy=max(p[1] for p in cells)
                            mask=np.zeros((hy-ly+1,hx-lx+1),dtype=bool)
                            for u,v in cells:mask[v-ly,u-lx]=True
                            cycles=boundary_cycles(mask,lx,ly)
                            if len(cycles)!=1:continue
                            border=cycles[0];changes=defaultdict(set)
                            for a,b in zip(border,border[1:]+border[:1]):
                                changes[a].add(b);changes[b].add(a)
                            valid=True
                            for p,ns in changes.items():
                                result=self.adj.get(p,set()) ^ ns
                                if len(result)!=2 or any(tuple(sorted((p,q))) in self.protected for q in ns):valid=False;break
                            if not valid:continue
                            other_root=owner[other]
                            for p,ns in changes.items():self.adj[p]^=ns;self.ids[p]=root
                            self.parent[other_root]=root
                            return True
                    continue
                if other not in empty:continue
                owner[other]=root;prev[other]=cell;queue.append(other)
        return False

    def paths(self):
        seen = set(); paths = []
        for a in self.adj:
            if a in seen: continue
            path = [a]; last = None; b = a
            while True:
                ns = self.adj[b]
                if len(ns) != 2: raise ValueError(('degree',b,ns))
                c = next(v for v in ns if v != last)
                if c == a: break
                path.append(c); last,b = b,c
                if len(path)>len(self.adj): raise ValueError('cycle tracing')
            seen.update(path); paths.append(path)
        return paths


def room_cycles(domain, ri, pitch=4, offset=0, bounds=None, balanced=False):
    nodes = domain.room_mask(ri)
    if bounds is not None:
        x,y=domain.xs[None,:],domain.ys[:,None]
        rects=[bounds] if isinstance(bounds[0],(int,float)) else bounds
        region=np.zeros_like(nodes)
        for r in rects:region|=(x>=r[0]-1e-7)&(x<=r[2]+1e-7)&(y>=r[1]-1e-7)&(y<=r[3]+1e-7)
        nodes &= region
    mask = nodes[:-1,:-1] & nodes[1:,:-1] & nodes[:-1,1:] & nodes[1:,1:]
    if isinstance(offset,(tuple,list)):
        ox,oy=offset
        if ox:mask=binary_erosion(mask,structure=np.ones((1,3)),iterations=ox)
        if oy:mask=binary_erosion(mask,structure=np.ones((3,1)),iterations=oy)
    elif offset:mask=binary_erosion(mask,structure=np.ones((3,3)),iterations=offset)
    cycles=[];ring=0
    while mask.any():
        cycles.extend(boundary_cycles(mask,domain.x0,domain.y0))
        erosion=max(1,math.floor((ring+1)*pitch+1e-7)-math.floor(ring*pitch+1e-7));ring+=1
        mask=binary_erosion(mask,structure=np.ones((3,3)),iterations=erosion)
    c = Cycles(cycles)
    def allowed(p):
        return domain.x0<=p[0]<=domain.x1 and domain.y0<=p[1]<=domain.y1 and nodes[p[1]-domain.y0,p[0]-domain.x0]
    width=max(1,round(pitch))
    if balanced in (5,6):width+=1;balanced=balanced==6
    left = c.merge_all(allowed,width,balanced)
    return c.paths(), left
