from __future__ import annotations
import math, heapq
from collections import defaultdict
import numpy as np
from contours import Cycles, line


def simplify(path):
    out=[]
    for p in path:
        if out and p==out[-1]: continue
        if len(out)>1:
            a,b=out[-2:]
            if (a[0]==b[0]==p[0] and (b[1]-a[1])*(p[1]-b[1])>0) or (a[1]==b[1]==p[1] and (b[0]-a[0])*(p[0]-b[0])>0):
                out[-1]=p;continue
        out.append(p)
    return out


def connections(d):
    out=[]
    for passage in d.passages:
        x0,y0,x1,y1=passage
        adj=[]
        for ri,r in enumerate(d.rooms):
            if x0 == x1:
                hit=(abs(r[0]-x0)<1e-6 or abs(r[2]-x0)<1e-6) and max(r[1],y0)<min(r[3],y1)-1e-6
            else:
                hit=(abs(r[1]-y0)<1e-6 or abs(r[3]-y0)<1e-6) and max(r[0],x0)<min(r[2],x1)-1e-6
            if hit:adj.append(ri)
        if len(adj)==2:out.append((adj[0],adj[1],passage))
    return out


def edge_lists(adj, passage, width=1):
    x0,y0,x1,y1=passage
    vertical=abs(x0-x1)<1e-6
    edges=defaultdict(list)
    for a,ns in adj.items():
        for b in ns:
            if a>=b:continue
            if vertical:
                if a[0] != b[0]:continue
                b=(a[0],a[1]+width)
                if a[1]<=y0+1e-6 or b[1]>=y1-1e-6:continue
            else:
                if a[1] != b[1]:continue
                b=(a[0]+width,a[1])
                if a[0]<=x0+1e-6 or b[0]>=x1-1e-6:continue
            cut=line(a,b)
            if not all(v in adj.get(u,()) for u,v in zip(cut,cut[1:])):continue
            edges[a[1] if vertical else a[0]].append((a,b))
    return edges


def join_rooms(d, first, second, passage):
    c=Cycles([first,second])
    candidates=[]
    for width in (2,1):
        e1=edge_lists(Cycles([first]).adj,passage,width)
        e2=edge_lists(Cycles([second]).adj,passage,width)
        for u in e1.keys() & e2.keys():
            for a,b in e1[u]:
                for e,f in e2[u]:
                    dist=abs(a[0]-e[0])+abs(a[1]-e[1])
                    candidates.append((2-width,dist,a,b,e,f))
    candidates.sort()
    for _,_,a,b,e,f in candidates:
        inner=line(a,e)[1:-1]+line(b,f)[1:-1]
        if any(p in c.adj or not d.is_allowed(p) for p in inner):continue
        c.merge(a,b,e,f)
        return c.paths()[0]
    raise ValueError('cannot join rooms')


def entrance(d, cycle, ri, passage, fixed_ends=None, region_bounds=None):
    h=d.rooms[d.hall];r=d.rooms[ri]
    vertical=abs(passage[0]-passage[2])<1e-6
    if vertical:
        wall=passage[0]
        side=1 if (h[0]+h[2])/2>wall else -1
        target_coord=math.floor(wall+1e-7)+1 if side>0 else math.ceil(wall-1e-7)-1
    else:
        wall=passage[1]
        side=1 if (h[1]+h[3])/2>wall else -1
        target_coord=math.floor(wall+1e-7)+1 if side>0 else math.ceil(wall-1e-7)-1
    adj=Cycles([cycle]).adj
    center=(passage[1]+passage[3])/2 if vertical else (passage[0]+passage[2])/2
    candidates=[]
    if fixed_ends is not None:
        width=abs(fixed_ends[0][0]-fixed_ends[1][0])+abs(fixed_ends[0][1]-fixed_ends[1][1])
        widths=(width,)
    else:widths=(2,1)
    for width in widths:
        edges=edge_lists(adj,passage,width)
        for u,es in edges.items():
            for a,b in es:
                if not all(r[0]-1e-7<=v[0]<=r[2]+1e-7 and r[1]-1e-7<=v[1]<=r[3]+1e-7 for v in (a,b)):continue
                ca=(target_coord,a[1]) if vertical else (a[0],target_coord)
                cb=(target_coord,b[1]) if vertical else (b[0],target_coord)
                if fixed_ends is not None and {ca,cb}!=set(fixed_ends):continue
                dist=abs(a[0]-ca[0])+abs(a[1]-ca[1])
                candidates.append((2-width,dist+0.02*abs(u+width*.5-center),width,a,b,ca,cb))
    candidates.sort()
    for _,_,width,a,b,ca,cb in candidates:
        inner=line(a,ca)[1:]+line(b,cb)[1:]
        if any(p in adj or not d.is_allowed(p) for p in inner):continue
        pos=cycle.index(a);size=len(cycle)
        direction=-1 if cycle[(pos+width)%size]==b else 1
        path=[cycle[(pos+direction*k)%size] for k in range(size-width+1)]
        assert path[-1]==b
        return {'path':line(ca,a)[:-1]+path+line(b,cb)[1:], 'ends':(ca,cb),'room':ri,'passage':passage}
    if fixed_ends is None:
        lo,hi=(passage[1],passage[3]) if vertical else (passage[0],passage[2])
        tangent_lo,tangent_hi=(r[1],r[3]) if vertical else (r[0],r[2])
        for width in (2,1):
            choices=sorted(range(math.ceil(lo+1e-7),math.floor(hi-width-1e-7)+1),key=lambda t:abs(t+width*.5-center))
            for u in choices[:5]:
                if u<=tangent_lo+1e-7 or u+width>=tangent_hi-1e-7:continue
                ca=(target_coord,u) if vertical else (u,target_coord)
                cb=(target_coord,u+width) if vertical else (u+width,target_coord)
                if not d.is_allowed(ca) or not d.is_allowed(cb):continue
                try:return entrance(d,cycle,ri,passage,(ca,cb),region_bounds)
                except ValueError:pass
    if fixed_ends is not None:
        # Start with a small doorway ribbon and splice it to the room cycle.
        ca,cb=sorted(fixed_ends)
        direction=-side
        aa=(ca[0]+direction,ca[1]) if vertical else (ca[0],ca[1]+direction)
        bb=(cb[0]+direction,cb[1]) if vertical else (cb[0],cb[1]+direction)
        stub=line(ca,cb)[:-1]+line(cb,bb)[:-1]+line(bb,aa)[:-1]+line(aa,ca)[:-1]
        if len(set(stub))==len(stub) and not any(p in adj or not d.is_allowed(p) for p in stub):
            bounds=region_bounds or [d.rooms[ri]]
            stub_set=set(stub)
            def allowed(p):
                if not d.is_allowed(p):return False
                if p in stub_set:return True
                r=d.rooms[ri]
                if not (r[0]+1e-7<p[0]<r[2]-1e-7 and r[1]+1e-7<p[1]<r[3]-1e-7):return False
                return any(q[0]-1e-7<=p[0]<=q[2]+1e-7 and q[1]-1e-7<=p[1]<=q[3]+1e-7 for q in bounds)
            c=Cycles([cycle,stub]);cut=line(ca,cb)
            c.protected={tuple(sorted((u,v))) for u,v in zip(cut,cut[1:])}
            if c.merge_all(allowed,2,False,True)==0:
                for u,v in zip(cut,cut[1:]):c.remove_edge(u,v)
                for p in cut[1:-1]:
                    if not c.adj[p]:del c.adj[p]
                path=[ca];a=ca;prev=None
                while a!=cb:
                    nxt=next(p for p in c.adj[a] if p!=prev)
                    path.append(nxt);prev,a=a,nxt
                return {'path':path,'ends':(ca,cb),'room':ri,'passage':passage}
    raise ValueError(('cannot find entrance',ri,passage))


def target_order(d, target):
    h=d.rooms[d.hall];x0,y0,x1,y1=h
    a,b=target['ends'];cx=(a[0]+b[0])/2;cy=(a[1]+b[1])/2
    # Counterclockwise from immediately left of the manifold.
    ds=[(abs(cy-y0),0),(abs(cx-x0),1),(abs(cy-y1),2),(abs(cx-x1),3)]
    side=min(ds)[1]
    if side==0:
        if cx<0:return (0,-cx)
        return (4,-cx)
    if side==1:return (1,cy)
    if side==2:return (2,cx)
    return (3,-cy)


def astar(start, goal, allowed, occupied, reserved):
    directions=((1,0),(0,1),(-1,0),(0,-1))
    # States include the current axis to prefer fewer bends.
    heap=[(abs(start[0]-goal[0])+abs(start[1]-goal[1]),0,0,start[0],start[1],1)]
    best={(start[0],start[1],1):0.}; prev={}
    count=0
    while heap:
        _,cost,_,x,y,axis=heapq.heappop(heap)
        state=(x,y,axis)
        if cost>best.get(state,1e99)+1e-8:continue
        if (x,y)==goal:
            path=[goal]
            while state in prev:
                state=prev[state];path.append((state[0],state[1]))
            path.reverse();return path
        for dx,dy in directions:
            p=(x+dx,y+dy);na=0 if dx else 1
            if p in occupied or p in reserved or p not in allowed:continue
            nc=cost+1+(0.12 if na!=axis else 0)
            ns=(p[0],p[1],na)
            if nc+1e-8>=best.get(ns,1e99):continue
            best[ns]=nc;prev[ns]=state;count+=1
            heur=abs(p[0]-goal[0])+abs(p[1]-goal[1])
            heapq.heappush(heap,(nc+heur,nc,count,p[0],p[1],na))
    return None


def route_hall(d, targets, omit=5):
    targets=sorted(targets,key=lambda t:target_order(d,t))
    slots=[k for k in range(d.plan['manifold']['ports']//2) if k!=omit]
    slots=slots[:len(targets)]
    pitch=d.plan['pipe']['port_pitch']/d.unit
    if abs(pitch-round(pitch))>1e-7:raise ValueError('port pitch')
    pitch=round(pitch)
    mask=d.room_mask(d.hall)
    allowed={(int(x+d.x0),int(y+d.y0)) for y,x in zip(*np.nonzero(mask))}
    starts=[];ends=[];parts=[]
    for slot,t in zip(slots,targets):
        a,b=t['ends'];key=target_order(d,t)
        # endpoints in the same order as their position around the hall boundary
        if key[0] in (0,4):want=(a[0]>b[0])
        elif key[0]==2:want=(a[0]<b[0])
        elif key[0]==1:want=(a[1]<b[1])
        else:want=(a[1]>b[1])
        part=t['path'] if want else t['path'][::-1]
        a,b=part[0],part[-1]
        starts.extend([(2*slot*pitch,0),((2*slot+1)*pitch,0)])
        ends.extend([a,b]);parts.append(part)
    n=len(starts)
    occupied=set();hallpaths=[]
    for i,(start,end) in enumerate(zip(starts,ends)):
        reserved=set()
        for j in range(i+1,n):
            x,y=starts[j]
            for y in range(0,n-j+1):reserved.add((x,y))
        # Unused target points must remain reachable.
        reserved.update(ends[i+1:])
        hp=None
        left_count=sum(end[0]<start[0] for start,end in zip(starts,ends))
        hmax=max(2,math.ceil(d.rooms[d.hall][3])-3)
        if end[0]<start[0]:
            preferred=round(i*hmax/max(1,left_count-1))
        else:
            preferred=round((n-i-1)*hmax/max(1,n-left_count-1))
        bend_rows=sorted(range(0,hmax+1),key=lambda y:(abs(y-preferred),y))
        for bend_y in bend_rows:
            trial=line(start,(start[0],bend_y))[:-1]+line((start[0],bend_y),(end[0],bend_y))[:-1]+line((end[0],bend_y),end)
            if len(set(trial))!=len(trial):continue
            if all(p in allowed and p not in occupied and p not in reserved for p in trial):
                hp=trial;break
        if hp is None:hp=astar(start,end,allowed,occupied,reserved)
        if hp is None:
            return None
        hallpaths.append(hp);occupied.update(hp)
    loops=[]
    for i,(slot,part,t) in enumerate(zip(slots,parts,targets)):
        path=hallpaths[2*i][:-1]+part+hallpaths[2*i+1][-2::-1]
        loops.append({'slot':slot,'path':path,'rooms':t.get('rooms',[t['room']]),'hall_paths':hallpaths[2*i:2*i+2],'room_path':part,'bounds':t.get('bounds'),'components':t.get('components'),'join':t.get('join'),'passage':t['passage']})
    return loops
