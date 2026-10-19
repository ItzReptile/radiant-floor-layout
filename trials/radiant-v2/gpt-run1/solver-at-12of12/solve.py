"""Design a heating layout: python3 solve.py PLAN.json LAYOUT.json.

Construct noncrossing circuits, solve the coupled floor/water heat balance,
and tune the routes and room coverage within a wall-clock budget.
"""
import sys, json, math, heapq, time, os
START_TIME=time.monotonic()
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, label, distance_transform_cdt, distance_transform_edt

# The construction is carried out in cells whose side is one port pitch.  A
# circuit is the outline of a connected, hole-free collection of these cells.
# Its bottom edge at the manifold is omitted, giving the two pipe endpoints.

class Designer:
    def __init__(self, plan):
        self.original = plan
        g = self.g = plan['grid']
        self.c = int(round(plan['pipe']['port_pitch']/g))
        m = plan['manifold']; self.origin = np.array(m['origin'])
        self.ux = np.array(m['direction'], dtype=int)
        hr = next((r for r in plan['rooms'] if r['id']=='hall'), None)
        if hr is None:
            hr = next(r for r in plan['rooms'] if r['rect'][0]<=m['origin'][0]<=r['rect'][2] and r['rect'][1]<=m['origin'][1]<=r['rect'][3])
        hc = np.array([(hr['rect'][0]+hr['rect'][2])/2,(hr['rect'][1]+hr['rect'][3])/2])
        self.uy = np.array([-self.ux[1],self.ux[0]])
        if np.dot(hc-self.origin,self.uy)<0: self.uy *= -1
        def point(p):
            d=np.array(p)-self.origin
            return [round(float(np.dot(d,self.ux)/g)), round(float(np.dot(d,self.uy)/g))]
        def rect(r):
            a=point(r[:2]); b=point(r[2:]); return [min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1])]
        self.p = json.loads(json.dumps(plan))
        self.p['building']=rect(plan['building'])
        self.p['passages']=[rect(r) for r in plan['passages']]
        self.p['keepouts']=[rect(r) for r in plan.get('keepouts',[])]
        for r,rr in zip(self.p['rooms'],plan['rooms']): r['rect']=rect(rr['rect'])
        self.rooms=self.p['rooms']; self.hall=plan['rooms'].index(hr)
        self.bx0,self.by0,self.bx1,self.by1=self.p['building']
        self.ix0=math.floor(self.bx0/self.c); self.iy0=math.floor(self.by0/self.c)
        self.ix1=math.ceil(self.bx1/self.c); self.iy1=math.ceil(self.by1/self.c)
        self.nx=self.ix1-self.ix0; self.ny=self.iy1-self.iy0
        self.shape=(self.ny,self.nx)
        self.xs=(np.arange(self.nx)+self.ix0+.5)*self.c
        self.ys=(np.arange(self.ny)+self.iy0+.5)*self.c
        self.roomof=np.full(self.shape,-1,int)
        for ri,r in enumerate(self.rooms):
            x0,y0,x1,y1=r['rect']
            self.roomof[(self.ys[:,None]>=y0)&(self.ys[:,None]<y1)&(self.xs[None,:]>=x0)&(self.xs[None,:]<x1)]=ri
        # Exact test of every cell edge, including the fine lattice transitions
        # that straddle room walls.
        margin=round(plan['edge_margin']/g)
        vx=(np.arange(self.nx+1)+self.ix0)*self.c
        vy=(np.arange(self.ny+1)+self.iy0)*self.c
        valid=(vx[None,:]>=self.bx0+margin)&(vx[None,:]<=self.bx1-margin)&(vy[:,None]>=self.by0+margin)&(vy[:,None]<=self.by1-margin)
        for x0,y0,x1,y1 in self.p['keepouts']:
            valid &= ~((vx[None,:]>=x0)&(vx[None,:]<=x1)&(vy[:,None]>=y0)&(vy[:,None]<=y1))
        self.allowed=valid[:-1,:-1]&valid[1:,:-1]&valid[:-1,1:]&valid[1:,1:]
        # Rectangular keepouts can cut through an edge despite valid corners.
        for x0,y0,x1,y1 in self.p['keepouts']:
            self.allowed &= ~((vx[:-1][None,:]<=x1)&(vx[1:][None,:]>=x0)&(vy[:-1,None]<=y1)&(vy[1:,None]>=y0))
        orig_rects=[tuple(round(v/g) for v in r['rect']) for r in plan['rooms']]
        orig_b=tuple(round(v/g) for v in plan['building'])
        orig_origin=np.rint(self.origin/g).astype(int)
        def rat(x,y):
            ox=orig_origin[0]+x*int(self.ux[0])+y*int(self.uy[0])
            oy=orig_origin[1]+x*int(self.ux[1])+y*int(self.uy[1])
            for ri,(x0,y0,x1,y1) in enumerate(orig_rects):
                if x0<=ox and (ox<x1 or x1==orig_b[2] and ox==x1) and y0<=oy and (oy<y1 or y1==orig_b[3] and oy==y1): return ri
            return -1
        def edgeok(x,y,dx,dy):
            a=rat(x,y)
            for t in range(self.c):
                b=rat(x+dx,y+dy)
                if a!=b:
                    mx=x+dx*.5; my=y+dy*.5
                    if not any((p[1]==p[3] and p[0]+.5<=mx<=p[2]-.5 and abs(my-p[1])<=.5) or (p[0]==p[2] and p[1]+.5<=my<=p[3]-.5 and abs(mx-p[0])<=.5) for p in self.p['passages']): return False
                a=b; x+=dx; y+=dy
            return True
        # Only edges near an internal wall need this relatively expensive test.
        for j in range(self.ny):
            for i in range(self.nx):
                if not self.allowed[j,i]: continue
                x=int(vx[i]); y=int(vy[j])
                rs={rat(x,y),rat(x+self.c,y),rat(x,y+self.c),rat(x+self.c,y+self.c)}
                if len(rs)>1 and not (edgeok(x,y,1,0) and edgeok(x,y+self.c,1,0) and edgeok(x,y,0,1) and edgeok(x+self.c,y,0,1)):
                    self.allowed[j,i]=False
        self.nslots=m['ports']//2
        self.roots=[(0-self.iy0,2*k-self.ix0) for k in range(self.nslots)]
        self.adj={i:[] for i in range(len(self.rooms))}
        self.doors={}
        for d in self.p['passages']:
            x0,y0,x1,y1=d; cx=(x0+x1)/2; cy=(y0+y1)/2
            if x0==x1: a=rat(cx-1,cy); b=rat(cx+1,cy)
            else: a=rat(cx,cy-1); b=rat(cx,cy+1)
            if a!=b and a>=0 and b>=0:
                self.adj[a].append(b);self.adj[b].append(a);self.doors[a,b]=self.doors[b,a]=d
        self.parent={self.hall:None}; queue=[self.hall]
        for a in queue:
            for b in self.adj[a]:
                if b not in self.parent:self.parent[b]=a;queue.append(b)
        self.children=[i for i in range(len(self.rooms)) if i!=self.hall and self.parent[i]!=self.hall]

    def cell(self,x,y): return (int(round(y/self.c-.5))-self.iy0,int(round(x/self.c-.5))-self.ix0)
    def empty(self):return np.zeros(self.shape,bool)
    def dilation(self,a):return binary_dilation(a,structure=np.ones((3,3),bool))

    def astar(self,start,goals,allowed,cost=None):
        # start and goals are masks; paths include the first goal cell.
        ss=np.flatnonzero(start & allowed); gg=np.flatnonzero(goals & allowed)
        if not len(ss) or not len(gg):return None
        distgo=distance_transform_cdt(~(goals&allowed),metric='taxicab').ravel()
        W=self.nx; H=self.ny; N=W*H
        al=allowed.ravel(); cc=cost.ravel() if cost is not None else None
        dst=np.full(N,np.inf); prev=np.full(N,-1,int); goal=goals.ravel()
        heap=[]
        for k in ss:
            k=int(k);dst[k]=0;heapq.heappush(heap,(float(distgo[k]),0.,k))
        found=-1
        while heap:
            _,d,k=heapq.heappop(heap)
            if d!=dst[k]:continue
            if goal[k]:found=k;break
            y,x=divmod(k,W)
            for q in ((k-1 if x else -1),(k+1 if x+1<W else -1),(k-W if y else -1),(k+W if y+1<H else -1)):
                if q<0 or not al[q]:continue
                nd=d+(cc[q] if cc is not None else 1.)
                if nd<dst[q]:
                    dst[q]=nd;prev[q]=k;heapq.heappush(heap,(nd+float(distgo[q]),nd,q))
        if found<0:return None
        out=[]
        while found>=0:
            out.append(divmod(found,W));found=int(prev[found])
        out.reverse();return out

    def room_distance(self,ri):
        x0,y0,x1,y1=self.rooms[ri]['rect']
        return np.minimum(np.minimum(self.xs[None,:]-x0,x1-self.xs[None,:]),np.minimum(self.ys[:,None]-y0,y1-self.ys[:,None]))/self.c

    def pattern(self,ri,feed,others,width=3,pitch=6,orient=0,offset=0,bbox=None,inset=0.5,root_slot=None,inset_x=None,inset_y=None,pitch_x=None,pitch_y=None,style=0,entry=0,clip=0,zone_mask=None,connection_zone=None,end_mode=0,end_point=None,attach=0,reserve_side=None):
        room=(self.roomof==ri)
        connection_cost=None
        if entry and self.parent[ri]!=self.hall:
            door=self.doors[self.parent[ri],ri]
            connection_cost=np.ones(self.shape)
            if door[1]==door[3]:
                aim=door[0]+self.c if entry<0 else door[2]-self.c
                connection_cost+=np.where(abs(self.ys[:,None]-door[1])<self.c*1.6,abs(self.xs[None,:]-aim)/self.c*4,0)
            else:
                aim=door[1]+self.c if entry<0 else door[3]-self.c
                connection_cost+=np.where(abs(self.xs[None,:]-door[0])<self.c*1.6,abs(self.ys[:,None]-aim)/self.c*4,0)
        allowed=self.allowed & ~self.dilation(others) & room
        x0,y0,x1,y1=self.rooms[ri]['rect'] if bbox is None else bbox
        allowed &= (self.xs[None,:]>=x0)&(self.xs[None,:]<=x1)&(self.ys[:,None]>=y0)&(self.ys[:,None]<=y1)
        # Keep all room outline pipes at least one cell inside internal walls.
        ix=(inset if inset_x is None else inset_x)*self.c
        iy=(inset if inset_y is None else inset_y)*self.c
        lo_x=math.ceil((x0+ix)/self.c)-self.ix0
        hi_x=math.floor((x1-ix)/self.c)-self.ix0
        lo_y=math.ceil((y0+iy)/self.c)-self.iy0
        hi_y=math.floor((y1-iy)/self.c)-self.iy0
        if reserve_side==0:lo_x=max(lo_x,math.ceil((x0+2.5*self.c)/self.c)-self.ix0)
        elif reserve_side==2:hi_x=min(hi_x,math.floor((x1-2.5*self.c)/self.c)-self.ix0)
        elif reserve_side==1:lo_y=max(lo_y,math.ceil((y0+2.5*self.c)/self.c)-self.iy0)
        elif reserve_side==3:hi_y=min(hi_y,math.floor((y1-2.5*self.c)/self.c)-self.iy0)
        aj,ai=np.where(allowed)
        if not len(ai):return None
        lo_x=max(lo_x,int(ai.min()));hi_x=min(hi_x,int(ai.max())+1)
        lo_y=max(lo_y,int(aj.min()));hi_y=min(hi_y,int(aj.max())+1)
        width=int(width);pitch=float(pitch)
        def positions(lo,hi,pitch):
            if hi-lo<width:return []
            nn=max(1,int(round((hi-lo-width)/pitch))+1)
            nn=min(nn,(hi-lo-width)//(width+1)+1)
            if nn==1:return [int(round((lo+hi-width)/2))]
            return list(dict.fromkeys(int(round(t)) for t in np.linspace(lo,hi-width,nn)))
        xx=positions(lo_x,hi_x,pitch if pitch_x is None else pitch_x);yy=positions(lo_y,hi_y,pitch if pitch_y is None else pitch_y)
        nodes={}; nd=[]; patches=[]
        def validrect(a,b,c,d):return a<c and b<d and allowed[b:d,a:c].all()
        for iy,y in enumerate(yy):
            for ix,x in enumerate(xx):
                patch=allowed[y:y+width,x:x+width]
                if not patch.all():
                    if not clip or not patch.any():continue
                    labs,nlabs=label(patch)
                    counts=np.bincount(labs.ravel());counts[0]=0
                    patch=labs==np.argmax(counts)
                nodes[ix,iy]=len(nd);nd.append((x,y,ix,iy));patches.append(patch.copy())
        if not nd:return None
        edges=[]
        for i,(x,y,ix,iy) in enumerate(nd):
            for key,di in (((ix+1,iy),0),((ix,iy+1),1)):
                if key not in nodes:continue
                j=nodes[key];X,Y,_,_=nd[j]
                a=min(x,X);b=min(y,Y);c=max(x,X)+width;d=max(y,Y)+width
                bp=allowed[b:d,a:c]
                if not bp.all():
                    if not clip:continue
                    labs,nlabs=label(bp)
                    maska=np.zeros(bp.shape,bool);maska[y-b:y-b+width,x-a:x-a+width]=patches[i]
                    maskb=np.zeros(bp.shape,bool);maskb[Y-b:Y-b+width,X-a:X-a+width]=patches[j]
                    la=set(np.unique(labs[maska]));lb=set(np.unique(labs[maskb]));both=la&lb-{0}
                    if not both:continue
                    chosenlab=max(both,key=lambda q:int(np.sum(labs==q)));bp=labs==chosenlab
                edges.append((0 if di==orient else 1,i,j,(a,b,c,d,bp.copy())))
        # Choose the reachable component with the most useful floor area.
        par=list(range(len(nd)))
        def find(i):
            while par[i]!=i:par[i]=par[par[i]];i=par[i]
            return i
        for _,i,j,_ in edges:
            a,b=find(i),find(j)
            if a!=b:par[a]=b
        comps={}
        for i in range(len(nd)):comps.setdefault(find(i),[]).append(i)
        candidates=sorted(comps.values(),key=len,reverse=True)
        chosen=None; connector=None
        # The short connection from the feed to the heating pattern is allowed
        # to enter any tile of a connected component.
        for comp in candidates[:5]:
            target=self.empty()
            for i in comp:
                x,y,_,_=nd[i];target[y:y+width,x:x+width]|=patches[i]
            al=(self.allowed & ~self.dilation(others)) & (np.isin(self.roomof,[ri,self.parent[ri]]) if self.parent[ri]!=self.hall else ((self.roomof==ri)|feed))
            if zone_mask is not None:al &= zone_mask|feed
            if connection_zone is not None:al &= (self.roomof!=self.parent[ri])|connection_zone
            if attach and self.parent[ri]!=self.hall:
                parent=self.parent[ri];pr=self.rooms[parent]['rect'];door=self.doors[parent,ri]
                pm=feed&(self.roomof==parent)
                if door[0]==door[2]:
                    side=0 if abs(door[0]-pr[0])<1 else 2
                    normal=np.broadcast_to(self.xs[None,:]-pr[0] if side==0 else pr[2]-self.xs[None,:],self.shape)
                    tangent=np.broadcast_to(self.ys[:,None],self.shape);dy,dx=0,(-1 if side==0 else 1)
                else:
                    side=1 if abs(door[1]-pr[1])<1 else 3
                    normal=np.broadcast_to(self.ys[:,None]-pr[1] if side==1 else pr[3]-self.ys[:,None],self.shape)
                    tangent=np.broadcast_to(self.xs[None,:],self.shape);dy,dx=(-1 if side==1 else 1),0
                candidates=pm&(normal>=2.5*self.c-1e-6)
                if not candidates.any():continue
                closest=normal[candidates].min();candidates &= normal<=closest+1e-6
                if attach in (1,2):vals=np.where(candidates,tangent if attach==1 else -tangent,np.inf)
                else:
                    fraction={3:.5,4:.25,5:.75,6:.125,7:.375,8:.625,9:.875}.get(attach,.5)
                    tmin=tangent[candidates].min();tmax=tangent[candidates].max()
                    vals=np.where(candidates,abs(tangent-(tmin+fraction*(tmax-tmin))),np.inf)
                tj,ti=np.unravel_index(np.argmin(vals),self.shape)
                original_al=al.copy();al &= ~self.dilation(feed)
                gate=[]
                for step in range(3):
                    jj,ii=tj+step*dy,ti+step*dx
                    if not (0<=jj<self.ny and 0<=ii<self.nx and original_al[jj,ii]):break
                    gate.append((jj,ii))
                if len(gate)<3:continue
                for jj,ii in gate:al[jj,ii]=True
            conn=self.astar(feed,target,al,connection_cost)
            if conn is not None:chosen=set(comp);connector=conn;break
        if chosen is None:return None
        pat=self.empty()
        for i in chosen:
            x,y,_,_=nd[i];pat[y:y+width,x:x+width]|=patches[i]
        graph={i:[] for i in chosen}
        for di,i,j,box in edges:
            if i in chosen and j in chosen:
                graph[i].append((j,box));graph[j].append((i,box))
        rj,rix=connector[-1]
        root=next(i for i in chosen if nd[i][0]<=rix<nd[i][0]+width and nd[i][1]<=rj<nd[i][1]+width)
        comb_selected=None
        if style==5:
            side=2 if reserve_side is None else reserve_side
            horizontal=side in (0,2)
            parents={i:i for i in chosen}
            def cfind(i):
                while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
                return i
            def edge_weight(edge):
                _,i,j,box=edge;x,y,_,_=nd[i];X,Y,_,_=nd[j]
                main=(y==Y) if horizontal else (x==X)
                coord=min(x,X) if horizontal else min(y,Y)
                return (0 if main else 1,coord if side in (2,3) else -coord,i,j)
            comb_selected=[]
            for edge in sorted(edges,key=edge_weight):
                _,i,j,box=edge
                if i not in chosen or j not in chosen:continue
                aa,bb=cfind(i),cfind(j)
                if aa==bb:continue
                parents[aa]=bb;comb_selected.append(box)
        inlet_node=root
        deferred=None
        if end_mode and end_point is not None:
            ex,ey=end_point
            outlet_node=min(chosen,key=lambda i:(((nd[i][0]+self.ix0+width*.5)*self.c-ex)**2+((nd[i][1]+self.iy0+width*.5)*self.c-ey)**2))
            if outlet_node!=inlet_node:
                root=outlet_node if end_mode==1 else inlet_node
                deferred=inlet_node if end_mode==1 else outlet_node
        def grow_tree(start,defer):
            visited={start};stack=[(start,None)];selected=[];depth={start:0}
            while stack:
                i,prev=stack[-1]
                ns=[(j,box) for j,box in graph[i] if j not in visited and (j!=defer or len(visited)==len(chosen)-1)]
                if not ns:stack.pop();continue
                x,y,_,_=nd[i]
                def priority(item):
                    j,box=item;X,Y,_,_=nd[j]
                    straight=0
                    if prev is not None:
                        a,b,_,_=nd[prev];straight=0 if (X-x)*(x-a)+(Y-y)*(y-b)>0 else 1
                    else:straight=0 if (y==Y)==(orient==0) else 1
                    degree=sum(q not in visited for q,_ in graph[j])
                    tie=j if offset%2==0 else -j
                    if style==1:return (degree,straight,tie)
                    if style in (2,3):
                        xi,yi=nd[i][2:];xj,yj=nd[j][2:]
                        if style==2:
                            si=yi*len(xx)+(xi if (yi+offset)%2==0 else len(xx)-1-xi)
                            sj=yj*len(xx)+(xj if (yj+offset)%2==0 else len(xx)-1-xj)
                        else:
                            si=xi*len(yy)+(yi if (xi+offset)%2==0 else len(yy)-1-yi)
                            sj=xj*len(yy)+(yj if (xj+offset)%2==0 else len(yy)-1-yj)
                        return (abs(sj-si),degree,tie)
                    if style>=4:
                        rnd=((i*73856093)^(j*19349663)^((style-3)*83492791)^((offset+1)*2654435761))%65521
                        return (degree,rnd,straight)
                    return (straight,degree,tie)
                j,box=min(ns,key=priority)
                visited.add(j);depth[j]=depth[i]+1;stack.append((j,i));selected.append(box)
            if defer is not None and defer not in visited and len(visited)==len(chosen)-1:
                choices=[(depth[j],box) for j,box in graph[defer] if j in visited]
                if choices:
                    _,box=max(choices,key=lambda q:q[0]);selected.append(box);visited.add(defer)
            return selected,visited
        if comb_selected is not None:selected,visited=comb_selected,chosen
        else:
            selected,visited=grow_tree(root,deferred)
            if len(visited)<len(chosen):selected,visited=grow_tree(root,None)
        for box in selected:
            x,y,X,Y,bp=box;pat[y:Y,x:X]|=bp
        # Stop on the first edge of the tree.  Retaining the old feed's tail
        # could bridge two spiral turns and accidentally enclose a cold hole.
        near=binary_dilation(pat,structure=np.array([[0,1,0],[1,1,1],[0,1,0]],bool))
        if root_slot is not None:
            st=self.empty();st[self.roots[root_slot]]=True
            connector=self.astar(st,near,al|feed)
            if connector is None:return None
            feed=self.empty()
        else:
            connector=self.astar(feed,near,al,connection_cost)
            if connector is None:return None
        for j,i in connector:pat[j,i]=True
        pat |= feed
        filled=binary_fill_holes(pat)
        pat=filled
        # Eliminate a diagonal pinch by adding its missing corner cell.  This
        # gives one simple orthogonal outline, with segments at least one cell.
        for _ in range(8):
            a=pat[:-1,:-1];b=pat[:-1,1:];c=pat[1:,:-1];d=pat[1:,1:]
            bad=(a&d&~b&~c)|(b&c&~a&~d)
            if not bad.any():break
            for j,i in zip(*np.nonzero(bad)):
                for jj,ii in ((j,i),(j,i+1),(j+1,i),(j+1,i+1)):
                    if not pat[jj,ii] and self.allowed[jj,ii] and not self.dilation(others)[jj,ii]:pat[jj,ii]=True;break
        if (pat & self.dilation(others)).any() or (pat & ~self.allowed).any():return None
        return pat

    def outline(self,mask,k):
        # Directed boundary edges have the occupied region to their left.
        edges={}
        for j,i in zip(*np.nonzero(mask)):
            if j==0 or not mask[j-1,i]:edges[i,j]=(i+1,j)
            if i+1==self.nx or not mask[j,i+1]:edges[i+1,j]=(i+1,j+1)
            if j+1==self.ny or not mask[j+1,i]:edges[i+1,j+1]=(i,j+1)
            if i==0 or not mask[j,i-1]:edges[i,j+1]=(i,j)
        rj,ri=self.roots[k]
        # The lower edge between the manifold ports is the open end of the loop.
        start=(ri,rj);end=(ri+1,rj)
        if edges.get(start)!=end:return None
        del edges[start]
        # Traverse in reverse, so the supply endpoint comes first.
        rev={b:a for a,b in edges.items()}
        out=[start];q=start
        for _ in range(len(edges)+1):
            if q==end:break
            if q not in rev:return None
            q=rev[q];out.append(q)
        if q!=end or len(out)!=len(edges)+1:return None
        simp=[]
        for p in out:
            if len(simp)>=2 and ((simp[-2][0]==simp[-1][0]==p[0]) or (simp[-2][1]==simp[-1][1]==p[1])):simp[-1]=p
            else:simp.append(p)
        points=[]
        for i,j in simp:
            x=(i+self.ix0)*self.c*self.g;y=(j+self.iy0)*self.c*self.g
            pp=self.origin+x*self.ux+y*self.uy
            points.append([round(float(pp[0]),8),round(float(pp[1]),8)])
        return {'slot':k,'path':points}

    def circuits(self):
        h=self.hall;hx0,hy0,hx1,hy1=self.rooms[h]['rect']
        def rank(x,y,d):
            if d[1]==d[3] and abs(y-hy0)<1:return -x if x<0 else 1e5-x
            if d[0]==d[2] and abs(x-hx0)<1:return 1e4+y
            if d[1]==d[3] and abs(y-hy1)<1:return 2e4+x
            return 3e4-y
        living=next((i for i,r in enumerate(self.rooms) if r['id']=='living' and i in self.adj[h]),None)
        def pressure(ri):
            ids=[ri]+[i for i in self.adj[ri] if i!=h]
            demand=0.;area=0.
            for ir in ids:
                room=self.rooms[ir];a,b,c,e=room['rect']
                ar=(c-a)*(e-b)*self.g*self.g
                for ka,kb,kc,kd in self.p['keepouts']:ar-=max(0,min(c,kc)-max(a,ka))*max(0,min(e,kd)-max(b,kb))*self.g*self.g
                area+=max(.1,ar)
                inlet=self.original['physics']['supply_temp']
                demand+=room['required_w']*(.2+room.get('floor_r',0.))*(inlet-20.)/max(1.,inlet-room['air_temp'])
            return demand/max(1.,area)**.35
        largest=max(self.adj[h],key=pressure)
        if living is None or pressure(largest)>pressure(living)*1.15:living=largest
        if len(self.adj[h])>=self.nslots:living=None
        if hasattr(self,'split_room'):living=self.split_room
        split_rooms=[] if living is None else [living]
        max_splits=min(self.nslots-len(self.adj[h]),getattr(self,'split_limit',self.nslots))
        for ri in sorted(self.adj[h],key=pressure,reverse=True):
            if len(split_rooms)>=max_splits:break
            members=[ri]+[ch for ch in self.adj[ri] if ch!=h]
            demand=sum(self.rooms[ch]['required_w'] for ch in members)
            resistance=max(self.rooms[ch].get('floor_r',0.) for ch in members)
            if demand<(400. if resistance>=.09 else 500.):continue
            if ri not in split_rooms:split_rooms.append(ri)
        circuits=[]
        for ri in self.adj[h]:
            d=self.doors[h,ri];x=(d[0]+d[2])/2;y=(d[1]+d[3])/2
            r=self.rooms[ri]['rect']; children=[i for i in self.adj[ri] if i!=h]
            if ri in split_rooms:
                horizontal=d[1]==d[3]
                frac=((x-r[0])/(r[2]-r[0])) if horizontal else ((y-r[1])/(r[3]-r[1]))
                balanced=True
                child_owner={}
                body_split=(r[0]+r[2])/2 if horizontal else (r[1]+r[3])/2
                if children:
                    middle=body_split
                    for child in children:
                        cd=self.doors[ri,child];coord=(cd[0]+cd[2])/2 if horizontal else (cd[1]+cd[3])/2
                        child_owner[child]=-1 if coord<=middle else 1
                    load=sum(self.rooms[ch]['required_w']*child_owner[ch] for ch in children)
                    fraction=min(.8,max(.2,.5+.5*load/max(1,self.rooms[ri]['required_w'])))
                    ax=0 if horizontal else 1
                    body_split=r[ax]+fraction*(r[ax+2]-r[ax])
                    for child in children:
                        cd=self.doors[ri,child]
                        if (horizontal and cd[1]==cd[3]) or (not horizontal and cd[0]==cd[2]):
                            if child_owner[child]<0:body_split=max(body_split,cd[ax]+2.5*self.c)
                            else:body_split=min(body_split,cd[ax+2]-2.5*self.c)
                for side in (-1,1):
                    bbox=r.copy();zone=None
                    if horizontal:
                        split=body_split if balanced else x
                        if side<0:bbox[2]=split
                        else:bbox[0]=split
                        tx=x+side*1.5*self.c;ty=y
                        if balanced:
                            near_low=abs(y-r[1])<1
                            transition=y+(3*self.c if near_low else -3*self.c)
                            if near_low:bbox[1]=transition
                            else:bbox[3]=transition
                            front=(self.ys[:,None]<transition) if near_low else (self.ys[:,None]>transition)
                            front_side=(self.xs[None,:]<x) if side<0 else (self.xs[None,:]>=x)
                            body_side=(self.xs[None,:]<split) if side<0 else (self.xs[None,:]>=split)
                            zone=(self.roomof==ri)&((front&front_side)|(~front&body_side))
                    else:
                        split=body_split if balanced else y
                        if side<0:bbox[3]=split
                        else:bbox[1]=split
                        tx=x;ty=y+side*1.5*self.c
                        if balanced:
                            near_low=abs(x-r[0])<1
                            transition=x+(3*self.c if near_low else -3*self.c)
                            if near_low:bbox[0]=transition
                            else:bbox[2]=transition
                            front=(self.xs[None,:]<transition) if near_low else (self.xs[None,:]>transition)
                            front_side=(self.ys[:,None]<y) if side<0 else (self.ys[:,None]>=y)
                            body_side=(self.ys[:,None]<split) if side<0 else (self.ys[:,None]>=split)
                            zone=(self.roomof==ri)&((front&front_side)|(~front&body_side))
                    owned=[]
                    for child in children:
                        cd=self.doors[ri,child];cx=(cd[0]+cd[2])/2;cy=(cd[1]+cd[3])/2
                        if horizontal:
                            near_front=abs(cy-y)<3*self.c
                            cut=x if near_front else split
                            owner=-1 if cx<=cut else 1
                        else:
                            near_front=abs(cx-x)<3*self.c
                            cut=y if near_front else split
                            owner=-1 if cy<=cut else 1
                        if child_owner.get(child,owner)==side:owned.append(child)
                    circuits.append(dict(ri=ri,rooms=[ri]+owned,bbox=bbox,zone=zone,rank=rank(tx,ty,d),door=d,target=(tx,ty)))
            else:circuits.append(dict(ri=ri,rooms=[ri]+children,bbox=None,rank=rank(x,y,d),door=d,target=(x,y)))
        circuits.sort(key=lambda a:a['rank'])
        self.cs=circuits
        self.assignment={i:k for k,a in enumerate(circuits) for i in a['rooms']}

    def feeds_new(self,order=0,shift=0):
        self.circuits();cs=self.cs;h=self.hall
        masks=[self.empty() for _ in cs]
        for k in range(len(cs)):masks[k][self.roots[k]]=True
        indices=list(range(len(cs)))
        if order==0:indices=sorted(indices,key=lambda k:min(k,len(cs)-1-k))
        elif order==1:indices=list(reversed(indices))
        elif order==2:indices=sorted(indices,key=lambda k:abs(k-(len(cs)-1)/2))
        for k in indices:
            a=cs[k];ri=a['ri'];d=a['door'];r=self.rooms[ri]['rect'];x,y=a['target']
            if d[1]==d[3]:y+=(self.c*1.8 if abs(y-r[1])<1 else -self.c*1.8);x+=shift*self.c
            else:x+=(self.c*1.8 if abs(x-r[0])<1 else -self.c*1.8);y+=shift*self.c
            other=np.logical_or.reduce([v for l,v in enumerate(masks) if l!=k])
            allowed=self.allowed&~self.dilation(other)&np.isin(self.roomof,[h,ri])
            allowed &= ~((self.roomof==h)&(self.ys[:,None]<0)&(self.xs[None,:]>=-self.c)&(self.xs[None,:]<2*self.nslots*self.c))
            if a.get('zone') is not None:
                allowed &= (self.roomof!=ri)|a['zone']
            elif a['bbox'] is not None:
                x0,y0,x1,y1=a['bbox'];allowed &= (self.roomof!=ri)|((self.xs[None,:]>=x0)&(self.xs[None,:]<=x1)&(self.ys[:,None]>=y0)&(self.ys[:,None]<=y1))
            target=(self.roomof==ri)&allowed
            ds=(self.xs[None,:]-x)**2+(self.ys[:,None]-y)**2
            if not target.any():return None
            best=np.min(ds[target]);goals=target&(ds<=best+self.c*self.c*.5)
            path=self.astar(masks[k],goals,allowed)
            if path is None:return None
            for j,i in path:masks[k][j,i]=True
        return masks

    def add_hall(self,masks,reach=3.0,max_add=200,biases=None):
        masks=[a.copy() for a in masks]
        hall=self.roomof==self.hall
        # Occupied hall points, sampled on this construction lattice.
        rd=self.room_distance(self.hall)
        target=hall & self.allowed & (rd>=1.5)
        cross=np.array([[0,1,0],[1,1,1],[0,1,0]],bool)
        added=0
        rejected=self.empty()
        while added<max_add:
            allm=np.logical_or.reduce(masks)
            dist=distance_transform_edt(~allm)
            score=np.where(target&~rejected,dist,0)
            j,i=np.unravel_index(np.argmax(score),self.shape)
            if score[j,i]<=reach:break
            st=self.empty();st[j,i]=True
            options=[]
            for k,old in enumerate(masks):
                other=np.logical_or.reduce([a for l,a in enumerate(masks) if l!=k])
                al=self.allowed & hall & ~self.dilation(other) & ~old
                goal=binary_dilation(old,structure=cross)&al
                if not al[j,i]:continue
                path=self.astar(st,goal,al)
                if path is None:continue
                cand=old.copy()
                for y,x in path:cand[y,x]=True
                cand=binary_fill_holes(cand)
                if (cand&self.dilation(other)).any() or (cand&~self.allowed).any():continue
                lp=self.outline(cand,k)
                if lp is None:continue
                length=sum(abs(a[0]-b[0])+abs(a[1]-b[1]) for a,b in zip(lp['path'],lp['path'][1:]))
                if length>self.original['pipe']['max_loop_length']-.2:continue
                # Share the new heating load among circuits with room to spare.
                options.append((len(path)+length*.12+(biases[k] if biases is not None else 0.),k,cand))
            if not options:
                rejected[j,i]=True;continue
            _,k,cand=min(options,key=lambda x:x[0])
            added+=int(cand.sum()-masks[k].sum());masks[k]=cand
        return masks


def main():
    with open(sys.argv[1]) as f:plan=json.load(f)
    from optimize import Optimizer
    opt=Optimizer(plan,START_TIME+38.8)
    layout=opt.run()
    with open(sys.argv[2],'w') as f:json.dump(layout,f)
    if os.environ.get('DEBUG'):print('total time',time.monotonic()-START_TIME,file=sys.stderr)

if __name__=='__main__':main()
