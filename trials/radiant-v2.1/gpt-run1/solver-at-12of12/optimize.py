from __future__ import annotations
import math,time,random
from collections import defaultdict
import numpy as np
from scipy.fft import irfft2,rfft2
import thermal
from contours import line
from routing import simplify


def greens(plan, floor_r, h=None, radius=12):
    p=plan['physics'];g=plan['grid'];area=g*g
    size=128
    kx=np.arange(size//2+1)[None,:]*2*np.pi/size
    ky=np.arange(size)[:,None]*2*np.pi/size
    lam=4-2*np.cos(kx)-2*np.cos(ky)
    gv=p['k_screed']/p['layer_gap']*area
    gr=area/(floor_r+1/(h or p['h_core']))
    aa=p['k_screed']*p['pipe_layer']*lam+gv+p['insulation_u']*area
    bb=p['k_screed']*p['surface_layer']*lam+gv+gr
    inv=irfft2(gv/(aa*bb-gv*gv),s=(size,size))
    inds=np.arange(-radius,radius+1)%size
    return inv[inds[:,None],inds[None,:]]


def canonical_array(d,a):
    bx0,by0,bx1,by1=(round(v/d.g) for v in d.plan['building'])
    x0,y0,x1,y1=(round(v*d.step) for v in d.building)
    xx=np.arange(x0,x1+1)[None,:];yy=np.arange(y0,y1+1)[:,None]
    x=d.origin[0]+xx*d.d[0]+yy*d.normal[0]
    y=d.origin[1]+xx*d.d[1]+yy*d.normal[1]
    return a[y-by0,x-bx0].copy()


def layout_of(d, paths):
    return {'loops':[{'slot':k,'path':[d.output(p) for p in simplify(path)]} for k,path in paths.items()]}


class Optimizer:
    def __init__(self,d,loops,hall_prepare=False):
        self.d=d;self.p=d.plan;self.step=d.step;self.unit=d.unit;self.hall_prepare=hall_prepare
        self.prepared_rooms={d.hall}|{ri for lp in loops for ri in lp['rooms'] if sum(ri in q['rooms'] for q in loops)>1}
        self.paths={lp['slot']:lp['path'] for lp in loops}
        self.ends={k:(v[0],v[-1]) for k,v in self.paths.items()}
        self.mouths={q[-1] for lp in loops for q in lp['hall_paths']}
        self.loop_bounds={lp['slot']:lp['bounds']+[d.rooms[d.hall]] for lp in loops}
        self.children={ri for lp in loops for ri in lp['rooms'][1:]}
        self.adj={};self.occ={};self.length={}
        for k,path in self.paths.items():
            adj=defaultdict(set)
            for a,b in zip(path,path[1:]):
                if a in self.occ and self.occ[a]!=k:raise ValueError('initial overlap')
                adj[a].add(b);adj[b].add(a)
                self.occ[a]=self.occ[b]=k
            self.adj[k]=adj;self.length[k]=len(path)-1
        self.floor=thermal.Floor(self.p)
        self.roomof=canonical_array(d,self.floor.room_of)
        self.ny,self.nx=self.roomof.shape
        self.x0,self.y0,_,_=(round(v*self.step) for v in d.building)
        self.allowed_edges=set()
        self.plaqs=[]
        self.prepare_graph()
        self.prepare_fields()
        self.rng=random.Random(314159)
        self.radius=12
        self.kernels=[greens(self.p,r.get('floor_r',0),radius=self.radius) for r in self.p['rooms']]
        self.basepatterns={};self.configs={}
        step=self.step;r=self.radius
        for shape,(w,h) in enumerate(self.shapes):
            border=[(x,0) for x in range(w)]+[(w,y) for y in range(h)]+[(x,h) for x in range(w,0,-1)]+[(0,y) for y in range(h,0,-1)]
            m=len(border);cfg={}
            for length in range(1,m):
                for start in range(m):
                    mask=sum(1<<((start+j)%m) for j in range(length))
                    olddeg=[int(bool(mask&(1<<j)))+int(bool(mask&(1<<((j-1)%m)))) for j in range(m)]
                    cfg[mask]=([j for j,v in enumerate(olddeg) if v==0],[j for j,v in enumerate(olddeg) if v==2],m-2*length)
            self.configs[shape]=cfg
            for ri,kernel in enumerate(self.kernels):
                for mask in cfg:
                    ar=np.zeros((2*r+h*step+1,2*r+w*step+1))
                    for e,(a,b) in enumerate(zip(border,border[1:]+border[:1])):
                        ax,ay=a[0]*step,a[1]*step;bx,by=b[0]*step,b[1]*step
                        sign=-1 if mask&(1<<e) else 1
                        for cx,cy,weight in ((ax,ay,.25),((ax+bx)//2,(ay+by)//2,.5),(bx,by,.25)):
                            ar[cy:cy+2*r+1,cx:cx+2*r+1]+=sign*weight*kernel
                    self.basepatterns[ri,shape,mask]=ar

    def prepare_graph(self):
        d=self.d
        floor=self.floor
        orig_ks=[tuple(round(v/d.g) for v in q) for q in self.p.get('keepouts',[])]
        passages=[tuple(v/d.g for v in q) for q in self.p.get('passages',[])]
        def original(p):
            x,y=p[0]*self.step,p[1]*self.step
            return (d.origin[0]+x*d.d[0]+y*d.normal[0],d.origin[1]+x*d.d[1]+y*d.normal[1])
        def valid_edge(a,b):
            oa,ob=original(a),original(b)
            lx,ly=min(oa[0],ob[0]),min(oa[1],ob[1]);hx,hy=max(oa[0],ob[0]),max(oa[1],ob[1])
            if any(lx<=q[2] and hx>=q[0] and ly<=q[3] and hy>=q[1] for q in orig_ks):return False
            walk=line(oa,ob)
            for a,b in zip(walk,walk[1:]):
                ra,rb=floor.room_at(*a),floor.room_at(*b)
                if ra!=rb:
                    mx,my=(a[0]+b[0])/2,(a[1]+b[1])/2
                    if not any(p[0]-.5<=mx<=p[2]+.5 and p[1]-.5<=my<=p[3]+.5 and (p[0]<=mx<=p[2] or p[1]<=my<=p[3]) for p in passages):return False
            return True
        valid_nodes=np.logical_or.reduce([d.room_mask(ri) for ri in self.prepared_rooms]) if self.hall_prepare else d.allowed
        iy,ix=np.nonzero(valid_nodes)
        xmin,xmax=int(ix.min())+d.x0,int(ix.max())+d.x0
        ymin,ymax=int(iy.min())+d.y0,int(iy.max())+d.y0
        for y in range(ymin,ymax+1):
            for x in range(xmin,xmax+1):
                a=(x,y)
                if not valid_nodes[y-d.y0,x-d.x0]:continue
                for b in ((x+1,y),(x,y+1)):
                    if d.is_allowed(b) and valid_nodes[b[1]-d.y0,b[0]-d.x0] and valid_edge(a,b):
                        self.allowed_edges.add((a,b))
        self.shapes=[(1,1),(2,1),(1,2),(2,2)]
        self.plaq_ranges=[]
        for shape,(w,h) in enumerate(self.shapes):
            begin=len(self.plaqs)
            offsets=[(x,0) for x in range(w)]+[(w,y) for y in range(h)]+[(x,h) for x in range(w,0,-1)]+[(0,y) for y in range(h,0,-1)]
            for y in range(ymin,ymax-h+1):
                for x in range(xmin,xmax-w+1):
                    c=tuple((x+dx,y+dy) for dx,dy in offsets)
                    es=tuple(tuple(sorted((c[i],c[(i+1)%len(c)]))) for i in range(len(c)))
                    if all(e in self.allowed_edges for e in es):
                        corners=[self.roomof[(y+dy)*self.step-self.y0,(x+dx)*self.step-self.x0] for dx,dy in ((0,0),(w,0),(0,h),(w,h))]
                        if len(set(corners))==1:self.plaqs.append((x,y,c,es,shape))
            self.plaq_ranges.append((begin,len(self.plaqs)))
        self.plaq_room=np.array([self.roomof[y*self.step-self.y0,x*self.step-self.x0] for x,y,_,_,_ in self.plaqs])
        self.plaq_hall=np.zeros(len(self.plaqs),dtype=bool)
        self.hall_only=False
        hall_mask=d.room_mask(d.hall)
        self.plaq_allowed=np.zeros(len(self.plaqs),dtype=np.uint16)
        for k,rects in self.loop_bounds.items():
            xx=d.xs[None,:];yy=d.ys[:,None];mask=np.zeros_like(d.allowed)
            for r in rects:mask|=(xx>=r[0]-1e-7)&(xx<=r[2]+1e-7)&(yy>=r[1]-1e-7)&(yy<=r[3]+1e-7)
            for shape,(begin,end) in enumerate(self.plaq_ranges):
                if begin==end:continue
                cs=np.array([self.plaqs[j][2] for j in range(begin,end)])
                valid=mask[cs[:,:,1]-d.y0,cs[:,:,0]-d.x0].all(axis=1)
                if k==next(iter(self.loop_bounds)):
                    self.plaq_hall[begin:end]=hall_mask[cs[:,:,1]-d.y0,cs[:,:,0]-d.x0].all(axis=1)
                self.plaq_allowed[begin:end]|=valid.astype(np.uint16)*(1<<k)

    def prepare_fields(self):
        d=self.d;p=self.p;g=d.g
        xx=(np.arange(self.nx)+self.x0)*g
        yy=(np.arange(self.ny)+self.y0)*g
        bx0,by0,bx1,by1=[v*self.unit for v in d.building]
        edge=np.minimum(np.minimum(xx-bx0,bx1-xx)[None,:],np.minimum(yy-by0,by1-yy)[:,None])
        h=np.where(edge<=p['physics']['perim_width']+1e-6,p['physics']['h_perim'],p['physics']['h_core'])
        fr=np.array([r.get('floor_r',0) for r in p['rooms']])[self.roomof]
        self.groom=g*g/(fr+1/h)
        self.zone=np.zeros_like(self.roomof,dtype=bool)
        self.room_masks=[];self.zone_masks=[]
        self.nroom=len(p['rooms'])
        for ri,room in enumerate(p['rooms']):
            r=np.array(d.rooms[ri])*self.unit
            zm=room.get('zone_margin',0)
            zone=(self.roomof==ri)&(xx[None,:]>=r[0]+zm-1e-6)&(xx[None,:]<=r[2]-zm+1e-6)&(yy[:,None]>=r[1]+zm-1e-6)&(yy[:,None]<=r[3]-zm+1e-6)
            for q in d.keepouts:
                q=np.array(q)*self.unit
                zone&=~((xx[None,:]>=q[0]-zm-1e-6)&(xx[None,:]<=q[2]+zm+1e-6)&(yy[:,None]>=q[1]-zm-1e-6)&(yy[:,None]<=q[3]+zm+1e-6))
            self.zone|=zone;self.room_masks.append(self.roomof==ri);self.zone_masks.append(zone)
        rooms=p['rooms']
        self.air=np.array([r['air_temp'] for r in rooms])[self.roomof]
        self.limits=np.array([r['max_surface'] for r in rooms])[self.roomof]-.15
        self.spread=np.array([r['max_spread'] for r in rooms])[self.roomof]
        self.aim=np.array([.5*(r['required_w']+r.get('max_w',r['required_w']*1.3)) for r in rooms])
        self.width=np.array([.5*(r.get('max_w',r['required_w']*1.3)-r['required_w']) for r in rooms])
        self.power_low=np.array([r['required_w']+.08*(r.get('max_w',r['required_w']*1.3)-r['required_w']) for r in rooms])
        self.power_high=np.array([r.get('max_w',r['required_w']*1.3)-.04*(r.get('max_w',r['required_w']*1.3)-r['required_w']) for r in rooms])
        self.lower_guard=np.array([r['required_w']+max(2.,.05*(r.get('max_w',r['required_w']*1.3)-r['required_w'])) for r in rooms])
        self.upper_guard=np.array([r.get('max_w',1e99)-max(2.,.05*(r.get('max_w',1e99)-r['required_w'])) for r in rooms])
        self.croom=np.array([self.groom[sel].sum() for sel in self.room_masks])
        self.nz=np.array([max(1,z.sum()) for z in self.zone_masks])
        self.ow=self.nz/(self.width**2)*.15

    def reset_paths(self,loops):
        self.paths={lp['slot']:lp['path'] for lp in loops}
        self.ends={k:(v[0],v[-1]) for k,v in self.paths.items()}
        self.adj={};self.occ={};self.length={}
        for k,path in self.paths.items():
            adj=defaultdict(set)
            for a,b in zip(path,path[1:]):
                adj[a].add(b);adj[b].add(a);self.occ[a]=self.occ[b]=k
            self.adj[k]=adj;self.length[k]=len(path)-1

    def trace(self):
        paths={}
        for k,adj in self.adj.items():
            a,end=self.ends[k];last=None;path=[a]
            while a!=end:
                ns=adj[a]
                b=next(b for b in ns if b!=last)
                path.append(b);last,a=a,b
            paths[k]=path
        self.paths=paths
        return paths

    def evaluate(self, verbose=False):
        paths=self.trace();layout=layout_of(self.d,paths)
        loops,problems=thermal.check_geometry(self.p,layout,self.floor)
        if problems:raise ValueError(problems)
        t0=time.perf_counter()
        heat=thermal.solve_heat(self.p,loops,self.floor)
        return self.install_heat(heat,loops,paths,verbose,t0)

    def install_heat(self,heat,loops,paths=None,verbose=False,t0=0.):
        paths=self.paths if paths is None else paths
        self.temp=canonical_array(self.d,heat['surface'])
        self.out=np.array([r['output_w'] for r in heat['rooms']])
        self.active_rooms=[]
        self.freeze_children=set()
        self.aim=self.out.copy()
        for ri,(room,res) in enumerate(zip(self.p['rooms'],heat['rooms'])):
            lo=room['required_w'];hi=room.get('max_w',lo*1.3);width=hi-lo
            active=(res['output_w']<lo or res['output_w']>hi or res['max_surface']>room['max_surface'] or res['zone_spread']>room['max_spread'])
            if active:
                if self.out[ri]<lo:self.aim[ri]=lo+width*(.35 if ri==self.d.hall else .15)
                elif self.out[ri]>hi:self.aim[ri]=hi-width*.02
                elif room['max_spread']<=2.0:
                    if self.out[ri]<lo+width*.08:self.aim[ri]=lo+width*.15
                    elif self.out[ri]>hi-width*.08:self.aim[ri]=hi-width*.15
            self.active_rooms.append(active)
            if ri in self.children and lo<=self.out[ri]<=hi and res['zone_spread']<=room['max_spread']*1.15:self.freeze_children.add(ri)
        if any(h['output_w']<r['required_w'] or h['output_w']>r.get('max_w',1e99) for r,h in zip(self.p['rooms'],heat['rooms'])):self.active_rooms[self.d.hall]=True
        self.centers=np.array([float(self.temp[z].mean()) for z in self.zone_masks])
        # Predict the occupied-zone temperature after reaching the output target.
        self.centers += (self.aim-self.out)/self.croom
        self.center=self.centers[self.roomof]
        self.qperm={}
        self.localq={}
        self.order={}
        self.uaout={}
        self.flow_loss={}
        self.order_range={}
        self.mcp={}
        self.uniform_shift=np.zeros(self.nroom)
        for lp in loops:
            k=lp['slot'];water=heat['water'][k];length=len(water)*self.p['grid']
            mcp=self.p['physics']['flow_constant']/(length+self.p['physics']['flow_offset'])
            prev=np.r_[self.p['physics']['supply_temp'],water[:-1]]
            qs=mcp*(prev-water)
            idx=np.array([self.floor.room_at(*a) for a in lp['walk'][1:]])
            ua=np.zeros(self.nroom);flo=np.zeros(self.nroom);lo=np.full(self.nroom,2.);hi=np.full(self.nroom,-1.)
            for j,point in enumerate(paths[k]):
                self.order[point]=j/max(1,len(paths[k])-1)
                js=max(0,j*self.step-self.step);je=min(len(qs),(j+1)*self.step)
                self.localq[point]=float(qs[js:je].mean()/self.p['grid']) if je>js else 0.
            for ri in range(self.nroom):
                sel=idx==ri;vals=qs[sel]
                if len(vals):
                    self.qperm[k,ri]=float(vals.mean()/self.p['grid'])
                    tw=float(water[sel].mean());air=self.p['rooms'][ri]['air_temp']
                    ua[ri]=float(vals.sum())*.78/max(2.,tw-air)
                    flo[ri]=ua[ri]*(self.p['physics']['supply_temp']-tw)/(length+self.p['physics']['flow_offset'])
                    inds=np.flatnonzero(sel)/len(water);lo[ri]=inds[0];hi[ri]=inds[-1]
            self.uaout[k]=ua;self.flow_loss[k]=flo;self.order_range[k]=(lo,hi);self.mcp[k]=mcp
        if verbose:
            print('HEAT',round(time.perf_counter()-t0,3),'lengths',self.length,file=__import__('sys').stderr)
            for r in heat['rooms']:print(r,file=__import__('sys').stderr)
        # Cooling one circuit of a shared room must not unbalance its two halves.
        self.pair_balance={};self.pair_shifts={}
        xx=(np.arange(self.nx)+self.x0)[None,:]/self.step;yy=(np.arange(self.ny)+self.y0)[:,None]/self.step
        for ri in range(self.nroom):
            members=[lp['slot'] for lp in loops if (lp['slot'],ri) in self.qperm]
            if ri==self.d.hall or len(members)!=2:continue
            stats=[]
            for k in members:
                region=np.zeros_like(self.roomof,dtype=bool)
                for b in self.loop_bounds[k]:region|=(xx>=b[0]-1e-7)&(xx<=b[2]+1e-7)&(yy>=b[1]-1e-7)&(yy<=b[3]+1e-7)
                sel=region&self.zone_masks[ri];full=region&self.room_masks[ri]
                stats.append((k,float(self.temp[sel].mean()) if sel.any() else self.centers[ri],max(1,int(sel.sum())),max(1.,float(self.groom[full].sum()))))
                self.pair_shifts[k,ri]=0.
            self.pair_balance[ri]=stats
        self.heat=heat
        self.parsed_loops=loops
        return heat

    def cost_pixel(self,t,c,sp,lim,z):
        diff=t-c
        v=z*(diff*diff+24*np.maximum(0,np.abs(diff)-sp*.35)**2)/(np.minimum(sp,3.)**2)
        v+=8*np.maximum(0,t-lim)**2
        return v

    def optimize(self, seconds=2.0, max_iterations=100000, anneal=.1, max_accept=100000):
        rng=self.rng;n=len(self.plaqs);t0=time.perf_counter();accepted=0;tested=0
        maxlen=int(self.p['pipe']['max_loop_length']/self.unit+1e-6)
        locked={a for ends in self.ends.values() for a in ends}|self.mouths
        temp=self.temp;roomof=self.roomof;groom=self.groom
        basepatterns=self.basepatterns;radius=self.radius;step=self.step
        active=np.asarray(self.active_rooms,dtype=bool)
        for ri in self.freeze_children:active[ri]=False
        candidates=[]
        for shape,(begin,end) in enumerate(self.plaq_ranges):
            candidates.extend([j for j in range(begin,end) if active[self.plaq_room[j]] and (not self.hall_only or self.plaq_hall[j])]*(3 if shape==0 else 1))
        if not candidates:return {'tested':0,'accepted':0,'iterations':0,'time':0}
        rng.shuffle(candidates);n=len(candidates)
        for it in range(max_iterations):
            if accepted>=max_accept:break
            if it%128==0 and time.perf_counter()-t0>seconds:break
            j=candidates[it%n] if it<n else candidates[rng.randrange(n)]
            x,y,c,es,shape=self.plaqs[j];m=len(c)
            ids=[self.occ.get(p,-1) for p in c]
            mask=0;k=-1;different=False
            for e,(a,b) in enumerate(es):
                ka=self.occ.get(a,-1)
                if ka>=0 and b in self.adj[ka][a]:
                    if k>=0 and k!=ka:different=True;break
                    k=ka;mask|=1<<e
            if different or mask not in self.configs[shape]:continue
            if not (int(self.plaq_allowed[j])&(1<<k)):continue
            add_inds,remove_inds,dl=self.configs[shape][mask]
            if any(ids[z]>=0 for z in add_inds) or any(c[z] in locked for z in remove_inds):continue
            if self.length[k]+dl>maxlen:continue
            added=[c[z] for z in add_inds];removed=[c[z] for z in remove_inds]
            # Thermal response to a changed pipe section.
            fx=x*step-self.x0;fy=y*step-self.y0
            ri=int(roomof[min(max(fy,0),self.ny-1),min(max(fx,0),self.nx-1)])
            qvals=[self.localq[p] for p in c if p in self.localq]
            qperm=sum(qvals)/len(qvals) if qvals else self.qperm.get((k,ri),10.)
            birth_gain=.65 if ri==self.d.hall and self.heat['rooms'][ri]['output_w']<self.p['rooms'][ri]['required_w'] else 1.0
            qeff=qperm*(birth_gain if dl>0 else .72 if dl<0 else .9)*self.unit
            kern=basepatterns[ri,shape,mask]*qeff
            bx,by=fx-radius,fy-radius
            ex,ey=bx+kern.shape[1],by+kern.shape[0]
            sx0,sy0=max(0,bx),max(0,by);sx1,sy1=min(self.nx,ex),min(self.ny,ey)
            if sx0>=sx1 or sy0>=sy1:continue
            dt=kern[sy0-by:sy1-by,sx0-bx:sx1-bx]
            sl=np.s_[sy0:sy1,sx0:sx1]
            old=temp[sl]+self.uniform_shift[roomof[sl]];new=old+dt
            cc=self.center[sl];sp=self.spread[sl];lim=self.limits[sl];zz=self.zone[sl]
            delta=float(np.sum(self.cost_pixel(new,cc,sp,lim,zz)-self.cost_pixel(old,cc,sp,lim,zz)))
            dout=np.bincount(roomof[sl].ravel(),weights=(groom[sl]*dt).ravel(),minlength=self.nroom)
            cool=np.zeros(self.nroom)
            if dl:
                order=sum(self.order.get(p,.5) for p in c if p in self.order)/max(1,sum(p in self.order for p in c))
                lo,hi=self.order_range[k]
                frac=np.clip((hi-order)/np.maximum(.001,hi-lo),0,1)
                cool=-1.25*self.uaout[k]*(float(dout.sum())*1.25)/self.mcp[k]*frac-self.flow_loss[k]*(dl*self.unit)
            balance_updates=[]
            if self.plaq_hall[j]:
                for rj,stats in self.pair_balance.items():
                    selected=next((v for v in stats if v[0]==k),None)
                    if selected is None or abs(cool[rj])<1e-8:continue
                    other=next(v for v in stats if v[0]!=k)
                    _,ta,na,capacity=selected;kb,tb,nb,_=other
                    shift=cool[rj]/capacity
                    difference=ta+self.pair_shifts[k,rj]-tb-self.pair_shifts[kb,rj]
                    delta+=3.*na*nb/(na+nb)*(2*difference*shift+shift*shift)/(min(3.,self.p['rooms'][rj]['max_spread'])**2)
                    balance_updates.append((k,rj,shift))
            total_dout=dout+cool
            if self.hall_only or self.plaq_hall[j]:
                guarded=((total_dout<-.02)&(self.out+total_dout<self.lower_guard))|((total_dout>.02)&(self.out+total_dout>self.upper_guard))
                guarded[self.d.hall]=False
                if guarded.any():continue
            after=self.out+total_dout
            old_power=np.maximum(0.,self.power_low-self.out)**2+np.maximum(0.,self.out-self.power_high)**2
            new_power=np.maximum(0.,self.power_low-after)**2+np.maximum(0.,after-self.power_high)**2
            delta+=float(np.sum(self.ow*(new_power-old_power)))
            tested+=1
            # Very small length penalty prevents gratuitous tight detours.
            delta+=.03*dl
            progress=min(1.,(time.perf_counter()-t0)/seconds)
            noise=anneal*(1-progress)
            if delta>0 and (noise<=0 or rng.random()>math.exp(-delta/max(.00001,noise))):continue
            temp[sl]+=dt;self.out+=total_dout;self.uniform_shift+=cool/self.croom
            for kk,rr,shift in balance_updates:self.pair_shifts[kk,rr]+=shift
            adj=self.adj[k]
            for e,(a,b) in enumerate(es):
                if mask&(1<<e):adj[a].remove(b);adj[b].remove(a)
                else:adj[a].add(b);adj[b].add(a)
            for a in removed:
                del self.occ[a];del adj[a];self.order.pop(a,None);self.localq.pop(a,None)
            for a in added:
                self.occ[a]=k;self.localq[a]=qperm;self.order[a]=order if dl else sum(self.order.get(p,.5) for p in c)/m
            self.length[k]+=dl;accepted+=1
        return {'tested':tested,'accepted':accepted,'iterations':it+1,'time':time.perf_counter()-t0,'predicted':self.out.tolist()}
