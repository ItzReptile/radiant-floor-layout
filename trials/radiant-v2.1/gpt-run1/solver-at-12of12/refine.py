from __future__ import annotations
import math,time,os,sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import thermal
from scipy.fft import rfft2,irfft2,next_fast_len


class LocalModel:
    """Exact local floor and a complete water circuit, with frozen floor boundary data."""
    def __init__(self,plan,floor,loops,heat,ri,slot,pad=.2,room_ids=None,region_mask=None):
        self.p=plan;self.floor=floor;self.slot=slot;self.ri=ri
        self.slots=set(slot) if isinstance(slot,(tuple,list,set)) else {slot}
        g=plan['grid'];ph=plan['physics'];area=g*g
        self.g=g;self.ph=ph
        self.room_ids=room_ids or [ri]
        rects=[plan['rooms'][i]['rect'] for i in self.room_ids]
        rect=[min(r[0] for r in rects),min(r[1] for r in rects),max(r[2] for r in rects),max(r[3] for r in rects)]
        self.x0=max(floor.bx0,round((rect[0]-pad)/g));self.x1=min(floor.bx1,round((rect[2]+pad)/g))
        self.y0=max(floor.by0,round((rect[1]-pad)/g));self.y1=min(floor.by1,round((rect[3]+pad)/g))
        self.nx=self.x1-self.x0+1;self.ny=self.y1-self.y0+1;self.n=self.nx*self.ny
        n=self.n
        xx=np.arange(self.x0,self.x1+1)[None,:];yy=np.arange(self.y0,self.y1+1)[:,None]
        idx=np.arange(n).reshape(self.ny,self.nx)
        rows=[];cols=[];vals=[];diag=np.zeros(2*n);rhs=np.zeros(2*n)
        kb=ph['k_screed']*ph['pipe_layer'];kt=ph['k_screed']*ph['surface_layer']
        gv=ph['k_screed']/ph['layer_gap']*area;gd=ph['insulation_u']*area
        room_of=floor.room_of[self.y0-floor.by0:self.y1-floor.by0+1,self.x0-floor.bx0:self.x1-floor.bx0+1]
        self.room_of=room_of
        b=plan['building'];xm,ym=xx*g,yy*g
        edge=np.minimum(np.minimum(xm-b[0],b[2]-xm),np.minimum(ym-b[1],b[3]-ym))
        h=np.where(edge<=ph['perim_width']+1e-6,ph['h_perim'],ph['h_core'])
        fr=np.array([r.get('floor_r',0.) for r in plan['rooms']])[room_of]
        self.groom=area/(fr+1/h)
        air=np.array([r['air_temp'] for r in plan['rooms']])[room_of]
        gr=self.groom.ravel()
        diag[:n]+=gv+gd;diag[n:]+=gv+gr
        rhs[:n]+=gd*ph['below_temp'];rhs[n:]+=gr*air.ravel()
        rows.extend([np.arange(n),n+np.arange(n)]);cols.extend([n+np.arange(n),np.arange(n)]);vals.extend([np.full(n,-gv),np.full(n,-gv)])
        for layer,k,ts in ((0,kb,heat['bottom']),(n,kt,heat['surface'])):
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                ax=np.broadcast_to(xx+dx,(self.ny,self.nx));ay=np.broadcast_to(yy+dy,(self.ny,self.nx))
                inbuilding=(ax>=floor.bx0)&(ax<=floor.bx1)&(ay>=floor.by0)&(ay<=floor.by1)
                diag[layer+idx[inbuilding]]+=k
                inside=inbuilding&(ax>=self.x0)&(ax<=self.x1)&(ay>=self.y0)&(ay<=self.y1)
                rows.append(layer+idx[inside]);cols.append(layer+(ay[inside]-self.y0)*self.nx+(ax[inside]-self.x0));vals.append(np.full(int(inside.sum()),-k))
                outside=inbuilding&~inside
                rhs[layer+idx[outside]]+=k*ts[ay[outside]-floor.by0,ax[outside]-floor.bx0]
        self.global_bottom=heat['bottom']
        self.sleeve=round(ph['sleeve_length']/g)
        for lp in loops:
            if lp['slot'] in self.slots:continue
            walk=lp['walk'][1:];steps=len(walk)
            xy=np.array(walk);x=xy[:,0];y=xy[:,1]
            sel=(x>=self.x0)&(x<=self.x1)&(y>=self.y0)&(y<=self.y1)
            js=np.flatnonzero(sel);cells=(y[sel]-self.y0)*self.nx+x[sel]-self.x0
            gks=np.full(steps,ph['g_pipe']*g);gks[:self.sleeve]*=ph['sleeve_factor'];gks[-self.sleeve:]*=ph['sleeve_factor']
            np.add.at(diag,cells,gks[sel]);np.add.at(rhs,cells,gks[sel]*heat['water'][lp['slot']][js])
        self.rows=rows;self.cols=cols;self.vals=vals;self.diag=diag;self.rhs=rhs
        self.infos=[];self.whole_infos={};self.fractions={}
        region=None if region_mask is None else region_mask[self.y0-floor.by0:self.y1-floor.by0+1,self.x0-floor.bx0:self.x1-floor.bx0+1]
        for rj in self.room_ids:
            sel=room_of==rj
            room=plan['rooms'][rj];x0,y0,x1,y1=room['rect'];zm=room.get('zone_margin',0)
            zone=sel&(xm>=x0+zm-1e-6)&(xm<=x1-zm+1e-6)&(ym>=y0+zm-1e-6)&(ym<=y1-zm+1e-6)
            for q in plan.get('keepouts',[]):
                zone&=~((xm>=q[0]-zm-1e-6)&(xm<=q[2]+zm+1e-6)&(ym>=q[1]-zm-1e-6)&(ym<=q[3]+zm+1e-6))
            zone=zone if zone.any() else sel
            self.whole_infos[rj]=(sel.copy(),zone.copy())
            if region is not None:
                sel=sel&region;zone=zone&region
                if not zone.any():zone=sel
            self.fractions[rj]=1. if region is None else float(sel.sum())/max(1,int((floor.room_of==rj).sum()))
            zy,zx=np.nonzero(zone)
            self.infos.append((rj,sel,zone,zy,zx))
        _,self.sel,self.zone,self.zy,self.zx=self.infos[0]
        self.otherloops=loops

    def evaluate(self,path,ret_surface=False,ret_state=False):
        paths=path if isinstance(path,dict) else {self.slot:path}
        n=self.n
        diag=self.diag.copy();rows=self.rows.copy();cols=self.cols.copy();vals=self.vals.copy()
        water_rhs=[];water_ranges={};offset=2*n
        for slot,walk in paths.items():
            xy=np.asarray(walk,dtype=np.int32)
            x,y=xy[1:,0],xy[1:,1];steps=len(x)
            mcp=self.ph['flow_constant']/(steps*self.g+self.ph['flow_offset'])
            gks=np.full(steps,self.ph['g_pipe']*self.g)
            if self.sleeve:
                gks[:self.sleeve]*=self.ph['sleeve_factor'];gks[-self.sleeve:]*=self.ph['sleeve_factor']
            inside=(x>=self.x0)&(x<=self.x1)&(y>=self.y0)&(y<=self.y1)
            iin=np.flatnonzero(inside);iout=np.flatnonzero(~inside)
            cells=(y[inside]-self.y0)*self.nx+x[inside]-self.x0
            nodes=offset+np.arange(steps)
            np.add.at(diag,cells,gks[inside])
            rows.extend([nodes,nodes[1:],nodes[iin],cells])
            cols.extend([nodes,nodes[:-1],cells,nodes[iin]])
            vals.extend([mcp+gks,np.full(steps-1,-mcp),-gks[inside],-gks[inside]])
            wrhs=np.zeros(steps);wrhs[0]=mcp*self.ph['supply_temp']
            wrhs[iout]+=gks[~inside]*self.global_bottom[y[~inside]-self.floor.by0,x[~inside]-self.floor.bx0]
            water_rhs.append(wrhs);water_ranges[slot]=(offset,offset+steps);offset+=steps
        rows.append(np.arange(2*n));cols.append(np.arange(2*n));vals.append(diag)
        matrix=coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(offset,offset)).tocsc()
        t=spsolve(matrix,np.concatenate([self.rhs]+water_rhs),permc_spec='MMD_AT_PLUS_A')
        surf=t[n:2*n].reshape(self.ny,self.nx)
        results=[]
        for rj,sel,zone,zy,zx in self.infos:
            room=self.p['rooms'][rj]
            out=float((self.groom[sel]*(surf[sel]-room['air_temp'])).sum())
            z=surf[zone];zz=z-float(z.mean())
            soft=(np.log(np.exp(8*(zz-zz.max())).mean())+8*zz.max()+np.log(np.exp(-8*(zz-zz.min())).mean())-8*zz.min())/8
            res={'ri':rj,'output_w':out,'max_surface':float(surf[sel].max()),'zone_spread':float(z.max()-z.min()),'zone_std':float(z.std()),'soft_spread':float(soft)}
            res['hot']=(int(zx[z.argmax()]+self.x0),int(zy[z.argmax()]+self.y0))
            res['cold']=(int(zx[z.argmin()]+self.x0),int(zy[z.argmin()]+self.y0))
            if self.fractions[rj]<1.-1e-9:
                res['fraction']=self.fractions[rj];res['spread_scale']=.80
                ws,wz=self.whole_infos[rj]
                res['_whole']={'output_w':float((self.groom[ws]*(surf[ws]-room['air_temp'])).sum()),'max_surface':float(surf[ws].max()),'zone_spread':float(surf[wz].max()-surf[wz].min())}
            results.append(res)
        res=results[0] if len(results)==1 else {'rooms':results}
        if ret_surface:res['surface']=surf
        if ret_state:
            res['_surface']=surf;res['_bottom']=t[:n].reshape(self.ny,self.nx);res['_water']={k:t[a:b] for k,(a,b) in water_ranges.items()} if isinstance(path,dict) else t[2*n:]
            air=np.array([r['air_temp'] for r in self.p['rooms']])[self.room_of]
            res['_local_outputs']=np.bincount(self.room_of.ravel(),weights=(self.groom*(surf-air)).ravel(),minlength=len(self.p['rooms']))
        return res


def water_powers(plan,floor,walk,water,outside=None):
    xy=np.asarray(walk[1:],dtype=np.int32)
    ids=floor.room_of[xy[:,1]-floor.by0,xy[:,0]-floor.bx0]
    mcp=plan['physics']['flow_constant']/((len(walk)-1)*plan['grid']+plan['physics']['flow_offset'])
    q=mcp*(np.r_[plan['physics']['supply_temp'],water[:-1]]-water)
    if outside is not None:
        x0,y0,x1,y1=outside
        q=q*((xy[:,0]<x0)|(xy[:,0]>x1)|(xy[:,1]<y0)|(xy[:,1]>y1))
    return np.bincount(ids,weights=q,minlength=len(plan['rooms']))


def original_paths(d,paths):
    out={}
    for k,pts in paths.items():
        corners=[]
        for p in pts:
            x,y=p[0]*d.step,p[1]*d.step
            corners.append((d.origin[0]+x*d.d[0]+y*d.normal[0],d.origin[1]+x*d.d[1]+y*d.normal[1]))
        out[k]=thermal.lattice_walk(thermal.simplify(corners))
    return out


def score(room,res):
    width=max(10.,room.get('max_w',room['required_w']*1.3)-room['required_w'])
    out=res['output_w'];spread=res['zone_spread'];peak=res['max_surface'];lim=room['max_spread']
    es=[max(0.,room['required_w']-out)/width,max(0.,out-room.get('max_w',1e99))/width,max(0.,spread-lim)/lim,max(0.,peak-room['max_surface'])/3.]
    return sum(1000*x+10000*x*x for x in es)+(spread+.25*res.get('soft_spread',0)+.1*res.get('zone_std',0))/lim+20*(max(0.,spread-lim+.08)/lim)**2+5*(max(0.,room['required_w']+width*.03-out)/width)**2+5*(max(0.,out-room.get('max_w',1e99)+width*.03)/width)**2


class Geometry:
    def __init__(self,plan,floor,paths):
        self.p=plan;self.floor=floor;self.paths=paths
        self.corners={k:thermal.simplify(path) for k,path in paths.items()}
        self.keepouts=[tuple(round(v/plan['grid']) for v in q) for q in plan.get('keepouts',[])]
        self.passages=[tuple(v/plan['grid'] for v in q) for q in plan.get('passages',[])]
        self.margin=round(plan['edge_margin']/plan['grid'])
        self.spacing=round(plan['pipe']['min_spacing']/plan['grid'])
        self.minseg=round(plan['pipe']['min_segment']/plan['grid'])
        self.maxlen=plan['pipe']['max_loop_length']/plan['grid']
        self.refresh()

    def refresh(self):
        rows=[]
        for k,pts in self.corners.items():
            for i,(a,b) in enumerate(zip(pts,pts[1:])):
                rows.append((min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1]),k,i))
        self.rows=np.asarray(rows,dtype=np.int32)

    def try_shift(self,k,j,delta,end=None):
        pts=self.corners[k]
        end=j if end is None else end
        if j==0 or end>=len(pts)-2:return None
        vertical=pts[j][0]==pts[j+1][0]
        if (pts[end][0]==pts[end+1][0])!=vertical:return None
        pp=pts.copy()
        for t in range(j,end+2):
            a=pts[t];pp[t]=(a[0]+delta,a[1]) if vertical else (a[0],a[1]+delta)
        for i in (j-1,end+1):
            new=abs(pp[i][0]-pp[i+1][0])+abs(pp[i][1]-pp[i+1][1])
            if new<(1 if i==0 or i==len(pts)-2 else self.minseg):return None
            va=(pts[i+1][0]-pts[i][0],pts[i+1][1]-pts[i][1]);vb=(pp[i+1][0]-pp[i][0],pp[i+1][1]-pp[i][1])
            if va[0]*vb[0]+va[1]*vb[1]<=0:return None
        length=sum(abs(a[0]-b[0])+abs(a[1]-b[1]) for a,b in zip(pp,pp[1:]))
        if length>self.maxlen+1e-7:return None
        f=self.floor;m=self.margin;sp=self.spacing;s=self.rows
        changed=range(j-1,end+2)
        boxes=[]
        for i in changed:
            a,b=pp[i],pp[i+1]
            x0,y0,x1,y1=min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1])
            boxes.append((x0,y0,x1,y1))
            if x0<f.bx0+m or x1>f.bx1-m or y0<f.by0+m or y1>f.by1-m:return None
            if any(x0<=q[2] and x1>=q[0] and y0<=q[3] and y1>=q[1] for q in self.keepouts):return None
            dx=np.maximum(0,np.maximum(x0,s[:,0])-np.minimum(x1,s[:,2]));dy=np.maximum(0,np.maximum(y0,s[:,1])-np.minimum(y1,s[:,3]))
            bad=(dx*dx+dy*dy<sp*sp)&~((s[:,4]==k)&((np.abs(s[:,5]-i)<=1)|((s[:,5]>=j-1)&(s[:,5]<=end+1))))
            if bad.any():return None
            walk=thermal.lattice_walk([a,b])
            for u,v in zip(walk,walk[1:]):
                if f.room_at(*u)!=f.room_at(*v):
                    mx,my=(u[0]+v[0])/2,(u[1]+v[1])/2
                    if not any(p[0]-.5<=mx<=p[2]+.5 and p[1]-.5<=my<=p[3]+.5 and (p[0]<=mx<=p[2] or p[1]<=my<=p[3]) for p in self.passages):return None
        for i,a in enumerate(boxes):
            for b in boxes[i+2:]:
                dx=max(0,max(a[0],b[0])-min(a[2],b[2]));dy=max(0,max(a[1],b[1])-min(a[3],b[3]))
                if dx*dx+dy*dy<sp*sp:return None
        return pp

    def propagated_shift(self,k,j,delta,region):
        pts=self.corners[k];vertical=pts[j][0]==pts[j+1][0];move=(delta,0) if vertical else (0,delta)
        left=right=j
        while True:
            changed=False
            for side in (0,1):
                t=left-1 if side==0 else right+1
                a,b=pts[t],pts[t+1]
                vx,vy=b[0]-a[0],b[1]-a[1]
                sign=1 if side==0 else -1;nx,ny=vx+sign*move[0],vy+sign*move[1]
                minimum=1 if t==0 or t==len(pts)-2 else self.minseg
                if abs(nx)+abs(ny)<minimum or vx*nx+vy*ny<=0:
                    if side==0:left-=2
                    else:right+=2
                    if left<1 or right>=len(pts)-2:return None
                    changed=True
            if not changed:break
        if any(not(region[0]<=q[0]<=region[2] and region[1]<=q[1]<=region[3]) for q in pts[left:right+2]):return None
        return self.try_shift(k,left,delta,right)

    def try_replace(self,k,j,end,piece):
        piece=[(int(x),int(y)) for x,y in piece]
        pts=self.corners[k];sp=self.spacing;s=self.rows;f=self.floor;m=self.margin
        if j<=0 or end>=len(pts)-2 or piece[0]!=pts[j] or piece[-1]!=pts[end+1]:return None
        boxes=[]
        for t,(a,b) in enumerate(zip(piece,piece[1:])):
            if a[0]!=b[0] and a[1]!=b[1]:return None
            if abs(a[0]-b[0])+abs(a[1]-b[1])<self.minseg:return None
            x0,y0,x1,y1=min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1]);boxes.append((x0,y0,x1,y1))
            if x0<f.bx0+m or x1>f.bx1-m or y0<f.by0+m or y1>f.by1-m:return None
            if any(x0<=q[2] and x1>=q[0] and y0<=q[3] and y1>=q[1] for q in self.keepouts):return None
            dx=np.maximum(0,np.maximum(x0,s[:,0])-np.minimum(x1,s[:,2]));dy=np.maximum(0,np.maximum(y0,s[:,1])-np.minimum(y1,s[:,3]))
            exempt=(s[:,4]==k)&(((s[:,5]>=j)&(s[:,5]<=end))|((s[:,5]==j-1)&(t==0))|((s[:,5]==end+1)&(t==len(piece)-2)))
            if ((dx*dx+dy*dy<sp*sp)&~exempt).any():return None
            walk=thermal.lattice_walk([a,b])
            for u,v in zip(walk,walk[1:]):
                if f.room_at(*u)!=f.room_at(*v):
                    mx,my=(u[0]+v[0])/2,(u[1]+v[1])/2
                    if not any(q[0]-.5<=mx<=q[2]+.5 and q[1]-.5<=my<=q[3]+.5 and (q[0]<=mx<=q[2] or q[1]<=my<=q[3]) for q in self.passages):return None
        for t,a in enumerate(boxes):
            for b in boxes[t+2:]:
                dx=max(0,max(a[0],b[0])-min(a[2],b[2]));dy=max(0,max(a[1],b[1])-min(a[3],b[3]))
                if dx*dx+dy*dy<sp*sp:return None
        pp=thermal.simplify(pts[:j]+piece+pts[end+2:])
        for a,b,c in zip(pp,pp[1:],pp[2:]):
            if (b[0]-a[0])*(c[0]-b[0])+(b[1]-a[1])*(c[1]-b[1])<0:return None
        if sum(abs(a[0]-b[0])+abs(a[1]-b[1]) for a,b in zip(pp,pp[1:]))>self.maxlen+1e-7:return None
        return pp

    def accept(self,k,corners):
        self.corners[k]=corners;self.paths[k]=thermal.lattice_walk(corners);self.refresh()

    def layout(self):
        g=self.p['grid']
        return {'loops':[{'slot':k,'path':[[round(x*g,8),round(y*g,8)] for x,y in pts]} for k,pts in self.corners.items()]}


def refine_room(plan,floor,loops,heat,paths,ri,deadline):
    # Try small movements of straight runs with the exact local heat model.
    room=plan['rooms'][ri]
    geometry=Geometry(plan,floor,paths)
    if not hasattr(floor,'occupied_masks'):
        xm=floor.xs[None,:]*plan['grid'];ym=floor.ys[:,None]*plan['grid'];zones=[]
        for rj,rr in enumerate(plan['rooms']):
            x0,y0,x1,y1=rr['rect'];zm=rr.get('zone_margin',0)
            zone=(floor.room_of==rj)&(xm>=x0+zm-1e-6)&(xm<=x1-zm+1e-6)&(ym>=y0+zm-1e-6)&(ym<=y1-zm+1e-6)
            for q in plan.get('keepouts',[]):zone&=~((xm>=q[0]-zm-1e-6)&(xm<=q[2]+zm+1e-6)&(ym>=q[1]-zm-1e-6)&(ym<=q[3]+zm+1e-6))
            zones.append(zone if zone.any() else floor.room_of==rj)
        floor.occupied_masks=zones
    # Only pipes near the hottest or coldest occupied cells can reduce the spread.
    probe=LocalModel(plan,floor,loops,heat,ri,loops[0]['slot'])
    region=heat['surface'][probe.y0-floor.by0:probe.y1-floor.by0+1,probe.x0-floor.bx0:probe.x1-floor.bx0+1]
    iy,ix=np.nonzero(probe.zone)
    vals=region[iy,ix]
    hottest=(int(ix[vals.argmax()]+probe.x0),int(iy[vals.argmax()]+probe.y0))
    coldest=(int(ix[vals.argmin()]+probe.x0),int(iy[vals.argmin()]+probe.y0))
    contributions=[];active_slots=[]
    room_units=tuple(round(v/plan['grid']) for v in room['rect'])
    own_loops=[lp for lp in loops if any(floor.room_at(*v)==ri for v in lp['walk'])]
    own_slots={lp['slot'] for lp in own_loops}
    borrow=(room['air_temp']>=19 and (heat['rooms'][ri]['output_w']>room.get('max_w',1e99) or (heat['rooms'][ri]['output_w']<room['required_w'] and len(own_loops)==1 and (len(own_loops[0]['walk'])-1)*plan['grid']>plan['pipe']['max_loop_length']-4)))
    pad=.65 if borrow else .4 if room['air_temp']<19 else .2;halo=round(pad/plan['grid'])
    power_margin=.12 if borrow else .3
    for lp in loops:
        xy=np.array(lp['walk'])
        own=(floor.room_of[xy[:,1]-floor.by0,xy[:,0]-floor.bx0]==ri).any()
        if not own and not (borrow and ((xy[:,0]>=room_units[0]-halo)&(xy[:,0]<=room_units[2]+halo)&(xy[:,1]>=room_units[1]-halo)&(xy[:,1]<=room_units[3]+halo)).any()):continue
        active_slots.append(lp['slot'])
        dh=float(np.min((xy[:,0]-hottest[0])**2+(xy[:,1]-hottest[1])**2))
        dc=float(np.min((xy[:,0]-coldest[0])**2+(xy[:,1]-coldest[1])**2))
        influence=1/(1+dh)+1/(1+dc)
        if borrow:
            rr=heat['rooms'][ri]['output_w'];gap=max(0.,room['required_w']-rr,rr-room.get('max_w',1e99))/max(10.,room.get('max_w',1e99)-room['required_w'])
            influence+=(.15 if gap<.02 else .07)*min(12.,max(0.,plan['pipe']['max_loop_length']-(len(lp['walk'])-1)*plan['grid']))
        if room['air_temp']<19 and (heat['rooms'][ri]['output_w']<room['required_w'] or heat['rooms'][ri]['output_w']>room.get('max_w',1e99)):
            other=set(floor.room_of[xy[:,1]-floor.by0,xy[:,0]-floor.bx0].tolist())-{ri}
            under=heat['rooms'][ri]['output_w']<room['required_w']
            margin=min((heat['rooms'][j]['output_w']-plan['rooms'][j]['required_w'] if under else plan['rooms'][j].get('max_w',1e99)-heat['rooms'][j]['output_w'] for j in other),default=30.)
            headroom=plan['pipe']['max_loop_length']-(len(lp['walk'])-1)*plan['grid'] if under else 30.
            influence+=.1*max(0.,min(headroom,margin*1.5))
        contributions.append((influence,lp['slot']))
    contributions.sort(reverse=True)
    from optimize import greens
    radius=12;kernel=greens(plan,room.get('floor_r',0),radius=radius)
    fft_shape=(next_fast_len(floor.ny+2*radius),next_fast_len(floor.nx+2*radius))
    kernel_fft=rfft2(kernel,s=fft_shape)
    capture_global=None
    if borrow:
        capture_global=irfft2(rfft2((floor.room_of==ri).astype(float),s=fft_shape)*kernel_fft,s=fft_shape)[radius:radius+floor.ny,radius:radius+floor.nx]
        capture_global/=max(1e-12,float(capture_global.max()))
        capture_h=np.pad(np.cumsum(capture_global,axis=1),((0,0),(1,0)));capture_v=np.pad(np.cumsum(capture_global,axis=0),((1,0),(0,0)))
    def capture_line(a,b):
        x,y=a[0]-floor.bx0,a[1]-floor.by0;u,v=b[0]-floor.bx0,b[1]-floor.by0
        value=capture_h[y,max(x,u)+1]-capture_h[y,min(x,u)] if y==v else capture_v[max(y,v)+1,x]-capture_v[min(y,v),x]
        return value-.5*(capture_global[y,x]+capture_global[v,u])
    model=None;cached_state=None
    paired=len(own_loops)==2;idle_visits=0
    if paired:contributions=contributions*6
    for contribution_index,(_,slot) in enumerate(contributions):
        if time.perf_counter()>=deadline:break
        if model is None:model=LocalModel(plan,floor,loops,heat,ri,active_slots if len(active_slots)>1 else slot,pad=pad)
        def evaluate_path(path):
            if len(active_slots)>1:
                selected={k:geometry.paths[k] for k in active_slots};selected[slot]=path
                res=model.evaluate(selected,ret_state=True);res['_waters']=res['_water'];res['_water']=res['_waters'][slot]
                return res
            return model.evaluate(path,ret_state=True)
        if cached_state is None:base=evaluate_path(geometry.paths[slot])
        else:
            base={**cached_state}
            if '_waters' in base:base['_water']=base['_waters'][slot]
        cached_state=base;current_score=score(room,base)
        if os.environ.get('FINE_TRACE'):print('REFINE',ri,slot,base['output_w'],base['zone_spread'],deadline-time.perf_counter(),file=sys.stderr)
        local_bounds=(model.x0,model.y0,model.x1,model.y1)
        old_powers=sum((water_powers(plan,floor,geometry.paths[k],heat['water'][k],outside=local_bounds) for k in active_slots),np.zeros(len(plan['rooms'])))
        old_local=base['_local_outputs'].copy()
        sl=np.s_[model.y0-floor.by0:model.y1-floor.by0+1,model.x0-floor.bx0:model.x1-floor.bx0+1]
        other_limits=[]
        for rj,rr in enumerate(plan['rooms']):
            if rj==ri:continue
            local_sel=model.room_of==rj
            if not local_sel.any():continue
            zone=floor.occupied_masks[rj];local_zone=zone[sl]
            outside_zone=zone.copy();outside_zone[sl]=False;zout=heat['surface'][outside_zone]
            outside_sel=floor.room_of==rj;outside_sel[sl]=False;sout=heat['surface'][outside_sel]
            zmin=float(zout.min()) if len(zout) else math.inf;zmax=float(zout.max()) if len(zout) else -math.inf
            smax=float(sout.max()) if len(sout) else -math.inf
            rect=rr['rect'];capacity=max(1.,(rect[2]-rect[0])*(rect[3]-rect[1])/(rr.get('floor_r',0)+1/plan['physics']['h_core']))
            other_limits.append((rj,local_sel,local_zone,zmin,zmax,smax,capacity,max(rr['max_spread'],heat['rooms'][rj]['zone_spread']+(.02*rr['max_spread'] if heat['rooms'][rj]['zone_spread']>rr['max_spread'] else 0.)),max(rr['max_surface'],heat['rooms'][rj]['max_surface']+(.04 if heat['rooms'][rj]['max_surface']>rr['max_surface'] else 0.))))
        hottest,coldest=base['hot'],base['cold']
        # Rank proposals by their distance from the occupied-zone extrema.
        accepted_res=None
        slot_deadline=deadline
        slow_moves=0
        for sweep in range(2 if paired else 20):
            if time.perf_counter()>=slot_deadline:break
            low_goal=room['required_w']+(power_margin if borrow else 0.)
            high_goal=room.get('max_w',1e99)-(power_margin if borrow else 0.)
            power_bad=base['output_w']<low_goal or base['output_w']>high_goal
            power_error=max(0.,low_goal-base['output_w'],base['output_w']-high_goal)/max(10.,room.get('max_w',1e99)-room['required_w'])
            brief=(power_bad and power_error<.10) or (paired and base['zone_spread']<room['max_spread']*1.12)
            deltas=(-1,1,-2,2,-4,4,-6,6) if power_bad else (-1,1,-2,2)
            best=None
            pts=geometry.corners[slot]
            candidates=[]
            generation_stop=time.perf_counter()+max(.01,min(.05,(deadline-time.perf_counter())*.12))
            if base['max_surface']>room['max_surface']:
                zy,zx=np.nonzero(model.sel);tt=base['_surface'][model.sel];ii=int(tt.argmax());hottest=(int(zx[ii]+model.x0),int(zy[ii]+model.y0))
            r=room_units
            if borrow:r=(r[0]-halo,r[1]-halo,r[2]+halo,r[3]+halo)
            inside=[all(r[0]<=v[0]<=r[2] and r[1]<=v[1]<=r[3] for v in (a,b)) for a,b in zip(pts,pts[1:])]
            start=None
            for j,ok in enumerate(inside+[False]):
                if ok and start is None:start=j
                if not ok and start is not None:
                    end=j-1
                    if end-start>=2:
                        for delta in deltas:
                            pp=geometry.try_shift(slot,start,delta,end)
                            if pp is not None:candidates.append((-1.,pp))
                    start=None
            def proximity(j):
                a,b=pts[j],pts[j+1];ax,bx=sorted((a[0],b[0]));ay,by=sorted((a[1],b[1]))
                return min(max(0,ax-q[0],q[0]-bx)**2+max(0,ay-q[1],q[1]-by)**2 for q in (hottest,coldest))
            indices=range(1,len(pts)-2)
            if not power_bad:indices=sorted(indices,key=proximity)
            for number,j in enumerate(indices):
                if number%4==0 and time.perf_counter()>generation_stop:break
                for count in (1,3,5,7):
                    end=j+count-1
                    if end>=len(pts)-2:continue
                    chain=pts[j:end+2]
                    if not all(r[0]<=v[0]<=r[2] and r[1]<=v[1]<=r[3] for v in chain):continue
                    ax=min(v[0] for v in chain);ay=min(v[1] for v in chain)
                    bx=max(v[0] for v in chain);by=max(v[1] for v in chain)
                    if bx<r[0] or ax>r[2] or by<r[1] or ay>r[3]:continue
                    if count==1 and bx-ax+by-ay<3:continue
                    def dist(p):
                        dx=max(0,ax-p[0],p[0]-bx);dy=max(0,ay-p[1],p[1]-by)
                        return dx*dx+dy*dy
                    rank=min(dist(hottest),dist(coldest))
                    if rank>225 and not power_bad:continue
                    for delta in deltas:
                        pp=geometry.try_shift(slot,j,delta,end)
                        if pp is not None:candidates.append((rank+.01*abs(delta)+.001*count,pp))
            expanded=(r[0]-3,r[1]-3,r[2]+3,r[3]+3)
            for j in range(1,len(pts)-2):
                if time.perf_counter()>generation_stop+.03:break
                a,b=pts[j],pts[j+1]
                if not all(r[0]<=q[0]<=r[2] and r[1]<=q[1]<=r[3] for q in (a,b)):continue
                for delta in deltas:
                    pp=geometry.propagated_shift(slot,j,delta,expanded)
                    if pp is not None:candidates.append((0.,pp))
            lengths=np.array([abs(a[0]-b[0])+abs(a[1]-b[1]) for a,b in zip(pts,pts[1:])],dtype=int)
            cum=np.r_[0,np.cumsum(lengths)];water=base['_water']
            xy=np.asarray(geometry.paths[slot][1:],dtype=int);sel=floor.room_of[xy[:,1]-floor.by0,xy[:,0]-floor.bx0]==ri
            twmean=float(water[sel].mean()) if sel.any() else float(water.mean())
            direction=1 if base['output_w']<low_goal else -1 if base['output_w']>high_goal else 0
            room_after=np.cumsum(sel[::-1])[::-1]/max(1,int(sel.sum()))
            mcp=plan['physics']['flow_constant']/(len(water)*plan['grid']+plan['physics']['flow_offset'])
            ua=water_powers(plan,floor,geometry.paths[slot],water)[ri]*.8/max(2.,twmean-room['air_temp'])
            def hydraulic_bonus(index,change,point):
                if not borrow or not len(water):return 0.
                ix=min(len(water)-1,max(0,int(index)-1));x,y=point
                if model.x0<=x<=model.x1 and model.y0<=y<=model.y1:tb=float(base['_bottom'][y-model.y0,x-model.x0])
                else:tb=float(heat['bottom'][y-floor.by0,x-floor.bx0])
                heat_effect=.55*(float(water[ix])-tb)*change*ua/mcp*room_after[ix]
                flow_effect=ua*(plan['physics']['supply_temp']-twmean)/(len(water)*plan['grid']+plan['physics']['flow_offset'])*change/plan['physics']['g_pipe']
                return direction*(heat_effect+flow_effect)
            def strict(q):return r[0]<q[0]<r[2] and r[1]<q[1]<r[3]
            def segment_rank(j):
                a,b=pts[j],pts[j+1];mx,my=(a[0]+b[0])//2,(a[1]+b[1])//2
                if power_bad:
                    ix=min(len(water)-1,max(0,int((cum[j]+cum[j+1])*.5)-1))
                    return -(float(water[ix])-float(heat['bottom'][my-floor.by0,mx-floor.bx0]))
                x0,x1=sorted((a[0],b[0]));y0,y1=sorted((a[1],b[1]))
                return min(max(0,x0-q[0],q[0]-x1)**2+max(0,y0-q[1],q[1]-y1)**2 for q in (hottest,coldest))
            generation_stop=time.perf_counter()+max(.015,min(.025 if brief and len(candidates)>12 else .10,(deadline-time.perf_counter())*.18))
            for j in sorted(range(1,len(pts)-2),key=segment_rank):
                if time.perf_counter()>generation_stop:break
                for count in (2,3,4,5,6,8):
                    if time.perf_counter()>generation_stop:break
                    end=j+count-1
                    if end>=len(pts)-2:continue
                    chain=pts[j:end+2]
                    if not all(expanded[0]<=q[0]<=expanded[2] and expanded[1]<=q[1]<=expanded[3] for q in chain):continue
                    a,b=chain[0],chain[-1]
                    if a==b:continue
                    if a[0]==b[0] or a[1]==b[1]:options=[[a,b]]
                    else:options=[[a,(a[0],b[1]),b],[a,(b[0],a[1]),b]]
                    for piece in options:
                        pp=geometry.try_replace(slot,j,end,piece)
                        if pp is not None and pp!=pts:
                            new_length=sum(abs(q[0]-v[0])+abs(q[1]-v[1]) for q,v in zip(piece,piece[1:]))
                            change=new_length-int(cum[end+1]-cum[j]);point=((a[0]+b[0])//2,(a[1]+b[1])//2)
                            candidates.append((0.,pp,hydraulic_bonus((cum[j]+cum[end+1])*.5,change,point)))
            generation_stop=time.perf_counter()+max(.015,min(.03 if brief and len(candidates)>12 else .10,(deadline-time.perf_counter())*.20))
            for j in sorted(range(1,len(pts)-2),key=segment_rank):
                a,b=pts[j],pts[j+1]
                if time.perf_counter()>generation_stop:break
                if j==0 or j>=len(pts)-2 or not strict(a) or not strict(b):continue
                length=lengths[j];dx=(b[0]>a[0])-(b[0]<a[0]);dy=(b[1]>a[1])-(b[1]<a[1])
                if not power_bad:
                    dd=min((max(0,min(a[0],b[0])-q[0],q[0]-max(a[0],b[0])))**2+(max(0,min(a[1],b[1])-q[1],q[1]-max(a[1],b[1])))**2 for q in (hottest,coldest))
                    if dd>144:continue
                for width in (2,4,6,8,12):
                    if time.perf_counter()>generation_stop:break
                    if length<width+4:continue
                    anchors={2,length-width-2,(length-width)//2}
                    if power_bad:anchors.update(range(2,length-width-1,6))
                    else:
                        for q in (hottest,coldest):
                            t=(q[0]-a[0])*dx+(q[1]-a[1])*dy;anchors.add(max(2,min(length-width-2,round(t-width*.5))))
                    for t in sorted(anchors):
                        if time.perf_counter()>generation_stop:break
                        u=(a[0]+dx*t,a[1]+dy*t);v=(u[0]+dx*width,u[1]+dy*width)
                        idx=min(len(water)-1,max(0,int(cum[j]+t+width*.5)-1));tw=float(water[idx])
                        if idx<model.sleeve+2 or idx>len(water)-model.sleeve-2:continue
                        for depth in (2,-2,4,-4,6,-6,8,-8,10,-10):
                            if cum[-1]+2*abs(depth)>geometry.maxlen+1e-7:continue
                            nu=(u[0]-dy*depth,u[1]+dx*depth);nv=(v[0]-dy*depth,v[1]+dx*depth)
                            if not strict(nu) or not strict(nv):continue
                            pp=geometry.try_replace(slot,j,j,[a,u,nu,nv,v,b])
                            if pp is not None:
                                bonus=hydraulic_bonus(idx+1,2*abs(depth),nu) if borrow else -direction*(tw-twmean)*2*abs(depth)
                                if borrow:
                                    capture=float(np.mean([capture_global[q[1]-floor.by0,q[0]-floor.bx0] for q in (u,nu,nv,v)]))
                                    bonus-=direction*(tw-twmean)*2*abs(depth)*capture
                                candidates.append((0.,pp,bonus))
                if j+2<len(pts)-2 and all(strict(q) for q in pts[j:j+4]):
                    c=pts[j+3]
                    if a[0]==c[0] or a[1]==c[1]:
                        removed=int(cum[j+3]-cum[j]-abs(a[0]-c[0])-abs(a[1]-c[1]))
                        if removed>0 and (not power_bad or removed*plan['grid']<max(1.,abs(base['output_w']-room['required_w'] if direction==1 else base['output_w']-room.get('max_w',1e99))*.6)):
                            pp=geometry.try_replace(slot,j,j+2,[a,c])
                            if pp is not None:
                                idx=min(len(water)-1,max(0,int((cum[j]+cum[j+3])*.5)-1));tw=float(water[idx]);bonus=hydraulic_bonus(idx+1,-removed,a) if borrow else direction*(tw-twmean)*removed;candidates.append((0.,pp,bonus))
            pressure=np.zeros((floor.ny,floor.nx))
            zone=model.sel if base['max_surface']>room['max_surface'] else model.zone
            temperatures=base['_surface'][zone]
            hot=np.exp(16*(temperatures-temperatures.max()));hot/=hot.sum()
            cold=np.exp(16*(temperatures.min()-temperatures));cold/=cold.sum()
            local=pressure[model.y0-floor.by0:model.y1-floor.by0+1,model.x0-floor.bx0:model.x1-floor.bx0+1]
            local[zone]=hot-cold
            potential=irfft2(rfft2(pressure,s=fft_shape)*kernel_fft,s=fft_shape)[radius:radius+floor.ny,radius:radius+floor.nx]
            potential/=max(1e-12,float(np.abs(potential).max()))
            use_prediction=(not power_bad and len(own_loops)==2)
            if not use_prediction:
                ax=np.arange(floor.nx)[None,:]+floor.bx0;ay=np.arange(floor.ny)[:,None]+floor.by0
                potential=np.exp(-np.hypot(ax-hottest[0],ay-hottest[1])/6.)-np.exp(-np.hypot(ax-coldest[0],ay-coldest[1])/6.)
            if power_bad:
                bottom=heat['bottom'].copy();bottom[model.y0-floor.by0:model.y1-floor.by0+1,model.x0-floor.bx0:model.x1-floor.bx0+1]=base['_bottom']
                potential=direction*(bottom-twmean)+potential*.5
                if borrow:potential*=capture_global
            hp=np.pad(np.cumsum(potential,axis=1),((0,0),(1,0)));vp=np.pad(np.cumsum(potential,axis=0),((1,0),(0,0)))
            def line_value(a,b):
                x,y=a[0]-floor.bx0,a[1]-floor.by0;u,v=b[0]-floor.bx0,b[1]-floor.by0
                if y==v:val=hp[y,max(x,u)+1]-hp[y,min(x,u)]
                else:val=vp[max(y,v)+1,x]-vp[min(y,v),x]
                return val-.5*(potential[y,x]+potential[v,u])
            old_values=[line_value(a,b) for a,b in zip(pts,pts[1:])]
            old_prefix=np.r_[0,np.cumsum(old_values)]
            old_capture_prefix=np.r_[0,np.cumsum([capture_line(a,b) for a,b in zip(pts,pts[1:])])] if borrow else None
            def proposal_rank(item):
                old_rank,pp=item[:2];bonus=item[2] if len(item)>2 else 0.
                if len(pp)!=len(pts):
                    effect=sum(line_value(a,b) for a,b in zip(pp,pp[1:]))-old_prefix[-1]
                    return effect+bonus+old_rank*.0001
                js=[j for j,(a,b) in enumerate(zip(pts,pp)) if a!=b]
                if not js:return 1e99
                j=max(0,js[0]-1);end=min(len(pts)-1,js[-1]+1)
                effect=sum(line_value(a,b) for a,b in zip(pp[j:end],pp[j+1:end+1]))-(old_prefix[end]-old_prefix[j])
                if borrow and direction:
                    change=sum(capture_line(a,b) for a,b in zip(pp[j:end],pp[j+1:end+1]))-(old_capture_prefix[end]-old_capture_prefix[j])
                    local_water=float(water[max(0,cum[j]-1):min(len(water),cum[end]+1)].mean())
                    effect-=direction*(local_water-twmean)*change
                return effect+bonus+old_rank*.0001
            # Rank small displacements by a full surface estimate, so that moving
            # a hot spot cannot simply create a new one beside it.
            mcp=plan['physics']['flow_constant']/((len(water))*plan['grid']+plan['physics']['flow_offset'])
            qs=mcp*(np.r_[plan['physics']['supply_temp'],water[:-1]]-water)
            def predicted_score(pp,changed):
                left=max(0,changed[0]-1);right=min(len(pts)-1,changed[-1]+1)
                old_walk=thermal.lattice_walk(pts[left:right+1])[1:]
                new_walk=thermal.lattice_walk(pp[left:right+1])[1:]
                oldq=qs[cum[left]:cum[right]]
                if not len(oldq) or not len(new_walk):return 1e9
                newq=np.interp(np.linspace(0,len(oldq)-1,len(new_walk)),np.arange(len(oldq)),oldq)
                sources={}
                for q,w in zip(old_walk,oldq):sources[q]=sources.get(q,0.)-float(w)
                for q,w in zip(new_walk,newq):sources[q]=sources.get(q,0.)+float(w)
                surf=base['_surface'].copy()
                for (x,y),w in sources.items():
                    if abs(w)<1e-8:continue
                    x-=model.x0;y-=model.y0
                    x0=max(0,x-radius);x1=min(model.nx,x+radius+1);y0=max(0,y-radius);y1=min(model.ny,y+radius+1)
                    if x0<x1 and y0<y1:surf[y0:y1,x0:x1]+=.72*w*kernel[y0-y+radius:y1-y+radius,x0-x+radius:x1-x+radius]
                z=surf[model.zone]
                rr={**base,'zone_spread':float(z.max()-z.min()),'max_surface':float(surf[model.sel].max()),'output_w':float((model.groom[model.sel]*(surf[model.sel]-room['air_temp'])).sum())}
                return score(room,rr)
            def trust_rank(item):
                value=proposal_rank(item);pp=item[1]
                power_error=max(0.,room['required_w']-base['output_w'],base['output_w']-room.get('max_w',1e99))/max(10.,room.get('max_w',1e99)-room['required_w'])
                if not use_prediction and not (power_bad and power_error<.12):return (0,value)
                if len(pp)!=len(pts):return (2,value)
                changed=[j for j,(a,b) in enumerate(zip(pts,pp)) if a!=b]
                extent=sum(abs(pts[j][0]-pp[j][0])+abs(pts[j][1]-pp[j][1]) for j in changed)
                if extent<=4 and not power_bad:return (0,predicted_score(pp,changed))
                return (0 if power_bad and extent<=4 else 1,value)
            ranked=[(trust_rank(c),c) for c in candidates];ranked.sort(key=lambda x:x[0])
            groups=[[],[],[]];seen_candidates=set()
            for rank,candidate in ranked:
                signature=tuple(candidate[1])
                if signature in seen_candidates:continue
                seen_candidates.add(signature);groups[rank[0]].append(candidate)
            candidates=[]
            schedule=((0,4),(2,4),(1,2)) if borrow and slot not in own_slots and power_bad else ((0,6),(1,3),(2,3))
            while any(groups):
                for index,count in schedule:
                    group=groups[index];candidates.extend(group[:count]);del group[:count]
            tested_paths=set();tested_count=0
            for candidate in candidates:
                pp=candidate[1]
                signature=tuple(pp)
                if signature in tested_paths:continue
                tested_paths.add(signature)
                tested_count+=1
                trial_limit=(4 if power_error<.01 and slot in own_slots else 12 if slot not in own_slots else 8) if borrow and power_error<.02 and base['zone_spread']<room['max_spread']-.05 else 24 if int(model.sel.sum())<3500 else 18
                if tested_count>trial_limit or time.perf_counter()>=slot_deadline:break
                path=thermal.lattice_walk(pp);res=evaluate_path(path);s=score(room,res)
                waters=res.get('_waters',{slot:res['_water']})
                powers=sum((water_powers(plan,floor,path if k==slot else geometry.paths[k],waters[k],outside=local_bounds) for k in active_slots),np.zeros(len(plan['rooms'])))
                delta=res['_local_outputs']-old_local+.8*(powers-old_powers)
                res['_other_outputs']=[q['output_w']+delta[j] for j,q in enumerate(heat['rooms'])]
                unsafe=False
                for rj,rr in enumerate(plan['rooms']):
                    if rj==ri:continue
                    old=heat['rooms'][rj]['output_w'];predicted=old+delta[rj]
                    tol=max(.35,.02*(rr.get('max_w',old*1.3)-rr['required_w'])) if old<rr['required_w'] or old>rr.get('max_w',1e99) else .05
                    if predicted<min(old,rr['required_w']+.5)-tol or predicted>max(old,rr.get('max_w',1e99)-.5)+tol:unsafe=True;break
                if not unsafe:
                    for rj,local_sel,local_zone,zmin,zmax,smax,capacity,spread_limit,surface_limit in other_limits:
                        shift=.8*(powers[rj]-old_powers[rj])/capacity
                        peak=max(smax+shift,float(res['_surface'][local_sel].max()))
                        if local_zone.any():
                            zz=res['_surface'][local_zone];spread=max(zmax+shift,float(zz.max()))-min(zmin+shift,float(zz.min()))
                        else:spread=zmax-zmin
                        if peak>surface_limit+1e-5 or spread>spread_limit+1e-4:unsafe=True;break
                if os.environ.get('FINE_TRACE'):print('TRY',slot,sweep,round(res['output_w'],3),round(res['zone_spread'],4),round(s,3),unsafe,file=sys.stderr)
                if unsafe:continue
                if s<current_score-1e-5 and (best is None or s<best[0]):
                    best=(s,pp,res)
                    break
            if best is None:break
            old_score=current_score;current_score,pp,res=best;geometry.accept(slot,pp);accepted_res=res
            gain=(old_score-current_score)/max(1.,old_score)
            slow_moves=slow_moves+1 if gain<.001 else 0
            hottest,coldest=res['hot'],res['cold'];base=res
            if slow_moves>=2:break
            if res['zone_spread']<room['max_spread']-.01 and room['required_w']+power_margin<res['output_w']<room.get('max_w',1e99)-power_margin and res['max_surface']<room['max_surface']-.03:break
        if accepted_res is not None:
            res=accepted_res;cached_state=res
            sl=np.s_[model.y0-floor.by0:model.y1-floor.by0+1,model.x0-floor.bx0:model.x1-floor.bx0+1]
            heat['surface'][sl]=res['_surface'];heat['bottom'][sl]=res['_bottom'];heat['water'].update(res.get('_waters',{slot:res['_water']}))
            heat['rooms'][ri].update({key:res[key] for key in ('output_w','zone_spread','max_surface')})
            for rj,out in enumerate(res['_other_outputs']):
                if rj!=ri:heat['rooms'][rj]['output_w']=float(out)
            for j,old in enumerate(loops):
                if old['slot']==slot:
                    fine=geometry.paths[slot];corners=thermal.simplify(fine)
                    loops[j]={**old,'walk':fine,'points':corners,'segments':list(zip(corners,corners[1:])),'length_units':len(fine)-1}
            if res['zone_spread']<room['max_spread']-.01 and room['required_w']+power_margin<res['output_w']<room.get('max_w',1e99)-power_margin and res['max_surface']<room['max_surface']-.03:break
            idle_visits=0
        else:
            idle_visits+=1
            if paired and idle_visits>=len(active_slots):break
    return geometry.paths,geometry.layout()
