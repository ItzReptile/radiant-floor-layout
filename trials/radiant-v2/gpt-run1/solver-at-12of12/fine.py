import time,itertools
import numpy as np
from scipy.ndimage import gaussian_filter
from model import lattice_walk,check_geometry

class FineSearch:
    def __init__(self,opt):
        self.o=opt;self.p=opt.p;self.th=opt.th;f=self.f=opt.th.floor;g=self.g=self.p['grid']
        self.spacing=round(self.p['pipe']['min_spacing']/g);self.minseg=round(self.p['pipe']['min_segment']/g)
        self.maxlen=self.p['pipe']['max_loop_length']/g
        self.nx=f.nx;self.ny=f.ny
        self.sels=[np.flatnonzero(s) for s in self.th.sels];self.zones=[np.flatnonzero(z) for z in self.th.zones]
        self.sigma=.14/g;self.response_gain=1.2/g
        self.nrooms=len(self.p['rooms'])
        self.room_masks=np.stack([self.th.rflat.reshape(self.ny,self.nx)==i for i in range(self.nrooms)])
        self.zone_masks=np.stack([z.reshape(self.ny,self.nx) for z in self.th.zones])
        self.zone_owner=np.full((self.ny,self.nx),-1,int)
        for i,z in enumerate(self.zone_masks):self.zone_owner[z]=i
        self.gr=self.th.gr.reshape(self.ny,self.nx)
        x=f.xs[None,:];y=f.ys[:,None];margin=round(self.p['edge_margin']/g)
        valid=(x>=f.bx0+margin)&(x<=f.bx1-margin)&(y>=f.by0+margin)&(y<=f.by1-margin)
        for r in self.p.get('keepouts',[]):
            a,b,c,d=[round(t/g) for t in r]
            valid &= ~((x>=a)&(x<=c)&(y>=b)&(y<=d))
        hs=f.room_of[:,:-1]==f.room_of[:,1:];vs=f.room_of[:-1,:]==f.room_of[1:,:]
        hp=np.zeros(hs.shape,bool);vp=np.zeros(vs.shape,bool)
        def passage(mx,my,r):
            a,b,c,d=[t/g for t in r]
            return (mx>=a-.5)&(mx<=c+.5)&(my>=b-.5)&(my<=d+.5)&((mx>=a)&(mx<=c)|(my>=b)&(my<=d))
        for r in self.p['passages']:
            hp |= passage(f.xs[None,:-1]+.5,f.ys[:,None],r)
            vp |= passage(f.xs[None,:],f.ys[:-1,None]+.5,r)
        self.hlegal=valid[:,:-1]&valid[:,1:]&(hs|hp)
        self.vlegal=valid[:-1,:]&valid[1:,:]&(vs|vp)

    def legal(self,a,b):
        f=self.f
        x,y=int(a[0])-f.bx0,int(a[1])-f.by0;X,Y=int(b[0])-f.bx0,int(b[1])-f.by0
        if min(x,X)<0 or max(x,X)>=self.nx or min(y,Y)<0 or max(y,Y)>=self.ny:return False
        if y==Y:return bool(self.hlegal[y,min(x,X):max(x,X)].all())
        return bool(self.vlegal[min(y,Y):max(y,Y),x].all())

    def prepare(self,layout):
        paths=[];rows=[];offsets=[];lengths=[]
        for k,l in enumerate(layout['loops']):
            pts=np.rint(np.array(l['path'])/self.g).astype(int);paths.append(pts);offsets.append(len(rows));ln=0
            for j,(a,b) in enumerate(zip(pts,pts[1:])):
                rows.append([min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1]),k,j]);ln+=int(abs(b-a).sum())
            lengths.append(ln)
        return paths,np.array(rows,int),offsets,lengths

    def cells(self,pts):
        w=lattice_walk([tuple(int(v) for v in x) for x in pts])[1:]
        return np.array([(y-self.f.by0)*self.nx+x-self.f.bx0 for x,y in w],int)

    def response_patch(self,oldcells,newcells):
        both=np.r_[oldcells,newcells];ys=both//self.nx;xs=both%self.nx
        radius=int(np.ceil(4*self.sigma))+1
        x0=max(0,int(xs.min())-radius);x1=min(self.nx,int(xs.max())+radius+1)
        y0=max(0,int(ys.min())-radius);y1=min(self.ny,int(ys.max())+radius+1)
        w=x1-x0;h=y1-y0
        new=(newcells//self.nx-y0)*w+newcells%self.nx-x0
        old=(oldcells//self.nx-y0)*w+oldcells%self.nx-x0
        load=np.bincount(new,minlength=h*w)-np.bincount(old,minlength=h*w)
        local=gaussian_filter(load.reshape(h,w).astype(float),self.sigma,mode='reflect')*self.response_gain
        return local,(x0,y0,x1,y1)

    def response(self,oldcells,newcells):
        local,(x0,y0,x1,y1)=self.response_patch(oldcells,newcells)
        delta=np.zeros((self.ny,self.nx));delta[y0:y1,x0:x1]=local
        return delta.ravel()

    def limits(self,surface,masks,minimum=False):
        initial=np.inf if minimum else -np.inf
        op=np.minimum if minimum else np.maximum
        reduce=np.min if minimum else np.max
        values=np.where(masks,surface[None,:,:],initial)
        rows=reduce(values,axis=2);cols=reduce(values,axis=1)
        pad=np.full((self.nrooms,1),initial)
        return (np.c_[pad,op.accumulate(rows,axis=1)],
                np.c_[op.accumulate(rows[:,::-1],axis=1)[:,::-1],pad],
                np.c_[pad,op.accumulate(cols,axis=1)],
                np.c_[op.accumulate(cols[:,::-1],axis=1)[:,::-1],pad])

    def prediction_limits(self,surface):
        return (self.limits(surface,self.zone_masks,True),self.limits(surface,self.zone_masks),self.limits(surface,self.room_masks))

    def predict_patch(self,surface,local,bounds,limits):
        x0,y0,x1,y1=bounds;sl=np.s_[y0:y1,x0:x1]
        pv=(surface[sl]+local).ravel();owners=self.f.room_of[sl].ravel();zones=self.zone_owner[sl].ravel()
        output=np.bincount(owners,weights=(local*self.gr[sl]).ravel(),minlength=self.nrooms)
        peak=np.full(self.nrooms,-np.inf);np.maximum.at(peak,owners,pv)
        zmin=np.full(self.nrooms,np.inf);zmax=np.full(self.nrooms,-np.inf);valid=zones>=0
        np.minimum.at(zmin,zones[valid],pv[valid]);np.maximum.at(zmax,zones[valid],pv[valid])
        def outside(lim,op):
            return op.reduce([lim[0][:,y0],lim[1][:,y1],lim[2][:,x0],lim[3][:,x1]])
        zmin=np.minimum(zmin,outside(limits[0],np.minimum));zmax=np.maximum(zmax,outside(limits[1],np.maximum))
        peak=np.maximum(peak,outside(limits[2],np.maximum))
        return output,zmax-zmin,peak

    def run(self,layout,res,seconds=8.):
        opt=self.o;deadline=min(opt.deadline-.5,time.monotonic()+seconds)
        stall=0
        for iteration in range(80):
            if time.monotonic()>deadline-.25 or opt.passed(res):break
            paths,rows,offsets,lengths=self.prepare(layout)
            current_score=sum(opt.score_room(i,r) for i,r in enumerate(res['rooms']))
            failing=[i for i,r in enumerate(res['rooms']) if not opt.passed_room(i,r)]
            active=failing
            if current_score>.2 and len(failing)>1:
                contributions=[(opt.score_room(i,res['rooms'][i]),i) for i in failing]
                worst=max(contributions)
                if worst[0]>.65*sum(v for v,i in contributions):active=[worst[1]]
            dominant=len(active)<len(failing)
            surface=res['surface'];flat=surface.ravel();n=flat.size
            counts=[]
            for pts in paths:
                c=self.cells(pts);counts.append(np.bincount(self.th.rflat[c],minlength=len(self.p['rooms'])))
            counts=np.array(counts,float);fractions=counts/np.maximum(1.,counts.sum(axis=0))
            limits=self.prediction_limits(surface)
            options=[]
            for k,pts in enumerate(paths):
                for j in range(1,len(pts)-2):
                    if time.monotonic()>deadline-.2:return layout,res
                    b,c=pts[j],pts[j+1];axis=1 if b[1]==c[1] else 0
                    # Only circuits touching a failed constraint need edits.
                    related=any(fractions[k,ri]>.015 for ri in active)
                    if not related:continue
                    ids=np.array([j-1,j,j+1]);globalids=offsets[k]+ids
                    oldpieces=pts[j-1:j+3];oldcells=self.cells(oldpieces)
                    oldlens=np.abs(np.diff(oldpieces,axis=0)).sum(axis=1)
                    mx,my=np.rint((b+c)*.5).astype(int)
                    room=int(self.f.room_of[my-self.f.by0,mx-self.f.bx0])
                    rp0=self.p['rooms'][room];rr0=res['rooms'][room]
                    heat_bad=not (rp0['required_w']<=rr0['output_w']<=rp0.get('max_w',float('inf')))
                    directions=(-6,-4,-2,-1,1,2,4,6) if heat_bad else (-2,-1,1,2)
                    for direction in directions:
                        cand=pts.copy();cand[j:j+2,axis]+=direction
                        pieces=cand[j-1:j+3];lens=np.abs(np.diff(pieces,axis=0)).sum(axis=1)
                        if (lens==0).any():continue
                        if any(lens[t]<self.minseg for t,index in enumerate(ids) if 0<index<len(pts)-2):continue
                        delta_units=int(lens.sum()-oldlens.sum());length=lengths[k]+delta_units
                        if length>self.maxlen+1e-6:continue
                        if not all(self.legal(a,b) for a,b in zip(pieces,pieces[1:])):continue
                        cs=np.c_[np.minimum(pieces[:-1],pieces[1:]),np.maximum(pieces[:-1],pieces[1:]),np.full(3,k),ids]
                        segs=rows.copy();segs[globalids]=cs
                        dx=np.maximum(0,np.maximum(cs[:,None,0],segs[None,:,0])-np.minimum(cs[:,None,2],segs[None,:,2]))
                        dy=np.maximum(0,np.maximum(cs[:,None,1],segs[None,:,1])-np.minimum(cs[:,None,3],segs[None,:,3]))
                        exempt=(cs[:,None,4]==segs[None,:,4])&(abs(cs[:,None,5]-segs[None,:,5])<=1)
                        if ((dx*dx+dy*dy<self.spacing*self.spacing)&~exempt).any():continue
                        newcells=self.cells(pieces)
                        patch,bounds=self.response_patch(oldcells,newcells);ld=delta_units*self.g
                        outputs,spreads,peaks=self.predict_patch(surface,patch,bounds,limits)
                        prediction=0.
                        for ri,(rp,rr,sel,zone) in enumerate(zip(self.p['rooms'],res['rooms'],self.sels,self.zones)):
                            flow=1.-.008*ld*fractions[k,ri]
                            rpred=rr.copy();rpred['output_w']+=float(outputs[ri])-rr['output_w']*.012*ld*fractions[k,ri]
                            rpred['zone_spread']=float(spreads[ri]*flow)
                            rpred['max_surface']=float(rp['air_temp']+(peaks[ri]-rp['air_temp'])*flow)
                            prediction+=opt.score_room(ri,rpred)
                        options.append((prediction,k,j,direction,cand,ld))
            options.sort(key=lambda q:q[0])
            failed_heat=[i for i,(rp,rr) in enumerate(zip(self.p['rooms'],res['rooms'])) if not (rp['required_w']<=rr['output_w']<=rp.get('max_w',float('inf')))]
            failed_rooms=[i for i,rr in enumerate(res['rooms']) if not opt.passed_room(i,rr)]
            hall=opt.d.hall;hrp=self.p['rooms'][hall];hrr=res['rooms'][hall]
            hall_pairs=hrr['zone_spread']>hrp['max_spread'] and hrr['output_w']>.95*hrp.get('max_w',1e9)
            coupled=((len(failed_heat)>=2 or (current_score<1. and bool(failed_heat) and len(failed_rooms)>=2)) and not dominant) or hall_pairs
            def changed_room(option):
                _,k,j,_,_=option[:5];x,y=np.rint((paths[k][j]+paths[k][j+1])*.5).astype(int)
                return int(self.f.room_of[y-self.f.by0,x-self.f.bx0])
            selected=options[:(60 if stall else 20)]
            if coupled and not stall:
                columns=[options[:8]]
                for ri in failed_rooms:columns.append([q for q in options if changed_room(q)==ri][:8])
                if hall_pairs:
                    columns.append([q for q in options if changed_room(q)==hall and q[5]>0][:8])
                    columns.append([q for q in options if changed_room(q)==hall and q[5]<0][:8])
                selected=[];seen=set()
                for row in itertools.zip_longest(*columns):
                    for q in row:
                        if q is None:continue
                        key=q[1],q[2],q[3]
                        if key not in seen:selected.append(q);seen.add(key)
            best=None;bestscore=current_score;tested=0;measured=[]
            for option in selected:
                prediction,k,j,direction,cand=option[:5]
                if time.monotonic()>deadline-.15:break
                trial={'loops':list(layout['loops'])}
                trial['loops'][k]=dict(slot=layout['loops'][k]['slot'],path=np.round(cand*self.g,8).tolist())
                rr=opt.evaluate(layout=trial)
                if rr is None:continue
                tested+=1;score=sum(opt.score_room(i,r) for i,r in enumerate(rr['rooms']))
                measured.append((option,trial,rr))
                if opt.passed(rr) or score<bestscore:
                    bestscore=score;best=(trial,rr)
                    if opt.passed(rr):break
                if tested>=(max(7,2*len(failed_rooms)+3) if coupled else 5) and best is not None and bestscore<current_score*.94:break
            # A change that cools one room can warm the next room on the same
            # circuit.  Pair measured edits when both constraints need attention.
            if coupled and len(measured)>=2 and time.monotonic()<deadline-.4 and not (best is not None and opt.passed(best[1])):
                pairs=[]
                for ia,(qa,la,ra) in enumerate(measured):
                    for qb,lb,rb in measured[ia+1:]:
                        if qa[1]==qb[1] and abs(qa[2]-qb[2])<=2:continue
                        if changed_room(qa)==changed_room(qb) and not (hall_pairs and changed_room(qa)==hall):continue
                        sp=ra['surface'].ravel()+rb['surface'].ravel()-flat
                        predicted=[]
                        for ri,(rp,r0,xa,xb,sel,zone) in enumerate(zip(self.p['rooms'],res['rooms'],ra['rooms'],rb['rooms'],self.sels,self.zones)):
                            pr=r0.copy();pr['output_w']=xa['output_w']+xb['output_w']-r0['output_w'];pr['zone_spread']=float(sp[zone].max()-sp[zone].min());pr['max_surface']=float(sp[sel].max());predicted.append(pr)
                        score=sum(opt.score_room(i,r) for i,r in enumerate(predicted))
                        if score<bestscore*.98 or all(opt.passed_room(i,r) for i,r in enumerate(predicted)):pairs.append((score,qa,qb))
                pairs.sort(key=lambda q:q[0])
                for _,qa,qb in pairs[:(6 if current_score<.2 else 4)]:
                    if time.monotonic()>deadline-.2:break
                    trial={'loops':list(layout['loops'])}
                    ka,kb=qa[1],qb[1]
                    if ka==kb:
                        pts=qa[4]+qb[4]-paths[ka]
                        trial['loops'][ka]=dict(slot=layout['loops'][ka]['slot'],path=np.round(pts*self.g,8).tolist())
                    else:
                        for q in (qa,qb):
                            k=q[1];trial['loops'][k]=dict(slot=layout['loops'][k]['slot'],path=np.round(q[4]*self.g,8).tolist())
                    _,problems=check_geometry(self.p,trial,self.f)
                    if problems:continue
                    rr=opt.evaluate(layout=trial)
                    if rr is None:continue
                    score=sum(opt.score_room(i,r) for i,r in enumerate(rr['rooms']))
                    if opt.passed(rr) or score<bestscore:
                        bestscore=score;best=(trial,rr)
                        if opt.passed(rr):break
            if best is None:
                if stall:break
                stall=1;continue
            stall=0;layout,res=best
            opt.report('fine '+str(iteration),res)
        return layout,res
