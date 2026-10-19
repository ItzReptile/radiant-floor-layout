import numpy as np
from scipy.ndimage import binary_fill_holes,gaussian_filter,label

def edge_arrays(a):
    h=np.zeros((a.shape[0]+1,a.shape[1]),bool);v=np.zeros((a.shape[0],a.shape[1]+1),bool)
    h[:-1]^=a;h[1:]^=a;v[:,:-1]^=a;v[:,1:]^=a
    return h,v

def improve(opt,masks,res,seconds=9.,iterations=60):
    import time
    d=opt.d;stop=min(opt.deadline-1.,time.monotonic()+seconds)
    # Search edits of straight pipe runs.  Grid-aligned cell edits retain the
    # minimum bend length; the outline/topology checks reject pinches.
    stalled=0
    for iteration in range(iterations):
        if time.monotonic()>stop-.2:break
        failing=[i for i,r in enumerate(res['rooms']) if not opt.passed_room(i,r)]
        if not failing:break
        failing.sort(key=lambda i:-opt.score_room(i,res['rooms'][i]))
        hot_slots=set()
        for g in opt.groups:
            if any(not (opt.p['rooms'][i]['required_w']<=res['rooms'][i]['output_w']<=opt.p['rooms'][i].get('max_w',float('inf'))) for i in g['rooms']):hot_slots.update(g['slots'])
        active=failing[:2]
        if hot_slots and d.hall not in active:active=active+[d.hall]
        ts=opt.cell_surface(res['surface']);fullscore=sum(opt.score_room(i,r) for i,r in enumerate(res['rooms']))
        options=[];seen=set()
        allmask=np.logical_or.reduce(masks)
        for ri in active:
            r=opt.p['rooms'][ri];rr=res['rooms'][ri];rm=d.roomof==ri
            zone=opt.cell_surface(opt.th.zones[ri].reshape(res['surface'].shape).astype(float))>.5
            vals=ts[zone]
            if not len(vals):continue
            avg=float(vals.mean());mn=float(vals.min());mx=float(vals.max())
            target=(r['required_w']+r.get('max_w',r['required_w']*1.25))/2
            shift=(target-rr['output_w'])/max(opt.th.gr[opt.th.sels[ri]].sum(),1)
            if r['required_w']<=rr['output_w']<=r.get('max_w',float('inf')):shift*=.2
            E=np.where(zone,avg-ts,0.)+shift*(2.5 if stalled==0 else 7.)
            E=np.where(rm,E,0.)
            gr=opt.cell_surface(opt.th.gr.reshape(res['surface'].shape))*d.c*d.c
            coarse_spread=float(vals.max()-vals.min())
            coarse_peak=float(ts[rm].max())
            eh=np.zeros((d.ny+1,d.nx));ev=np.zeros((d.ny,d.nx+1))
            eh[:-1]+=E*.5;eh[1:]+=E*.5;ev[:,:-1]+=E*.5;ev[:,1:]+=E*.5
            slots=(sorted(hot_slots) if ri==d.hall and ri not in failing else range(len(masks))) if ri==d.hall else next(g['slots'] for g in opt.groups if ri in g['rooms'])
            for k in slots:
                old=masks[k];lp=d.outline(old,k)
                if lp is None:continue
                pts=[]
                for x,y in lp['path']:
                    dx=x-d.origin[0];dy=y-d.origin[1]
                    pts.append((round((dx*d.ux[0]+dy*d.ux[1])/(d.c*d.g))-d.ix0,round((dx*d.uy[0]+dy*d.uy[1])/(d.c*d.g))-d.iy0))
                others=np.logical_or.reduce([a for l,a in enumerate(masks) if l!=k]);blocked=d.dilation(others)
                oh,ov=edge_arrays(old)
                for a,b in zip(pts,pts[1:]):
                    if time.monotonic()>stop-.15:return masks,res
                    x,y=a;X,Y=b
                    horizontal=y==Y
                    lo,hi=sorted((x,X) if horizontal else (y,Y))
                    if hi-lo<1:continue
                    intervals=[(lo,hi)]
                    if stalled or hi-lo<=7:
                        intervals += [(t,t+1) for t in range(lo,hi)]
                        intervals += [(lo,min(hi,lo+2)),(max(lo,hi-2),hi)]
                    else:
                        intervals += [(lo,lo+1),(hi-1,hi),((lo+hi)//2,(lo+hi)//2+1)]
                    if hi-lo>4:
                        for t in range(lo,hi,4):intervals.append((t,min(hi,t+4)))
                    for l,h in intervals:
                        for add in (False,True):
                          for depth in ((1,2,3) if stalled or (ri==d.hall and hot_slots) else (1,)):
                            mid=(l+h-1)//2
                            if horizontal:
                                inside=y if y<d.ny and old[y,mid] else y-1
                                outside=y-1 if inside==y else y
                                j=outside if add else inside
                                if not 0<=j<d.ny:continue
                                direction=-1 if outside<inside else 1
                                if not add:direction=-direction
                                j2=j+direction*(depth-1)
                                if not 0<=j2<d.ny:continue
                                sl=(slice(min(j,j2),max(j,j2)+1),slice(l,h))
                            else:
                                inside=x if x<d.nx and old[mid,x] else x-1
                                outside=x-1 if inside==x else x
                                i=outside if add else inside
                                if not 0<=i<d.nx:continue
                                direction=-1 if outside<inside else 1
                                if not add:direction=-direction
                                i2=i+direction*(depth-1)
                                if not 0<=i2<d.nx:continue
                                sl=(slice(l,h),slice(min(i,i2),max(i,i2)+1))
                            if not rm[sl].all():continue
                            if add and (not d.allowed[sl].all() or blocked[sl].any()):continue
                            if (old[sl]==add).all():continue
                            cand=old.copy();cand[sl]=add
                            # A disk-shaped region has neither a disconnected
                            # corner nor a hole.  Filling a newly closed gap is
                            # also a useful way of shortening an over-hot coil.
                            cand=binary_fill_holes(cand)
                            if (cand&blocked).any() or (cand&~d.allowed).any():continue
                            rj,rix=d.roots[k]
                            if not cand[rj,rix] or (rj>0 and cand[rj-1,rix]):continue
                            aa=cand[:-1,:-1];bb=cand[:-1,1:];cc=cand[1:,:-1];dd=cand[1:,1:]
                            if ((aa&dd&~bb&~cc)|(bb&cc&~aa&~dd)).any():continue
                            if label(cand)[1]!=1:continue
                            key=(k,cand.tobytes())
                            if key in seen:continue
                            seen.add(key)
                            ch,cv=edge_arrays(cand)
                            if (int(ch.sum())+int(cv.sum())-1)*d.c*d.g>opt.p['pipe']['max_loop_length']+1e-6:continue
                            benefit=float((eh*(ch.astype(float)-oh)).sum()+(ev*(cv.astype(float)-ov)).sum())
                            if benefit<=0 and not (ri==d.hall and hot_slots):continue
                            dh=ch.astype(float)-oh;dv=cv.astype(float)-ov
                            load=.5*(dh[:-1]+dh[1:]+dv[:,:-1]+dv[:,1:])
                            # A local floor response predicts whether a large
                            # edit would replace a hot spot with a cold hole.
                            delta=gaussian_filter(load,1.4,mode='constant')*13.
                            pred=ts+delta
                            pr=rr.copy();pr['output_w']+=float((gr[rm]*delta[rm]).sum())
                            pr['zone_spread']+=float(pred[zone].max()-pred[zone].min())-coarse_spread
                            pr['max_surface']+=float(pred[rm].max())-coarse_peak
                            proxy=fullscore-opt.score_room(ri,rr)+opt.score_room(ri,pr)
                            length_delta=float(dh.sum()+dv.sum())*d.c*d.g
                            if ri==d.hall:
                                for gg in opt.groups:
                                    if k not in gg['slots']:continue
                                    for ir in gg['rooms']:
                                        rp=res['rooms'][ir].copy()
                                        rp['output_w']*=max(.5,1.-.012*length_delta)
                                        rp['zone_spread']*=max(.5,1.-.008*length_delta)
                                        proxy+=opt.score_room(ir,rp)-opt.score_room(ir,res['rooms'][ir])
                            options.append((proxy,benefit,k,cand))
        options.sort(key=lambda a:(a[0],-a[1]));best=None;bestscore=fullscore
        metrics=[]
        for proxy,benefit,k,cand in options[:(64 if stalled else 32)]:
            if time.monotonic()>stop-.1:break
            trial=masks.copy();trial[k]=cand
            rr=opt.evaluate(trial)
            if rr is None:continue
            score=sum(opt.score_room(i,r) for i,r in enumerate(rr['rooms']))
            metrics.append((score,benefit,k,[(r['output_w'],r['zone_spread']) for r in rr['rooms']]))
            if score<bestscore:
                bestscore=score;best=(trial,rr)
                if opt.passed(rr):break
            if len(metrics)>=8 and best is not None and bestscore<fullscore*.94:break
        if best is None:
            if opt.debug:
                print('local stalled',stalled,'options',len(options),'best trials',sorted(metrics,key=lambda a:a[0])[:3],file=__import__('sys').stderr)
            if stalled>=1:break
            stalled+=1;continue
        stalled=0
        masks,res=best
        opt.report('local '+str(iteration),res)
    return masks,res
