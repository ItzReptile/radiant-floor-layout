"""Coupled pattern changes for rooms sharing a water circuit."""
import copy,time,itertools
import numpy as np

def improve(opt,roommasks,hall,res,seconds=5.):
    stop=min(opt.deadline-13.,time.monotonic()+seconds)
    def score(rr):return sum(opt.score_room(i,r) for i,r in enumerate(rr['rooms']))
    for g in opt.groups:
        if len(g['rooms'])!=2 or all(opt.passed_room(i,res['rooms'][i]) for i in g['rooms']):continue
        if time.monotonic()>stop-.2:break
        severity=0.
        for i in g['rooms']:
            rp=opt.p['rooms'][i];rr=res['rooms'][i]
            severity=max(severity,(rp['required_w']-rr['output_w'])/max(30,rp['required_w']),(rr['output_w']-rp.get('max_w',1e9))/max(30,rp.get('max_w',1e9)),rr['zone_spread']/rp['max_spread']-1.)
        if severity<.04:continue
        base=copy.deepcopy(opt.params);baseline=res;best=score(res);bestdata=None
        cols=[]
        for ri in g['rooms']:
            old=base[ri];choices=[old.copy()]
            if opt.d.parent[ri]!=opt.d.hall:
                for (ix,iy),st,ori in itertools.product(((1.,.5),(.5,1.)),(0,1),(old['orient'],1-old['orient'])):
                    v=dict(width=2,pitch=4,orient=ori,offset=0,inset=.5,inset_x=ix,inset_y=iy,style=st,entry=0,attach=0);choices.append(v)
            if opt.d.parent[ri]!=opt.d.hall:
                densities=list(dict.fromkeys(((2,4),(old['width'],old['pitch']))))
                for (w,pi),ins,off,st in itertools.product(densities,(1.5,1.,2.),(1-old.get('offset',0),old.get('offset',0)),dict.fromkeys((1,old.get('style',0),0))):
                    v=dict(width=w,pitch=pi,orient=old['orient'],offset=off,inset=ins,style=st,entry=0,attach=0)
                    choices.append(v)
            for w,pi,ori,st in [(w,pi,ori,st) for w,pi in ((1,3),(2,4),(2,5),(3,5),(3,6),(3,7)) for ori,st in ((old['orient'],0),(old['orient'],1),(1-old['orient'],0),(1-old['orient'],1),(old['orient'],old.get('style',0)))]:
                v=old.copy()
                for key in ('pitch_x','pitch_y','inset_x','inset_y'):v.pop(key,None)
                v.update(width=w,pitch=pi,orient=ori,style=st,end_mode=0,entry=0,attach=0);choices.append(v)
            pit=old['pitch']
            for px,py in ((pit,pit-.5),(pit-.5,pit),(pit,pit+.5),(pit+.5,pit),(pit,pit-1),(pit-1,pit),(pit,pit+1),(pit+1,pit),(pit+1,pit+1)):
                if min(px,py)<=old['width']:continue
                v=old.copy();v.update(pitch_x=px,pitch_y=py);choices.append(v)
            for ins in (1.,1.5):
                v=old.copy();v['inset']=ins;choices.append(v)
            if opt.d.parent[ri]!=opt.d.hall:
                for att in (0,1,2,3,4,5):
                    v=old.copy();v['attach']=att;choices.append(v)
            vals=[];seen=set()
            for v in choices:
                if time.monotonic()>stop-.1:break
                pp=copy.deepcopy(base);pp[ri]=v;geo=opt.geometry(g,pp)
                if geo is None:continue
                key=tuple(mm.tobytes() for mm in geo.values())
                if key in seen:continue
                seen.add(key);trial=roommasks.copy()
                for k,mm in geo.items():trial[k]=mm
                rr=opt.evaluate(opt.combine(trial,hall))
                if rr is None:continue
                vals.append((v,rr))
                ss=score(rr)
                if ss<best:best=ss;bestdata=(pp,trial,rr)
                if opt.passed(rr):return pp,trial,rr
                if ss<score(baseline) and all(opt.passed_room(i,rr['rooms'][i]) for i in g['rooms']):return pp,trial,rr
            cols.append(vals)
        if len(cols)<2 or not all(cols):continue
        ids=g['rooms'];sels=opt.th.sels;zones=opt.th.zones
        na,nb=len(cols[0]),len(cols[1]);proxy=np.zeros((na,nb))
        for i in (*ids,opt.d.hall):
            rp=opt.p['rooms'][i];r0=baseline['rooms'][i]
            oa=np.array([r['rooms'][i]['output_w'] for v,r in cols[0]])
            ob=np.array([r['rooms'][i]['output_w'] for v,r in cols[1]])
            out=oa[:,None]+ob[None,:]-r0['output_w']
            sel=sels[i];zone=zones[i]
            za=np.stack([r['surface'].ravel()[zone] for v,r in cols[0]])
            zb=np.stack([r['surface'].ravel()[zone] for v,r in cols[1]])
            ma=np.stack([r['surface'].ravel()[sel] for v,r in cols[0]])
            mb=np.stack([r['surface'].ravel()[sel] for v,r in cols[1]])
            z0=baseline['surface'].ravel()[zone];m0=baseline['surface'].ravel()[sel]
            spread=np.empty((na,nb));peak=np.empty((na,nb))
            for ia in range(na):
                zz=za[ia]+zb-z0;spread[ia]=zz.max(axis=1)-zz.min(axis=1)
                peak[ia]=(ma[ia]+mb-m0).max(axis=1)
            low=rp['required_w'];high=rp.get('max_w',low*2);gap=high-low;mg=.005*gap
            errors=[np.maximum(0,(low+mg-out)/max(low,30)),np.maximum(0,(out-high+mg)/max(high,30)),np.maximum(0,(spread-rp['max_spread']+.008)/rp['max_spread']),np.maximum(0,(peak-rp['max_surface'])/4)]
            proxy+=1000*sum(e*e for e in errors)+.001*((out-(low+high)/2)/max(gap,20))**2+.00005*spread/rp['max_spread']
        pairs=[(float(proxy[ia,ib]),v,w) for ia,(v,ra) in enumerate(cols[0]) for ib,(w,rb) in enumerate(cols[1])]
        pairs.sort(key=lambda a:a[0]);count=0
        for _,v,w in pairs:
            if time.monotonic()>stop-.1:break
            pp=copy.deepcopy(base);pp[ids[0]]=v;pp[ids[1]]=w;geo=opt.geometry(g,pp)
            if geo is None:continue
            trial=roommasks.copy()
            for k,mm in geo.items():trial[k]=mm
            before=opt.evals;rr=opt.evaluate(opt.combine(trial,hall));count+=opt.evals-before
            if rr is None:continue
            ss=score(rr)
            if ss<best:
                best=ss;bestdata=(pp,trial,rr)
                if opt.debug:opt.report('paired pattern',rr)
            if opt.passed(rr):return pp,trial,rr
            if opt.group_score(g,rr)<.5 and all(rr['rooms'][i]['zone_spread']<opt.p['rooms'][i]['max_spread'] for i in ids):break
            if count>=60:break
        if bestdata is not None:
            pp0,trial0,rr0=bestdata
            if opt.group_score(g,rr0)<.5:
                for ri in ids:
                    if opt.passed_room(ri,rr0['rooms'][ri]):continue
                    old=pp0[ri]
                    for dx,dy in ((.5,0),(0,.5),(-.5,0),(0,-.5)):
                        if time.monotonic()>stop-.1:break
                        pp=copy.deepcopy(pp0);pp[ri].update(inset_x=old.get('inset_x',old['inset'])+dx,inset_y=old.get('inset_y',old['inset'])+dy)
                        geo=opt.geometry(g,pp)
                        if geo is None:continue
                        trial=roommasks.copy()
                        for k,mm in geo.items():trial[k]=mm
                        rr=opt.evaluate(opt.combine(trial,hall))
                        if rr is None:continue
                        ss=score(rr)
                        if ss<best:best=ss;bestdata=(pp,trial,rr)
                        if opt.passed(rr):return pp,trial,rr
            opt.params,roommasks,res=bestdata;opt.report('coupled patterns',res)
    return opt.params,roommasks,res
