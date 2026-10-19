import time,os,sys,itertools,copy
from collections import OrderedDict
import numpy as np
from scipy.ndimage import binary_fill_holes,binary_dilation,minimum_filter
from thermal import Thermal

class Optimizer:
    def __init__(self,p,deadline):
        from solve import Designer
        self.p=p;self.deadline=deadline;self.d=d=Designer(p);self.th=Thermal(p);self.th.deadline=deadline
        self.feeds=None
        for limit in (d.nslots,1):
            d.split_limit=limit
            for shift in (0,-1,1):
                for order in range(4):
                    self.feeds=d.feeds_new(order,shift)
                    if self.feeds is not None:break
                if self.feeds is not None:break
            if self.feeds is not None:break
        if self.feeds is None:raise RuntimeError('No manifold routing')
        self.groups=[]
        for k,a in enumerate(d.cs):
            related=[gg for gg in self.groups if set(gg['rooms'])&set(a['rooms'])]
            rooms=list(a['rooms']);slots=[k]
            for gg in related:
                rooms=list(dict.fromkeys(list(gg['rooms'])+rooms));slots=gg['slots']+slots;self.groups.remove(gg)
            self.groups.append(dict(rooms=tuple(rooms),slots=sorted(slots)))
        self.groups.sort(key=lambda gg:min(gg['slots']))
        self.cache={};self.outlinecache=OrderedDict();self.heatcache=OrderedDict();self.evals=0;self.best=None;self.bestscore=float('inf')
        self.baseparams={i:dict(width=2,pitch=4,orient=0,offset=0,inset=.5) for i in d.assignment}
        self.params=copy.deepcopy(self.baseparams)
        self.debug=bool(os.environ.get('DEBUG'))

    def timeleft(self):return self.deadline-time.monotonic()
    def score_room(self,i,r,margin=True):
        p=self.p['rooms'][i];a=p['required_w'];b=p.get('max_w',a*2);out=r['output_w']
        gap=b-a;m=.005*gap if margin else 0
        errors=[max(0,(a+m-out)/max(a,30)),max(0,(out-b+m)/max(b,30)),max(0,(r['zone_spread']-p['max_spread']+(.008 if margin else 0))/p['max_spread']),max(0,(r['max_surface']-p['max_surface'])/4)]
        hard=sum(x*x for x in errors)
        soft=.001*((out-(a+b)/2)/max(gap,20))**2+.00005*r['zone_spread']/p['max_spread']
        return 1000*hard+soft
    def group_score(self,g,res):return sum(self.score_room(i,res['rooms'][i]) for i in g['rooms'])
    def passed_room(self,i,r):
        p=self.p['rooms'][i]
        return p['required_w']+1e-4<=r['output_w']<=p.get('max_w',float('inf'))-1e-4 and r['zone_spread']<=p['max_spread']-1e-4 and r['max_surface']<=p['max_surface']-1e-4
    def passed(self,res):return all(self.passed_room(i,r) for i,r in enumerate(res['rooms']))
    def key(self,g,params):return (tuple(g['slots']),tuple((i,tuple(sorted(params[i].items()))) for i in g['rooms']))
    def geometry(self,g,params):
        key=self.key(g,params)
        if key in self.cache:return self.cache[key]
        d=self.d;result={}
        for k in g['slots']:
            a=d.cs[k];mask=self.feeds[k].copy()
            other=np.logical_or.reduce([f for l,f in enumerate(self.feeds) if l!=k])
            for ri in a['rooms']:
                pp=params[ri].copy()
                if ri==a['ri']:
                    pp['bbox']=a['bbox'];pp['root_slot']=k;pp['zone_mask']=a.get('zone')
                    if len(a['rooms'])>1:
                        child=a['rooms'][1];door=d.doors[ri,child];pp['end_point']=((door[0]+door[2])/2,(door[1]+door[3])/2)
                        if params[child].get('attach',0):
                            pr=d.rooms[ri]['rect']
                            if door[0]==door[2]:side=0 if abs(door[0]-pr[0])<1 else 2
                            else:side=1 if abs(door[1]-pr[1])<1 else 3
                            pp['reserve_side']=side;pp['style']=5
                else:pp['connection_zone']=a.get('zone')
                mask=d.pattern(ri,mask,other,**pp)
                if mask is None:self.cache[key]=None;return None
            result[k]=mask
        self.cache[key]=result;return result
    def layout(self,masks):
        loops=[]
        for k,mask in enumerate(masks):
            key=(k,mask.tobytes())
            if key in self.outlinecache:
                lp,length=self.outlinecache[key];self.outlinecache.move_to_end(key)
            else:
                lp=self.d.outline(mask,k)
                length=(sum(abs(a[0]-b[0])+abs(a[1]-b[1]) for a,b in zip(lp['path'],lp['path'][1:])) if lp is not None else float('inf'))
                self.outlinecache[key]=(lp,length)
                if len(self.outlinecache)>768:self.outlinecache.popitem(last=False)
            if lp is None or length>self.p['pipe']['max_loop_length']+1e-6:return None
            loops.append(lp)
        return dict(loops=loops)
    def evaluate(self,masks=None,layout=None):
        lay=self.layout(masks) if layout is None else layout
        if lay is None:return None
        key=tuple((a['slot'],tuple(tuple(p) for p in a['path'])) for a in lay['loops'])
        accuracy=2e-7 if self.bestscore<.5 else 1e-5
        if key in self.heatcache and self.heatcache[key].get('_rtol',1.)<=accuracy:
            res=self.heatcache[key];self.heatcache.move_to_end(key)
        else:
            try:res=self.th.evaluate(lay,rtol=accuracy)
            except (TimeoutError,ArithmeticError):return None
            self.evals+=1;self.heatcache[key]=res
            if len(self.heatcache)>128:self.heatcache.popitem(last=False)
        if self.passed(res) and res.get('_rtol',1.)>2e-10:
            try:res=self.th.evaluate(lay,rtol=2e-10)
            except (TimeoutError,ArithmeticError):return None
            self.evals+=1;self.heatcache[key]=res
        if self.passed(res):
            from model import check_geometry
            _,problems=check_geometry(self.p,lay,self.th.floor)
            if problems:
                if self.debug:print('geometry rejection',problems[:3],file=sys.stderr)
                return None
        score=sum(self.score_room(i,r) for i,r in enumerate(res['rooms']))
        if (self.passed(res) and (self.best is None or not self.passed(self.bestres))) or (score<self.bestscore and (self.best is None or not self.passed(self.bestres) or self.passed(res))):
            self.bestscore=score;self.best=lay;self.bestres=res
            if masks is not None:self.bestmasks=[a.copy() for a in masks]
        return res
    def combine(self,roommasks,hall):return [a|b for a,b in zip(roommasks,hall)]
    def report(self,what,res):
        if self.debug:
            print(what,'eval',self.evals,'left',round(self.timeleft(),1),'score',round(sum(self.score_room(i,r) for i,r in enumerate(res['rooms'])),3),file=sys.stderr)
            print(' '.join(f"{r['id']}:{r['output_w']:.0f}/{r['zone_spread']:.2f}" for r in res['rooms']),file=sys.stderr)

    def refine(self,roommasks,hall,res,max_trials=60):
        tested={tuple(g['slots']):set() for g in self.groups};stalled=set()
        for sweep in range(2):
            for g in sorted(self.groups,key=lambda g:-self.group_score(g,res)):
                if self.timeleft()<3:return roommasks,hall,res
                if all(self.passed_room(i,res['rooms'][i]) for i in g['rooms']):continue
                bestscore=self.group_score(g,res);starting_score=bestscore
                trial_limit=min(max_trials,16) if sweep and tuple(g['slots']) in stalled else max_trials
                base=copy.deepcopy(self.params)
                trial_columns=[]
                # Search density and the path through the room separately.  The
                # small child room has its own density on the shared circuit.
                for ri in g['rooms']:
                    b=base[ri]
                    values=[]
                    if self.d.parent[ri]!=self.d.hall:
                        for att in (0,1,2,3):
                            v=b.copy();v['attach']=att;values.append(v)
                    if self.d.parent[ri]==self.d.hall and len(self.d.adj[ri])>1:
                        for em,st in itertools.product((0,1,2),(b.get('style',0),1)):
                            v=b.copy();v.update(end_mode=em,style=st);values.append(v)
                    low=res['rooms'][ri]['output_w']<self.p['rooms'][ri]['required_w']
                    hot=res['rooms'][ri]['output_w']>self.p['rooms'][ri].get('max_w',float('inf'))
                    if hot and len(g['slots'])>1 and len(g['rooms'])==1:
                        for ins in (1.5,2.5,3.5,2.,3.,4.):
                            v=b.copy();v.pop('inset_x',None);v.pop('inset_y',None);v['inset']=ins;values.append(v)
                    if low:families=((1,3),(2,3),(2,4),(3,5),(3,6))
                    elif hot:families=((3,7),(4,7),(4,8),(3,6),(2,5),(3,5),(2,4),(1,3))
                    else:families=((3,6),(2,5),(3,5),(2,4),(1,3))
                    for w,pi in families:
                        for st in dict.fromkeys((b.get('style',0),1,0)):
                            v=b.copy()
                            for key in ('pitch_x','pitch_y','inset_x','inset_y'):v.pop(key,None)
                            v.update(width=w,pitch=pi,style=st);values.append(v)
                    if not (self.p['rooms'][ri]['required_w']<=res['rooms'][ri]['output_w']<=self.p['rooms'][ri].get('max_w',float('inf'))):
                        pit=b['pitch']
                        for px,py in ((pit,pit-.5),(pit-.5,pit),(pit,pit+.5),(pit+.5,pit),(pit,pit-1),(pit-1,pit),(pit-.5,pit+.5),(pit+.5,pit-.5)):
                            if min(px,py)<=b['width']:continue
                            v=b.copy();v.update(pitch_x=px,pitch_y=py);values.append(v)
                        for ix,iy in ((.5,1.),(1.,.5),(.5,1.5),(1.5,.5)):
                            v=b.copy();v.update(inset_x=ix,inset_y=iy);values.append(v)
                    for clipping in (0,1):
                        for orient in (b['orient'],1-b['orient']):
                            v=b.copy();v.update(clip=clipping,orient=orient);values.append(v)
                    if self.d.parent[ri]!=self.d.hall:
                        for entry,orient in itertools.product((-1,0,1),(0,1)):
                            v=b.copy();v.update(entry=entry,orient=orient);values.append(v)
                    for style,orient,off in itertools.product((1,2,3,4),(b['orient'],1-b['orient']),(b['offset'],1-b['offset'])):
                        v=b.copy();v.update(style=style,orient=orient,offset=off);values.append(v)
                    for inset in (.5,1.,1.5,2.):
                        for orient in (b['orient'],1-b['orient']):
                            v=b.copy();v.update(inset=inset,orient=orient);values.append(v)
                    for pitch in (b['pitch']+1,b['pitch']-1,b['pitch']+2,b['pitch']-2):
                        if pitch<3:continue
                        for width in sorted(set([max(1,pitch//2),b['width']])):
                            if width>=pitch:continue
                            v=b.copy()
                            for key in ('pitch_x','pitch_y'):v.pop(key,None)
                            v.update(pitch=pitch,width=width);values.append(v)
                    for orient,off in itertools.product((0,1),(0,1)):
                        v=b.copy();v.update(orient=orient,offset=off);values.append(v)
                    for pitch in sorted(range(3,8),key=lambda x:abs(x-b['pitch'])):
                        for width in sorted(set([max(1,pitch//2),max(1,(pitch+1)//2),1,2,3])):
                            if width>=pitch:continue
                            for orient in (b['orient'],1-b['orient']):
                                v=b.copy()
                                for key in ('pitch_x','pitch_y'):v.pop(key,None)
                                v.update(pitch=pitch,width=width,orient=orient);values.append(v)
                    for inset in (.5,1.,1.5):
                        for orient in (0,1):
                            v=b.copy();v.update(inset=inset,orient=orient);values.append(v)
                    trial_columns.append([(ri,v) for v in values])
                trials=[item for row in itertools.zip_longest(*trial_columns) for item in row if item is not None]
                count=0
                for ri,v in trials:
                    pp=copy.deepcopy(self.params);pp[ri]=v
                    if self.timeleft()<3 or count>=trial_limit:break
                    if self.key(g,pp)==self.key(g,self.params):continue
                    geo=self.geometry(g,pp)
                    if geo is None:continue
                    signature=tuple((k,m.tobytes()) for k,m in geo.items())
                    if signature in tested[tuple(g['slots'])]:continue
                    tested[tuple(g['slots'])].add(signature)
                    trial=roommasks.copy()
                    for k,a in geo.items():trial[k]=a
                    before=self.evals
                    rr=self.evaluate(self.combine(trial,hall));count+=self.evals-before
                    if rr is None:continue
                    score=self.group_score(g,rr)
                    if score<bestscore:
                        bestscore=score;self.params=pp;roommasks=trial;res=rr
                        if all(self.passed_room(i,rr['rooms'][i]) for i in g['rooms']):break
                if bestscore>=starting_score*.999:stalled.add(tuple(g['slots']))
                else:stalled.discard(tuple(g['slots']))
                self.report('refined '+str(g['rooms']),res)
        return roommasks,hall,res

    def hall_refine(self,roommasks,hall,res):
        d=self.d;hi=d.hall
        if self.passed(res):return hall,res
        if self.passed_room(hi,res['rooms'][hi]):
            severity=0.
            for i,(rp,rr) in enumerate(zip(self.p['rooms'],res['rooms'])):
                if i==hi:continue
                severity=max(severity,(rp['required_w']-rr['output_w'])/max(30,rp['required_w']),(rr['output_w']-rp.get('max_w',1e9))/max(30,rp.get('max_w',1e9)),rr['zone_spread']/rp['max_spread']-1.,(rr['max_surface']-rp['max_surface'])/4)
            if severity<.04:return hall,res
        bestscore=sum(self.score_room(i,r) for i,r in enumerate(res['rooms']))
        biases=np.zeros(len(roommasks))
        for gg in self.groups:
            bias=0.
            for ir in gg['rooms']:
                rp=self.p['rooms'][ir];out=res['rooms'][ir]['output_w'];lo=rp['required_w'];hiw=rp.get('max_w',lo*1.25)
                bias+=(.5*(lo+hiw)-out)/max(hiw-lo,20)
            for k in gg['slots']:biases[k]=10*bias
        base_res=res;best_roommasks=roommasks.copy();best_params=copy.deepcopy(self.params)
        alternatives=[(3.,biases),(3.,biases*2),(3.,biases*5),(3.,biases*10),(3.8,biases*5),(4.2,biases*10)]+[(r,None) for r in (3.4,3.8,4.2,2.7,2.4,4.7)]
        for reach,bias in alternatives:
            if self.timeleft()<3:break
            trial=d.add_hall(roommasks,reach=reach,biases=bias)
            rr=self.evaluate(trial)
            if rr is None:continue
            trial_rm=roommasks.copy();trial_hall=[a&(d.roomof==hi) for a in trial];trial_params=copy.deepcopy(self.params)
            # A changed feed load changes the right spacing in the destination.
            # Try a short set of room patterns before rejecting that allocation.
            for gg in self.groups:
                changed=[ir for ir in gg['rooms'] if abs(rr['rooms'][ir]['output_w']-base_res['rooms'][ir]['output_w'])>.04*max(30,self.p['rooms'][ir]['required_w'])]
                if not changed or all(self.passed_room(ir,rr['rooms'][ir]) for ir in gg['rooms']):continue
                gbest=self.group_score(gg,rr)+.05*self.score_room(hi,rr['rooms'][hi])
                for ir in changed:
                    for w,pi,orient,style in ((2,4,0,0),(2,4,1,0),(2,4,0,1),(2,4,1,1),(3,5,0,0),(2,5,1,1),(3,6,0,1),(1,3,0,1)):
                        if self.timeleft()<9:break
                        pp=copy.deepcopy(trial_params);v=pp[ir]
                        for key in ('pitch_x','pitch_y','inset_x','inset_y'):v.pop(key,None)
                        v.update(width=w,pitch=pi,orient=orient,style=style)
                        geo=self.geometry(gg,pp)
                        if geo is None:continue
                        candidate=trial_rm.copy()
                        for k,a in geo.items():candidate[k]=a
                        rrr=self.evaluate(self.combine(candidate,trial_hall))
                        if rrr is None:continue
                        ss=self.group_score(gg,rrr)+.05*self.score_room(hi,rrr['rooms'][hi])
                        if ss<gbest:
                            gbest=ss;trial_rm=candidate;trial_params=pp;rr=rrr
                        if all(self.passed_room(jr,rr['rooms'][jr]) for jr in gg['rooms']):break
            trial=self.combine(trial_rm,trial_hall)
            score=sum(self.score_room(i,r) for i,r in enumerate(rr['rooms']))
            if score<bestscore:
                bestscore=score;hall=trial_hall;res=rr;best_roommasks=trial_rm;best_params=trial_params
            if self.passed(res):break
        roommasks[:]=best_roommasks;self.params=best_params
        self.report('hall alternatives',res)
        if self.passed_room(hi,res['rooms'][hi]):return hall,res
        # Temperature-directed fingers address cool corners missed by a uniform
        # distance-based covering of the hall.
        for _ in range(10):
            if self.timeleft()<3:break
            current=self.combine(roommasks,hall);allm=np.logical_or.reduce(current)
            zone=self.th.zones[hi].reshape(res['surface'].shape)
            ts=self.cell_surface(minimum_filter(np.where(zone,res['surface'],np.inf),size=3,mode='constant',cval=np.inf))
            eligible=(d.roomof==hi)&d.allowed&(d.room_distance(hi)>=1.)&~allm
            if not eligible.any():break
            ranking=np.argsort(np.where(eligible,ts,np.inf).ravel())
            targets=[]
            for flat in ranking:
                j,i=divmod(int(flat),d.nx)
                if not np.isfinite(ts[j,i]):break
                if all(abs(j-y)+abs(i-x)>=3 for y,x in targets):targets.append((j,i))
                if len(targets)>=10:break
            opts=[]
            for j,i in targets:
                if self.timeleft()<3:break
                st=d.empty();st[j,i]=True
                for k,old in enumerate(current):
                    other=np.logical_or.reduce([a for l,a in enumerate(current) if l!=k]);al=d.allowed&(d.roomof==hi)&~d.dilation(other)&~old
                    goal=binary_dilation(old,structure=np.array([[0,1,0],[1,1,1],[0,1,0]],bool))&al
                    path=d.astar(st,goal,al)
                    if path is None:continue
                    cand=old.copy()
                    for y,x in path:cand[y,x]=True
                    cand=binary_fill_holes(cand)
                    if (cand&d.dilation(other)).any() or (cand&~d.allowed).any():continue
                    trial=current.copy();trial[k]=cand
                    rr=self.evaluate(trial)
                    if rr is None:continue
                    score=sum(self.score_room(i,r) for i,r in enumerate(rr['rooms']))
                    opts.append((score,trial,rr))
                if opts and min(x[0] for x in opts)<bestscore*.2:break
            if not opts:break
            score,trial,rr=min(opts,key=lambda a:a[0])
            if score>=bestscore:break
            bestscore=score;hall=[a&(d.roomof==hi) for a in trial];res=rr
            if self.passed_room(hi,res['rooms'][hi]):break
        self.report('hall temperatures',res)
        return hall,res

    def cell_surface(self,surface):
        d=self.d;cx=np.broadcast_to(d.xs[None,:],d.shape);cy=np.broadcast_to(d.ys[:,None],d.shape)
        ox=np.rint(d.origin[0]/d.g+cx*d.ux[0]+cy*d.uy[0]).astype(int)-self.th.floor.bx0
        oy=np.rint(d.origin[1]/d.g+cx*d.ux[1]+cy*d.uy[1]).astype(int)-self.th.floor.by0
        return surface[np.clip(oy,0,surface.shape[0]-1),np.clip(ox,0,surface.shape[1]-1)]

    def run(self,initial_only=False,short=False):
        d=self.d;roommasks=[x.copy() for x in self.feeds]
        for g in self.groups:
            geo=None
            for pi,w,o in ((4,2,0),(5,2,0),(5,3,0),(6,3,0),(4,2,1),(6,2,1)):
                pp=copy.deepcopy(self.params)
                for i in g['rooms']:pp[i].update(pitch=pi,width=w,orient=o)
                candidate=self.geometry(g,pp)
                if candidate is None:continue
                okay=True
                for k,a in candidate.items():
                    lp=d.outline(a,k)
                    if lp is None:okay=False;break
                    length=sum(abs(x[0]-y[0])+abs(x[1]-y[1]) for x,y in zip(lp['path'],lp['path'][1:]))
                    if length>self.p['pipe']['max_loop_length']-.2:okay=False;break
                if okay:geo=candidate;self.params=pp;break
            if geo is None:raise RuntimeError('No initial room pattern')
            for k,a in geo.items():roommasks[k]=a
        masks=d.add_hall(roommasks)
        hall=[a&(d.roomof==d.hall) for a in masks]
        res=self.evaluate(masks)
        if res is None:raise RuntimeError('Initial circuit too long')
        self.report('initial',res)
        if self.passed(res):return (roommasks,hall,res) if initial_only else self.best
        bestgroups=[(self.group_score(g,res),self.geometry(g,self.params),copy.deepcopy(self.params)) for g in self.groups]
        variants=[(2,4,o,f,ins) for ins in (.5,1.) for o in (0,1) for f in (0,1)]
        variants += [(w,pi,o,0,.5) for w,pi in ((2,5),(3,5),(3,6),(2,3),(1,3)) for o in (0,1)]
        variants += [(w,pi,o,0,.5) for w,pi in ((3,7),(4,7),(4,8)) for o in (0,1)]
        variants += [(2,4,o,0,.5,1) for o in (0,1)]
        variants += [(2,5,o,0,.5,1) for o in (0,1)]
        variants += [(w,pi,o,0,.5,0,1) for w,pi in ((1,3),(2,4),(2,5),(3,6)) for o in (0,1)]
        variants += [(2,4,0,off,.5,0,st) for st in (2,3) for off in (0,1)]
        variants += [(w,pi,0,0,.5,0,st,em) for w,pi in ((2,4),(2,5),(3,6)) for st in (0,1) for em in (1,2)]
        variants += [(w,pi,0,0,.5,0,0,0,att) for w,pi in ((2,4),(2,5),(3,6)) for att in (1,2)]
        if short:variants=variants[:40]
        for variant in variants:
            w,pi,o,off,ins=variant[:5]
            clipping=variant[5] if len(variant)>5 else 0
            style=variant[6] if len(variant)>6 else 0
            end_mode=variant[7] if len(variant)>7 else 0
            attach=variant[8] if len(variant)>8 else 0
            if self.timeleft()<12:break
            pp={i:dict(width=w,pitch=pi,orient=o,offset=off,inset=ins,clip=clipping,style=style,end_mode=end_mode,attach=attach) for i in d.assignment}
            trial=roommasks.copy();geos=[]
            for g in self.groups:
                geo=self.geometry(g,pp)
                if geo is not None:
                    sub=trial.copy()
                    for k,a in geo.items():sub[k]=a
                    if self.layout(self.combine(sub,hall)) is None:geo=None
                geos.append(geo)
                if geo is not None:
                    for k,a in geo.items():trial[k]=a
            rr=self.evaluate(self.combine(trial,hall))
            if rr is None:continue
            if self.passed(rr):
                if initial_only:self.params=pp;return trial,hall,rr
                return self.best
            for gi,g in enumerate(self.groups):
                score=self.group_score(g,rr)
                if geos[gi] is not None and score<bestgroups[gi][0]:bestgroups[gi]=(score,geos[gi],copy.deepcopy(pp))
        for g,(_,geo,pp) in zip(self.groups,bestgroups):
            for k,a in geo.items():roommasks[k]=a
            for i in g['rooms']:self.params[i]=pp[i]
        res=self.evaluate(self.combine(roommasks,hall));self.report('combined',res)
        if self.passed(res):return (roommasks,hall,res) if initial_only else self.best
        if initial_only:return roommasks,hall,res
        # Reconsider the extra circuit if one unsplit destination still needs
        # substantially more heat.  Most apartments keep their first allocation.
        split=next((g for g in self.groups if len(g['slots'])>1),None)
        if split is not None and self.timeleft()>24.:
            splitload=sum(self.p['rooms'][i]['required_w'] for i in split['rooms'])
            cold=[]
            for g in self.groups:
                need=sum(self.p['rooms'][i]['required_w'] for i in g['rooms'])
                got=sum(res['rooms'][i]['output_w'] for i in g['rooms'])
                if len(g['slots'])==1 and need>=.75*splitload and got<.96*need:
                    cold.append(((need-got)/max(30,need),d.cs[g['slots'][0]]['ri']))
            if cold:
                target=max(cold)[1]
                alternate=Optimizer(self.p,self.deadline)
                alternate.d.split_room=target
                alternate.feeds=None
                for shift in (0,-1,1):
                    for order in range(4):
                        alternate.feeds=alternate.d.feeds_new(order,shift)
                        if alternate.feeds is not None:break
                    if alternate.feeds is not None:break
                if alternate.feeds is not None:
                    alternate.groups=[]
                    for k,entry in enumerate(alternate.d.cs):
                        related=[gg for gg in alternate.groups if set(gg['rooms'])&set(entry['rooms'])]
                        rooms=list(entry['rooms']);slots=[k]
                        for gg in related:
                            rooms=list(dict.fromkeys(list(gg['rooms'])+rooms));slots=gg['slots']+slots;alternate.groups.remove(gg)
                        alternate.groups.append(dict(rooms=tuple(rooms),slots=sorted(slots)))
                    alternate.groups.sort(key=lambda gg:min(gg['slots']))
                    alternate.cache={}
                    try:
                        am,ah,ar=alternate.run(initial_only=True,short=True)
                    except RuntimeError:ar=None
                    if ar is not None and (alternate.passed(alternate.bestres) or alternate.bestscore<self.bestscore):
                        self.__dict__.update(alternate.__dict__)
                        d=self.d;roommasks,hall,res=am,ah,ar
                        self.report('changed circuit allocation',res)
                        if self.passed(self.bestres):return self.best
        roommasks,hall,res=self.refine(roommasks,hall,res,max_trials=38)
        if self.passed(res):return self.best
        from joint import improve as improve_joint
        self.params,roommasks,res=improve_joint(self,roommasks,hall,res,seconds=7.5)
        if self.passed(res):return self.best
        hall,res=self.hall_refine(roommasks,hall,res)
        if self.passed(res):return self.best
        from local import improve
        severity=0.
        for rp,rr in zip(self.p['rooms'],self.bestres['rooms']):
            severity=max(severity,(rp['required_w']-rr['output_w'])/max(30,rp['required_w']),(rr['output_w']-rp.get('max_w',1e9))/max(30,rp.get('max_w',1e9)),rr['zone_spread']/rp['max_spread']-1.)
        heat_fails=sum(not (rp['required_w']<=rr['output_w']<=rp.get('max_w',float('inf'))) for rp,rr in zip(self.p['rooms'],self.bestres['rooms']))
        if (severity>.08 or heat_fails>=2) and self.timeleft()>14:
            current=[a.copy() for a in self.bestmasks];res=self.bestres
            current,res=improve(self,current,res,seconds=min(6.,self.timeleft()-14.))
            if self.passed(self.bestres):return self.best
        from fine import FineSearch
        _,res=FineSearch(self).run(self.best,self.bestres,seconds=max(0.,self.timeleft()-.2))
        self.report('final',res)
        if self.debug:
            print('parameters',self.params,file=sys.stderr)
        return self.best
