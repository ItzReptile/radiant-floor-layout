#!/usr/bin/env python3
import sys,json,time,math,os,copy
START=time.perf_counter()
for _threads in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_threads]='1'
import numpy as np
from contours import Domain,room_cycles
from routing import connections,join_rooms,entrance,route_hall,simplify


def generate(p,pitches=None,offsets=None):
    d=Domain(p)
    links=connections(d)
    direct={b if a==d.hall else a:(a,b,pa) for a,b,pa in links if d.hall in (a,b)}
    children={}
    for a,b,pa in links:
        if d.hall in (a,b):continue
        if a not in direct:a,b=b,a
        children[a]=(b,pa)
    spare=max(0,p['manifold']['ports']//2-len(direct))
    split=set(sorted((ri for ri in direct if ri not in children),key=lambda ri:p['rooms'][ri]['required_w'],reverse=True)[:spare])
    cs={};components={}
    for ri,r in enumerate(p['rooms']):
        if ri==d.hall or ri in split:continue
        pitch=pitches.get(ri,3) if pitches else 3
        offset=offsets.get(ri,1) if offsets else 1
        paths,left=room_cycles(d,ri,pitch,offset)
        if left or len(paths)!=1:raise ValueError(('disconnected',ri,pitch,offset,left))
        cs[ri]=paths[0]
        components[ri]={'room':ri,'bounds':[d.rooms[ri]],'cycle':paths[0],'pitch':pitch,'offset':offset,'balanced':False}
    for a,(b,pa) in children.items():cs[a]=join_rooms(d,cs[a],cs[b],pa)
    targets=[]
    for ri,(_,_,pa) in direct.items():
        if ri in split:
            r=d.rooms[ri];vertical=abs(pa[0]-pa[2])<1e-6
            lo,hi=(pa[1],pa[3]) if vertical else (pa[0],pa[2])
            cut=round((lo+hi)/2-.5)+.5
            if vertical:
                bounds=[(r[0],r[1],r[2],cut),(r[0],cut,r[2],r[3])]
                passages=[(pa[0],pa[1],pa[2],cut),(pa[0],cut,pa[2],pa[3])]
            else:
                bounds=[(r[0],r[1],cut,r[3]),(cut,r[1],r[2],r[3])]
                passages=[(pa[0],pa[1],cut,pa[3]),(cut,pa[1],pa[2],pa[3])]
            bounds=[[b] for b in bounds]
            t0,t1=(r[1],r[3]) if vertical else (r[0],r[2])
            n0,n1=(r[0],r[2]) if vertical else (r[1],r[3])
            wall=pa[0] if vertical else pa[1]
            if abs(cut-(t0+t1)/2)>3 and n1-n0>12:
                inward=1 if abs(wall-n0)<1e-6 else -1
                transition=round(wall+inward*max(2.,min(3.,p['rooms'][ri].get('zone_margin',.5)/d.unit*.5))-.5)+.5
                near=(n0,transition) if inward>0 else (transition,n1)
                far=(transition,n1) if inward>0 else (n0,transition)
                node_mask=d.room_mask(ri)
                yy,xx=np.nonzero(node_mask)
                ts=yy+d.y0 if vertical else xx+d.x0
                ns=xx+d.x0 if vertical else yy+d.y0
                in_near=(ns<transition) if inward>0 else (ns>transition)
                candidates=[v+.5 for v in range(math.ceil(t0+2),math.floor(t1-2))]
                balance=min(candidates,key=lambda v:abs(int(((ts<cut)&in_near|((ts<v)&~in_near)).sum())-len(ts)/2))
                def rect(tl,th,nl,nh):return (nl,tl,nh,th) if vertical else (tl,nl,th,nh)
                bounds=[[rect(t0,cut,*near),rect(t0,balance,*far)],[rect(cut,t1,*near),rect(balance,t1,*far)]]
            for bound,passage in zip(bounds,passages):
                paths,left=room_cycles(d,ri,pitches.get(ri,3) if pitches else 3,offsets.get(ri,1) if offsets else 1,bound)
                if left or len(paths)!=1:raise ValueError(('split disconnected',ri,left))
                t=entrance(d,paths[0],ri,passage,region_bounds=bound);t['rooms']=[ri];t['bounds']=bound;t['components']=[{'room':ri,'bounds':bound,'cycle':paths[0],'pitch':pitches.get(ri,3) if pitches else 3,'offset':offsets.get(ri,1) if offsets else 1,'balanced':False}];targets.append(t)
        else:
            t=entrance(d,cs[ri],ri,pa);t['rooms']=[ri]+([children[ri][0]] if ri in children else []);t['bounds']=[d.rooms[rj] for rj in t['rooms']];t['components']=[components[rj] for rj in t['rooms']];t['join']=children[ri][1] if ri in children else None;targets.append(t)
    loops=route_hall(d,targets,None)
    if loops is None:raise ValueError('hall routing failed')
    layout={'loops':[{'slot':lp['slot'],'path':[d.output(v) for v in simplify(lp['path'])]} for lp in loops]}
    return layout,(d,loops)


def status(plan,heat):
    errors=[];bad=[]
    for i,(r,h) in enumerate(zip(plan['rooms'],heat['rooms'])):
        width=max(10.,r.get('max_w',r['required_w']*1.3)-r['required_w'])
        es=[max(0.,r['required_w']-h['output_w'])/width,
            max(0.,h['output_w']-r.get('max_w',float('inf')))/width,
            max(0.,h['max_surface']-r['max_surface'])/4.,
            max(0.,h['zone_spread']-r['max_spread'])/r['max_spread']]
        if max(es)>1e-8:bad.append((max(es),i))
        errors.extend(es)
    score=sum(errors)*2+max(errors,default=0)+10*sum(v*v for v in errors)+.25*len(bad)
    return score,sorted(bad,reverse=True)


def initial_parameters(p):
    d=Domain(p);links=connections(d)
    adjacent={a if b==d.hall else b for a,b,_ in links if d.hall in (a,b)}
    pitches={};offsets={}
    for ri in range(len(p['rooms'])):
        if ri==d.hall or ri in adjacent:continue
        mask=d.room_mask(ri);yy,xx=np.nonzero(mask)
        span=min(int(xx.max()-xx.min()),int(yy.max()-yy.min()))
        pitches[ri]=2
        offsets[ri]=2 if span%4 in (2,3) else 1
    return pitches,offsets


def main():
    if len(sys.argv)!=3:raise SystemExit('Usage: solve.py PLAN.json LAYOUT.json')
    from optimize import Optimizer,layout_of
    import thermal
    p=json.load(open(sys.argv[1]));pitches,offsets=initial_parameters(p)
    for attempt in range(3):
        try:
            layout,(d,loops)=generate(p,pitches,offsets)
            break
        except ValueError:
            if attempt==2:raise
            if attempt==0:offsets={i:0 for i in range(len(p['rooms']))}
            else:pitches={i:2 for i in range(len(p['rooms']))}
    for _ in range(10):
        long_loops=[lp for lp in loops if (len(lp['path'])-1)*d.unit>p['pipe']['max_loop_length']+1e-6]
        if not long_loops:break
        for ri in {ri for lp in long_loops for ri in lp['rooms']}:pitches[ri]=pitches.get(ri,3)+.5
        layout,(d,loops)=generate(p,pitches,offsets)
    optimizer=Optimizer(d,loops,hall_prepare=True)
    debug=bool(os.environ.get('SOLVER_DEBUG'))
    deadline=START+14.3
    best=None;bestscore=float('inf');bestkey=float('inf');eval_time=.8

    def evaluate_current():
        nonlocal best,bestscore,bestkey,eval_time
        t=time.perf_counter();heat=optimizer.evaluate()
        eval_time=max(.55,eval_time*.96,time.perf_counter()-t)
        score,bad=status(p,heat)
        power_bad=sum(h['output_w']<r['required_w']-2 or h['output_w']>r.get('max_w',1e99)+2 for r,h in zip(p['rooms'],heat['rooms']))
        key=score
        if key<bestkey:
            bestkey=key;bestscore=score
            best=(layout_of(d,optimizer.paths),heat,optimizer.parsed_loops,bad,{k:v.copy() for k,v in optimizer.paths.items()},copy.deepcopy(loops))
        if debug:
            print('heat',round(time.perf_counter()-START,3),'bad',[(p['rooms'][i]['id'],round(e,4)) for e,i in bad],file=sys.stderr)
            print('linear',getattr(optimizer.floor,'last_schur_status',None),getattr(optimizer.floor,'direct_only',False),file=sys.stderr)
        return bad

    bad=evaluate_current()
    from patterns import optimize_patterns,sync_hall

    def search_patterns(seconds,only_failed=False,paired_seconds=0.):
        sync_hall(loops,optimizer.paths)
        h=optimizer.heat
        local_heat={**h,'rooms':[r.copy() for r in h['rooms']],'surface':h['surface'].copy(),'bottom':h['bottom'].copy(),'water':dict(h['water'])}
        local_parsed=optimizer.parsed_loops.copy()
        optimize_patterns(d,loops,optimizer.floor,local_parsed,local_heat,time.perf_counter()+seconds,max_trials=7 if not only_failed else 6,debug=debug,only_failed=only_failed,exclude_pairs=True)
        if paired_seconds:
            from pair import optimize_pairs
            optimize_pairs(d,loops,optimizer.floor,local_parsed,local_heat,time.perf_counter()+paired_seconds,debug)
        optimizer.reset_paths(loops)
        return evaluate_current()

    def search_pairs(seconds):
        from pair import optimize_pairs
        sync_hall(loops,optimizer.paths)
        h=optimizer.heat
        local_heat={**h,'rooms':[r.copy() for r in h['rooms']],'surface':h['surface'].copy(),'bottom':h['bottom'].copy(),'water':dict(h['water'])}
        if optimize_pairs(d,loops,optimizer.floor,optimizer.parsed_loops.copy(),local_heat,time.perf_counter()+seconds,debug):
            optimizer.reset_paths(loops)
            return evaluate_current()
        return status(p,optimizer.heat)[1]

    if bad and time.perf_counter()<START+5.6:
        bad=search_patterns(max(.1,min(3.0,START+4.1-time.perf_counter())),paired_seconds=1.25)
    if any(i==d.hall for _,i in bad) and time.perf_counter()<START+7.:
        from hall import optimize_ends
        sync_hall(loops,optimizer.paths)
        h=optimizer.heat
        local_heat={**h,'rooms':[r.copy() for r in h['rooms']],'surface':h['surface'].copy(),'bottom':h['bottom'].copy(),'water':dict(h['water'])}
        if optimize_ends(d,loops,optimizer.floor,optimizer.parsed_loops.copy(),local_heat,time.perf_counter()+1.45,debug):
            optimizer.reset_paths(loops)
            bad=evaluate_current()
    pair_rooms={ri for lp in loops for ri in lp['rooms'] if sum(ri in q['rooms'] for q in loops)>1}
    def near_capacity_under(ri):
        h=optimizer.heat['rooms'][ri];r=p['rooms'][ri]
        members=[lp['slot'] for lp in loops if ri in lp['rooms']]
        return (len(members)==1 and optimizer.length[members[0]]*d.unit>p['pipe']['max_loop_length']-4 and h['output_w']<r['required_w'] and h['zone_spread']<=r['max_spread']+.025 and h['max_surface']<r['max_surface'])
    def minor_hall(ri,error):
        h=optimizer.heat['rooms'][d.hall];r=p['rooms'][d.hall]
        return ri==d.hall and error<.025 and h['zone_spread']<=r['max_spread'] and h['max_surface']<=r['max_surface']
    optimizer.hall_only=True
    for iteration in range(5):
        if not bad:break
        if all((i in pair_rooms and e<.03) or (near_capacity_under(i) and e<.08) or minor_hall(i,e) for e,i in bad):break
        if len(bad)==1 and bad[0][1]==d.hall and bad[0][0]<.025 and optimizer.heat['rooms'][d.hall]['zone_spread']<=p['rooms'][d.hall]['max_spread'] and optimizer.heat['rooms'][d.hall]['max_surface']<=p['rooms'][d.hall]['max_surface']:break
        if len(bad)<=2 and all(e<.002 and p['rooms'][i]['required_w']-3<=optimizer.heat['rooms'][i]['output_w']<=p['rooms'][i].get('max_w',1e99)+3 for e,i in bad):break
        now=time.perf_counter();reserve=eval_time*1.35+.15
        fine_reserve=(.65 if len(bad)==1 and bad[0][1] in pair_rooms else 1.1)+eval_time if len(bad)<=2 else .3
        if any(i in pair_rooms and e>.08 for e,i in bad):fine_reserve=.35
        available=deadline-now-reserve-fine_reserve
        if available<.4:break
        if len(bad)<=2 and all(i not in optimizer.prepared_rooms and e<.08 for e,i in bad):break
        heat=optimizer.heat
        hall_failed=any(i==d.hall for _,i in bad)
        power_failed=any(h['output_w']<r['required_w'] or h['output_w']>r.get('max_w',1e99) for r,h in zip(p['rooms'],heat['rooms']))
        room_failed=any(i!=d.hall for _,i in bad)
        hall_severity=max((e for e,i in bad if i==d.hall),default=0.)
        prepared_failed=any(i!=d.hall and i in optimizer.prepared_rooms for _,i in bad)
        unprepared={i for _,i in bad if i!=d.hall and i not in optimizer.prepared_rooms and not near_capacity_under(i)}
        if unprepared and hall_severity<.2 and available>2.2:
            optimizer.prepared_rooms.update(unprepared);optimizer.allowed_edges=set();optimizer.plaqs=[];optimizer.prepare_graph()
            available=deadline-time.perf_counter()-reserve-fine_reserve
            prepared_failed=True
        optimizer.hall_only=(hall_severity>.2) or (optimizer.hall_prepare and not prepared_failed)
        defer_hall=hall_failed and minor_hall(d.hall,hall_severity) and prepared_failed
        if defer_hall:optimizer.hall_only=False
        if prepared_failed and (not hall_failed or defer_hall):optimizer.active_rooms[d.hall]=False
        if hall_failed or power_failed or (room_failed and (prepared_failed or not optimizer.hall_prepare)):
            previous_score=status(p,optimizer.heat)[0]
            stats=optimizer.optimize(seconds=min(.7,available),max_iterations=100000,anneal=.035,max_accept=100000 if hall_failed and not defer_hall and not p['rooms'][d.hall]['required_w']<=optimizer.heat['rooms'][d.hall]['output_w']<=p['rooms'][d.hall].get('max_w',1e99) else 250 if hall_failed and not defer_hall else 100)
            if debug:print('moves',optimizer.hall_only,stats['accepted'],stats['tested'],file=sys.stderr)
            bad=evaluate_current()
            if status(p,optimizer.heat)[0]>bestscore+.35:
                loops[:]=copy.deepcopy(best[5])
                for lp in loops:lp['path']=best[4][lp['slot']].copy()
                sync_hall(loops,best[4]);optimizer.reset_paths(loops)
                optimizer.install_heat(best[1],best[2]);bad=best[3]
            if not bad:break
            if not hall_failed and not power_failed and status(p,optimizer.heat)[0]>=previous_score-1e-5:break
        now=time.perf_counter();available=deadline-now-eval_time*1.35-.2-fine_reserve
        need_refresh=any(i!=d.hall and (e>.18 or i not in pair_rooms and e>.05 and not near_capacity_under(i)) for e,i in bad)
        hall_power_bad=not p['rooms'][d.hall]['required_w']<=optimizer.heat['rooms'][d.hall]['output_w']<=p['rooms'][d.hall].get('max_w',1e99)
        if hall_power_bad and max((e for e,i in bad if i!=d.hall),default=0.)<.25:need_refresh=False
        if need_refresh and available>.3:
            worst=max(((e,i) for e,i in bad if i!=d.hall and (e>.18 or i not in pair_rooms and e>.05 and not near_capacity_under(i))),default=(0,None))[1]
            bad=search_pairs(min(1.55,available)) if worst in pair_rooms else search_patterns(min(1.35,available),True)
        elif not (hall_failed or power_failed or prepared_failed):break
    pending=None
    if bestscore>1e-8 and len(best)>=6:
        targets=[i for e,i in best[3] if i in pair_rooms and e>.03]
        available=deadline-time.perf_counter()-eval_time*1.35-.15
        if targets and available>.3:
            from regional import polish_pair
            loops[:]=copy.deepcopy(best[5])
            for lp in loops:lp['path']=best[4][lp['slot']].copy()
            sync_hall(loops,best[4]);optimizer.reset_paths(loops);optimizer.install_heat(best[1],best[2])
            if polish_pair(optimizer,targets[0],time.perf_counter()+min(1.05,available),debug):
                pending=(layout_of(d,optimizer.paths),optimizer.heat,optimizer.parsed_loops,status(p,optimizer.heat)[1])
    if bestscore>1e-8:
        from refine import refine_room
        layout,heat,parsed,bad=pending if pending is not None else best[:4]
        for epoch in range(2):
            heat={**heat,'rooms':[r.copy() for r in heat['rooms']],'surface':heat['surface'].copy(),'bottom':heat['bottom'].copy(),'water':dict(heat['water'])}
            parsed=parsed.copy()
            paths={lp['slot']:lp['walk'] for lp in parsed}
            fine_deadline=deadline-eval_time*1.35-.12
            available=fine_deadline-time.perf_counter()
            if epoch==0 and available>3.2:fine_deadline=time.perf_counter()+available*.60
            for j,(_,ri) in enumerate(bad):
                remaining=fine_deadline-time.perf_counter()
                if remaining<.15:break
                budget=remaining/max(1,len(bad)-j)
                paths,layout=refine_room(p,optimizer.floor,parsed,heat,paths,ri,time.perf_counter()+budget)
            if time.perf_counter()>=deadline-eval_time*1.35-.05:break
            parsed2,problems=thermal.check_geometry(p,layout,optimizer.floor)
            if problems:break
            t=time.perf_counter();heat2=thermal.solve_heat(p,parsed2,optimizer.floor)
            eval_time=max(eval_time*.96,time.perf_counter()-t)
            score,bad2=status(p,heat2)
            if score<bestscore:
                bestscore=score;best=(layout,heat2,parsed2,bad2)
            if debug:print('fine',round(time.perf_counter()-START,3),'bad',[(p['rooms'][i]['id'],round(e,4)) for e,i in bad2],file=sys.stderr)
            if not bad2:break
            layout,heat,parsed,bad=best[:4]
            if time.perf_counter()>deadline-eval_time*1.35-.35:break
    with open(sys.argv[2],'w') as f:json.dump(best[0],f,separators=(',',':'))
    if debug:print('finished',round(time.perf_counter()-START,3),'score',bestscore,file=sys.stderr)


if __name__=='__main__':main()
