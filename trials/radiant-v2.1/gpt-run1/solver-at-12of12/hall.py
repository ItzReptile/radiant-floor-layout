from __future__ import annotations
import copy,time
import numpy as np
from contours import Cycles,room_cycles
from refine import LocalModel,original_paths,water_powers
from patterns import sync_hall
import thermal


def branch_score(room,res,surface,zone):
    spread=room['max_spread'];z=surface[zone];target=float(z.max())-spread+.22
    width=max(10.,room.get('max_w',1e99)-room['required_w']);out=res['output_w']
    lo=room['required_w']+.14*width;hi=room.get('max_w',1e99)-.12*width
    return (25*max(0.,res['zone_spread']-spread+.12)+8*float(np.mean(np.maximum(0.,target-z)**2))+
            8*(max(0.,lo-out)/width)**2+40*(max(0.,out-hi)/width)**2+
            100*max(0.,out-room.get('max_w',1e99))/width)


def splice(d,lp,path,curve,mask,occ):
    c=Cycles([path,curve]);ia=path.index(lp['hall_paths'][0][-1]);ib=path.index(lp['hall_paths'][1][-1])
    for j,(a,b) in enumerate(zip(path,path[1:]+path[:1])):
        if ia<=j<ib or j==len(path)-1:c.protected.add(tuple(sorted((a,b))))
    own=set(path)
    def allowed(q):return d.x0<=q[0]<=d.x1 and d.y0<=q[1]<=d.y1 and bool(mask[q[1]-d.y0,q[0]-d.x0]) and (q not in occ or q in own)
    if c.merge_all(allowed,2,False,True):return None
    c.remove_edge(path[0],path[-1]);a=path[0];last=None;out=[a]
    while a!=path[-1]:
        b=next(z for z in c.adj[a] if z!=last);out.append(b);last,a=a,b
    if (len(out)-1)*d.unit>d.plan['pipe']['max_loop_length']+1e-7:return None
    return out


def optimize_ends(d,loops,floor,parsed,heat,deadline,debug=False):
    hall=d.hall;room=d.plan['rooms'][hall];r=d.rooms[hall];paths={lp['slot']:lp['path'] for lp in loops};slots=list(paths)
    if heat['rooms'][hall]['output_w']>room.get('max_w',1e99):return 0
    mask=d.room_mask(hall);changed=0
    for turn in range(2):
        if time.perf_counter()>deadline-.15:break
        turn_deadline=time.perf_counter()+(deadline-time.perf_counter())/(2-turn)
        model=LocalModel(d.plan,floor,parsed,heat,hall,slots)
        surf=heat['surface'][model.y0-floor.by0:model.y1-floor.by0+1,model.x0-floor.bx0:model.x1-floor.bx0+1]
        yy,xx=np.nonzero(model.zone);z=surf[model.zone];j=int(z.argmin());ox,oy=int(xx[j]+model.x0),int(yy[j]+model.y0)
        cold=((ox-d.origin[0])*d.d[0]+(oy-d.origin[1])*d.d[1])/d.step;left=cold<(r[0]+r[2])/2
        occ={q for path in paths.values() for q in path};free=mask.copy()
        for x,y in occ:free[y-d.y0,x-d.x0]=False
        newd=copy.copy(d);newd.allowed=free
        base_score=branch_score(room,heat['rooms'][hall],surf,model.zone)
        old_fine={q['slot']:q['walk'] for q in parsed}
        local_bounds=(model.x0,model.y0,model.x1,model.y1)
        old_power=sum((water_powers(d.plan,floor,q['walk'],heat['water'][q['slot']],outside=local_bounds) for q in parsed),np.zeros(len(d.plan['rooms'])))
        air=np.array([rr['air_temp'] for rr in d.plan['rooms']])[model.room_of]
        old_local=np.bincount(model.room_of.ravel(),weights=(model.groom*(surf-air)).ravel(),minlength=len(d.plan['rooms']))
        room_out=np.array([q['output_w'] for q in heat['rooms']])
        def outside_score(out):
            total=0.
            for ri,rr in enumerate(d.plan['rooms']):
                if ri==hall:continue
                width=max(10.,rr.get('max_w',1e99)-rr['required_w'])
                total+=(max(0.,rr['required_w']-out[ri])+max(0.,out[ri]-rr.get('max_w',1e99)))/width
            return total
        base_score+=15*outside_score(room_out)
        banks=[];bulk=[];ends=[]
        if heat['rooms'][hall]['output_w']<room['required_w']+10:
            for offset in (0,1):
                try:cs,rem=room_cycles(newd,hall,2.5,offset,[r])
                except (ValueError,IndexError):continue
                bulk.extend(sorted((c for c in cs if len(c)*d.unit>1.),key=len,reverse=True)[:4])
        if min(cold-r[0],r[2]-cold)<max(20.,(r[2]-r[0])*.25):
            for w,offset in ((12,1),(18,1),(8,0),(12,2),(24,1)):
                bounds=(r[0],r[1],min(r[2],cold+w),r[3]) if left else (max(r[0],cold-w),r[1],r[2],r[3])
                try:cs,rem=room_cycles(newd,hall,2.5,offset,[bounds])
                except (ValueError,IndexError):continue
                ends.extend(sorted((c for c in cs if len(c)*d.unit>.6),key=len,reverse=True)[:2])
        if turn==0 and heat['rooms'][hall]['output_w']<room['required_w']-15:banks=bulk+ends
        else:banks=ends+bulk
        best=None;seen=set();seen_cycles=set()
        for curve in banks:
            if time.perf_counter()>turn_deadline-.08:break
            sig=frozenset(curve)
            if sig in seen_cycles:continue
            seen_cycles.add(sig)
            cx=sum(v[0] for v in curve)/len(curve);cy=sum(v[1] for v in curve)/len(curve)
            ordered=sorted(loops,key=lambda lp:min(abs(q[0]-cx)+abs(q[1]-cy) for q in lp['hall_paths'][0]+lp['hall_paths'][1]))
            for lp in ordered:
                if time.perf_counter()>turn_deadline-.08:break
                slot=lp['slot'];path=paths[slot]
                if len(curve)*d.unit>(d.plan['pipe']['max_loop_length']-(len(path)-1)*d.unit)+.6:continue
                out=splice(d,lp,path,curve,mask,occ)
                if out is None:continue
                sig=(slot,tuple(out))
                if sig in seen:continue
                seen.add(sig)
                fine=original_paths(d,{slot:out})[slot];fine_paths={**old_fine,slot:fine}
                res=model.evaluate(fine_paths,ret_state=True)
                power=sum((water_powers(d.plan,floor,fine_paths[k],res['_water'][k],outside=local_bounds) for k in slots),np.zeros(len(d.plan['rooms'])))
                predicted=room_out+res['_local_outputs']-old_local+.8*(power-old_power)
                unsafe=False
                for ri,rr in enumerate(d.plan['rooms']):
                    if ri==hall:continue
                    old=room_out[ri]
                    if predicted[ri]<min(old,rr['required_w']+1.5)-.2 or predicted[ri]>max(old,rr.get('max_w',1e99)-1.5)+.2:unsafe=True;break
                if unsafe or res['max_surface']>room['max_surface']-.04:continue
                if room_out[hall]<=room.get('max_w',1e99) and res['output_w']>room.get('max_w',1e99)-1:continue
                sc=branch_score(room,res,res['_surface'],model.zone)+15*outside_score(predicted)+.001*(len(out)-len(path))
                if sc<base_score-1e-5 and (best is None or sc<best[0]):best=(sc,lp,out,fine,res,predicted)
        if best is None:break
        _,lp,out,fine,res,predicted=best;slot=lp['slot'];paths[slot]=out;lp['path']=out;sync_hall(loops,paths)
        sl=np.s_[model.y0-floor.by0:model.y1-floor.by0+1,model.x0-floor.bx0:model.x1-floor.bx0+1]
        heat['surface'][sl]=res['_surface'];heat['bottom'][sl]=res['_bottom'];heat['water'].update(res['_water'])
        heat['rooms'][hall].update({key:res[key] for key in ('output_w','max_surface','zone_spread')})
        for ri in range(len(d.plan['rooms'])):
            if ri!=hall:heat['rooms'][ri]['output_w']=float(predicted[ri])
        for j,old in enumerate(parsed):
            if old['slot']==slot:
                corners=thermal.simplify(fine);parsed[j]={**old,'walk':fine,'points':corners,'segments':list(zip(corners,corners[1:])),'length_units':len(fine)-1}
        changed+=1
        if debug:print('hall-fill',slot,round(res['output_w']),round(res['zone_spread'],3),file=__import__('sys').stderr)
    return changed
