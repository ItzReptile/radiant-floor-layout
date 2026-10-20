from __future__ import annotations
import time
import numpy as np
import thermal
from refine import LocalModel,original_paths,water_powers,score


def polish_pair(optimizer,ri,deadline,debug=False):
    """Recalibrate small grid edits against the exact two-circuit room model."""
    o=optimizer;plan=o.p;room=plan['rooms'][ri];floor=o.floor
    base_heat=o.heat;parsed=o.parsed_loops
    slots=[lp['slot'] for lp in parsed if any(floor.room_at(*q)==ri for q in lp['walk'])]
    if len(slots)!=2:return False
    initial_paths={k:v.copy() for k,v in o.paths.items()}
    model=LocalModel(plan,floor,parsed,base_heat,ri,slots,pad=.35)
    bounds=(model.x0,model.y0,model.x1,model.y1)
    fine=original_paths(o.d,{k:o.paths[k] for k in slots})
    res=model.evaluate(fine,ret_state=True)
    initial_score=score(room,res);best_score=initial_score
    old_local=res['_local_outputs'].copy()
    old_out=np.array([h['output_w'] for h in base_heat['rooms']])
    old_power=sum((water_powers(plan,floor,fine[k],base_heat['water'][k],outside=bounds) for k in slots),np.zeros(len(plan['rooms'])))
    region=np.s_[model.y0-floor.by0:model.y1-floor.by0+1,model.x0-floor.bx0:model.x1-floor.bx0+1]
    def objective(h):return score(room,h)
    current_objective=objective(res)
    current_paths=initial_paths;current_heat=base_heat;current_parsed=parsed
    best=(initial_paths,base_heat,parsed)
    def restore(paths,heat,loops):
        o.reset_paths([{'slot':k,'path':v.copy()} for k,v in paths.items()])
        o.install_heat(heat,loops,paths)
    for epoch in range(7):
        remaining=deadline-time.perf_counter()
        if remaining<.13:break
        o.active_rooms=[j==ri for j in range(len(plan['rooms']))]
        o.freeze_children=set();o.hall_only=False
        stats=o.optimize(seconds=min(.14,max(.025,remaining-.11)),max_iterations=4000,max_accept=35,anneal=.015)
        if not stats['accepted']:break
        paths={k:v.copy() for k,v in o.trace().items()}
        fine=original_paths(o.d,{k:paths[k] for k in slots})
        result=model.evaluate(fine,ret_state=True)
        power=sum((water_powers(plan,floor,fine[k],result['_water'][k],outside=bounds) for k in slots),np.zeros(len(plan['rooms'])))
        predicted=old_out+result['_local_outputs']-old_local+.8*(power-old_power)
        safe=True
        for j,rr in enumerate(plan['rooms']):
            if j==ri:continue
            if predicted[j]<min(old_out[j],rr['required_w']+.8)-.15 or predicted[j]>max(old_out[j],rr.get('max_w',1e99)-.8)+.15:safe=False;break
        candidate_objective=objective(result)
        if debug:print('regional',epoch,round(result['output_w']),round(result['zone_spread'],4),safe,file=__import__('sys').stderr)
        if not safe:
            restore(current_paths,current_heat,current_parsed)
            continue
        heat={**base_heat,'surface':base_heat['surface'].copy(),'bottom':base_heat['bottom'].copy(),'water':{**base_heat['water'],**result['_water']},'rooms':[h.copy() for h in base_heat['rooms']]}
        heat['surface'][region]=result['_surface'];heat['bottom'][region]=result['_bottom']
        for j,h in enumerate(heat['rooms']):h['output_w']=float(predicted[j])
        heat['rooms'][ri].update({k:result[k] for k in ('output_w','max_surface','zone_spread')})
        loops=parsed.copy()
        for j,lp in enumerate(loops):
            if lp['slot'] in slots:
                walk=fine[lp['slot']];corners=thermal.simplify(walk)
                loops[j]={**lp,'walk':walk,'points':corners,'segments':list(zip(corners,corners[1:])),'length_units':len(walk)-1}
        actual_score=score(room,result)
        if actual_score<best_score-1e-5:
            best_score=actual_score;best=(paths,heat,loops)
        if candidate_objective<current_objective:
            current_objective=candidate_objective;current_paths,current_heat,current_parsed=paths,heat,loops
            o.install_heat(heat,loops,paths)
        else:restore(current_paths,current_heat,current_parsed)
        if result['zone_spread']<room['max_spread']-.04 and room['required_w']+1<result['output_w']<room.get('max_w',1e99)-1 and result['max_surface']<room['max_surface']-.05:break
    restore(*best)
    return best_score<initial_score-1e-5
