from __future__ import annotations
import time,math
import numpy as np
from contours import room_cycles
from patterns import strip_cycle,assemble,results_of
from refine import LocalModel,original_paths
import thermal


def pair_score(room,res):
    width=max(10.,room.get('max_w',1e99)-room['required_w']);out=res['output_w'];spread=res['zone_spread'];lim=room['max_spread']
    es=[max(0,room['required_w']-out)/width,max(0,out-room.get('max_w',1e99))/width,max(0,spread-lim)/lim,max(0,res['max_surface']-room['max_surface'])/3]
    cost=sum(400*x+4000*x*x for x in es)
    cost+=15*(max(0,spread-lim+.18)/lim)**2+3*(res['zone_std']/lim)**2+.3*res['soft_spread']/lim
    cost+=5*((out-room['required_w']-.45*width)/width)**2
    good=(room['required_w']+2<out<room.get('max_w',1e99)-2 and spread<lim-.08 and res['max_surface']<room['max_surface']-.05)
    return cost,good


def pair_options(comp,out,room):
    p=comp['pitch'];off=comp['offset'];bal=comp.get('balanced',False);opts=[]
    if isinstance(p,str) and p[0]=='z':
        spec=p[1:].split('g');pitch=float(spec[0]);grade=float(spec[1]) if len(spec)>1 else 0.
        direction=1 if out>room['required_w']+.6*(room.get('max_w',1e99)-room['required_w']) else -1
        for q in (.35,.65):opts.append(('z'+str(max(1.,pitch+direction*q))+('g'+str(grade) if grade else ''),off,bal))
        if isinstance(off,(tuple,list)):
            opts.extend([(p,(off[0],off[1]+1),bal),(p,(off[0]+1,off[1]),bal)])
        opts.extend([('z'+str(pitch)+(('g'+str(z)) if z else ''),off,bal) for z in (0.,.25,-.25,.5,-.5)])
    elif isinstance(p,(int,float)):
        direction=1 if out>room['required_w']+.6*(room.get('max_w',1e99)-room['required_w']) else -1
        opts.extend([(max(1.,p+direction*.5),off,True),(p,off,True),('z2.5',(0,0),True),('z2.5',(0,0),False),(p,off,3),(p,off,4)])
    opts.extend([(2.5,1,False),(2,1,False),(2,2,False),(2.5,2,False),(3,0,False)])
    for pitch in (2.5,3.,2.,3.5,1.75):
        for mirror in (True,False):opts.extend([('z'+str(pitch),(0,0),mirror),('z'+str(pitch),(0,1),mirror)])
    if room.get('zone_margin',0)>=.4:
        inset=max(1,round(room.get('zone_margin',.5)/.25))
        inner=[('z2g0.25',(inset,max(0,inset-1)),False),('z2g0.5',(inset,0),False),('z2g0.25',(max(0,inset-1),max(0,inset-1)),False)]
        opts=inner+opts
    if isinstance(p,str) and p.startswith('z'):
        simple='z'+p[1:].split('g')[0]
        opts=[(simple,off,False),(simple,off,True)]+opts
    return opts


def optimize_pairs(d,loops,floor,parsed,heat,deadline,debug=False):
    groups={}
    for lp in loops:
        if len(lp['rooms'])==1:groups.setdefault(lp['rooms'][0],[]).append(lp)
    changed=0
    for ri,lps in groups.items():
        if len(lps)!=2:continue
        room=d.plan['rooms'][ri];h=heat['rooms'][ri]
        if room['required_w']<=h['output_w']<=room.get('max_w',1e99) and h['zone_spread']<=room['max_spread'] and h['max_surface']<=room['max_surface']:continue
        slots=[q['slot'] for q in lps];model=LocalModel(d.plan,floor,parsed,heat,ri,slots)
        cur_paths={lp['slot']:lp['path'] for lp in lps};fine=original_paths(d,cur_paths)
        res=model.evaluate(fine,ret_state=True);cur_score,good=pair_score(room,res);seen=set();accepted=False
        rect=room['rect'];area=(rect[2]-rect[0])*(rect[3]-rect[1]);pitch=2. if room['required_w']/area<30 else 1.75
        inset=max(1,round(room.get('zone_margin',.5)/.25))
        a=(inset,max(0,inset-1),.25);b=(inset,0,.5)
        seeds=[(pitch,a,b,False),(pitch,b,a,False),(max(1.25,pitch-.25),a,b,False),(pitch+.25,a,b,False),(pitch,a,b,True)]
        seed_deadline=time.perf_counter()+max(.12,(deadline-time.perf_counter())*.48)
        for pp,aa,bb,mirror in seeds:
            if good or time.perf_counter()>seed_deadline-.06:break
            cs=[];new_paths={};new_parts={}
            try:
                for lp,spec in zip(lps,(aa,bb)):
                    c=lp['components'][0];li,si,grade=spec;off=(li,si)
                    curve=strip_cycle(d,c,pp,off,mirror,grade)
                    cc={**c,'cycle':curve,'pitch':'z'+str(pp)+'g'+str(grade),'offset':off,'balanced':mirror}
                    path,part=assemble(d,lp,[cc])
                    if (len(path)-1)*d.unit>d.plan['pipe']['max_loop_length']+1e-7:raise ValueError('long seed')
                    cs.append(cc);new_paths[lp['slot']]=path;new_parts[lp['slot']]=part
                trial_fine=original_paths(d,new_paths);result=model.evaluate(trial_fine,ret_state=True);score,passed=pair_score(room,result)
                if score<cur_score-1e-5:
                    cur_score=score;res=result;fine=trial_fine;accepted=True;changed+=2;good=passed
                    for lp,cc in zip(lps,cs):lp['path']=new_paths[lp['slot']];lp['room_path']=new_parts[lp['slot']];lp['components']=[cc]
            except (ValueError,KeyError,IndexError):continue
        for sweep in range(2):
            for index,lp in enumerate(lps):
                if good or time.perf_counter()>deadline-.1:break
                stop=min(deadline,time.perf_counter()+(deadline-time.perf_counter())/max(1,len(lps)-index))
                comp=lp['components'][0];opts=pair_options(comp,res['output_w'],room)
                for pitch,off,bal in opts:
                    if time.perf_counter()>stop-.06:break
                    try:
                        if isinstance(pitch,str):
                            spec=pitch[1:].split('g');cycle=strip_cycle(d,comp,float(spec[0]),off,bal,float(spec[1]) if len(spec)>1 else 0.)
                        else:
                            cs,rem=room_cycles(d,ri,pitch,off,comp['bounds'],bal)
                            if rem or len(cs)!=1:continue
                            cycle=cs[0]
                        cc={**comp,'cycle':cycle,'pitch':pitch,'offset':off,'balanced':bal}
                        path,part=assemble(d,lp,[cc]);slot=lp['slot']
                        if (len(path)-1)*d.unit>d.plan['pipe']['max_loop_length']+1e-7:continue
                        sig=(slot,tuple(path))
                        if sig in seen:continue
                        seen.add(sig)
                        trial_fine={**fine,slot:original_paths(d,{slot:path})[slot]};result=model.evaluate(trial_fine,ret_state=True);score,passed=pair_score(room,result)
                    except (ValueError,KeyError,IndexError):continue
                    if score<cur_score-1e-5:
                        cur_score=score;res=result;fine=trial_fine;lp['path']=path;lp['room_path']=part;lp['components']=[cc];accepted=True;changed+=1;good=passed
                        if good:break
            if good:break
        if accepted:
            sl=np.s_[model.y0-floor.by0:model.y1-floor.by0+1,model.x0-floor.bx0:model.x1-floor.bx0+1]
            heat['surface'][sl]=res['_surface'];heat['bottom'][sl]=res['_bottom'];heat['water'].update(res['_water']);heat['rooms'][ri].update({key:res[key] for key in ('output_w','max_surface','zone_spread')})
            for j,old in enumerate(parsed):
                if old['slot'] in slots:
                    path=fine[old['slot']];corners=thermal.simplify(path);parsed[j]={**old,'walk':path,'points':corners,'segments':list(zip(corners,corners[1:])),'length_units':len(path)-1}
        if debug:print('pair',ri,[(q['components'][0]['pitch'],q['components'][0]['offset'],q['components'][0]['balanced']) for q in lps],round(res['output_w']),round(res['zone_spread'],3),file=__import__('sys').stderr)
    return changed
