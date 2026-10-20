from __future__ import annotations
import math,time
import numpy as np
from scipy.ndimage import binary_opening
import thermal
from contours import room_cycles,line
from routing import join_rooms,entrance
from refine import LocalModel,original_paths


def results_of(res):
    return res['rooms'] if 'rooms' in res else [res]


def room_for_result(plan,h):
    r=plan['rooms'][h['ri']]
    if 'fraction' in h:
        f=h['fraction'];r={**r,'required_w':r['required_w']*f,'max_w':r.get('max_w',r['required_w']*1.3)*f,'max_spread':r['max_spread']*h.get('spread_scale',1.)}
    return r


def pattern_score(plan,res):
    cost=0.;passed=True
    for h in results_of(res):
        r=room_for_result(plan,h);width=max(10.,r.get('max_w',r['required_w']*1.3)-r['required_w'])
        lo=r['required_w']+.20*width;hi=r.get('max_w',float('inf'))-.12*width
        spread=r['max_spread']-.12
        es=(max(0.,lo-h['output_w'])/width,max(0.,h['output_w']-hi)/width,
            max(0.,h['max_surface']-r['max_surface']+.10)/3.,max(0.,h['zone_spread']-spread)/r['max_spread'])
        actual=(max(0.,r['required_w']-h['output_w'])/width,max(0.,h['output_w']-r.get('max_w',1e99))/width,
                max(0.,h['max_surface']-r['max_surface'])/3.,max(0.,h['zone_spread']-r['max_spread'])/r['max_spread'])
        cost+=sum(1000*e+10000*e*e for e in actual)
        cost+=sum(30*e+150*e*e for e in es)
        cost+=.1*(h['zone_spread']/r['max_spread'])**2+.01*((h['output_w']-(lo+hi)*.5)/width)**2
        passed &= (h['output_w']>=r['required_w']+(.03 if 'fraction' in h else .08 if r['max_spread']<=2. else .20)*width and h['output_w']<=r.get('max_w',1e99)-.03*width and h['max_surface']<=r['max_surface']-.03 and h['zone_spread']<=r['max_spread']-.04)
    return cost,passed


def snake_cycle(domain,component,nrows,insets=(0,1),mirror=False,central_pitch=None,fixed_spine=False):
    ri=component['room'];mask=domain.room_mask(ri)
    xx=domain.xs[None,:];yy=domain.ys[:,None];region=np.zeros_like(mask)
    for b in component['bounds']:region|=(xx>=b[0]-1e-7)&(xx<=b[2]+1e-7)&(yy>=b[1]-1e-7)&(yy<=b[3]+1e-7)
    mask&=region
    yy,xx=np.nonzero(mask);xx=xx+domain.x0;yy=yy+domain.y0
    vertical=(yy.max()-yy.min())>(xx.max()-xx.min())
    us,vs=(yy,xx) if vertical else (xx,yy)
    u0,u1,v0,v1=int(us.min()),int(us.max()),int(vs.min()),int(vs.max())
    long_inset,short_inset=(insets,insets) if isinstance(insets,(int,float)) else insets
    spine=u0 if fixed_spine else u0+long_inset;left=u0+long_inset+1;right=u1-long_inset
    if central_pitch is None:
        first=v0+short_inset;last=v1-short_inset
        rows=[round(first+j*(last-first)/(nrows-1)) for j in range(nrows)]
    else:
        center=round((v0+v1)/2)+short_inset
        rows=[round(center+(j-(nrows-1)/2)*central_pitch) for j in range(nrows)]
        first,last=rows[0],rows[-1]
    if right-left<2 or last-first<nrows-1 or first<v0 or last>v1:raise ValueError('small snake')
    points=[(left,rows[0])]
    for j,v in enumerate(rows):
        target=(right,v) if j%2==0 else (left,v)
        points.extend(line(points[-1],target)[1:])
        if j+1<nrows:points.extend(line(points[-1],(target[0],rows[j+1]))[1:])
    points.extend(line(points[-1],(spine,rows[-1]))[1:])
    points.extend(line(points[-1],(spine,rows[0]))[1:])
    points.extend(line(points[-1],(left,rows[0]))[1:-1])
    if mirror:points=[(u0+u1-u,v) for u,v in points]
    if vertical:points=[(v,u) for u,v in points]
    if not all(mask[p[1]-domain.y0,p[0]-domain.x0] for p in points):raise ValueError('blocked snake')
    return points


def config_trials(plan,component,baseline,refresh=False):
    ri=component['room'];p=component['pitch'];off=component['offset'];bal=component.get('balanced',False)
    res=next(h for h in results_of(baseline) if h['ri']==ri);r=room_for_result(plan,res)
    trials=[]
    if r['max_spread']<=2.0 and 'fraction' not in res:
        bounds=component['bounds'][0];short_y=bounds[3]-bounds[1]<bounds[2]-bounds[0]
        ox,oy=(off,off) if isinstance(off,(int,float)) else off
        nn=int(p[1:].split('c')[0]) if isinstance(p,str) and p.startswith('s') else max(4,int(math.ceil((min(bounds[2]-bounds[0],bounds[3]-bounds[1])/2+1)/2))*2)
        trials.extend([('s'+str(nn),(0,1),True),('s'+str(nn),(0,2),True),('s'+str(nn),(0,1),False),('s'+str(nn),(0,2),False),('s'+str(nn),(0,0),True),('s'+str(nn),(0,0),False),('s'+str(max(4,nn-2)),(0,1),True),('s'+str(max(4,nn-2)),(0,1),False),('s'+str(nn+2),(0,0),True),('s'+str(nn),(1,1),True),('s4c2',(0,0),False),('s4c2.5',(0,0),False),('s4c2',(0,0),True),('s4c3',(0,0),False),('s6c2',(0,0),False),('s4c2.5',(0,0),True),('s6',(0,2),False),('s4',(0,0),False)])
        if res['output_w']<r['required_w']+.15*(r.get('max_w',1e99)-r['required_w']):
            trials[0:0]=[('s'+str(nn+2),(0,0),True),('s'+str(nn+2),(0,1),True)]
        elif res['output_w']>r.get('max_w',1e99)-.15*(r.get('max_w',1e99)-r['required_w']):
            trials[0:0]=[('s'+str(max(4,nn-2)),(0,2),True),('s'+str(nn),(0,2),True)]
        trials.append((p,(ox,oy),not bal))
        for val in (0,1,2,3):trials.append((p,(val,oy) if short_y else (ox,val),bal))
        trials.extend([(2.5,off,bal),(2,max(0,min(ox,oy)-1),bal),(2,max(ox,oy)+1,bal),(1.5,off,bal)])
    else:
        if 'fraction' in res:
            trials.extend([('z2',(0,0),False),('z2',(0,0),True),('z2g0.5',(0,0),False),('z2g-0.5',(0,0),False),('z2g0.5',(0,0),True),('z2g-0.5',(0,0),True),(2,2,False),(2.5,1,False),(2.5,2,False),(2,1,False),(2.25,2,False),(2.25,1,False),(1.75,2,False),(3,0,False),('z2.5',(0,0),False),('z2.5',(0,0),True),('z2',(0,1),False),('z2',(0,1),True)])
        elif r['air_temp']>=23:
            if res['output_w']<r['required_w']-.6*(r.get('max_w',1e99)-r['required_w']):
                trials.extend([('z1.75',(0,0),False),('z1.75',(0,0),True),('z2',(0,0),False),('z2',(0,0),True)])
            trials.extend([('z2.5',(0,0),False),('z2.5',(0,0),True),('z2',(0,0),False),('z2',(0,0),True),('z1.75',(0,0),False),('z1.75',(0,0),True),('z1.5',(0,0),False),('z1.5',(0,0),True),(2,1,False),(2,0,False),(1.5,0,False)])
        elif r.get('floor_r',0)<=.02:
            trials.extend([(2,1,False),(2.25,1,False),(2.25,0,False),(2,(1,2),False),(2,(2,1),False),(2.5,1,False),(3,0,False),(2,2,False),(1.75,1,False),(2.5,2,False),(2.25,(1,2),False),(2.25,(2,1),False)])
        elif r.get('floor_r',0)>=.08:
            if res['output_w']<r['required_w']-.4*(r.get('max_w',1e99)-r['required_w']):trials.extend([(2,1,True),(2,1,False)])
            trials.extend([(2.5,1,False),(2,1,False),(2,1,True),(2.25,0,5),(2.25,0,6),(2,(0,1),False),(2,(1,0),False),(2,0,5),(2,0,6),(1.75,1,False)])
        else:
            trials.extend([(2.5,1,False),(2,1,False),(2.25,0,False),(2.5,0,False),('z2.5',(0,0),False),('z2.5',(0,0),True),('z2',(0,0),False),('z2',(0,0),True),(2,2,False),(3,0,False),(2.25,1,False),(2.5,2,False)])
        if isinstance(off,(tuple,list)):
            ox,oy=off;lowoff=tuple(max(0,int(v)-1) for v in off);highoff=tuple(int(v)+1 for v in off)
        else:ox=oy=off;lowoff=max(0,off-1);highoff=off+1
        if isinstance(p,str) and p.startswith('z'):
            current=float(p[1:].split('g')[0]);grading=('g'+p.split('g')[1]) if 'g' in p else '';direction=-1 if res['output_w']<(r['required_w']+r.get('max_w',1e99))*.5 else 1
            trials[0:0]=[('z'+str(max(1.,current+direction*.25))+grading,off,bal),('z'+str(max(1.,current+direction*.5))+grading,off,bal)]
            window=r.get('max_w',1e99)-r['required_w']
            if res['output_w']>r.get('max_w',1e99)-.15*window:
                trials[0:0]=[(p,(ox+1,oy),bal),(p,(ox,oy+1),bal),(p,(ox+1,oy+1),bal)]
            elif res['output_w']<r['required_w']+.15*window:
                trials[0:0]=[(p,(max(0,ox-1),oy),bal),(p,(ox,max(0,oy-1)),bal)]
        trials.extend([(p,lowoff,bal),(p,highoff,bal),(p,(max(0,ox-1),oy),bal),(p,(ox,max(0,oy-1)),bal),(p,(ox+1,oy),bal),(p,(ox,oy+1),bal)])
        if not isinstance(p,str):trials.extend([(p,off,not bal),(p,off,3),(p,off,4),(max(1.,p-.25),off,bal),(p+.25,off,bal)])
    if refresh and isinstance(p,(int,float)) and res['zone_spread']>r['max_spread'] and r['required_w']<=res['output_w']<=r.get('max_w',1e99):
        trials[0:0]=[(p,off,bb) for bb in (3,4,True,5,6) if bb!=bal]
    if isinstance(p,(int,float)) and res['output_w']>r.get('max_w',1e99):
        ox,oy=(off,off) if isinstance(off,(int,float)) else off
        trials[0:0]=[(p,(ox+1,oy),bal),(p,(ox,oy+1),bal),(p,tuple((ox+1,oy+1)),bal),(p+.25,off,bal),(p+.5,off,bal)]
    unique=[];seen={(p,tuple(off) if isinstance(off,(tuple,list)) else off,bal)}
    for pp,oo,bb in trials:
        key=(pp,tuple(oo) if isinstance(oo,(tuple,list)) else oo,bb)
        if key not in seen:seen.add(key);unique.append((pp,oo,bb))
    return unique


def assemble(domain,lp,components):
    if len(components)==2:
        cycle=join_rooms(domain,components[0]['cycle'],components[1]['cycle'],lp['join'])
    else:cycle=components[0]['cycle']
    fixed=(lp['hall_paths'][0][-1],lp['hall_paths'][1][-1])
    ent=entrance(domain,cycle,lp['rooms'][0],lp['passage'],fixed,components[0]['bounds'])
    part=ent['path']
    if part[0]!=fixed[0]:part=part[::-1]
    path=lp['hall_paths'][0][:-1]+part+lp['hall_paths'][1][-2::-1]
    return path,part


def optimize_patterns(domain,loops,floor,parsed,heat,deadline,max_trials=9,debug=False,only_failed=False,exclude_pairs=False):
    plan=domain.plan
    # Solve each whole room circuit; the two halves of a large room share the floor model.
    xx=floor.xs[None,:]-domain.origin[0];yy=floor.ys[:,None]-domain.origin[1]
    cx=(xx*domain.d[0]+yy*domain.d[1])/domain.step
    cy=(xx*domain.normal[0]+yy*domain.normal[1])/domain.step
    def priority(lp):
        duplicate=len(lp['rooms'])==1 and sum(q['rooms']==lp['rooms'] for q in loops)>1
        if not duplicate:return (0 if len(lp['components'])>1 else 1,0)
        ri=lp['rooms'][0];r=domain.rooms[ri];zm=plan['rooms'][ri].get('zone_margin',0)/domain.unit
        sel=(floor.room_of==ri)&(cx>=r[0]+zm-1e-7)&(cx<=r[2]-zm+1e-7)&(cy>=r[1]+zm-1e-7)&(cy<=r[3]-zm+1e-7)
        region=np.zeros_like(sel)
        for b in lp['bounds']:region|=(cx>=b[0]-1e-7)&(cx<=b[2]+1e-7)&(cy>=b[1]-1e-7)&(cy<=b[3]+1e-7)
        sel&=region
        for k in domain.keepouts:sel&=~((cx>=k[0]-zm-1e-7)&(cx<=k[2]+zm+1e-7)&(cy>=k[1]-zm-1e-7)&(cy<=k[3]+zm+1e-7))
        return (2,float(heat['surface'][sel].mean()) if sel.any() else 0)
    ordered=sorted([lp for lp in loops if not exclude_pairs or not(len(lp['rooms'])==1 and sum(q['rooms']==lp['rooms'] for q in loops)>1)],key=priority)
    changed=0;trials_count=0
    for position,lp in enumerate(ordered):
        if time.perf_counter()>=deadline:break
        remaining=sum(2 if len(q['components'])>1 else 1 for q in ordered[position:] if not only_failed or any(plan['rooms'][ri]['required_w']>heat['rooms'][ri]['output_w'] or heat['rooms'][ri]['output_w']>plan['rooms'][ri].get('max_w',1e99) or heat['rooms'][ri]['max_surface']>plan['rooms'][ri]['max_surface'] or heat['rooms'][ri]['zone_spread']>plan['rooms'][ri]['max_spread'] for ri in q['rooms']))
        share=(2 if len(lp['components'])>1 else 1)/max(1,remaining)
        loop_deadline=min(deadline,time.perf_counter()+max(.12,(deadline-time.perf_counter())*share))
        slot=lp['slot'];ris=lp['rooms']
        if only_failed and all(plan['rooms'][ri]['required_w']<=heat['rooms'][ri]['output_w']<=plan['rooms'][ri].get('max_w',1e99) and heat['rooms'][ri]['max_surface']<=plan['rooms'][ri]['max_surface'] and heat['rooms'][ri]['zone_spread']<=plan['rooms'][ri]['max_spread'] for ri in ris):continue
        region_mask=None
        if len(ris)==1 and sum(q['rooms']==ris for q in loops)>1:
            rr=domain.rooms[ris[0]];region_mask=np.zeros_like(floor.room_of,dtype=bool)
            for b in lp['bounds']:
                maxx=(cx<b[2]-1e-7) if b[2]<rr[2]-1e-7 else (cx<=b[2]+1e-7)
                maxy=(cy<b[3]-1e-7) if b[3]<rr[3]-1e-7 else (cy<=b[3]+1e-7)
                region_mask|=(cx>=b[0]-1e-7)&(cy>=b[1]-1e-7)&maxx&maxy
        model=LocalModel(plan,floor,parsed,heat,ris[0],slot,room_ids=ris,region_mask=region_mask)
        fine=next(q['walk'] for q in parsed if q['slot']==slot)
        baseline=model.evaluate(fine,ret_state=True);cur_score,good=pattern_score(plan,baseline)
        if good:continue
        cur_path=lp['path'];cur_part=lp['room_path'];components=[dict(c) for c in lp['components']]
        seen_paths={tuple(cur_path)}
        if len(components)==2:
            parent,child=components
            mask=domain.room_mask(child['room']);child_y,child_x=np.nonzero(mask)
            span=min(int(child_x.max()-child_x.min()),int(child_y.max()-child_y.min()))
            child_n=max(4,int(math.ceil((span/2+1)/2))*2)
            child_curves={}
            joint_trials=[(child_n,2.5,0,False,0),(child_n,2.5,0,True,0),(child_n,2.75,0,False,0),(child_n,2.25,0,False,0),(max(4,child_n-2),3,1,True,2),(max(4,child_n-2),3,1,True,0),(max(4,child_n-2),3,0,False,0),(child_n,2,0,False,0),(child_n,2,0,True,0),(child_n,3,1,False,0),(child_n,3,1,True,0),(max(4,child_n-2),2.5,0,True,0),(child_n,3,0,False,0),(child_n,3,0,True,0),(child_n,2.5,1,False,0),(child_n,2.5,1,True,0)]
            joint_trials=[(*q,0) for q in joint_trials[:2]]+[(child_n,3,0,False,0,1),(child_n,2,0,False,0,1),(child_n,2,0,False,0,2),(max(4,child_n-2),3,1,True,2,0),(child_n,2.5,0,False,0,1)]+[(*q,0) for q in joint_trials[2:]]
            pres,cres=results_of(baseline)
            if isinstance(parent['pitch'],str) and parent['pitch'].startswith('z'):
                pp=float(parent['pitch'][1:].split('g')[0]);li,si=parent['offset'];mir=parent.get('balanced',False)
                cp=child['pitch'];cn=int(cp[1:].split('c')[0]) if isinstance(cp,str) and cp.startswith('s') else child_n
                cli,csi=child['offset'] if isinstance(child['offset'],(tuple,list)) else (0,child['offset'])
                pr=plan['rooms'][parent['room']];cr=plan['rooms'][child['room']]
                dynamic=[]
                if pres['output_w']>pr.get('max_w',1e99)-.1*(pr.get('max_w',1e99)-pr['required_w']):
                    dynamic.extend([(cn,pp,si,mir,csi,li+1),(cn,pp,si,mir,csi,li+2),(cn,pp+.25,si,mir,csi,li)])
                elif pres['output_w']<pr['required_w']+.1*(pr.get('max_w',1e99)-pr['required_w']):
                    dynamic.extend([(cn,pp,si,mir,csi,max(0,li-1)),(cn,max(1,pp-.25),si,mir,csi,li)])
                if cres['output_w']>cr.get('max_w',1e99)-.1*(cr.get('max_w',1e99)-cr['required_w']):
                    dynamic.extend([(cn,pp,si,mir,csi+1,li),(max(4,cn-2),pp,si,mir,csi,li)])
                elif cres['output_w']<cr['required_w']+.1*(cr.get('max_w',1e99)-cr['required_w']):
                    dynamic.extend([(cn,pp,si,mir,max(0,csi-1),li),(cn+2,pp,si,mir,csi,li)])
                joint_trials=dynamic+joint_trials
            for cn,pp,ss,mirror,child_inset,parent_inset in joint_trials:
                if time.perf_counter()>=loop_deadline:break
                try:
                    if (cn,child_inset) not in child_curves:child_curves[cn,child_inset]=snake_cycle(domain,child,cn,(0,child_inset),True)
                    child_curve=child_curves[cn,child_inset]
                    cycle=strip_cycle(domain,parent,pp,(parent_inset,ss),mirror)
                    pc={**parent,'cycle':cycle,'pitch':'z'+str(pp),'offset':(parent_inset,ss),'balanced':mirror}
                    cc={**child,'cycle':child_curve,'pitch':'s'+str(cn),'offset':(0,child_inset),'balanced':True}
                    cs=[pc,cc];path,part=assemble(domain,lp,cs)
                    if (len(path)-1)*domain.unit>plan['pipe']['max_loop_length']+1e-7:continue
                    signature=tuple(path)
                    if signature in seen_paths:continue
                    seen_paths.add(signature)
                    fp=original_paths(domain,{slot:path})[slot];res=model.evaluate(fp,ret_state=True);s,passed=pattern_score(plan,res);trials_count+=1
                except (ValueError,KeyError,IndexError):continue
                if s<cur_score:
                    cur_score=s;components=cs;cur_path=path;cur_part=part;fine=fp;baseline=res;changed+=1;good=passed
                    if good:break
        for ci,component in enumerate(components):
            if good or time.perf_counter()>=loop_deadline:break
            options=config_trials(plan,component,baseline,refresh=only_failed)
            turn=component.get('search_round',0)
            if options:
                shift=(turn*max_trials)%len(options)
                options=options[shift:]+options[:shift]
            component['search_round']=turn+1
            best=None
            ntested=0
            for pitch,offset,balanced in options:
                if ntested>=max_trials or time.perf_counter()>=loop_deadline:break
                try:
                    if isinstance(pitch,str):
                        if pitch[0]=='z':
                            spec=pitch[1:].split('g');paths,left=[strip_cycle(domain,component,float(spec[0]),offset,balanced,float(spec[1]) if len(spec)>1 else 0.)],0
                        else:
                            spec=pitch[1:].split('c');paths,left=[snake_cycle(domain,component,int(spec[0]),offset,balanced,float(spec[1]) if len(spec)>1 else None)],0
                    else:paths,left=room_cycles(domain,component['room'],pitch,offset,component['bounds'],balanced)
                    if left or len(paths)!=1:continue
                    candidate={**component,'cycle':paths[0],'pitch':pitch,'offset':offset,'balanced':balanced}
                    cs=components.copy();cs[ci]=candidate
                    path,part=assemble(domain,lp,cs)
                    if (len(path)-1)*domain.unit>plan['pipe']['max_loop_length']+1e-7:continue
                    signature=tuple(path)
                    if signature in seen_paths:continue
                    seen_paths.add(signature)
                    fine_path=original_paths(domain,{slot:path})[slot]
                    res=model.evaluate(fine_path,ret_state=True);s,passed=pattern_score(plan,res);trials_count+=1;ntested+=1
                except (ValueError,KeyError,IndexError):continue
                if s<cur_score-1e-7:
                    cur_score=s;best=(candidate,path,part,fine_path,res);good=passed
                    if good:break
            if best is not None:
                component,cur_path,cur_part,fine,baseline=best;component['search_round']=0;components[ci]=component;changed+=1
            if good:break
        lp['components']=components;lp['path']=cur_path;lp['room_path']=cur_part
        sy=slice(model.y0-floor.by0,model.y1-floor.by0+1);sx=slice(model.x0-floor.bx0,model.x1-floor.bx0+1)
        heat['surface'][sy,sx]=baseline['_surface'];heat['bottom'][sy,sx]=baseline['_bottom'];heat['water'][slot]=baseline['_water']
        for j,old in enumerate(parsed):
            if old['slot']==slot:
                corners=thermal.simplify(fine)
                parsed[j]={**old,'walk':fine,'points':corners,'segments':list(zip(corners,corners[1:])),'length_units':len(fine)-1}
                break
        for res in results_of(baseline):
            heat['rooms'][res['ri']].update({k:res.get('_whole',res)[k] for k in ('output_w','max_surface','zone_spread')})
        if debug:print('pattern',slot,[(c['room'],c['pitch'],c['offset'],c['balanced']) for c in components],[(h['ri'],round(h['output_w']),round(h['zone_spread'],2)) for h in results_of(baseline)],'score',round(cur_score,3),file=__import__('sys').stderr)
    return {'changed':changed,'trials':trials_count}


def strip_cycle(domain,component,pitch=2.5,insets=(0,0),mirror=False,grade=0.):
    """A meander with a separate return following the free room boundary."""
    ri=component['room'];mask=domain.room_mask(ri)
    xx=domain.xs[None,:];yy=domain.ys[:,None]
    region=np.zeros_like(mask)
    for r in component['bounds']:region|=(xx>=r[0]-1e-7)&(xx<=r[2]+1e-7)&(yy>=r[1]-1e-7)&(yy<=r[3]+1e-7)
    mask&=region
    core=binary_opening(mask,structure=np.ones((5,5)),iterations=1)
    if core.any():
        yy,xx=np.nonzero(core)
        mask&=(domain.xs[None,:]>=xx.min()+domain.x0)&(domain.xs[None,:]<=xx.max()+domain.x0)&(domain.ys[:,None]>=yy.min()+domain.y0)&(domain.ys[:,None]<=yy.max()+domain.y0)
    by,bx=np.nonzero(mask);bx=bx+domain.x0;by=by+domain.y0
    vertical=(by.max()-by.min())>(bx.max()-bx.min())
    uu,vv=(by,bx) if vertical else (bx,by)
    u0,u1=int(uu.min()),int(uu.max())
    if mirror:uu=u0+u1-uu
    free={(int(u),int(v)) for u,v in zip(uu,vv)}
    long_inset,short_inset=(insets,insets) if isinstance(insets,(int,float)) else insets
    v0,v1=int(vv.min())+short_inset,int(vv.max())-short_inset
    intervals={}
    for v in range(v0,v1+1):
        us=sorted(u for u,w in free if w==v)
        if not us or len(us)!=us[-1]-us[0]+1:raise ValueError('split strip')
        intervals[v]=(us[0],us[-1])
    nrows=max(2,round((v1-v0)/pitch+1))
    nrows+=nrows%2
    if nrows>v1-v0+1:nrows-=2
    rows=[]
    for j in range(nrows):
        t=j/(nrows-1);value=round(v0+(v1-v0)*(t+grade*t*(1-t)))
        rows.append(max(value,rows[-1]+1 if rows else v0))
    rows[-1]=min(rows[-1],v1)
    for j in range(nrows-2,-1,-1):rows[j]=min(rows[j],rows[j+1]-1)
    if rows[0]<v0:raise ValueError('crowded rows')
    base={v:intervals[v][0] for v in intervals}
    spine={v:max(base[w] for w in range(max(v0,v-1),min(v1,v+1)+1)) for v in intervals}
    left=[max(spine[w] for w in range(max(v0,v-1),min(v1,v+1)+1))+1+long_inset for v in rows]
    right=[intervals[v][1]-long_inset for v in rows]
    for j in range(0,nrows-1,2):
        end=min(intervals[v][1] for v in range(rows[j],rows[j+1]+1))-long_inset
        right[j]=right[j+1]=end
    for j in range(1,nrows-1,2):
        start=max(spine[v] for v in range(max(v0,rows[j]-1),min(v1,rows[j+1]+1)+1))+1+long_inset
        left[j]=left[j+1]=start
    if any(b-a<2 for a,b in zip(left,right)):raise ValueError('narrow strip')
    points=[(left[0],rows[0])]
    for j,v in enumerate(rows):
        target=(right[j],v) if j%2==0 else (left[j],v)
        points.extend(line(points[-1],target)[1:])
        if j+1<nrows:points.extend(line(points[-1],(target[0],rows[j+1]))[1:])
    points.extend(line(points[-1],(spine[v1],v1))[1:])
    for v in range(v1,v0,-1):
        u=spine[v];n=spine[v-1]
        corner=(n,v) if n>u else (u,v-1)
        points.extend(line(points[-1],corner)[1:]);points.extend(line(points[-1],(n,v-1))[1:])
    points.extend(line(points[-1],(left[0],v0))[1:-1])
    if len(points)!=len(set(points)) or not all(p in free for p in points):raise ValueError('blocked meander')
    if mirror:points=[(u0+u1-u,v) for u,v in points]
    if vertical:points=[(v,u) for u,v in points]
    return points


def sync_hall(loops,paths):
    for lp in loops:
        path=paths[lp['slot']];ca,cb=lp['hall_paths'][0][-1],lp['hall_paths'][1][-1]
        ia,ib=path.index(ca),path.index(cb)
        lp['hall_paths']=[path[:ia+1],path[ib:][::-1]];lp['room_path']=path[ia:ib+1];lp['path']=path
