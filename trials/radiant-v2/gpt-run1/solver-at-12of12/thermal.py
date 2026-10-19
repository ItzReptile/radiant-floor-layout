import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres
from scipy.signal import lfilter
import time
from collections import OrderedDict
from model import Floor, lattice_walk

class Thermal:
    def __init__(self,p):
        self.p=p;self.floor=f=Floor(p);g=p['grid'];ph=p['physics']
        nx,ny=f.nx,f.ny;n=nx*ny;self.n=n;self.ph=ph;self.nx=nx;self.ny=ny
        room_flat=f.room_of.ravel();self.rflat=room_flat
        air=np.array([r['air_temp'] for r in p['rooms']])[room_flat]
        bx0,by0,bx1,by1=p['building'];xs=f.xs*g;ys=f.ys*g
        edge=np.minimum(np.minimum(ys-by0,by1-ys)[:,None],np.minimum(xs-bx0,bx1-xs)[None,:])
        h=np.where(edge<=ph['perim_width']+1e-6,ph['h_perim'],ph['h_core']).ravel()
        floor_r=np.array([r.get('floor_r',0.) for r in p['rooms']])[room_flat]
        gr=g*g/(floor_r+1/h);self.gr=gr
        kb=ph['k_screed']*ph['pipe_layer'];kt=ph['k_screed']*ph['surface_layer'];gv=ph['k_screed']/ph['layer_gap']*g*g;gd=ph['insulation_u']*g*g
        self.gv=gv
        idx=np.arange(n).reshape(ny,nx);aa=[];bb=[]
        for a,b in ((idx[:,:-1],idx[:,1:]),(idx[:-1,:],idx[1:,:])):aa.append(a.ravel());bb.append(b.ravel())
        a=np.concatenate(aa);b=np.concatenate(bb);degree=np.bincount(np.r_[a,b],minlength=n)
        self.db=degree*kb+gv+gd;self.dt=degree*kt+gv+gr
        rows=np.r_[a,b,n+a,n+b,np.arange(n),n+np.arange(n),np.arange(2*n)]
        cols=np.r_[b,a,n+b,n+a,n+np.arange(n),np.arange(n),np.arange(2*n)]
        vals=np.r_[np.full(2*len(a),-kb),np.full(2*len(a),-kt),np.full(2*n,-gv),self.db,self.dt]
        self.base=coo_matrix((vals,(rows,cols)),shape=(2*n,2*n)).tocsr()
        self.rhs=np.r_[np.full(n,gd*ph['below_temp']),gr*air]
        self.air=air
        self.zones=[];self.sels=[]
        for ri,r in enumerate(p['rooms']):
            sel=f.room_of==ri;x0,y0,x1,y1=r['rect'];zm=r.get('zone_margin',0.)
            zone=sel & (xs[None,:]>=x0+zm-1e-6)&(xs[None,:]<=x1-zm+1e-6)&(ys[:,None]>=y0+zm-1e-6)&(ys[:,None]<=y1-zm+1e-6)
            for k in p.get('keepouts',[]):zone &= ~((xs[None,:]>=k[0]-zm-1e-6)&(xs[None,:]<=k[2]+zm+1e-6)&(ys[:,None]>=k[1]-zm-1e-6)&(ys[:,None]<=k[3]+zm+1e-6))
            self.sels.append(sel.ravel());self.zones.append((zone if zone.any() else sel).ravel())
        self.last=np.r_[air+4,air+3]
        self.loop_cache=OrderedDict()
        self.deadline=float('inf')

    def parse(self,layout):
        f=self.floor;p=self.p;g=p['grid'];ph=self.ph;n=self.n
        water=[];allcells=[];allg=[];allforce=[];offset=0
        for loop in layout['loops']:
            key=tuple(tuple(point) for point in loop['path'])
            if key in self.loop_cache:
                cells,gk,runs,force,length,mc,steps=self.loop_cache[key]
                self.loop_cache.move_to_end(key)
            else:
                pts=[(round(x/g),round(y/g)) for x,y in loop['path']]
                walk=lattice_walk(pts);steps=len(walk)-1
                cells=np.array([(y-f.by0)*f.nx+x-f.bx0 for x,y in walk[1:]],int)
                length=steps*g;mc=ph['flow_constant']/(length+ph['flow_offset'])
                sl=round(ph['sleeve_length']/g);gk=np.full(steps,ph['g_pipe']*g)
                gk[(np.arange(steps)<sl)|(np.arange(steps)>=steps-sl)]*=ph['sleeve_factor']
                a=mc/(mc+gk);changes=np.r_[0,np.flatnonzero(np.diff(a))+1,steps]
                runs=[(int(i),int(j),float(a[i])) for i,j in zip(changes,changes[1:])]
                force=ph['supply_temp']*np.cumprod(a)
                self.loop_cache[key]=(cells,gk,runs,force,length,mc,steps)
                if len(self.loop_cache)>256:self.loop_cache.popitem(last=False)
            water.append((loop['slot'],cells,gk,runs,force,length,mc,slice(offset,offset+steps)))
            allcells.append(cells);allg.append(gk);allforce.append(force);offset+=steps
        return water,np.concatenate(allcells),np.concatenate(allg),np.concatenate(allforce)

    def evaluate(self,layout,rtol=2e-8):
        n=self.n;gv=self.gv
        water,cells,gk,force=self.parse(layout)
        gp=np.bincount(cells,weights=gk,minlength=n)
        A=self.base+diags(np.r_[gp,np.zeros(n)])
        rhs=self.rhs.copy();rhs[:n]+=np.bincount(cells,weights=gk*force,minlength=n)
        out=np.empty(len(cells))
        def waterlinear(x):
            for slot,c,g,runs,force,ln,mc,sl in water:
                v=x[c];ww=out[sl];prev=0.
                for i,j,a in runs:
                    ww[i:j],zf=lfilter([1-a],[1.,-a],v[i:j],zi=[a*prev]);prev=ww[j-1]
            return out
        def matvec(x):
            z=A@x
            z[:n]-=np.bincount(cells,weights=gk*waterlinear(x[:n]),minlength=n)
            return z
        db=self.db+gp;dt=self.dt;inv=1/(db*dt-gv*gv)
        def precond(x):
            return np.r_[(dt*x[:n]+gv*x[n:])*inv,(gv*x[:n]+db*x[n:])*inv]
        op=LinearOperator((2*n,2*n),matvec=matvec,dtype=float);M=LinearOperator((2*n,2*n),matvec=precond,dtype=float)
        counter=0
        def tick(_):
            nonlocal counter
            counter+=1
            if counter%8==0 and time.monotonic()>self.deadline:raise TimeoutError('heat solve deadline')
        t,info=bicgstab(op,rhs,x0=self.last,rtol=rtol,atol=0.,maxiter=220,M=M,callback=tick)
        if info!=0:t,info=gmres(op,rhs,x0=t,rtol=rtol,atol=0.,maxiter=12,restart=35,M=M,callback=tick,callback_type='pr_norm')
        if info!=0:raise ArithmeticError('floor iteration did not converge')
        self.last=t
        surface=t[n:];results=[]
        for r,sel,zone in zip(self.p['rooms'],self.sels,self.zones):
            output=float((self.gr[sel]*(surface[sel]-r['air_temp'])).sum());zt=surface[zone]
            results.append(dict(id=r['id'],output_w=output,max_surface=float(surface[sel].max()),zone_spread=float(zt.max()-zt.min())))
        wlin=waterlinear(t[:n]).copy();loops=[]
        for slot,c,g,runs,force,ln,mc,sl in water:
            loops.append(dict(slot=slot,length_m=ln,flow_w_per_k=mc,return_temp=float(wlin[sl][-1]+force[-1])))
        return dict(rooms=results,loops=loops,surface=surface.reshape(self.ny,self.nx),converged=info==0,_state=t,_rtol=rtol)
