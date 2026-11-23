import sys; sys.path.insert(0,'src')
import numpy as np, torch
from glo.solver import integrate
from glo.evaluation import load_model, collect
torch.set_num_threads(4)
# (a) order of convergence on a harmonic oscillator, y'' = -w^2 y, exact solution known
w=1.3
f=lambda t,y: torch.stack([y[...,1], -w*w*y[...,0]],-1)
y0=torch.tensor([[1.0,0.0]],dtype=torch.float64)
exact=lambda t: torch.stack([torch.cos(w*t), -w*torch.sin(w*t)],-1)
tq_grid=torch.linspace(0,10,101,dtype=torch.float64)           # multiples of 0.1
tq_model=torch.linspace(0,10,200,dtype=torch.float64)          # like the model's evaluation grid (not on the step grid)
print('step   err@grid-aligned-times   err@200-point-grid (as used in evaluation)')
prev=None
for h in (0.4,0.2,0.1,0.05,0.025):
    e1=(integrate(f,y0,tq_grid,'rk4',step_size=h)[:,0]-exact(tq_grid)).abs().max().item()
    e2=(integrate(f,y0,tq_model,'rk4',step_size=h)[:,0]-exact(tq_model)).abs().max().item()
    print(f'{h:5.3f}  {e1:.3e} {"" if prev is None else f"(ratio {prev/e1:5.1f})"}   {e2:.3e}'); prev=e1
# (b) trained-model sensitivity to the integration step / solver, test split, same observations
for cell in ('A','D'):
    path={'A':'checkpoints/ablation_2x2/five_shapes_A_lat8_gru64_seed42.pt','D':'checkpoints/ablation_2x2/five_shapes_D_lat16_gru128_seed42.pt'}[cell]
    m,ck=load_model(path,'cpu'); res={}
    for name,(solver,step) in {'rk4_0.1':('rk4',0.1),'rk4_0.025':('rk4',0.025),'dopri5_tight':('dopri5',None)}.items():
        m.solver,m.step_size=solver,step
        if solver=='dopri5': m.rtol,m.atol=1e-7,1e-9
        res[name]=collect(m,ck,'data/processed','five_shapes')
    for k in ('recon_rmse','forecast_rmse'):
        base=res['rk4_0.1'][k]
        for other in ('rk4_0.025','dopri5_tight'):
            d=res[other][k]-base
            print(f'cell {cell} {k:13s} rk4_0.1 vs {other:12s}: mean|diff|={np.abs(d).mean():.2e}  max|diff|={np.abs(d).max():.2e}  mean value {base.mean():.4f} -> {res[other][k].mean():.4f}')
    dp=np.abs(res['rk4_0.1']['pred']-res['dopri5_tight']['pred']).max(); print(f'cell {cell} max |pred difference| rk4(0.1) vs dopri5: {dp:.2e} data units')
