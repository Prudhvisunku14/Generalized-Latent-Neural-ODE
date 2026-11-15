#!/usr/bin/env python3
"""Sample NEW sequences from the generative prior of a trained unconditional Latent ODE.

Sampling from an insufficiently trained model will NOT produce recognizable shapes.
The posterior reconstruction/evaluation script is the correct first performance test.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from glo.model import LatentODE
from glo.solver import integrate
from glo.data import load_split

p=argparse.ArgumentParser(description='Sample generative trajectories from z0~N(0,I)')
p.add_argument('--checkpoint',required=True)
p.add_argument('--n-samples',type=int,default=12)
p.add_argument('--seed',type=int,default=1234)
p.add_argument('--output-prefix',default='results/prior_generated')
args=p.parse_args()
if args.n_samples<1: p.error('n-samples must be >=1')
root=Path(__file__).resolve().parents[1]
ckpt_path=Path(args.checkpoint)
if not ckpt_path.is_absolute():ckpt_path=root/ckpt_path
ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False)
if ckpt['model']['cond_dim']:
    p.error('Prior sampling of conditional code is undefined in this prototype. Use the unconditional baseline.')
torch.manual_seed(args.seed)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=LatentODE(**ckpt['model']).to(device)
model.load_state_dict(ckpt['state_dict']);model.eval()
arrays,meta=load_split(root/'data/processed',ckpt['experiment'],'train')
t=torch.tensor(arrays['time'],device=device)
with torch.no_grad():
    z0=torch.randn(args.n_samples,ckpt['model']['latent_dim'],device=device)
    zs=integrate(model.dynamics,z0,t,method=model.solver,adjoint=False)
    pred=model.decoder(zs).transpose(0,1).cpu().numpy()
xy=pred*meta['normalization_scale']+np.asarray(meta['normalization_mean'])
prefix=Path(args.output_prefix)
if not prefix.is_absolute():prefix=root/prefix
prefix.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(prefix.with_suffix('.npz'),time=t.cpu().numpy(),generated_xy=xy,
                    initial_latents=z0.cpu().numpy())
cols=min(4,args.n_samples)
rows=(args.n_samples+cols-1)//cols
fig,axs=plt.subplots(rows,cols,figsize=(cols*3,rows*3),squeeze=False)
for i,ax in enumerate(axs.ravel()):
    if i>=args.n_samples:
        ax.axis('off');continue
    ax.plot(xy[i,:,0],xy[i,:,1],lw=1.8)
    ax.scatter(xy[i,0,0],xy[i,0,1],s=18,marker='o')
    ax.set_title(f'Sample {i+1}',fontsize=10)
    ax.set_aspect('equal',adjustable='datalim')
    ax.grid(alpha=.12)
fig.suptitle('Unconditional samples from Latent ODE prior (not shape-class controlled)',fontsize=12)
fig.tight_layout(rect=(0,0,1,.96))
fig.savefig(prefix.with_suffix('.png'),dpi=170)
plt.close(fig)
print('Generated',args.n_samples,'new trajectories to',prefix.with_suffix('.npz'))
print('WARNING: this generates samples from the prior; output shape quality depends on model convergence.')
