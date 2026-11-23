import sys, glob, math, json, hashlib; sys.path.insert(0,'src')
import numpy as np, torch
from glo.evaluation import load_model, collect
torch.set_num_threads(4)
cells={'A':'five_shapes_A_lat8_gru64_seed42','B':'five_shapes_B_lat16_gru64_seed42','C':'five_shapes_C_lat8_gru128_seed42','D':'five_shapes_D_lat16_gru128_seed42'}
cfg={}
for c,n in cells.items():
    h=np.genfromtxt(f'checkpoints/ablation_2x2/{n}_history.csv',delimiter=',',names=True); log=open(f'logs/ablation_2x2/{n}.log').read()
    ck=torch.load(f'checkpoints/ablation_2x2/{n}.pt',map_location='cpu',weights_only=True); cfg[c]=ck['config']['training']
    print(f'{c}: history rows={len(h)} (epochs {int(h["epoch"][0])}..{int(h["epoch"][-1])}) tracebacks={log.lower().count("traceback")} early-stop line={"Early stopping" in log} '
          f'final line has Checkpoint={"Checkpoint:" in log} selected epoch={ck["epoch"]} argmin(val_elbo)+1={int(np.argmin(h["val_elbo_loss"]))+1} nonfinite rows={int((~np.isfinite(h["train_loss"])).sum())} minutes={h["seconds"].sum()/60:.1f}')
diff={k:{c:cfg[c][k] for c in cfg} for k in cfg['A'] if len({str(cfg[c][k]) for c in cfg})>1}
print('training args that differ between A-D:',{k:v for k,v in diff.items() if k!='checkpoint'})
print('seed/data_seed/epochs/batch/lr/sigma/kl/supervision/n_obs/n_query (A):',{k:cfg['A'][k] for k in ('seed','data_seed','epochs','batch_size','lr','lr_schedule','min_lr','sigma','kl_weight','kl_anneal_epochs','supervision','n_obs','n_query','grad_clip','patience','solver','step_size')})
n_train=int(np.load('data/processed/five_shapes/train.npz')['family'].shape[0]); print('steps/epoch',math.ceil(n_train/32),'x 300 =',math.ceil(n_train/32)*300)
# selection code path
src=open('scripts/02_train_model.py').read(); i=src.index("val_loss = row['val_elbo_loss']"); print('selection code:',src[i:i+30].splitlines()[0],'| improvement test:',[l.strip() for l in src[i:i+400].splitlines() if 'if val_loss' in l])
# identical observations across cells, evaluated fresh
R={}
for c,n in cells.items():
    m,ck=load_model(f'checkpoints/ablation_2x2/{n}.pt','cpu'); R[c]=collect(m,ck,'data/processed','five_shapes')
ok=all(np.array_equal(R['A'][k],R[c][k]) for c in 'BCD' for k in ('obs_t','obs_mask','ids','family')) and all(np.allclose(R['A']['obs'],R[c]['obs'],atol=1e-5) for c in 'BCD')
print('fresh evaluation: observation times/masks/ids/families identical and noisy observations equal (atol 1e-5) across A-D:',ok,'| test n =',len(R['A']['ids']),'| ids unique:',len(set(R['A']['ids'].tolist())))
print('test ids are all from the test split of five_shapes:',set(R['A']['ids'].tolist())==set(np.load('data/processed/five_shapes/test.npz')['ids'].tolist()))
for k in ('recon_rmse','forecast_rmse'):
    print(' fresh-run mean',k,{c:round(float(R[c][k].mean()),4) for c in R})
