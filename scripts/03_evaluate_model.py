#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from glo.model import LatentODE
from glo.pipeline import (TrajectoryDataset, create_loader, make_query_indices, prepare_batch,
                          infer_test_metrics, seed_everything)
from glo.visualization import plot_prediction

p=argparse.ArgumentParser(description='Evaluate reconstruction and forecasting on held-out clean trajectories')
p.add_argument('--checkpoint', required=True, help='Path relative to project root or absolute')
p.add_argument('--n-obs', type=int, default=30)
p.add_argument('--n-query', type=int, default=200)
p.add_argument('--batch-size', type=int, default=24)
p.add_argument('--seed', type=int, default=123)
p.add_argument('--max-batches', type=int, default=0)
p.add_argument('--noise-std',type=float,default=None,help='Override dataset noise level')
p.add_argument('--output', default='results/evaluation.json')
p.add_argument('--figures-dir', default='results/figures', help='Where prediction figures are written')
args=p.parse_args()
root=Path(__file__).resolve().parents[1]
seed_everything(args.seed)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path=Path(args.checkpoint)
if not checkpoint_path.is_absolute(): checkpoint_path=root/checkpoint_path
checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
model=LatentODE(**checkpoint['model']).to(device)
model.load_state_dict(checkpoint['state_dict']); model.eval()
data=TrajectoryDataset(root/'data/processed',checkpoint['experiment'],'test')
meta=data.meta; qidx=make_query_indices(meta['n_points'],args.n_query)
time=torch.tensor(data.arrays['time'],dtype=torch.float32)
loader=create_loader(data,batch_size=args.batch_size,shuffle=False)
rng=torch.Generator().manual_seed(args.seed)
record=defaultdict(list)
plotted=set()
with torch.no_grad():
    for j,batch in enumerate(loader):
        if args.max_batches and j>=args.max_batches: break
        prepared=prepare_batch(batch,time,meta,qidx,args.n_obs,device,
                generator=rng,noise_std=args.noise_std,
                use_stored_noise=(args.noise_std is None))
        output=model(prepared['obs_xy'],prepared['obs_t'],prepared['query_t'],sample=False)
        metrics=infer_test_metrics(output['prediction'],prepared,meta['forecast_start'])
        fams=prepared['family'].cpu().tolist()
        for i,fid in enumerate(fams):
            entry={'family':int(fid)}
            for key,values in metrics.items(): entry[key]=float(values[i].cpu())
            record['all'].append(entry)
            if fid not in plotted:
                family=next(k for k,v in meta['family_to_id'].items() if v==fid)
                true_xy=prepared['raw_truth'][i].cpu().numpy()
                pred_xy=(output['prediction'][i]*prepared['scale']+prepared['mean']).cpu().numpy()
                obs_xy=(prepared['obs_xy'][i]*prepared['scale']+prepared['mean']).cpu().numpy()
                fig=root/args.figures_dir/f'{checkpoint["experiment"]}_{family}_prediction.png'
                plot_prediction(true_xy,pred_xy,obs_xy,prepared['query_t'].cpu().numpy(),
                                meta['forecast_start'],family,fig)
                plotted.add(fid)
keys=['recon_rmse','forecast_rmse','forecast_mae','forecast_ade','forecast_fde','chamfer_full']
def summarize(entries):
    return {'n':len(entries), **{name:{'mean':float(np.mean([e[name] for e in entries])),
                                  'std':float(np.std([e[name] for e in entries]))} for name in keys}}
report={'checkpoint':str(checkpoint_path.relative_to(root)) if checkpoint_path.is_relative_to(root) else str(checkpoint_path),
        'experiment':checkpoint['experiment'],'n_obs':args.n_obs,'n_query':args.n_query,
        'forecast_start':meta['forecast_start'],'noise_std_override':args.noise_std,
        'metric_units':'original trajectory coordinate units; sample-wise means',
        'status':'PARTIAL SMOKE EVALUATION' if args.max_batches else 'FULL TEST SPLIT',
        'overall':summarize(record['all']),
        'per_family':{f:summarize([e for e in record['all'] if e['family']==fid])
                      for f,fid in meta['family_to_id'].items() if f in meta['families']}}
out=Path(args.output)
if not out.is_absolute():out=root/out
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
print('Saved:',out)
