import sys, json, hashlib
sys.path.insert(0,'src')
import numpy as np
from glo.data import load_split, validate_dataset
R='data/processed'; out={}
for exp in ('two_shapes','three_shapes','five_shapes'):
    print('=====',exp); print('validate_dataset:',validate_dataset(R,exp))
    S={s:load_split(R,exp,s)[0] for s in ('train','val','test')}; meta=load_split(R,exp,'train')[1]
    fams=meta['families']; print('families',fams,'n_points',meta['n_points'],'t_end',meta['t_end'],'forecast_start',meta['forecast_start'],'n_obs',meta['default_n_obs'])
    t=S['train']['time']
    for s,a in S.items():
        cnt=np.bincount(a['family'],minlength=5)[:len(fams)]; noise=(a['noisy_xy']-a['xy']).std()
        print(f" {s:5s} n={len(a['ids']):5d} per-family={cnt.tolist()} noise_sd={noise:.4f} obs/traj={a['default_obs_mask'].sum(1).min()}..{a['default_obs_mask'].sum(1).max()} "
              f"max obs time={t[a['default_obs_mask'].any(0)].max():.3f} first_obs_always_t0={a['default_obs_mask'][:,0].all()}")
    # leakage: ids, exact-row hashes, nearest-neighbour distance test->train (same family)
    ids={s:set(a['ids'].tolist()) for s,a in S.items()}
    ov={(a,b):len(ids[a]&ids[b]) for a,b in (('train','val'),('train','test'),('val','test'))}
    h={s:{hashlib.sha1(x.tobytes()).hexdigest() for x in a['xy']} for s,a in S.items()}
    hv={(a,b):len(h[a]&h[b]) for a,b in (('train','val'),('train','test'),('val','test'))}
    print(' id overlap',ov,'| identical-path overlap',hv)
    from scipy.spatial.distance import cdist
    nn={}
    for f,name in enumerate(fams):
        tr=S['train']['xy'][S['train']['family']==f].reshape(-1,t.size*2); te=S['test']['xy'][S['test']['family']==f].reshape(-1,t.size*2)
        d=cdist(te,tr); nn[name]=float(d.min(1).min())
        d_tt=cdist(tr[:200],tr[:200]); np.fill_diagonal(d_tt,np.inf); nn[name+'_train_vs_train_min']=float(d_tt.min())
    print(' min test->train path distance (L2 over 200x2 coords):',{k:round(v,3) for k,v in nn.items()})
    out[exp]={'counts':{s:len(a['ids']) for s,a in S.items()},'id_overlap':{f'{a}-{b}':v for (a,b),v in ov.items()},'identical_paths':{f'{a}-{b}':v for (a,b),v in hv.items()}}
# five vs three consistency
d3={s:load_split(R,'three_shapes',s)[0] for s in ('train','val','test')}; d5={s:load_split(R,'five_shapes',s)[0] for s in ('train','val','test')}
ok=True
for s in d3:
    pos={k:i for i,k in enumerate(d5[s]['ids'].tolist())}; j=[pos[k] for k in d3[s]['ids'].tolist()]
    ok&=all(np.array_equal(d3[s][k],d5[s][k][j]) for k in('xy','noisy_xy','default_obs_mask'))
print('five_shapes contains the three_shapes trajectories unchanged, same splits:',ok)
# new-family geometry sanity
t=d5['train']['time']; a=d5['train']; 
for f,name in ((3,'figure_eight'),(4,'sinusoid')):
    x=a['xy'][a['family']==f]; print(name,'n_train',len(x),'x-range',float(x[...,0].min()),float(x[...,0].max()),'y-range',float(x[...,1].min()),float(x[...,1].max()))
print('dataset_checks.json failures:',json.load(open('results/five_shapes/dataset_checks.json')).get('failures'))
