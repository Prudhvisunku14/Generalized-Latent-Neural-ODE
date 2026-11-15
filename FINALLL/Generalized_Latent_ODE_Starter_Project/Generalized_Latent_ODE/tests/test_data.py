import numpy as np
from glo.data import FAMILY_TO_ID,generate_project_data,load_split,validate_dataset,sample_trajectory
from glo.pipeline import TrajectoryDataset,make_query_indices,prepare_batch
import torch


def test_circle_constant_radius():
    rng=np.random.default_rng(42)
    t=np.linspace(0,10,200)
    xy,p=sample_trajectory('circle',t,rng)
    radii=np.linalg.norm(xy-np.asarray([p[3],p[4]]),axis=1)
    assert np.allclose(radii,p[0],atol=1e-6)


def test_spiral_monotonic_radius_and_direction():
    t=np.linspace(0,10,200)
    for seed in range(6):
        xy,p=sample_trajectory('spiral',t,np.random.default_rng(seed))
        r=np.linalg.norm(xy-np.array([p[3],p[4]]),axis=1)
        assert (np.diff(r)*np.sign(p[5])>-1e-6).all()
        assert np.isclose(r[-1]/r[0],np.exp(float(p[5])*10),rtol=1e-5)


def test_reproducibility_and_no_leaks(tmp_path):
    a=tmp_path/'a'; b=tmp_path/'b'
    generate_project_data(a,n_per_family=20,n_points=40,n_obs=9,seed=9)
    generate_project_data(b,n_per_family=20,n_points=40,n_obs=9,seed=9)
    for exp in ('two_shapes','three_shapes'):
        assert validate_dataset(a,exp) == validate_dataset(b,exp)
        da,ma=load_split(a,exp,'train'); db,_=load_split(b,exp,'train')
        for field in ('xy','noisy_xy','ids','default_obs_mask'):
            assert np.array_equal(da[field],db[field])
        assert not da['default_obs_mask'][:,da['time']>ma['forecast_start']].any()
    tw,_=load_split(a,'two_shapes','test')
    three,_=load_split(a,'three_shapes','test')
    assert set(tw['ids']).issubset(set(three['ids']))


def test_batch_no_future_inputs(tmp_path):
    generate_project_data(tmp_path,n_per_family=20,n_points=40,n_obs=9)
    d=TrajectoryDataset(tmp_path,'two_shapes','train')
    batch=next(iter(torch.utils.data.DataLoader(d,batch_size=3)))
    prepared=prepare_batch(batch,torch.from_numpy(d.arrays['time']),d.meta,make_query_indices(40,30),
                           n_obs=7,device=torch.device('cpu'),generator=torch.Generator().manual_seed(4))
    assert prepared['obs_xy'].shape==(3,7,2)
    assert torch.all(prepared['obs_t']<=d.meta['forecast_start'])
    assert torch.all(prepared['obs_t'][:,1:]>=prepared['obs_t'][:,:-1])
