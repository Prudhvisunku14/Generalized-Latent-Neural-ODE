import json
import numpy as np
import torch
from scipy import stats
from glo.data import generate_project_data
from glo.pipeline import TrajectoryDataset
from glo.evaluation import BaselineModel, t_ci95, collect, METRICS


def test_family_filter_keeps_only_requested_and_normalisation(tmp_path):
    generate_project_data(tmp_path, n_per_family=20, n_points=40, n_obs=9)
    full = TrajectoryDataset(tmp_path, 'three_shapes', 'test')
    ell = TrajectoryDataset(tmp_path, 'three_shapes', 'test', ['ellipse'])
    assert len(ell) == int((full.arrays['family'] == 2).sum()) and set(ell.arrays['family']) == {2}
    assert ell.meta['normalization_scale'] == full.meta['normalization_scale']
    assert np.array_equal(ell.arrays['time'], full.arrays['time'])


def test_baselines_interpolate_and_extrapolate():
    t = torch.linspace(0, 10, 101)
    obs_t = torch.tensor([[0., 2., 4., 6.]])
    obs = torch.stack([obs_t * 0.5, -obs_t], -1)                  # straight line, velocity (0.5, -1)
    cv = BaselineModel('const_vel')(obs, obs_t, t)['prediction'][0]
    assert torch.allclose(cv, torch.stack([t * 0.5, -t], -1), atol=1e-5)   # exact for linear motion
    cp = BaselineModel('const_pos')(obs, obs_t, t)['prediction'][0]
    assert torch.allclose(cp[t > 6], obs[0, -1].expand(int((t > 6).sum()), 2))
    assert torch.allclose(cp[t <= 6], cv[t <= 6], atol=1e-5)


def test_t_ci95_matches_scipy():
    v = [0.19, 0.21, 0.18, 0.25, 0.2]
    r = t_ci95(v)
    lo, hi = stats.t.interval(0.95, len(v) - 1, loc=np.mean(v), scale=stats.sem(v))
    assert np.isclose(r['ci95_low'], lo, atol=1e-3) and np.isclose(r['ci95_high'], hi, atol=1e-3)
    assert np.isclose(r['std'], np.std(v, ddof=1))


def test_collect_normalises_with_model_training_statistics(tmp_path):
    """A model trained on two_shapes must receive inputs normalised with two_shapes statistics even
    when evaluated on three_shapes data (OOD ellipses)."""
    generate_project_data(tmp_path, n_per_family=20, n_points=40, n_obs=9)
    seen = {}

    class Spy(torch.nn.Module):
        def forward(self, obs_xy, obs_t, query_t, sample=False):
            seen['obs'] = obs_xy.clone()
            return {'prediction': torch.zeros(obs_xy.shape[0], len(query_t), 2)}
    r = collect(Spy(), {'experiment': 'two_shapes'}, tmp_path, 'three_shapes', families=['ellipse'],
                n_obs=9, n_query=40)
    m2 = json.loads((tmp_path / 'two_shapes' / 'metadata.json').read_text())
    back = seen['obs'][-r['obs'].shape[0]:] * m2['normalization_scale'] + torch.tensor(m2['normalization_mean'])
    assert np.allclose(back.numpy(), r['obs'][-back.shape[0]:], atol=1e-5)
    assert set(r['family']) == {2} and all(k in r for k in METRICS)
