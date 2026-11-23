from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from glo.model import LatentODE
from glo.pipeline import TrajectoryDataset, create_loader, make_query_indices, prepare_batch, batch_objective
from test_capacity_scaling import expected_params

ROOT = Path(__file__).resolve().parents[1]
ABL = yaml.safe_load((ROOT / 'configs/ablation_2x2.yaml').read_text())
MODEL_D = yaml.safe_load((ROOT / 'configs/model_d.yaml').read_text())
needs_data = pytest.mark.skipif(not (ROOT / 'data/processed/five_shapes/train.npz').exists(), reason='five_shapes not generated')


def build(name):
    v = ABL['variants'][name]
    return LatentODE(latent_dim=v['latent_dim'], encoder_hidden=v['encoder_hidden'], dynamics_hidden=64, decoder_hidden=64,
                     dynamics_layers=2, decoder_layers=2, solver='rk4', step_size=.1)


def test_design_is_a_full_2x2_with_unique_checkpoints():
    v = ABL['variants']
    assert {(x['latent_dim'], x['encoder_hidden']) for x in v.values()} == {(8, 64), (16, 64), (8, 128), (16, 128)}
    assert len({x['checkpoint'] for x in v.values()}) == 4
    refs = set(ABL['references'].values())
    assert not refs & {x['checkpoint'] for x in v.values()}      # earlier results can never be overwritten


def test_fixed_settings_match_model_d():
    for k in ('dynamics_hidden', 'decoder_hidden', 'dynamics_layers', 'decoder_layers'):
        assert ABL['fixed'][k] == MODEL_D[k]
    assert ABL['fixed']['seed'] == MODEL_D['seed'] == 42


@pytest.mark.parametrize('name', 'ABCD')
def test_parameter_count_matches_layer_formula(name):
    v = ABL['variants'][name]
    assert sum(p.numel() for p in build(name).parameters()) == expected_params(v['latent_dim'], v['encoder_hidden'])


def test_a_and_d_match_the_earlier_baseline_and_medium_architectures():
    assert sum(p.numel() for p in build('A').parameters()) == 24602 and sum(p.numel() for p in build('D').parameters()) == 67250


@pytest.mark.parametrize('name', 'BC')
def test_forward_backward_finite_for_new_cells(name):
    torch.manual_seed(0)
    m = build(name).train(); B, n = 4, 30
    obs_t = torch.sort(torch.rand(B, n) * 7, 1).values; obs_t[:, 0] = 0
    out = m(torch.randn(B, n, 2) * .5, obs_t, torch.linspace(0, 10, 100), sample=True)
    idx = torch.sort(torch.randperm(70)[:n].expand(B, n), 1).values
    loss, _ = batch_objective(out, {'obs_indices': idx, 'obs_xy': torch.randn(B, n, 2) * .5, 'scale': .8}, .05, 1., 'observed')
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_launch_script_matches_config_and_uses_data_seed():
    sh = (ROOT / 'scripts/run_ablation_2x2.sh').read_text()
    assert '--data-seed 42' in sh and '--seed 42' in sh
    for v in ABL['variants'].values():
        line = next(l for l in sh.splitlines() if l.startswith('run ') and Path(v['checkpoint']).stem in l)
        assert f"--latent-dim {v['latent_dim']} " in line + ' ' and f"--encoder-hidden {v['encoder_hidden']}" in line


@needs_data
def test_data_seed_gives_identical_batches_and_observations_regardless_of_model_size():
    ds = TrajectoryDataset(ROOT / 'data/processed', 'five_shapes', 'train'); meta = ds.meta; t = torch.as_tensor(ds.arrays['time']).float()
    q = make_query_indices(meta['n_points'], 100)

    def stream(model_name):
        m = build(model_name)                                  # consumes the global RNG differently per size
        loader_gen = torch.Generator().manual_seed(42); obs_gen = torch.Generator().manual_seed(43)
        out = []
        for j, b in enumerate(create_loader(ds, 32, True, loader_gen)):
            if j == 3: break
            p = prepare_batch(b, t, meta, q, 30, 'cpu', generator=obs_gen)
            out.append((b['family'].clone().numpy(), p['obs_indices'].numpy().copy(), p['obs_xy'].numpy().copy()))
        return out
    sa, sd = stream('A'), stream('D')
    for (fa, ia, xa), (fd, i_d, xd) in zip(sa, sd):
        assert np.array_equal(fa, fd) and np.array_equal(ia, i_d) and np.array_equal(xa, xd)
