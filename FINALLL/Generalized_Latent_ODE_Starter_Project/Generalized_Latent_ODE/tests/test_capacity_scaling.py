from pathlib import Path
import pytest
import torch
import yaml
from glo.model import LatentODE
from glo.pipeline import batch_objective, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
CAP = yaml.safe_load((ROOT / 'configs/capacity_scaling.yaml').read_text())
MODEL_D = yaml.safe_load((ROOT / 'configs/model_d.yaml').read_text())


def build(name):
    v = CAP['variants'][name]
    return LatentODE(latent_dim=v['latent_dim'], encoder_hidden=v['encoder_hidden'], dynamics_hidden=MODEL_D['dynamics_hidden'],
                     decoder_hidden=MODEL_D['decoder_hidden'], dynamics_layers=MODEL_D['dynamics_layers'],
                     decoder_layers=MODEL_D['decoder_layers'], solver='rk4', step_size=MODEL_D['step_size'])


def expected_params(L, H, D=64):
    gru = 3 * H * (4 + H + 2)                      # weight_ih, weight_hh, bias_ih, bias_hh
    posterior = H * 2 * L + 2 * L
    mlp = lambda i, o: (i * D + D) + (D * D + D) + (D * o + o)
    return gru + posterior + mlp(L, L) + mlp(L, 2)


def test_fixed_widths_match_model_d():
    for k in ('dynamics_hidden', 'decoder_hidden', 'dynamics_layers', 'decoder_layers'):
        assert CAP['fixed'][k] == MODEL_D[k]


@pytest.mark.parametrize('name', ['baseline', 'medium', 'large'])
def test_parameter_count_matches_layer_formula(name):
    v = CAP['variants'][name]
    m = build(name)
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == expected_params(v['latent_dim'], v['encoder_hidden'])


def test_baseline_variant_is_the_existing_five_shape_model():
    ck = torch.load(ROOT / CAP['variants']['baseline']['checkpoint'], map_location='cpu', weights_only=True)
    m = build('baseline')
    m.load_state_dict(ck['state_dict'])                       # loads strictly
    assert sum(p.numel() for p in m.parameters()) == 24602
    assert ck['model']['latent_dim'] == 8 and ck['model']['encoder_hidden'] == 64


@pytest.mark.parametrize('name', ['medium', 'large'])
def test_shapes_forward_backward_and_finite_loss(name):
    torch.manual_seed(0)
    L = CAP['variants'][name]['latent_dim']
    m = build(name).train()
    B, n = 5, 30
    obs_t = torch.sort(torch.rand(B, n) * 7, 1).values; obs_t[:, 0] = 0
    obs = torch.randn(B, n, 2) * .5
    q = torch.linspace(0, 10, 100)
    out = m(obs, obs_t, q, sample=True)
    assert out['prediction'].shape == (B, 100, 2) and out['mu'].shape == (B, L) and out['logvar'].shape == (B, L)
    idx = torch.sort(torch.randperm(70)[:n].expand(B, n), 1).values
    prep = {'obs_indices': idx, 'obs_xy': obs, 'truth': torch.zeros(B, 100, 2), 'scale': .8}
    loss, parts = batch_objective(out, prep, .05, 1.0, 'observed')
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in m.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in m.encoder.parameters() for g in [g.grad])


@pytest.mark.parametrize('name', ['medium', 'large'])
def test_checkpoint_round_trip(tmp_path, name):
    m = build(name).eval()
    cfg = {'model': dict(latent_dim=m.latent_dim, encoder_hidden=m.encoder.gru.hidden_size, dynamics_hidden=64, decoder_hidden=64,
                         cond_dim=0, solver='rk4', adjoint=False, dynamics_layers=2, decoder_layers=2, rtol=1e-4, atol=1e-5, step_size=.1)}
    save_checkpoint(tmp_path / 'c.pt', m, {'dataset_name': 'five_shapes'}, cfg, 1, 0.)
    ck = torch.load(tmp_path / 'c.pt', map_location='cpu', weights_only=True)
    m2 = LatentODE(**ck['model']); m2.load_state_dict(ck['state_dict']); m2.eval()
    obs_t = torch.linspace(0, 7, 30)[None]; obs = torch.randn(1, 30, 2); q = torch.linspace(0, 10, 50)
    assert torch.equal(m(obs, obs_t, q, sample=False)['prediction'], m2(obs, obs_t, q, sample=False)['prediction'])


def test_launch_script_matches_variants():
    sh = (ROOT / 'scripts/run_capacity_scaling.sh').read_text()
    for name in ('medium', 'large'):
        v = CAP['variants'][name]
        line = next(l for l in sh.splitlines() if l.startswith('run ') and Path(v['checkpoint']).stem in l)
        assert f"--latent-dim {v['latent_dim']}" in line and f"--encoder-hidden {v['encoder_hidden']}" in line
