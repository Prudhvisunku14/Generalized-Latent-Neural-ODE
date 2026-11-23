#!/usr/bin/env python3
"""Package the five-family unified checkpoint under saved_models/five_shapes_unified/ and create example inputs
for the two families added in the five-family study (figure-eight, sinusoid).

No training and no new evaluation: the checkpoint is copied byte-for-byte; the recorded test metrics are read from
results/five_shapes/compare_three_vs_five.json (produced by scripts/31_evaluate_five_shapes.py).
"""
import hashlib, json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, torch
from glo.model import LatentODE

root = Path(__file__).resolve().parents[1]
src = root / 'checkpoints/five_shapes/five_shapes_seed42.pt'
dst = root / 'saved_models/five_shapes_unified'
dst.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

ck = torch.load(src, map_location='cpu', weights_only=True)
arch = ck['model']
model = LatentODE(**arch); model.load_state_dict(ck['state_dict'])
n_params = sum(p.numel() for p in model.parameters())
shutil.copyfile(src, dst / 'best_model.pt')
assert sha(src) == sha(dst / 'best_model.pt')

ref = json.loads((root / 'saved_models/three_shapes_unified/model_config.json').read_text())


def make_cfg(a):
    """model_config.json content with descriptions generated from the actual architecture."""
    L, H, D, E = a['latent_dim'], a['encoder_hidden'], a['dynamics_hidden'], a['decoder_hidden']
    desc = {'encoder': f'GRU(input 4 -> hidden {H}), reads the time-reversed sequence of [x, y, t/T, dt/T]; Linear({H} -> {2*L}) gives mean and log-variance of z0',
            'dynamics': f'dz/dt = f(z); tanh MLP with {a["dynamics_layers"]} hidden layers of width {D}; autonomous (no explicit time input)',
            'decoder': f'tanh MLP, {a["decoder_layers"]} hidden layers of width {E}, output 2 (x, y)',
            'solver': 'rk4, fixed step 0.1, integrated from t = 0'}
    return {'architecture': a, 'description': desc}


import torch.nn as _nn
(dst / 'model_config.json').write_text(json.dumps(make_cfg(arch), indent=2))
dmeta = json.loads((root / 'data/processed/five_shapes/metadata.json').read_text())
pre = json.loads((root / 'saved_models/three_shapes_unified/preprocessing.json').read_text())
pre['normalization_mean'] = dmeta['normalization_mean']
pre['normalization_scale'] = dmeta['normalization_scale']
pre['normalization_source'] = 'data/processed/five_shapes/metadata.json (train split statistics)'
pre['normalization_method'] = pre['normalization_method'].replace('three_shapes', 'five_shapes')
pre.pop('coordinate_abs_max_train', None)
(dst / 'preprocessing.json').write_text(json.dumps(pre, indent=2))

cmp_ = json.loads((root / 'results/five_shapes/compare_three_vs_five.json').read_text())
assert cmp_['five_shape_sha256'] == sha(src), 'checkpoint differs from the one that was evaluated'
perf = {k: {m: round(v['mean'], 5) for m, v in cmp_['five_shape_model'][k].items() if isinstance(v, dict)}
        for k in ['overall', *dmeta['families']]}
meta = {
    'model_name': 'five_shapes_unified',
    'description': 'Unified Latent ODE (Model D configuration) trained on circles, spirals, ellipses, figure-eights and sinusoids',
    'training_families': dmeta['families'], 'n_parameters': n_params, 'training_seed': 42,
    'best_checkpoint_epoch': int(ck['epoch']), 'epochs_configured': 300, 'training_n_obs': 30,
    'ode_solver': 'rk4 (fixed step 0.1)', 'original_checkpoint': 'checkpoints/five_shapes/five_shapes_seed42.pt',
    'sha256': sha(dst / 'best_model.pt'), 'file_size_bytes': (dst / 'best_model.pt').stat().st_size,
    'training_dataset': 'five_shapes', 'best_val_neg_elbo': cmp_['five_shape_best_val_neg_elbo'],
    'selection': 'single run (seed 42); checkpoint with the lowest validation -ELBO during training (test data never used)',
    'test_performance (n_obs=30, original units, posterior mean, per-trajectory RMSE averaged; recorded after training)': {
        **perf, 'n_test_trajectories': 750,
        'source': 'results/five_shapes/compare_three_vs_five.json'},
    'caveat': 'single training seed; see reports/five_shapes_report.md and docs/RESEARCH_LIMITATIONS.md',
    'torch_version': torch.__version__,
}
(dst / 'metadata.json').write_text(json.dumps(meta, indent=2))


# ---- controlled 2x2 ablation models (five_shapes, seed 42, data_seed 42); metrics live in results/ablation_2x2/ ----
ABL = {'A': ('lat8_gru64', 8, 64), 'B': ('lat16_gru64', 16, 64), 'C': ('lat8_gru128', 8, 128), 'D': ('lat16_gru128', 16, 128)}
for cell, (tag, lat, gru) in ABL.items():
    s_ = root / f'checkpoints/ablation_2x2/five_shapes_{cell}_{tag}_seed42.pt'
    d_ = root / f'saved_models/ablation_2x2/{cell}_{tag}'; d_.mkdir(parents=True, exist_ok=True)
    c = torch.load(s_, map_location='cpu', weights_only=True)
    assert c['model']['latent_dim'] == lat and c['model']['encoder_hidden'] == gru, (cell, c['model'])
    m_ = LatentODE(**c['model']); m_.load_state_dict(c['state_dict'])
    shutil.copyfile(s_, d_ / 'best_model.pt'); assert sha(s_) == sha(d_ / 'best_model.pt')
    (d_ / 'model_config.json').write_text(json.dumps(make_cfg(c['model']), indent=2))
    (d_ / 'preprocessing.json').write_text(json.dumps(pre, indent=2))
    (d_ / 'metadata.json').write_text(json.dumps({
        'model_name': f'ablation_2x2_{cell}', 'description': f'Controlled 2x2 ablation cell {cell}: latent dim {lat}, GRU width {gru} (five_shapes, seed 42, data_seed 42)',
        'training_families': dmeta['families'], 'n_parameters': sum(p.numel() for p in m_.parameters()), 'training_seed': 42,
        'best_checkpoint_epoch': int(c['epoch']), 'training_n_obs': 30, 'ode_solver': 'rk4 (fixed step 0.1)',
        'original_checkpoint': str(s_.relative_to(root)), 'sha256': sha(d_ / 'best_model.pt'),
        'file_size_bytes': (d_ / 'best_model.pt').stat().st_size, 'training_dataset': 'five_shapes',
        'selection': 'lowest validation -ELBO during training (test data never used)',
        'test_performance': 'see results/ablation_2x2/ and ABLATION_2X2_REPORT.md (not duplicated here)',
        'caveat': 'single training seed; differences between cells are of the same order as seed-to-seed variation',
        'torch_version': torch.__version__}, indent=2))
    print('packaged', cell, d_.name)

# example inputs for figure-eight and sinusoid (held-out test split of five_shapes; first test trajectory of each)
d = np.load(root / 'data/processed/five_shapes/test.npz')
t = d['time'].astype(np.float64); inp = root / 'input_trajectories'; gt = inp / 'ground_truth'
for fam, fid in (('figure_eight', 3), ('sinusoid', 4)):
    i = int(np.flatnonzero(d['family'] == fid)[0]); m = d['default_obs_mask'][i]
    assert m.sum() == 30 and m[0] and not m[t > 7].any()
    with open(inp / f'example_{fam}.csv', 'w') as f:
        f.write('t,x,y\n'); [f.write(f'{a:.9g},{b:.9g},{c:.9g}\n') for a, (b, c) in zip(t[m], d['noisy_xy'][i][m])]
    with open(gt / f'example_{fam}.csv', 'w') as f:
        f.write('t,x,y\n'); [f.write(f'{a:.9g},{b:.9g},{c:.9g}\n') for a, (b, c) in zip(t, d['xy'][i])]
print('packaged', dst, n_params, 'parameters')
