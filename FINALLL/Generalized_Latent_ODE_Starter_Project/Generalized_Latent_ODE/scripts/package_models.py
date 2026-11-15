#!/usr/bin/env python3
"""Select, copy and verify the pretrained checkpoints under ``saved_models/``. No training happens here.

Candidate pools are defined from each checkpoint's STORED configuration (never from file names). Within a pool the
winner is the checkpoint with the lowest validation -ELBO measured with IDENTICAL observation subsets for every
candidate (the training-time validation losses used different random subsets per seed and are not comparable).
Only the validation split drives selection; test metrics are computed afterwards for the metadata only.
"""
import argparse, glob, json, math, shutil, sys, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, torch
from glo.model import LatentODE
from glo.data import load_split
from glo.evaluation import collect, by_family
from glo.inference import (MODEL_REGISTRY, load_packaged_model, predict, sha256_file, state_dict_digest)
from glo.pipeline import TrajectoryDataset, create_loader, make_query_indices, prepare_batch, batch_objective

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--out', default='saved_models')
p.add_argument('--force', action='store_true', help='Replace an existing saved_models folder content (originals are never touched)')
p.add_argument('--val-seed', type=int, default=2027, help='Seed for the identical validation observation subsets')
a = p.parse_args()
torch.set_num_threads(4)
root = Path(__file__).resolve().parents[1]
data_root = root / 'data/processed'
out_root = root / a.out

ARCH = dict(latent_dim=8, encoder_hidden=64, dynamics_hidden=64, decoder_hidden=64, dynamics_layers=2, decoder_layers=2,
            solver='rk4', adjoint=False, rtol=1e-4, atol=1e-5, step_size=0.1)
BASE_TRAIN = dict(lr=0.003, lr_schedule='cosine', kl_weight=1.0, kl_anneal_epochs=0, n_obs=30, supervision='observed', sigma=0.05,
                  batch_size=32, n_query=100)

CATEGORIES = {   # cli name -> (experiment, families, cond_dim, epochs, n_obs_min>0?)
    'two_shapes_unified': dict(experiment='two_shapes', families=None, cond=0, epochs=300, var=False),
    'three_shapes_unified': dict(experiment='three_shapes', families=None, cond=0, epochs=300, var=False),
    'circle_specialist': dict(experiment='two_shapes', families=['circle'], cond=0, epochs=600, var=False),
    'spiral_specialist': dict(experiment='two_shapes', families=['spiral'], cond=0, epochs=600, var=False),
    'dynamics_conditioned': dict(experiment='two_shapes', families=None, cond=4, epochs=300, var=False),
    'variable_nobs_experimental': dict(experiment='two_shapes', families=None, cond=0, epochs=300, var=True),
}
DESCRIPTIONS = {
    'two_shapes_unified': 'M1: unified Latent ODE (Model D configuration) trained on circles + spirals',
    'three_shapes_unified': 'Unified Latent ODE (Model D configuration) trained on circles + spirals + ellipses',
    'circle_specialist': 'Circle-only specialist (Model D configuration)',
    'spiral_specialist': 'Spiral-only specialist (Model D configuration)',
    'dynamics_conditioned': 'EXPERIMENTAL M3: dynamics-conditioned Latent ODE, dz/dt = f(z, c), c in R^4, trained on circles + spirals',
    'variable_nobs_experimental': 'EXPERIMENTAL ablation: Model D configuration trained with a random 10-50 observations per batch (circles + spirals)',
}


def classify(ck):
    """Return (category or None, reason-if-excluded) from the checkpoint's stored configuration."""
    m, t = ck['model'], ck['config']['training']
    arch_ok = all(m.get(k) == v for k, v in ARCH.items())
    if not arch_ok:
        return None, 'architecture differs from Model D (latent/hidden/layers/solver)'
    if ck['epoch'] < 50:
        return None, 'smoke test (very few epochs)'
    for k, v in BASE_TRAIN.items():
        if t.get(k) != v and not (k == 'lr' and abs(t.get(k, 0) - v) < 1e-12):
            return None, f'training setting {k}={t.get(k)} differs from the Model D recipe ({v})'
    var = (t.get('n_obs_min') or 0) > 0
    fams = t.get('families')
    for name, c in CATEGORIES.items():
        if (c['experiment'] == ck['experiment'] and c['families'] == fams and c['cond'] == m['cond_dim'] and c['var'] == var
                and t['epochs'] == c['epochs']):
            return name, None
    return None, 'does not match any packaged category'


def validation_score(path, ck, families):
    """-ELBO (training criterion), reconstruction/forecast RMSE on the VALIDATION split, identical subsets for all candidates."""
    model = LatentODE(**ck['model']); model.load_state_dict(ck['state_dict']); model.eval()
    data = TrajectoryDataset(data_root, ck['experiment'], 'val', families)
    meta = data.meta
    qidx = make_query_indices(meta['n_points'], 100)
    time = torch.tensor(data.arrays['time'], dtype=torch.float32)
    gen = torch.Generator().manual_seed(a.val_seed)
    tot = {'loss': 0., 'nll': 0., 'kl': 0.}; n = 0; fsq = csq = fc = cc = 0.
    with torch.no_grad():
        for batch in create_loader(data, batch_size=100, shuffle=False):
            P = prepare_batch(batch, time, meta, qidx, 30, 'cpu', generator=gen)
            out = model(P['obs_xy'], P['obs_t'], P['query_t'], sample=False)
            _, parts = batch_objective(out, P, 0.05, 1.0, 'observed')
            B = P['obs_xy'].shape[0]
            for k in tot: tot[k] += parts[k] * B
            n += B
            d2 = ((out['prediction'] - P['truth']) * P['scale']).square()
            fut = P['query_t'] > meta['forecast_start']
            fsq += float(d2[:, fut].sum()); fc += d2[:, fut].numel(); csq += float(d2[:, ~fut].sum()); cc += d2[:, ~fut].numel()
    return {'val_neg_elbo': tot['loss'] / n, 'val_nll': tot['nll'] / n, 'val_kl': tot['kl'] / n,
            'val_context_rmse': math.sqrt(csq / cc), 'val_forecast_rmse': math.sqrt(fsq / fc), 'n_val_trajectories': n}


def relpath(path):
    return str(Path(path).resolve().relative_to(root))


# ------------------------------------------------------------------------------ discover and score
files = sorted(glob.glob(str(root / 'checkpoints/**/*.pt'), recursive=True))
cands = {name: [] for name in CATEGORIES}
excluded = []
sha_seen = {}
for f in files:
    ck = torch.load(f, map_location='cpu', weights_only=True)
    digest = sha256_file(f)
    cat, why = classify(ck)
    if cat is None:
        excluded.append({'checkpoint': relpath(f), 'reason': why}); continue
    if digest in sha_seen:
        excluded.append({'checkpoint': relpath(f), 'reason': f'byte-identical duplicate of {sha_seen[digest]}'}); continue
    sha_seen[digest] = relpath(f)
    fams = ck['config']['training'].get('families')
    sc = validation_score(f, ck, fams)
    cands[cat].append({'checkpoint': relpath(f), 'sha256': digest, 'seed': ck['config']['training']['seed'], 'epoch': int(ck['epoch']),
                       'stored_training_val_loss': ck['val_loss'], **sc})
    print(f"[{cat}] {relpath(f):55s} seed={cands[cat][-1]['seed']:5d} epoch={ck['epoch']:3d} val -ELBO={sc['val_neg_elbo']:8.2f} "
          f"val fc RMSE={sc['val_forecast_rmse']:.4f}", flush=True)

selection = {}
for name, lst in cands.items():
    if not lst:
        selection[name] = None; print(f'!! NO CHECKPOINT FOUND for {name}; it will not be packaged (nothing is retrained or invented)'); continue
    lst.sort(key=lambda r: r['val_neg_elbo'])
    selection[name] = lst[0]
    by_fc = min(lst, key=lambda r: r['val_forecast_rmse'])
    lst[0]['selected_by_criterion'] = 'validation -ELBO (identical observation subsets)'
    lst[0]['agrees_with_best_val_forecast_rmse'] = by_fc['checkpoint'] == lst[0]['checkpoint']
    print(f"==> {name}: {lst[0]['checkpoint']} (pool of {len(lst)}; best val forecast RMSE in pool: {by_fc['checkpoint']})")

# ------------------------------------------------------------------------------ copy + metadata
if out_root.exists() and any(out_root.iterdir()) and not a.force:
    sys.exit(f'{out_root} already has content; re-run with --force to rebuild it (original checkpoints are never modified).')
out_root.mkdir(exist_ok=True)
summary_rows = []
for name, sel in selection.items():
    if sel is None:
        continue
    cat = CATEGORIES[name]
    dest = out_root / MODEL_REGISTRY[name]
    dest.mkdir(parents=True, exist_ok=True)
    src = root / sel['checkpoint']
    shutil.copyfile(src, dest / 'best_model.pt')
    assert sha256_file(src) == sha256_file(dest / 'best_model.pt') == sel['sha256'], 'checksum mismatch after copy'
    ck = torch.load(dest / 'best_model.pt', map_location='cpu', weights_only=True)
    train_cfg = ck['config']['training']
    meta_ds = json.loads((data_root / cat['experiment'] / 'metadata.json').read_text())
    arrays, _ = load_split(data_root, cat['experiment'], 'train')
    fams_trained = cat['families'] or meta_ds['families']
    model = LatentODE(**ck['model'])
    n_params = sum(p_.numel() for p_ in model.parameters())
    # ---- model_config.json
    (dest / 'model_config.json').write_text(json.dumps({
        'architecture': ck['model'],
        'description': {
            'encoder': f"GRU(input 4 -> hidden {ck['model']['encoder_hidden']}), reads the time-reversed sequence of [x, y, t/T, dt/T]; "
                       f"Linear({ck['model']['encoder_hidden']} -> {2 * ck['model']['latent_dim']}) gives mean and log-variance of z0"
                       + (f"; Linear({ck['model']['encoder_hidden']} -> {ck['model']['cond_dim']}) + tanh gives the dynamics code c"
                          if ck['model']['cond_dim'] else ''),
            'dynamics': f"dz/dt = f(z{', c' if ck['model']['cond_dim'] else ''}); tanh MLP with {ck['model']['dynamics_layers']} hidden layers of width "
                        f"{ck['model']['dynamics_hidden']}; autonomous (no explicit time input)",
            'decoder': f"tanh MLP, {ck['model']['decoder_layers']} hidden layers of width {ck['model']['decoder_hidden']}, output 2 (x, y)",
            'solver': f"{ck['model']['solver']}, fixed step {ck['model']['step_size']}, integrated from t = 0"},
        'module_repr': str(model).splitlines(),
        'n_parameters': n_params}, indent=2))
    # ---- preprocessing.json
    pre = {'input_columns': ['t', 'x', 'y'],
           'normalization_method': 'affine, applied to x and y only: (xy - mean) / scale with one scalar scale = RMS deviation of the '
                                   f"{cat['experiment']} TRAINING-split coordinates about their mean; inverted as xy_hat * scale + mean",
           'normalization_mean': meta_ds['normalization_mean'], 'normalization_scale': meta_ds['normalization_scale'],
           'normalization_source': f"data/processed/{cat['experiment']}/metadata.json (train split statistics)",
           'time_scale_T': 10.0, 'time_horizon': float(meta_ds['t_end']), 'context_end': float(meta_ds['forecast_start']),
           'training_n_obs': int(train_cfg['n_obs']) if not (train_cfg.get('n_obs_min') or 0) else
           {'min': int(train_cfg['n_obs_min']), 'max': int(train_cfg['n_obs_max'])},
           'encoder_features': '[x_norm, y_norm, t/T, dt/T] per observation',
           'encoder_time_order': 'sequence is reversed (newest observation first) before the GRU; hidden state starts at zero',
           'time_gap_convention': 'on the reversed sequence gap[0] = 0 and gap[i] = t_rev[i-1] - t_rev[i] (positive: later minus earlier time)',
           'integration_origin_t': 0.0, 'latent_state_used': 'posterior mean z0 = mu (deterministic)',
           'training_observation_noise_std': meta_ds['base_noise_std'], 'training_time_grid': f"{meta_ds['n_points']} points on [0, {meta_ds['t_end']}]",
           'coordinate_abs_max_train': float(np.abs(arrays['xy']).max()),
           'first_observation_always_at_t0_in_training': True}
    (dest / 'preprocessing.json').write_text(json.dumps(pre, indent=2))
    # ---- performance (test metrics are for the record only; selection is already final)
    loaded_model = LatentODE(**ck['model']); loaded_model.load_state_dict(ck['state_dict']); loaded_model.eval()
    test_exp, test_fams = cat['experiment'], cat['families']
    res = collect(loaded_model, ck, data_root, test_exp, split='test', families=test_fams, n_obs=30, n_query=200, seed=123)
    test_perf = {fam: {k: round(v['mean'], 5) for k, v in d.items() if isinstance(v, dict)} for fam, d in by_family(res).items()}
    test_perf['n_test_trajectories'] = int(len(res['family']))
    sha = sel['sha256']
    meta = {'model_name': name, 'description': DESCRIPTIONS[name], 'training_families': fams_trained,
            'n_parameters': n_params, 'training_seed': int(train_cfg['seed']), 'best_checkpoint_epoch': int(ck['epoch']),
            'epochs_configured': int(train_cfg['epochs']), 'training_n_obs': pre['training_n_obs'],
            'time_range': {'full': [0.0, pre['time_horizon']], 'context': [0.0, pre['context_end']],
                           'forecast': [pre['context_end'], pre['time_horizon']]},
            'normalization': pre['normalization_method'], 'ode_solver': f"{ck['model']['solver']} (fixed step {ck['model']['step_size']})",
            'original_checkpoint': sel['checkpoint'], 'sha256': sha, 'file_size_bytes': (dest / 'best_model.pt').stat().st_size,
            'training_dataset': cat['experiment'],
            'stored_training_val_loss_neg_elbo': ck['val_loss'],
            'selection': {'criterion': 'lowest validation -ELBO with identical observation subsets (seed %d), validation split only' % a.val_seed,
                          'pool_size': len(cands[name]), 'validation_scores_of_selected': {k: round(sel[k], 5) for k in
                          ('val_neg_elbo', 'val_nll', 'val_kl', 'val_context_rmse', 'val_forecast_rmse')},
                          'agrees_with_best_val_forecast_rmse_in_pool': sel['agrees_with_best_val_forecast_rmse'],
                          'full_candidate_table': 'saved_models/selection_report.json'},
            'test_performance (n_obs=30, original units, posterior mean; recorded AFTER selection, not used for it)': test_perf,
            'training_config_keys': {k: train_cfg.get(k) for k in ('lr', 'lr_schedule', 'min_lr', 'batch_size', 'grad_clip', 'kl_weight',
                                     'kl_anneal_epochs', 'sigma', 'supervision', 'n_query', 'patience', 'families', 'data_seed')},
            'git_commit': 'unavailable (repository has no commits)', 'torch_version': torch.__version__,
            'packaged_on': datetime.date.today().isoformat(),
            'dataset_metadata_sha256': __import__('hashlib').sha256((data_root / cat['experiment'] / 'metadata.json').read_bytes()).hexdigest()}
    if name == 'dynamics_conditioned':
        meta['warning'] = 'experimental: in the conditioning study it did not beat the unified baseline on forecasting'
    if name == 'variable_nobs_experimental':
        meta['warning'] = ('experimental: robust to 10-50+ observations but its forecasts at 30 observations were about twice as '
                           'inaccurate as the unified Model D (see reports/validation_study_report.md, section 8)')
    if name.endswith('specialist'):
        meta['warning'] = f"valid only for {fams_trained[0]} trajectories"
    (dest / 'metadata.json').write_text(json.dumps(meta, indent=2))
    summary_rows.append((name, sel['checkpoint'], n_params, int(ck['epoch'])))

(out_root / 'selection_report.json').write_text(json.dumps({
    'criterion': 'lowest validation -ELBO, identical observation subsets (seed %d), validation split only' % a.val_seed,
    'note': 'stored training val_loss values used different random observation subsets per seed and are not comparable',
    'candidates': {k: v for k, v in cands.items()}, 'selected': {k: (v['checkpoint'] if v else None) for k, v in selection.items()},
    'excluded_checkpoints': excluded}, indent=2))

# ------------------------------------------------------------------------------ verify every packaged model
report = {}
for name, *_ in summary_rows:
    entry = {}
    lm = load_packaged_model(name, out_root)
    entry['architecture_rebuilt'] = True
    entry['missing_keys'], entry['unexpected_keys'] = lm.load_report['missing_keys'], lm.load_report['unexpected_keys']
    entry['n_parameters'] = lm.n_parameters
    entry['sha256_packaged'] = lm.sha256
    orig = root / lm.metadata['original_checkpoint']
    entry['sha256_original'] = sha256_file(orig)
    entry['checksums_identical'] = entry['sha256_original'] == entry['sha256_packaged']
    # forward-pass check on one validation trajectory of the model's own training dataset
    exp = CATEGORIES[name]['experiment']
    fams = CATEGORIES[name]['families']
    data = TrajectoryDataset(data_root, exp, 'val', fams)
    meta_ds = data.meta
    mask = data.arrays['default_obs_mask'][0]
    t_obs = data.arrays['time'][mask].astype(np.float64); xy_obs = data.arrays['noisy_xy'][0][mask].astype(np.float64)
    q = np.linspace(0, 10, 200)
    pr = predict(lm, t_obs, xy_obs, q)
    entry['forward_test'] = {'n_observations': int(mask.sum()), 'output_shape': list(pr['pred'].shape), 'finite': bool(np.isfinite(pr['pred']).all()),
                             'has_dynamics_code': pr['cond'] is not None, 'expected_has_code': bool(lm.model_config['architecture']['cond_dim'])}
    assert entry['checksums_identical'] and tuple(pr['pred'].shape) == (200, 2) and entry['forward_test']['finite']
    assert entry['forward_test']['has_dynamics_code'] == entry['forward_test']['expected_has_code']
    report[name] = entry
    print(f"verified {name}: params={lm.n_parameters} sha256 identical={entry['checksums_identical']} forward ok")
(out_root / 'verification_report.json').write_text(json.dumps(report, indent=2))
print('\nPackaged:')
for r in summary_rows:
    print(f'  {r[0]:28s} <- {r[1]}  ({r[2]} params, epoch {r[3]})')
missing = [n for n, s in selection.items() if s is None]
print('Not packaged (no checkpoint found):', missing if missing else 'none')
