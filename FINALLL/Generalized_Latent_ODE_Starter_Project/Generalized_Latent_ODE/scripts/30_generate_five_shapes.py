#!/usr/bin/env python3
"""Generate data/processed/five_shapes (circle, spiral, ellipse, figure-eight, sinusoid; 1,000 each) and verify it.

Checks (results/five_shapes/dataset_checks.json; the script exits non-zero if any fails):
  * split counts 3,500 / 750 / 750, 700 / 150 / 150 per family, no trajectory id in two splits (data.validate_dataset);
  * circle / spiral / ellipse are byte-identical to data/processed/three_shapes (trajectories, noisy observations,
    observation masks, parameters, split membership);
  * every figure-eight crosses itself at its centre with alternating crossing directions and y oscillates twice as fast
    as x; every sinusoid moves monotonically and linearly in x with the drawn amplitude in y;
  * coordinate scale per family, compared with the original families.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from glo import five_shapes as F
from glo.data import load_split, validate_dataset

root = Path(__file__).resolve().parents[1]
data_root = root / 'data/processed'
out = root / 'results/five_shapes'; fig_dir = out / 'figures'; fig_dir.mkdir(parents=True, exist_ok=True)
meta = F.generate_five_shapes(data_root, three_shapes_dir=data_root / 'three_shapes')
checks = {'split_counts': validate_dataset(data_root, 'five_shapes'), 'failures': []}
fail = checks['failures'].append

per_family = {}
for split in ('train', 'val', 'test'):
    a5, _ = load_split(data_root, 'five_shapes', split)
    per_family[split] = {f: int((a5['family'] == i).sum()) for f, i in F.FAMILY_TO_ID.items()}
    want = {'train': 700, 'val': 150, 'test': 150}[split]
    if any(v != want for v in per_family[split].values()):
        fail(f'{split}: per-family counts {per_family[split]}')
    a3, _ = load_split(data_root, 'three_shapes', split)
    pos = {k: i for i, k in enumerate(a5['ids'].tolist())}
    if not set(a3['ids'].tolist()) <= set(pos):
        fail(f'{split}: some three_shapes ids are not in the same five_shapes split')
        continue
    j = np.array([pos[k] for k in a3['ids'].tolist()])
    same = {k: bool(np.array_equal(a3[k], a5[k][j])) for k in ('xy', 'noisy_xy', 'default_obs_mask', 'family')}
    same['params'] = bool(np.array_equal(a3['params'], a5['params'][j][:, :9]))
    same['time'] = bool(np.array_equal(a3['time'], a5['time']))
    checks[f'identical_to_three_shapes_{split}'] = same
    if not all(same.values()):
        fail(f'{split}: original families differ from three_shapes: {same}')
checks['per_family_split_counts'] = per_family
all_ids = np.concatenate([load_split(data_root, 'five_shapes', s)[0]['ids'] for s in ('train', 'val', 'test')])
checks['n_trajectories'] = int(len(all_ids)); checks['n_unique_ids'] = int(len(set(all_ids.tolist())))
if checks['n_trajectories'] != 5000 or checks['n_unique_ids'] != 5000:
    fail('expected 5,000 unique trajectories')

# ------------------------------------------------------------------------------------- shape checks (all splits)
arrs = {s: load_split(data_root, 'five_shapes', s)[0] for s in ('train', 'val', 'test')}
xy = np.concatenate([arrs[s]['xy'] for s in arrs]); fam = np.concatenate([arrs[s]['family'] for s in arrs])
par = np.concatenate([arrs[s]['params'] for s in arrs]); t = arrs['train']['time']
fe = [F.figure_eight_checks(x, p, t) for x, p in zip(xy[fam == 3], par[fam == 3])]
si = [F.sinusoid_checks(x, p, t) for x, p in zip(xy[fam == 4], par[fam == 4])]
ratio = np.array([c['y_zero_crossings'] / max(c['x_zero_crossings'], 1) for c in fe])
checks['figure_eight'] = {
    'n': len(fe), 'all_cover_a_full_eight': all(c['covers_full_eight'] for c in fe),
    'all_have_alternating_centre_crossings': all(c['crossing_directions_alternate'] for c in fe),
    'centre_passes_min_median_max': [int(np.min([c['centre_passes'] for c in fe])), float(np.median([c['centre_passes'] for c in fe])),
                                     int(np.max([c['centre_passes'] for c in fe]))],
    'max_abs_y_offset_at_centre_over_B': float(max(c['max_abs_y_offset_at_centre_over_B'] for c in fe)),
    'y_to_x_zero_crossing_ratio_median': float(np.median(ratio)), 'y_to_x_zero_crossing_ratio_range': [float(ratio.min()), float(ratio.max())]}
if not (checks['figure_eight']['all_cover_a_full_eight'] and checks['figure_eight']['all_have_alternating_centre_crossings']
        and checks['figure_eight']['max_abs_y_offset_at_centre_over_B'] < 0.1):
    fail('figure-eight shape check failed')
hr = np.array([c['y_half_range_over_amplitude'] for c in si])
checks['sinusoid'] = {'n': len(si), 'all_x_monotone': all(c['x_monotone'] for c in si),
                      'x_linear_max_residual': float(max(c['x_linear_max_residual'] for c in si)),
                      'y_half_range_over_amplitude_min_median_max': [float(hr.min()), float(np.median(hr)), float(hr.max())]}
if not checks['sinusoid']['all_x_monotone'] or checks['sinusoid']['x_linear_max_residual'] > 1e-4:
    fail('sinusoid shape check failed')

# ------------------------------------------------------------------------------------------ coordinate scale
names = {v: k for k, v in F.FAMILY_TO_ID.items()}
scale = {}
for i, f in names.items():
    X = xy[fam == i].astype(np.float64)
    c = X.mean(1, keepdims=True)
    scale[f] = {'rms_radius_about_own_centroid_median': float(np.median(np.sqrt(((X - c) ** 2).sum(-1).mean(-1)))),
                'max_abs_coordinate_p95': float(np.percentile(np.abs(X).max((1, 2)), 95)),
                'bounding_box_diagonal_median': float(np.median(np.linalg.norm(X.max(1) - X.min(1), axis=-1)))}
checks['coordinate_scale_per_family'] = scale
checks['normalization'] = {'five_shapes_mean': meta['normalization_mean'], 'five_shapes_scale': meta['normalization_scale'],
                           'three_shapes_scale': json.loads((data_root / 'three_shapes/metadata.json').read_text())['normalization_scale']}
(out / 'dataset_checks.json').write_text(json.dumps(checks, indent=2))

# ------------------------------------------------------------------------------------------------- figures
tr = arrs['train']
fig, axes = plt.subplots(5, 4, figsize=(13, 15.5))
for r, (f, i) in enumerate(F.FAMILY_TO_ID.items()):
    for c, k in enumerate(np.flatnonzero(tr['family'] == i)[:4]):
        ax = axes[r, c]; m = tr['default_obs_mask'][k]
        ax.plot(*tr['xy'][k].T, color='#1965a5', lw=1.8, label='ground truth (t in [0, 10])')
        ax.plot(*tr['xy'][k][0], 'o', color='#1965a5', ms=4)
        ax.scatter(*tr['noisy_xy'][k][m].T, color='#e47b37', s=9, zorder=3, label='30 noisy observations (t <= 7)')
        ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.15); ax.set_title(f'{f.replace("_", "-")} / {c + 1}', fontsize=10)
        if r == 0 and c == 0: ax.legend(fontsize=7, frameon=False)
fig.suptitle('Five-shape dataset (training split, first four of each family; dot = start)', fontsize=13)
fig.tight_layout(rect=(0, 0, 1, .975)); fig.savefig(fig_dir / 'five_shapes_gallery.png', dpi=130); plt.close(fig)
for i, f in ((3, 'figure_eight'), (4, 'sinusoid')):
    rng = np.random.default_rng(7); pick = rng.choice(np.flatnonzero(fam == i), 12, replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(13, 9.5))
    for ax, k in zip(axes.flat, pick):
        sc = ax.scatter(*xy[k].T, c=t, cmap='viridis', s=6); ax.plot(*xy[k][0], 'k^', ms=6)
        ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.15)
        ax.set_title(f'A={par[k][6]:.2f} B={par[k][7]:.2f} w={par[k][1]:+.2f}' if i == 3 else
                     f'A={par[k][0]:.2f} w={par[k][1]:.2f} v={par[k][9]:+.2f}', fontsize=9)
    fig.colorbar(sc, ax=axes, shrink=.6, label='t')
    fig.suptitle(f'{f.replace("_", "-")}: 12 random trajectories (seed 7, not chosen by appearance; triangle = start, colour = time)')
    fig.savefig(fig_dir / f'{f}_samples.png', dpi=130, bbox_inches='tight'); plt.close(fig)

print(json.dumps({k: v for k, v in checks.items() if k not in ('coordinate_scale_per_family',)}, indent=1))
print(json.dumps(scale, indent=1))
print('FAILURES:', checks['failures'] or 'none')
sys.exit(1 if checks['failures'] else 0)
