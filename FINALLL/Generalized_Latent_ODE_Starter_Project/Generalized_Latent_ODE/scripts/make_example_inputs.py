#!/usr/bin/env python3
"""Create example input CSVs from HELD-OUT test trajectories (three_shapes test split).

For each family the first test trajectory is used. The input CSV holds 30 irregular, noisy observations at
t <= 7 (the dataset's stored observation mask, whose first point is t = 0): columns t, x, y ONLY, with no family
information. The dense clean trajectory (200 points on [0, 10]) goes to a separate ground-truth CSV that is
used only to score predictions after they are made.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from glo.data import load_split, FAMILY_TO_ID

root = Path(__file__).resolve().parents[1]
arrays, meta = load_split(root / 'data/processed', 'three_shapes', 'test')
inp = root / 'input_trajectories'; gt_dir = inp / 'ground_truth'
gt_dir.mkdir(parents=True, exist_ok=True)
t = arrays['time'].astype(np.float64)
prov = {}
for fam, fid in (('circle', 0), ('spiral', 1), ('ellipse', 2)):
    idx = int(np.flatnonzero(arrays['family'] == fid)[0])
    mask = arrays['default_obs_mask'][idx]
    assert mask.sum() == 30 and mask[0] and not mask[t > meta['forecast_start']].any()
    obs_t, obs_xy = t[mask], arrays['noisy_xy'][idx][mask].astype(np.float64)
    with open(inp / f'example_{fam}.csv', 'w') as fh:
        fh.write('t,x,y\n')
        for ti, (xi, yi) in zip(obs_t, obs_xy):
            fh.write(f'{ti:.9g},{xi:.9g},{yi:.9g}\n')
    with open(gt_dir / f'example_{fam}.csv', 'w') as fh:           # same stem: used by --ground-truth-dir in batch mode
        fh.write('t,x,y\n')
        for ti, (xi, yi) in zip(t, arrays['xy'][idx].astype(np.float64)):
            fh.write(f'{ti:.9g},{xi:.9g},{yi:.9g}\n')
    prov[f'example_{fam}'] = {'dataset': 'three_shapes', 'split': 'test', 'array_index': idx, 'trajectory_id': str(arrays['ids'][idx]),
                              'true_parameters (radius_or_r0, omega, phase, cx, cy, growth, a, b, angle)': [float(v) for v in arrays['params'][idx]]}
(gt_dir / 'provenance.json').write_text(json.dumps({
    'note': 'Provenance of the example trajectories; kept OUT of the input CSVs, never read by the inference code.',
    'noise': 'observations are the dataset noisy_xy (sigma 0.05) at the stored default_obs_mask times; ground truth is clean',
    'examples': prov}, indent=2))
print('wrote', sorted(p.name for p in inp.glob('*.csv')), 'and ground_truth/')
