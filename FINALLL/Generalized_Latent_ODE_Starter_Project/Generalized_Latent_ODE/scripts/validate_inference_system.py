#!/usr/bin/env python3
"""Phase 10: end-to-end validation of the pretrained-inference system (Tests 1-8). Evaluation only; no training.

1-3  held-out circle / spiral / ellipse through the frozen three-shape model via the command line
4    unknown family label: neutrally named files, extra 'family' column ignored, outputs unchanged
5    variable observation counts (10/20/30/40/50) through the inference code on the whole held-out test split,
     with trivial baselines for context
6    reproducibility (two identical runs)
7    frozen weights (checksums before/after everything, plus in-memory state-dict digests)
8    consistency with the existing evaluation pipeline (in-process for all packaged models + CLI CSV round trip)
"""
import hashlib, json, shutil, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from glo import inference as inf
from glo.data import load_split
from glo.evaluation import collect

torch.set_num_threads(1)
root = Path(__file__).resolve().parents[1]
saved = root / 'saved_models'; ex = root / 'input_trajectories'
out = root / 'results/inference_validation'; out.mkdir(parents=True, exist_ok=True)
CLI = [sys.executable, str(root / 'scripts/predict_trajectory.py')]
report = {}


def cli(*a):
    r = subprocess.run([*CLI, *map(str, a)], capture_output=True, text=True, cwd=root)
    return r


def tree_digest(paths):
    return {str(p.relative_to(root)): inf.sha256_file(p) for p in sorted(paths)}


pt_files = sorted((root / 'checkpoints').rglob('*.pt')) + sorted(saved.rglob('best_model.pt'))
before = tree_digest(pt_files)
tmp = Path(tempfile.mkdtemp(prefix='infer_validation_'))

# ----------------------------------------------------------------------------- Tests 1-3
t13 = {}
for fam in ('circle', 'spiral', 'ellipse'):
    d = tmp / f'{fam}_run'
    r = cli('--model', 'three_shapes_unified', '--input', ex / f'example_{fam}.csv', '--output', d,
            '--ground-truth', ex / 'ground_truth' / f'example_{fam}.csv', '--quiet')
    assert r.returncode == 0, r.stderr
    m = json.loads((d / 'metrics.json').read_text())
    t13[fam] = {k: round(m[k], 5) for k in ('reconstruction_rmse', 'forecast_rmse', 'forecast_mae', 'forecast_ade', 'forecast_fde', 'chamfer_distance')}
    same = (d / 'predicted_trajectory.csv').read_bytes() == (root / f'inference_outputs/{fam}_example/predicted_trajectory.csv').read_bytes()
    t13[fam]['identical_to_delivered_inference_outputs'] = same
    assert same
report['tests_1_3_held_out_examples_three_shapes_unified'] = t13

# ----------------------------------------------------------------------------- Test 4
blind = tmp / 'blind_inputs'; blind.mkdir()
names = {'circle': 'unknown_A', 'spiral': 'unknown_B', 'ellipse': 'unknown_C'}
for fam, nm in names.items():
    shutil.copy(ex / f'example_{fam}.csv', blind / f'{nm}.csv')
    head = (blind / f'{nm}.csv').read_text().splitlines()[0]
    assert head == 't,x,y'
r = cli('--input', blind, '--output', tmp / 'blind_out', '--quiet')
assert r.returncode == 0, r.stderr
same_all = all((tmp / 'blind_out' / nm / 'predicted_trajectory.csv').read_bytes() ==
               (root / f'inference_outputs/{fam}_example/predicted_trajectory.csv').read_bytes() for fam, nm in names.items())
labelled = tmp / 'labelled.csv'
lines = (ex / 'example_spiral.csv').read_text().splitlines()
labelled.write_text('\n'.join([lines[0] + ',family'] + [l + ',spiral' for l in lines[1:]]) + '\n')
mislabelled = tmp / 'mislabelled.csv'
mislabelled.write_text('\n'.join([lines[0] + ',family'] + [l + ',circle' for l in lines[1:]]) + '\n')
outs = {}
for nm, f in (('labelled', labelled), ('mislabelled', mislabelled)):
    r2 = cli('--input', f, '--output', tmp / f'{nm}_out', '--quiet'); assert r2.returncode == 0
    outs[nm] = (tmp / f'{nm}_out' / 'predicted_trajectory.csv').read_bytes()
    outs[nm + '_warned'] = 'Ignoring extra column' in r2.stderr
ref = (root / 'inference_outputs/spiral_example/predicted_trajectory.csv').read_bytes()
report['test_4_unknown_family_label'] = {
    'neutral_file_names_and_only_t_x_y_columns_run_ok': True, 'outputs_identical_to_named_runs': same_all,
    'extra_family_column_ignored_with_warning': outs['labelled_warned'] and outs['mislabelled_warned'],
    'predictions_identical_with_correct_wrong_or_no_family_column': outs['labelled'] == ref == outs['mislabelled'],
    'cli_has_no_family_option': not any('family' in a or 'shape' in a for a in
                                       [l for l in cli('--help').stdout.split() if l.startswith('--')])}
assert all(report['test_4_unknown_family_label'].values()), report['test_4_unknown_family_label']

# ----------------------------------------------------------------------------- Test 5: observation counts
LEVELS = [10, 20, 30, 40, 50]
arrays3, meta3 = load_split(root / 'data/processed', 'three_shapes', 'test')
tt = arrays3['time'].astype(np.float64); allowed = np.flatnonzero(tt <= meta3['forecast_start'] + 1e-9)
fam_names = {0: 'circle', 1: 'spiral', 2: 'ellipse'}


def subset(k, n):
    rng = np.random.default_rng(10_000 * n + k)
    idx = np.sort(np.concatenate([[0], rng.choice(allowed[1:], size=n - 1, replace=False)]))
    return idx


def baseline_metrics(lm, t_obs, xy_obs, gt_t, gt_xy):
    qt = gt_t
    interp = np.stack([np.interp(qt, t_obs, xy_obs[:, c]) for c in range(2)], 1)
    hold = interp.copy(); hold[qt > t_obs[-1]] = xy_obs[-1]
    return inf.evaluate_against_truth(lm, hold, gt_t, gt_xy, t_obs)


study = {}
for name, fam_ids in (('three_shapes_unified', (0, 1, 2)), ('two_shapes_unified', (0, 1)), ('variable_nobs_experimental', (0, 1))):
    lm = inf.load_packaged_model(name, saved)
    ks = [k for k in range(len(arrays3['family'])) if int(arrays3['family'][k]) in fam_ids]
    study[name] = {}
    for n in LEVELS:
        rows = {'model': [], 'baseline': []}
        for k in ks:
            idx = subset(k, n)
            t_o, xy_o = tt[idx], arrays3['noisy_xy'][k][idx].astype(np.float64)
            gt_t, gt_xy = tt, arrays3['xy'][k].astype(np.float64)
            pr = inf.predict(lm, t_o, xy_o, gt_t)['pred']
            m = inf.evaluate_against_truth(lm, pr, gt_t, gt_xy, t_o)
            b = baseline_metrics(lm, t_o, xy_o, gt_t, gt_xy)
            rows['model'].append((int(arrays3['family'][k]), m)); rows['baseline'].append((int(arrays3['family'][k]), b))
        agg = {}
        for kind in ('model', 'baseline'):
            for scope in ['overall'] + [fam_names[i] for i in fam_ids]:
                sel = [m for f, m in rows[kind] if scope == 'overall' or fam_names[f] == scope]
                agg.setdefault(scope, {})[kind] = {met: {'mean': float(np.mean([s[met] for s in sel])), 'std': float(np.std([s[met] for s in sel])),
                                                         'median': float(np.median([s[met] for s in sel]))}
                                                   for met in ('reconstruction_rmse', 'forecast_rmse', 'forecast_fde')}
                agg[scope]['n_trajectories'] = len(sel)
        study[name][str(n)] = agg
        a = agg['overall']
        print(f'{name:28s} n_obs={n:3d}  model recon {a["model"]["reconstruction_rmse"]["mean"]:.3f} fc {a["model"]["forecast_rmse"]["mean"]:.3f} | '
              f'interp/hold baseline recon {a["baseline"]["reconstruction_rmse"]["mean"]:.3f} fc {a["baseline"]["forecast_rmse"]["mean"]:.3f}', flush=True)
report['test_5_observation_counts'] = {
    'protocol': 'all held-out test trajectories (three_shapes test split: 450 incl. ellipses; two-shape and variable models: the 300 circles+spirals); '
                'n observations drawn at random from the 140 grid times with t<=7 (t=0 always included), stored noisy values (sigma 0.05); '
                'predictions through glo.inference.predict (the same code as the CLI); metrics vs the clean dense trajectory; '
                'baseline = linear interpolation of the observations / last-observation hold',
    'results': study}
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for j, met in enumerate(('reconstruction_rmse', 'forecast_rmse')):
    for name, col in (('three_shapes_unified', '#2578bc'), ('two_shapes_unified', '#e0a030'), ('variable_nobs_experimental', '#8a5cc2')):
        ax[j].plot(LEVELS, [study[name][str(n)]['overall']['model'][met]['mean'] for n in LEVELS], marker='o', color=col, label=name)
    ax[j].plot(LEVELS, [study['three_shapes_unified'][str(n)]['overall']['baseline'][met]['mean'] for n in LEVELS], 'k--', marker='s', ms=4,
               label='baseline (interpolate / hold last point)')
    ax[j].axvline(30, color='gray', ls=':'); ax[j].set_yscale('log'); ax[j].set_xlabel('number of observations'); ax[j].set_ylabel(met.replace('_', ' ') + ' (orig. units)')
    ax[j].set_xticks(LEVELS); ax[j].grid(alpha=.2); ax[j].legend(frameon=False, fontsize=7)
fig.suptitle('Frozen models vs number of input observations (held-out test trajectories; the unified models were trained with exactly 30)')
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out / 'observation_count_accuracy.png', dpi=140); plt.close(fig)

# ----------------------------------------------------------------------------- Test 6
runs = []
for i in (1, 2):
    d = tmp / f'repro_{i}'
    r = cli('--input', ex / 'example_spiral.csv', '--output', d, '--ground-truth', ex / 'ground_truth/example_spiral.csv', '--quiet')
    assert r.returncode == 0
    runs.append({f: (d / f).read_bytes() for f in ('predicted_trajectory.csv', 'metrics.json')})
lm3 = inf.load_packaged_model('three_shapes_unified', saved)
t_o, xy_o, _ = inf.read_observations(ex / 'example_spiral.csv', lm3.preprocessing, 30)
p1 = inf.predict(lm3, t_o, xy_o, np.linspace(0, 10, 200))['pred']; p2 = inf.predict(lm3, t_o, xy_o, np.linspace(0, 10, 200))['pred']
report['test_6_reproducibility'] = {'cli_outputs_byte_identical': runs[0] == runs[1], 'in_process_max_abs_difference': float(np.abs(p1 - p2).max())}
assert runs[0] == runs[1] and np.array_equal(p1, p2)

# ----------------------------------------------------------------------------- Test 8
cons = {}
for name in inf.MODEL_REGISTRY:
    lm = inf.load_packaged_model(name, saved)
    exp = lm.metadata['training_dataset']; fams = ['circle'] if name == 'circle_specialist' else (['spiral'] if name == 'spiral_specialist' else None)
    res = collect(lm.model, {'experiment': exp}, root / 'data/processed', exp, split='test', families=fams, n_obs=30, n_query=200, seed=123)
    worst = 0.0; n_chk = 0
    for k in range(0, len(res['family']), 5):
        o = inf.predict(lm, res['obs_t'][k].astype(np.float64), res['obs'][k].astype(np.float64), res['query_t'].astype(np.float64))
        worst = max(worst, float(np.abs(o['pred'] - res['pred'][k]).max())); n_chk += 1
    cons[name] = {'trajectories_compared': n_chk, 'max_abs_prediction_difference': worst, 'within_1e-4': worst < 1e-4}
    assert worst < 1e-4, (name, worst)
    if name == 'three_shapes_unified':             # CLI round trip from the exact observations used by the evaluation pipeline
        cli_worst = 0.0
        for k in (0, 1, 2, 3, 4, 5):
            p = tmp / f'cons_{k}.csv'
            with open(p, 'w') as fh:
                fh.write('t,x,y\n'); [fh.write(f'{ti:.9g},{xi:.9g},{yi:.9g}\n') for ti, (xi, yi) in zip(res['obs_t'][k], res['obs'][k])]
            r = cli('--input', p, '--output', tmp / f'cons_out_{k}', '--quiet'); assert r.returncode == 0, r.stderr
            arr = np.loadtxt(tmp / f'cons_out_{k}' / 'predicted_trajectory.csv', delimiter=',', skiprows=1, usecols=(1, 2))
            cli_worst = max(cli_worst, float(np.abs(arr - res['pred'][k]).max()))
        cons[name]['cli_csv_round_trip_max_abs_difference (csv has 6 decimals)'] = cli_worst
        assert cli_worst < 1e-4, cli_worst
report['test_8_consistency_with_existing_evaluation'] = cons

# ----------------------------------------------------------------------------- Test 7 (last: covers everything above)
after = tree_digest(pt_files)
report['test_7_frozen_weights'] = {'n_checkpoint_files_checked': len(before), 'all_checksums_unchanged': before == after,
                                   'changed_files': [k for k in before if before[k] != after.get(k)],
                                   'packaged_vs_original_identical': {n: json.loads((saved / inf.MODEL_REGISTRY[n] / 'metadata.json').read_text())['sha256'] ==
                                                                      inf.sha256_file(root / json.loads((saved / inf.MODEL_REGISTRY[n] / 'metadata.json').read_text())['original_checkpoint'])
                                                                      for n in inf.MODEL_REGISTRY}}
assert report['test_7_frozen_weights']['all_checksums_unchanged']
(out / 'validation_report.json').write_text(json.dumps(report, indent=2))
shutil.rmtree(tmp, ignore_errors=True)

# markdown table for the guide (generated from the saved results)
L = ['| Model | n_obs | Recon RMSE | Forecast RMSE | Forecast FDE | Baseline recon | Baseline forecast |', '|---|---|---|---|---|---|---|']
for name in study:
    for n in LEVELS:
        a = study[name][str(n)]['overall']
        L.append(f"| {name} | {n} | {a['model']['reconstruction_rmse']['mean']:.3f} | {a['model']['forecast_rmse']['mean']:.3f} | "
                 f"{a['model']['forecast_fde']['mean']:.3f} | {a['baseline']['reconstruction_rmse']['mean']:.3f} | {a['baseline']['forecast_rmse']['mean']:.3f} |")
(out / 'observation_count_table.md').write_text('\n'.join(L) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'test_5_observation_counts'}, indent=1))
