#!/usr/bin/env python3
"""Reconstruct and forecast a trajectory from sparse, noisy, time-stamped coordinates with a FROZEN pretrained
Latent Neural ODE. No training, no shape label.

Examples
--------
  python scripts/predict_trajectory.py --input input_trajectories/example_circle.csv
  python scripts/predict_trajectory.py --model three_shapes_unified \\
      --input input_trajectories/example_spiral.csv --n-query 200 --forecast-end 10 \\
      --output inference_outputs/spiral_example --ground-truth input_trajectories/ground_truth/example_spiral.csv
  python scripts/predict_trajectory.py --input input_trajectories/ --output inference_outputs/batch   # every *.csv
  python scripts/predict_trajectory.py --list-models
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from glo import inference as inf

root = Path(__file__).resolve().parents[1]
CHOICES = list(inf.MODEL_REGISTRY)

p = argparse.ArgumentParser(
    description='Reconstruct (0 <= t <= 7) and forecast (7 < t <= 10) a 2-D trajectory from sparse noisy observations using a '
                'frozen pretrained Latent Neural ODE. No retraining; no shape label is needed or used.',
    epilog='CSV format: header row "t,x,y", one observation per row, t in [0, 7] (the context interval), same coordinate units as '
           'training. Extra columns are ignored. See docs/PRETRAINED_INFERENCE_GUIDE.md.',
    formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--model', default=inf.DEFAULT_MODEL, choices=CHOICES, metavar='NAME',
               help=f"Pretrained model to use. One of: {', '.join(CHOICES)}. Default: {inf.DEFAULT_MODEL}. A model that is not "
                    f"packaged is reported as unavailable and is never silently replaced by another one.")
p.add_argument('--input', nargs='+', help='Observation CSV file(s) with columns t,x,y, or a directory (all *.csv inside are processed '
                                          '= batch mode). Each file is processed independently.')
p.add_argument('--output', help='Output directory. Single file: results go directly here (default inference_outputs/<csv name>). '
                                'Several files / a directory: one sub-folder per CSV is created inside it.')
p.add_argument('--n-query', type=int, default=200, help='Number of evenly spaced output timestamps over [0, forecast-end] (default 200).')
p.add_argument('--forecast-end', type=float, default=10.0,
               help='Last output time. The model was trained and validated on [0, 10]; larger values extrapolate beyond the training '
                    'horizon and trigger a warning (default 10).')
p.add_argument('--ground-truth', help='OPTIONAL dense ground-truth CSV (t,x,y) for ONE input file. Used only AFTER prediction to compute '
                                      'metrics.json and to draw the green curve; never given to the model.')
p.add_argument('--ground-truth-dir', help='OPTIONAL batch counterpart of --ground-truth: a directory containing <input name>.csv ground '
                                          'truth files matching the input file names.')
p.add_argument('--saved-models', default=str(root / 'saved_models'), help='Folder with the packaged pretrained models (default: saved_models/).')
p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Compute device (default cpu; the model is tiny).')
p.add_argument('--threads', type=int, default=1, help='CPU threads (1 gives bit-reproducible results; default 1).')
p.add_argument('--list-models', action='store_true', help='List the packaged pretrained models and exit.')
p.add_argument('--no-plots', action='store_true', help='Skip figure generation.')
p.add_argument('--quiet', action='store_true', help='Print only errors and the output location.')
args = p.parse_args()

saved_root = Path(args.saved_models)
if args.list_models:
    print(f'{"model":28s} available  description')
    for name, ok in inf.available_models(saved_root).items():
        desc = ''
        if ok:
            desc = json.loads((saved_root / inf.MODEL_REGISTRY[name] / 'metadata.json').read_text()).get('description', '')
        print(f'{name:28s} {"yes" if ok else "NO ":9s}  {desc}')
    sys.exit(0)
if not args.input:
    p.error('--input is required (or use --list-models)')
if args.device == 'cuda' and not torch.cuda.is_available():
    p.error('--device cuda requested but CUDA is not available')
torch.set_num_threads(args.threads)


def log(*m):
    if not args.quiet:
        print(*m)


def fail(msg, code=2):
    print(f'ERROR: {msg}', file=sys.stderr)
    sys.exit(code)


# ------------------------------------------------------------------ inputs
inputs: list[Path] = []
for item in args.input:
    ip = Path(item)
    if ip.is_dir():
        found = sorted(ip.glob('*.csv'))
        if not found:
            fail(f'no *.csv files in directory {ip}')
        inputs += found
    elif ip.is_file():
        inputs.append(ip)
    else:
        fail(f'input not found: {ip}')
batch = len(inputs) > 1 or Path(args.input[0]).is_dir()
if args.ground_truth and batch:
    fail('--ground-truth is for a single input file; use --ground-truth-dir in batch mode')

try:
    lm = inf.load_packaged_model(args.model, saved_root, device=args.device)
except inf.ModelUnavailableError as e:
    fail(str(e), 3)
pre = lm.preprocessing
log(f"Model: {args.model} ({lm.metadata['description']})")
log(f"  weights: {lm.checkpoint_path}  sha256={lm.sha256[:16]}…  params={lm.n_parameters}  epoch={lm.load_report['epoch']}")
if 'warning' in lm.metadata:
    log(f"  NOTE: {lm.metadata['warning']}")
try:
    out_times, q_warn = inf.make_query_times(args.n_query, args.forecast_end, pre)
except inf.InputError as e:
    fail(str(e))
trained_n = pre['training_n_obs'] if isinstance(pre['training_n_obs'], int) else None

failures = 0
for path in inputs:
    if len(inputs) == 1 and not batch:
        out_dir = Path(args.output) if args.output else root / 'inference_outputs' / path.stem
    else:
        out_dir = (Path(args.output) if args.output else root / 'inference_outputs' / 'batch') / path.stem
    log(f'\n== {path}')
    try:
        t_obs, xy_obs, warns = inf.read_observations(path, pre, trained_n)
        warns = q_warn + warns
        gt = None
        gt_path = Path(args.ground_truth) if args.ground_truth else (
            Path(args.ground_truth_dir) / path.name if args.ground_truth_dir and (Path(args.ground_truth_dir) / path.name).exists() else None)
        if gt_path is not None:
            gt = inf.read_ground_truth(gt_path)
        # integrate once on the union of output times and ground-truth times (ground truth is NOT an input to the model)
        times_all = out_times if gt is None else np.unique(np.concatenate([out_times, gt[0]]))
        res = inf.predict(lm, t_obs, xy_obs, times_all, device=args.device)
        pred_all = res['pred']
        if gt is None:
            pred_out = pred_all
        else:
            t32 = times_all.astype(np.float32)
            pred_out = pred_all[np.searchsorted(t32, out_times.astype(np.float32))]
        out_dir.mkdir(parents=True, exist_ok=True)
        region = inf.region_labels(out_times, pre['context_end'])
        with open(out_dir / 'predicted_trajectory.csv', 'w') as fh:
            fh.write('t,predicted_x,predicted_y,region\n')
            for ti, (px, py), r in zip(out_times, pred_out, region):
                fh.write(f'{ti:.6f},{px:.6f},{py:.6f},{r}\n')
        summary = {'input_file': str(path), 'model': args.model, 'checkpoint': str(lm.checkpoint_path),
                   'checkpoint_sha256': lm.sha256, 'weights_digest_after_inference': res['weights_digest'],
                   'n_observations': int(len(t_obs)), 'observation_time_range': [float(t_obs[0]), float(t_obs[-1])],
                   'n_query': int(len(out_times)), 'forecast_end': float(args.forecast_end), 'context_end': pre['context_end'],
                   'n_reconstruction_points': int((region == 'reconstruction').sum()), 'n_forecast_points': int((region == 'forecast').sum()),
                   'latent_posterior_mean_z0': [float(v) for v in res['mu']],
                   'dynamics_code_c': None if res['cond'] is None else [float(v) for v in res['cond']],
                   'ground_truth_used_for_metrics_only': str(gt_path) if gt_path else None,
                   'warnings': warns, 'shape_label_used': False}
        if gt is not None:
            pred_at_gt = pred_all[np.searchsorted(times_all.astype(np.float32), gt[0].astype(np.float32))]
            metrics = inf.evaluate_against_truth(lm, pred_at_gt, gt[0], gt[1], t_obs)
            metrics['model'] = args.model
            (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
            summary['metrics_file'] = 'metrics.json'
        (out_dir / 'run_summary.json').write_text(json.dumps(summary, indent=2))
        if not args.no_plots:
            inf.save_figures(out_dir, t_obs, xy_obs, out_times, pred_out, pre['context_end'],
                             title=f'{path.stem} | {args.model}', gt=gt)
        for w in warns:
            print(f'  WARNING: {w}', file=sys.stderr)
        log(f'  {len(t_obs)} observations -> {len(out_times)} output points '
            f"({summary['n_reconstruction_points']} reconstruction, {summary['n_forecast_points']} forecast)")
        if gt is not None:
            m = metrics
            log('  metrics vs ground truth: ' + ', '.join(f'{k}={m[k]:.4f}' for k in ('reconstruction_rmse', 'forecast_rmse', 'forecast_mae',
                                                                                     'forecast_ade', 'forecast_fde', 'chamfer_distance') if k in m))
        log(f'  saved to {out_dir}')
    except inf.InputError as e:
        failures += 1
        print(f'  ERROR: {e}', file=sys.stderr)
if failures:
    fail(f'{failures} of {len(inputs)} input file(s) failed', 1)
