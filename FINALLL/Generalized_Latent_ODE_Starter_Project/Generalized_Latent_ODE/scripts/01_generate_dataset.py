#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from glo.data import generate_project_data, validate_dataset
from glo.visualization import plot_gallery

p = argparse.ArgumentParser(description='Generate reproducible circle/spiral (+ellipse) datasets')
p.add_argument('--n-per-family', type=int, default=1000)
p.add_argument('--n-points', type=int, default=200)
p.add_argument('--n-obs', type=int, default=30)
p.add_argument('--noise-std', type=float, default=.05)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--out', default='data/processed')
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
out = root / args.out
summaries = generate_project_data(out, n_per_family=args.n_per_family,
    n_points=args.n_points, n_obs=args.n_obs, noise_std=args.noise_std, seed=args.seed)
for name, meta in summaries.items():
    counts = validate_dataset(out, name)
    fig = plot_gallery(out, name, root / 'results' / 'figures' / f'{name}_gallery.png')
    print(f'{name}: families={meta["families"]}, split={counts}; preview={fig}')
