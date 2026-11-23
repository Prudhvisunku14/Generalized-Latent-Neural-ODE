"""Shared loaders / helpers for the results-analysis build (reads SAVED results only; nothing is trained or modified).

Every accessor created with mk() records (claim, value, source file :: key path) so that RESULTS_SOURCE_INDEX.md can be generated
from exactly the values used in the documents.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path('.')
FAMS = ['circle', 'spiral', 'ellipse', 'figure_eight', 'sinusoid']
J = lambda p: json.load(open(p))
FILES = {
    'ab': 'results/ablation_2x2/ablation_2x2_results.json', 'cap': 'results/capacity_scaling/capacity_scaling_results.json',
    'c5': 'results/five_shapes/compare_three_vs_five.json', 'zs': 'results/five_shapes/zeroshot_three_shape_model.json',
    'sv': 'results/validation/shared_vs_separate/shared_vs_separate.json', 'ss': 'results/validation/seed_study/seed_study.json',
    'cond': 'results/conditioning/conditioning_study.json', 't3': 'results/validation/three_shapes/three_shapes_and_ood.json',
    'gl': 'results/geolife/main_evaluation/geolife_trained_summary.json', 'glf': 'results/geolife/main_evaluation/summary.json',
    'imp': 'results/improved_two_shapes/baseline_comparison.json', 'sel': 'results/ablation_2x2/checkpoint_selection_audit.json',
    'nobs': 'results/validation/nobs_ablation/nobs_ablation.json',
}
D = {k: J(v) for k, v in FILES.items()}
INDEX = []


def mk(key):
    data, file = D[key], FILES[key]

    def get(path, f=None, claim=None):
        v = data
        for p in path.split('.'):
            v = v[int(p)] if isinstance(v, list) else v[p]
        INDEX.append((claim or path, v, f'{file} :: {path}'))
        return v if f is None else format(v, f)
    return get


ab, cap, c5, zs, sv, ss, cond, t3, gl, glf, imp, sel, nobs = (mk(k) for k in FILES)
pct = lambda x, d=1: f'{x * 100:+.{d}f}%'
pv = lambda p: f'{p:.2g}'

# ---------------------------------------------------------------- presentation style (validated light-mode categorical slots 1-5)
BG, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e1'
SER = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
BASE = '#0b0b0b'


def style(ax, grid='y'):
    ax.set_facecolor(BG)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#b9b8b2')
    ax.tick_params(colors=INK2, labelsize=11)
    ax.grid(axis=grid, color=GRID, lw=.8); ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2); ax.yaxis.label.set_color(INK2); ax.title.set_color(INK)


def new_fig(w, h, **kw):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 12, 'figure.facecolor': BG, 'savefig.facecolor': BG})
    return plt.subplots(figsize=(w, h), **kw)


def fname(f):
    return f.replace('_', '-')
