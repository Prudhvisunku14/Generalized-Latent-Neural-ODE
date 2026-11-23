#!/usr/bin/env python3
"""Builds MTP_COMPLETE_PRESENTATION_MATERIALS/ from SAVED results, existing reports and the source code only.
No training, no new experiments; originals are copied, never modified. Run from the project root, after
audit/build_presentation_figures.py and audit/build_mtp_figures.py:   python audit/build_mtp_package.py
"""
import sys, shutil, hashlib, csv, json, textwrap
sys.path.insert(0, 'audit'); sys.path.insert(0, 'src')
from pathlib import Path
from results_values import *          # D, accessors (ab, cap, c5, zs, sv, ss, cond, t3, gl, glf, imp, sel, nobs), V, Z, CELLS, P, sig, ...
from results_common import INDEX
import numpy as np, torch
from glo.model import LatentODE

PK = Path('MTP_COMPLETE_PRESENTATION_MATERIALS')
SUB = ['01_PROJECT_OVERVIEW', '02_EXISTING_RESEARCH', '03_MODEL_ARCHITECTURE', '04_DATASET/DATASET_VISUALIZATIONS', '05_EXPERIMENTAL_METHODOLOGY', '06_EXPERIMENTAL_RESULTS/RESULTS_TABLES',
       '07_FIGURES/ARCHITECTURE', '07_FIGURES/DATASETS', '07_FIGURES/RECONSTRUCTION', '07_FIGURES/FORECASTING', '07_FIGURES/GENERALIZATION', '07_FIGURES/ABLATION', '07_FIGURES/EXISTING_RESEARCH',
       '08_DISCUSSION_AND_FUTURE_WORK', '09_PRESENTATION_CONTENT', '10_VERIFIED_SOURCE_REPORTS', '11_SOURCE_INDEX']
for s in SUB: (PK / s).mkdir(parents=True, exist_ok=True)
COPIES = []          # (destination, source) of every copied file, for the manifest
FIGSRC = {}          # figure destination -> description of how it was produced


def W(rel, text):
    p = PK / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(textwrap.dedent(text).strip('\n') + '\n'); return p


def COPY(src, dst_rel, how=None):
    src = Path(src); dst = PK / dst_rel; dst.parent.mkdir(parents=True, exist_ok=True)
    assert src.exists(), src
    shutil.copy2(src, dst); COPIES.append((dst_rel, str(src)))
    if dst_rel.startswith('07_FIGURES'): FIGSRC[dst_rel] = how or f'copy of {src}'


def CSV(rel, header, rows):
    p = PK / rel; p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', newline='') as f:
        w_ = csv.writer(f); w_.writerow(header); w_.writerows(rows)


def MD(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|'] + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


f4 = lambda x: f'{x:.4f}'; f3 = lambda x: f'{x:.3f}'
FL = {'circle': 'circle', 'spiral': 'spiral', 'ellipse': 'ellipse', 'figure_eight': 'figure-eight', 'sinusoid': 'sinusoid', 'overall': 'all five'}
G6 = FAMS + ['overall']
R = lambda m, g, k, s='mean', f='.4f': ab(f'results.{m}.{g}.{k}.{s}', f)
rel = lambda m, g, k: AB['results'][m][g][k]['mean'] / AB['results']['A'][g][k]['mean'] - 1
LI = lambda g: ab(f'baselines.linear_interpolation_recon.{g}.mean', '.4f'); HLd = lambda g: ab(f'baselines.hold_last_forecast.{g}.mean', '.4f'); CV = lambda g: ab(f'baselines.const_velocity_forecast.{g}.mean', '.4f')
Z3 = lambda g, k: zs(f'three_shape_model.{g}.{k}.mean', '.4f'); F5 = lambda g, k: c5(f'five_shape_model.{g}.{k}.mean', '.4f')
SVR = lambda fam, k: V['svt'][(fam, k)]
ssf = SS['across_seeds']['overall']['forecast_rmse']
cP = P('C', 'overall', 'forecast_rmse'); bP = P('B', 'overall', 'forecast_rmse'); dP = P('D', 'overall', 'forecast_rmse')
fI = AB['factorial']['overall/forecast_rmse/interaction ((D-C) - (B-A))']
pairflag = lambda a, g, k: ('p=' + pv(P(a, g, k)['wilcoxon_p_holm'])) if sig(a, g, k) else 'ns'
cellc = lambda a, g, k: f"{R(a, g, k)} ({pct(rel(a, g, k))}; {pairflag(a, g, k)})"

# ---- parameter counts (instantiating the model class; no training, nothing saved)
def counts(**kw):
    m = LatentODE(solver='rk4', step_size=.1, **kw)
    c = {'GRU encoder': sum(p.numel() for p in m.encoder.gru.parameters()), 'posterior layer': sum(p.numel() for p in m.encoder.posterior.parameters()),
         'ODE vector field': sum(p.numel() for p in m.dynamics.parameters()), 'decoder': sum(p.numel() for p in m.decoder.parameters())}
    c['total'] = sum(c.values()); assert c['total'] == sum(p.numel() for p in m.parameters()); return c
BASEKW = dict(dynamics_hidden=64, decoder_hidden=64, dynamics_layers=2, decoder_layers=2)
PC = {'original-size': counts(latent_dim=4, encoder_hidden=25, dynamics_hidden=20, decoder_hidden=20, dynamics_layers=1, decoder_layers=1),
      'A': counts(latent_dim=8, encoder_hidden=64, **BASEKW), 'B': counts(latent_dim=16, encoder_hidden=64, **BASEKW), 'C': counts(latent_dim=8, encoder_hidden=128, **BASEKW),
      'D': counts(latent_dim=16, encoder_hidden=128, **BASEKW), 'large': counts(latent_dim=32, encoder_hidden=128, **BASEKW)}
for m in CELLS: assert PC[m]['total'] == AB['models'][m]['n_parameters'], m
assert PC['large']['total'] == CAP['models']['large']['n_parameters']

for part in ('part1_overview_existing', 'part2_arch_dataset_method', 'part3_results', 'part4_discussion_slides', 'part5_copy_index'):
    exec(compile(open(f'audit/mtp/{part}.py').read(), f'audit/mtp/{part}.py', 'exec'))
print('package built:', PK, '| files:', sum(1 for _ in PK.rglob('*') if _.is_file()))
