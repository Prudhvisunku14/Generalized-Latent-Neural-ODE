"""All numbers used by the results-analysis documents, computed from SAVED results only (no model is run).

Importing this module gives a namespace V (plain dict) plus guards that assert every qualitative statement the documents make.
"""
import sys
sys.path.insert(0, 'audit')
import numpy as np, pandas as pd
from scipy import stats
from results_common import *

V = {}
AB, C5, ZS, SV, SS, COND, T3, GL, CAP, IMP, NOBS = (D[k] for k in ('ab', 'c5', 'zs', 'sv', 'ss', 'cond', 't3', 'gl', 'cap', 'imp', 'nobs'))
Z = np.load('results/ablation_2x2/predictions_test.npz')
CELLS = 'ABCD'; SEVEN = ['orig_baseline', 'A', 'B', 'C', 'D', 'orig_medium', 'orig_large']

# ------------------------------------------------------------------ five-family rankings (7 five-family models, one run each)
rk = {}
for m in SEVEN:
    r = {g: AB['results'][m][g]['recon_rmse']['mean'] for g in FAMS}; f = {g: AB['results'][m][g]['forecast_rmse']['mean'] for g in FAMS}
    rk[m] = {'recon_easiest': min(r, key=r.get), 'recon_hardest': max(r, key=r.get), 'fc_easiest': min(f, key=f.get), 'fc_hardest': max(f, key=f.get)}
V['rank'] = rk
count = lambda key, fam: sum(1 for m in SEVEN if rk[m][key] == fam)
V['n_recon_easiest_sinusoid'] = count('recon_easiest', 'sinusoid'); V['n_recon_hardest_fig8'] = count('recon_hardest', 'figure_eight')
V['n_fc_hardest_sinusoid'] = count('fc_hardest', 'sinusoid'); V['n_fc_hardest_spiral'] = count('fc_hardest', 'spiral')
assert (V['n_recon_easiest_sinusoid'], V['n_recon_hardest_fig8'], V['n_fc_hardest_sinusoid'], V['n_fc_hardest_spiral']) == (6, 6, 5, 2), V
assert [m for m in SEVEN if rk[m]['recon_easiest'] != 'sinusoid'] == ['A'] and [m for m in SEVEN if rk[m]['recon_hardest'] != 'figure_eight'] == ['orig_baseline']
assert [m for m in SEVEN if rk[m]['fc_hardest'] == 'spiral'] == ['orig_baseline', 'A']
A_c, A_s = AB['results']['A']['circle']['recon_rmse']['mean'], AB['results']['A']['sinusoid']['recon_rmse']['mean']
assert abs(A_c - A_s) < 1e-3                                                     # cell A: circle and sinusoid essentially tied

# ------------------------------------------------------------------ shared vs specialist
SVT = {}
for fam in ('circle', 'spiral'):
    for k in ('recon_rmse', 'forecast_rmse'):
        s = SV[fam]['seed_level'][k]; u, sp = s['unified'], s['specialist']
        SVT[(fam, k)] = {'u_mean': u['mean'], 'u_sd': u['std'], 'u_n': u['n_runs'], 's_mean': sp['mean'], 's_sd': sp['std'], 's_n': sp['n_runs'], 'rel': u['mean'] / sp['mean'] - 1,
                         'welch': s['welch_t_p'], 'welch_bonf': min(1., 4 * s['welch_t_p']), 'all_worse': min(u['values']) > max(sp['values']),
                         'p_rel': SV[fam]['paired_seed42'][k]['relative_diff_pct'] / 100, 'p_wil': SV[fam]['paired_seed42'][k]['wilcoxon_p'], 'p_frac': SV[fam]['paired_seed42'][k]['fraction_traj_unified_better']}
V['svt'] = SVT
assert SVT[('circle', 'forecast_rmse')]['all_worse'] and SVT[('spiral', 'forecast_rmse')]['all_worse'], 'every shared run forecasts worse than every specialist run'
assert SVT[('circle', 'forecast_rmse')]['welch_bonf'] < .05 and SVT[('spiral', 'forecast_rmse')]['welch_bonf'] > .05 and SVT[('spiral', 'recon_rmse')]['welch'] > .05

# ------------------------------------------------------------------ cross-family (two-shape models on ellipses; three-shape on figure-eight / sinusoid)
OOD = T3['C_out_of_distribution_ellipses']; V['ood'] = OOD
assert OOD['two_shape_models_on_unseen_ellipses']['forecast_rmse']['n_runs'] == 5
V['ell_fc_two'] = OOD['two_shape_models_on_unseen_ellipses']['forecast_rmse']; V['ell_rc_two'] = OOD['two_shape_models_on_unseen_ellipses']['recon_rmse']
assert V['ell_fc_two']['mean'] < OOD['const_pos_on_same_ellipses']['forecast_rmse']['mean'] and V['ell_rc_two']['mean'] > 3 * OOD['const_pos_on_same_ellipses']['recon_rmse']['mean']

# ------------------------------------------------------------------ ablation: windows and crossing times
qt, truth, fam = Z['query_t'], Z['truth'], Z['family']; SEL = {f: fam == i for i, f in enumerate(FAMS)}; SEL['overall'] = np.ones(len(fam), bool)
def window(m, lo, hi, sel):
    s = (qt > lo) & (qt <= hi); d2 = ((Z['pred_' + m] - truth) ** 2).sum(-1)[:, s]; return np.sqrt(d2.sum(1) / (2 * s.sum()))[sel]
WIN = [(7, 8), (8, 9), (9, 10)]; V['win'] = {(m, w): window(m, *w, SEL['overall']).mean() for m in CELLS for w in WIN}
et = AB['error_vs_time']; qte = np.array(et['query_t']); fut = qte > 7
def crossing(m):
    d = np.array(et[m]['overall']) - np.array(et['A']['overall']); return float(qte[fut][np.argmax(d[fut] > 0)])
V['cross'] = {m: crossing(m) for m in 'BCD'}
assert all(V['win'][(m, (7, 8))] < V['win']['A', (7, 8)] for m in 'BCD') and all(V['win'][(m, (9, 10))] > V['win']['A', (9, 10)] for m in 'BCD')
assert all(7.5 < V['cross'][m] < 9 for m in 'BCD')
P = lambda a, g, k: AB['paired'][f'{a}_minus_A/{g}/{k}']
sig = lambda a, g, k: P(a, g, k)['wilcoxon_p_holm'] < .05
assert [g for g in FAMS if sig('B', g, 'forecast_rmse')] == ['circle', 'sinusoid'] and [g for g in FAMS if sig('C', g, 'forecast_rmse')] == ['ellipse', 'sinusoid'] and not sig('C', 'overall', 'forecast_rmse')
assert all(AB['results']['A'][g]['forecast_rmse']['mean'] < AB['results']['D'][g]['forecast_rmse']['mean'] for g in FAMS)
HL = {m: [g for g in FAMS if AB['results'][m][g]['forecast_rmse']['mean'] > AB['baselines']['hold_last_forecast'][g]['mean']] for m in CELLS}
assert HL == {'A': [], 'B': ['sinusoid'], 'C': [], 'D': ['sinusoid']}

# ------------------------------------------------------------------ GeoLife: paired fine-tuned vs scratch (computed here from the existing per-segment output)
gd = pd.read_csv('results/geolife/main_evaluation/geolife_trained_per_segment.csv'); PA = {}
cols = ['interp_geolife_scratch', 'interp_geolife_finetuned_from_three_shapes', 'interp_linear', 'interp_pchip']
assert gd[cols].notna().all().all() and len(gd) == 5000
rng = np.random.default_rng(0)
for n in (10, 20, 30, 40, 50):
    s = gd[gd.n_obs == n]; a, b = s.interp_geolife_scratch.values, s.interp_geolife_finetuned_from_three_shapes.values; lr = np.log(b / a)
    g = pd.DataFrame({'u': s.user.values, 'lr': lr}).groupby('u').lr.agg(['sum', 'count']); su, cu = g['sum'].values, g['count'].values
    idx = rng.integers(0, len(su), (5000, len(su))); boot = np.exp(su[idx].sum(1) / cu[idx].sum(1))
    PA[n] = {'n': len(s), 'users': len(su), 'med_scratch': float(np.median(a)), 'med_fine': float(np.median(b)), 'gmr': float(np.exp(lr.mean())), 'ci': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
             'win_fine': float((b < a).mean()), 'wil_p': float(stats.wilcoxon(a, b).pvalue)}
V['pa'] = PA
assert len(gd[gd.n_obs == 30]) == 1000 and PA[30]['gmr'] < 1 and all(PA[n]['gmr'] > 1 for n in (10, 20, 40, 50))
V['pa_ci_excl1'] = {n: not (PA[n]['ci'][0] <= 1 <= PA[n]['ci'][1]) for n in PA}

import math
V['steps3'] = math.ceil(int(np.load('data/processed/three_shapes/train.npz')['family'].shape[0]) / 32) * 300
assert V['steps3'] == 19800 and AB['models']['A']['optimizer_steps'] == 33000
V['nll_gap'] = max(abs(AB['models'][m]['selected_train_nll'] - AB['models'][m]['selected_val_nll']) for m in CELLS)
V['cap_med_vs_base'] = AB['results']['orig_medium']['overall']['forecast_rmse']['mean'] / AB['results']['orig_baseline']['overall']['forecast_rmse']['mean'] - 1
V['cap_large_vs_base'] = AB['results']['orig_large']['overall']['forecast_rmse']['mean'] / AB['results']['orig_baseline']['overall']['forecast_rmse']['mean'] - 1
LIN = AB['baselines']['linear_interpolation_recon']
assert all(AB['results'][m][g]['recon_rmse']['mean'] < LIN[g]['mean'] for m in 'BD' for g in FAMS), 'B and D beat linear interpolation everywhere'
assert [g for g in FAMS if AB['results']['A'][g]['recon_rmse']['mean'] < LIN[g]['mean']] == ['circle'], 'A beats linear interpolation only on circle'
assert [g for g in FAMS if AB['results']['C'][g]['recon_rmse']['mean'] < LIN[g]['mean']] == ['circle', 'figure_eight', 'sinusoid']
V['d_vs_a'] = AB['results']['D']['overall']['forecast_rmse']['mean'] / AB['results']['A']['overall']['forecast_rmse']['mean'] - 1
