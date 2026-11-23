#!/usr/bin/env python3
"""Writes the corrected results/ablation_2x2/ABLATION_2X2_REPORT.md from SAVED results only (no model is run).

Every number in the report is read from: results/ablation_2x2/ablation_2x2_results.json, predictions_test.npz,
checkpoint_selection_audit.json, the per-epoch history CSVs, and results/validation/seed_study/seed_study.json.
Run from the project root. Refuses to run unless the uncorrected original has been backed up.
"""
import json
from pathlib import Path
import numpy as np

R = Path('results/ablation_2x2')
assert (R / 'ABLATION_2X2_REPORT.original_uncorrected.md').exists(), 'back up the original report first'
J = json.load(open(R / 'ablation_2x2_results.json')); AU = json.load(open(R / 'checkpoint_selection_audit.json'))['runs']
SEED = json.load(open('results/validation/seed_study/seed_study.json'))['across_seeds']['overall']['forecast_rmse']
Z = np.load(R / 'predictions_test.npz')
CELLS = 'ABCD'; FAMS = ['circle', 'spiral', 'ellipse', 'figure_eight', 'sinusoid']; G = FAMS + ['overall']
LAB = {'A': 'A (latent 8, GRU 64)', 'B': 'B (latent 16, GRU 64)', 'C': 'C (latent 8, GRU 128)', 'D': 'D (latent 16, GRU 128)'}
res = lambda m, g, k, s='mean': J['results'][m][g][k][s]
P = lambda m, g, k: J['paired'][f'{m}_minus_A/{g}/{k}']
pv = lambda p: f'{p:.2g}'
pct = lambda x: f'{x * 100:+.1f}%'
fr = lambda v: [x for k, x in v.items() if k.startswith('fraction_improved')][0]
fac = lambda g, k, c: J['factorial'][f'{g}/{k}/{c}']
RC = {m: J['relative_change_vs_A_of_mean'][m]['overall'] for m in 'BCD'}
RMIN = pct(max(RC[m]['recon_rmse'] for m in 'BCD')); RMAX = pct(min(RC[m]['recon_rmse'] for m in 'BCD'))
FB, FC, FD = pct(RC['B']['forecast_rmse']), pct(RC['C']['forecast_rmse']), pct(RC['D']['forecast_rmse'])
LATENT, GRU, INTER = ('latent_main_effect (16 vs 8, averaged over GRU)', 'gru_main_effect (128 vs 64, averaged over latent)', 'interaction ((D-C) - (B-A))')

# ---- horizon windows (derived from the saved predictions; mean over trajectories of per-trajectory RMSE inside the window)
qt, truth, fam = Z['query_t'], Z['truth'], Z['family']
WIN = [(7, 8), (8, 9), (9, 10)]
def window(m, lo, hi, sel):
    s = (qt > lo) & (qt <= hi); d2 = ((Z['pred_' + m] - truth) ** 2).sum(-1)[:, s]
    return np.sqrt(d2.sum(1) / (2 * s.sum()))[sel]
SEL = {f: fam == i for i, f in enumerate(FAMS)}; SEL['overall'] = np.ones(len(fam), bool)
W = {(m, w, g): window(m, *w, SEL[g]) for m in CELLS for w in WIN for g in G}
et = J['error_vs_time']; qte = np.array(et['query_t']); fut = qte > 7
def crossing(m):
    d = np.array(et[m]['overall']) - np.array(et['A']['overall'])
    first = float(qte[fut][np.argmax(d[fut] > 0)]); stay = None
    for i in np.flatnonzero(fut):
        if (d[i:] > 0).all(): stay = float(qte[i]); break
    return first, stay
CROSS = {m: crossing(m) for m in 'BCD'}

# ---- training trend (validation -ELBO, last 50 epochs)
def trend(m):
    h = np.genfromtxt(f"checkpoints/ablation_2x2/{J['models'][m]['checkpoint'].split('/')[-1][:-3]}_history.csv", delimiter=',', names=True)
    return float(h['val_elbo_loss'][250:275].mean()), float(h['val_elbo_loss'][275:300].mean())
TR = {m: trend(m) for m in CELLS}
au = {m: AU[f"cell {m} ({J['models'][m]['latent_dim']}/{J['models'][m]['encoder_hidden']})"] for m in CELLS}
rep = J['reproducibility_pairs']; rA = rep['A_vs_orig_baseline']['overall']['forecast_rmse']; rD = rep['D_vs_orig_medium']['overall']['forecast_rmse']
sv = SEED['values']; Mi = J['models']
bl = J['baselines']
# ---- guards: every qualitative claim in the prose is asserted against the data
sig = lambda m, g, k: P(m, g, k)['wilcoxon_p_holm'] < .05
assert all(sig(m, g, 'recon_rmse') and res(m, g, 'recon_rmse') < res('A', g, 'recon_rmse') for m in 'BCD' for g in G), 'recon claim'
assert [g for g in FAMS if sig('B', g, 'forecast_rmse')] == ['circle', 'sinusoid'], 'B significance claim'
assert not sig('C', 'overall', 'forecast_rmse') and [g for g in FAMS if sig('C', g, 'forecast_rmse')] == ['ellipse', 'sinusoid'], 'C claim'
assert P('C', 'ellipse', 'forecast_rmse')['mean_difference'] < 0 < P('C', 'sinusoid', 'forecast_rmse')['mean_difference'], 'C sign claim'
assert all(res('D', g, 'forecast_rmse') > res('A', g, 'forecast_rmse') for g in FAMS) and [g for g in FAMS if not sig('D', g, 'forecast_rmse')] == ['figure_eight'], 'D claim'
assert fac('overall', 'forecast_rmse', INTER)['mean'] > 0 > fac('sinusoid', 'forecast_rmse', INTER)['mean'], 'interaction sign claim'
assert abs(P('C', 'overall', 'forecast_rmse')['mean_difference']) < abs(J['paired']['D_minus_B/overall/forecast_rmse']['mean_difference']), 'GRU main effect carried by D vs B'
assert all(W[(m, (7, 8), 'overall')].mean() < W[('A', (7, 8), 'overall')].mean() for m in 'BCD'), 'early window claim'
assert all(W[(m, (9, 10), 'overall')].mean() > W[('A', (9, 10), 'overall')].mean() for m in 'BCD'), 'late window claim'
assert all(CROSS[m][0] > 7.5 for m in 'BCD') and all(CROSS[m][0] == CROSS[m][1] for m in 'BCD'), 'crossing claim'
assert all(max(FAMS, key=lambda g: P(m, g, 'forecast_rmse')['mean_difference']) == 'sinusoid' for m in 'BCD'), 'sinusoid dominates claim'
assert all(not sig(m, 'figure_eight', 'forecast_rmse') for m in 'BCD'), 'figure-eight claim'
assert all(au[m]['selected_epoch'] >= 294 for m in CELLS) and all(TR[m][1] < TR[m][0] for m in CELLS), 'still improving claim'
assert min(au['B']['min_val_forecast_rmse'], au['D']['min_val_forecast_rmse']) > au['A']['min_val_forecast_rmse'] * 1.1, 'selection claim'
assert max(abs(au[m]['train_nll_at_selected'] - au[m]['val_nll_at_selected']) for m in CELLS) < 10, 'train/val NLL similarity claim'
assert (np.array(J['pooled_rmse']['D']['overall']['forecast_rmse']) > J['pooled_rmse']['B']['overall']['forecast_rmse'] > J['pooled_rmse']['C']['overall']['forecast_rmse'] > J['pooled_rmse']['A']['overall']['forecast_rmse']), 'test order claim'
GAPMAX = max(abs(au[m]['train_nll_at_selected'] - au[m]['val_nll_at_selected']) for m in CELLS)

HL = {m: [g for g in FAMS if res(m, g, 'forecast_rmse') > bl['hold_last_forecast'][g]['mean']] for m in CELLS}
assert HL == {'A': [], 'B': ['sinusoid'], 'C': [], 'D': ['sinusoid']}, HL
assert all(res(m, 'overall', 'forecast_rmse') < min(bl['hold_last_forecast']['overall']['mean'], bl['const_velocity_forecast']['overall']['mean']) for m in CELLS)
HLTXT = f"**On the sinusoid family, however, B ({res('B', 'sinusoid', 'forecast_rmse'):.3f}) and D ({res('D', 'sinusoid', 'forecast_rmse'):.3f}) are worse than the hold-last baseline ({bl['hold_last_forecast']['sinusoid']['mean']:.3f}).**"
L = []; w = L.append

w('# Controlled 2x2 ablation: latent dimension x GRU encoder width (five-family dataset)')
w('')
w('> **Status: corrected version (2026-11-21).** This file replaces an earlier version that contained wrong reconstruction numbers and several unsupported')
w('> interpretations. The uncorrected original is preserved byte-for-byte as `ABLATION_2X2_REPORT.original_uncorrected.md` (read-only). The list of')
w('> corrections is in Appendix A. Every number below is read from `ablation_2x2_results.json`, `predictions_test.npz`, `checkpoint_selection_audit.json`,')
w('> the per-epoch history files and `results/validation/seed_study/seed_study.json` by `audit/build_corrected_ablation_report.py`; the table values were then')
w('> re-checked against an independent recomputation (`audit/verify_5_corrected_documents.py`).')
w('')
w('## 1. What this report is, and is not')
w('')
w('- **The project is an adapted re-implementation and experimental extension of Latent Neural ODEs** (Chen et al., 2018): same model family, different data, solver settings and objective details. It is **not an exact replication** of the paper, and the paper\'s numbers were not reproduced.')
w('- **It has not produced an improved forecasting model.** No method to improve forecasting has been developed or tested. This ablation studies how model size affects an existing model.')
w('- **Every result here comes from ONE training run per cell (seed 42). Nothing in this report has been confirmed across training seeds.** The only multi-seed evidence in the project (5 seeds, two-shape data, a different experiment) shows that the same architecture varies a lot between seeds (section 8).')
w('- All findings below are therefore **exploratory, single-seed observations**. Statements about causes are hypotheses, not conclusions.')
w('')
w('## 2. Design and what was verified')
w('')
w('Question: how do latent dimension and GRU encoder width, separately and together, affect reconstruction and forecasting of the same model on the five-family data?')
w('')
w('| Cell | Latent | GRU | Parameters | Selected epoch | Validation -ELBO | Training steps |')
w('|---|---:|---:|---:|---:|---:|---:|')
for m in CELLS:
    i = Mi[m]; w(f"| {LAB[m]} | {i['latent_dim']} | {i['encoder_hidden']} | {i['n_parameters']:,} | {i['selected_epoch']} | {i['selected_val_neg_elbo']:.2f} | {i['optimizer_steps']:,} |")
w('')
w('**Verified controls** (see `PROJECT_COMPLETION_AUDIT.md`): all four runs completed 300 epochs with no early stopping and no non-finite values; training arguments differ only in latent size and GRU width; seed 42 and `data_seed` 42 for every cell (identical batch order and identical training observation subsets, so these no longer differ between architectures); 110 optimiser steps per epoch, 33,000 in total; identical validation observation draws; identical test observations (750 trajectories, 150 per family) in a fresh evaluation; checkpoints chosen by validation -ELBO only, never by test results.')
w('')
w('**Not controlled:** initial weights and the reparameterisation noise differ between architectures (the networks have different shapes); there is one training run per cell; the ODE field and decoder widths (64 units, 2 layers) are held fixed, so most added parameters are in the encoder.')
w('')
w('## 3. Overall results (mean per-trajectory RMSE over the 750 test trajectories)')
w('')
w('Mean (median; SD) of per-trajectory RMSE in data units. "Better than A" is the share of the same 750 trajectories on which the cell has lower error than cell A.')
w('')
w('| Cell | Reconstruction | vs A | better than A | Forecast (7 < t <= 10) | vs A | better than A |')
w('|---|---|---:|---:|---|---:|---:|')
for m in CELLS:
    r = lambda k: f"{res(m, 'overall', k):.4f} ({res(m, 'overall', k, 'median'):.4f}; {res(m, 'overall', k, 'sd'):.4f})"
    if m == 'A': w(f"| {LAB[m]} | {r('recon_rmse')} | n/a | n/a | {r('forecast_rmse')} | n/a | n/a |")
    else:
        rc, fc = J['relative_change_vs_A_of_mean'][m]['overall'], J['fraction_of_trajectories_better_than_A'][m]['overall']
        w(f"| {LAB[m]} | {r('recon_rmse')} | {pct(rc['recon_rmse'])} | {fc['recon_rmse'] * 100:.0f}% | {r('forecast_rmse')} | {pct(rc['forecast_rmse'])} | {fc['forecast_rmse'] * 100:.0f}% |")
w('')
w(f"Reference baselines on the same trajectories: linear interpolation (reconstruction) {bl['linear_interpolation_recon']['overall']['mean']:.4f}; hold-last (forecast) {bl['hold_last_forecast']['overall']['mean']:.4f}; constant velocity (forecast) {bl['const_velocity_forecast']['overall']['mean']:.4f}. "
  f"Cell A's reconstruction ({res('A', 'overall', 'recon_rmse'):.4f}) is worse than linear interpolation; B, C and D are better. Overall, all four cells forecast far better than both naive baselines. {HLTXT}")
w('')
w('### Per family: forecast RMSE')
w('')
w('Mean per-trajectory forecast RMSE; for B, C and D the change relative to A and the Holm-adjusted Wilcoxon p-value for the paired difference (72 cell-contrast tests in total). "ns" = not significant after correction.')
w('')
w('| Family | A | B (vs A) | C (vs A) | D (vs A) |')
w('|---|---:|---|---|---|')
def cellstr(m, g, k):
    v = P(m, g, k); rel = v['relative_change_of_mean']; p = v['wilcoxon_p_holm']
    return f"{res(m, g, k):.4f} ({pct(rel)}; {'p=' + pv(p) if p < .05 else 'ns'})"
for g in G:
    w(f"| {g} | {res('A', g, 'forecast_rmse'):.4f} | " + ' | '.join(cellstr(m, g, 'forecast_rmse') for m in 'BCD') + ' |')
w('')
w('### Per family: reconstruction RMSE')
w('')
w('| Family | A | B (vs A) | C (vs A) | D (vs A) |')
w('|---|---:|---|---|---|')
for g in G:
    w(f"| {g} | {res('A', g, 'recon_rmse'):.4f} | " + ' | '.join(cellstr(m, g, 'recon_rmse') for m in 'BCD') + ' |')
w('')
w('Reconstruction: B, C and D have lower mean error than A in all five families, and every one of these differences is significant after correction.')
w('')
w('## 4. Separate effects of latent dimension and GRU width, and their interaction')
w('')
w('Effects are computed per trajectory from the four cells (positive = higher error). Intervals are 95% bootstrap intervals over test trajectories; p-values are Holm-adjusted Wilcoxon tests over 36 factorial tests. **These quantify test-set sampling only, not training-seed variability** (section 8).')
w('')
w('| Group | Metric | Latent 8 to 16 | GRU 64 to 128 | Interaction (D-C)-(B-A) |')
w('|---|---|---|---|---|')
for k, kn in (('forecast_rmse', 'forecast'), ('recon_rmse', 'reconstruction')):
    for g in ['overall'] + FAMS:
        c = lambda n: (lambda f: f"{f['mean']:+.4f} [{f['bootstrap_ci95'][0]:+.3f}, {f['bootstrap_ci95'][1]:+.3f}] (p={pv(f['wilcoxon_p_holm'])})")(fac(g, k, n))
        w(f"| {g} | {kn} | {c(LATENT)} | {c(GRU)} | {c(INTER)} |")
w('')
cm = P('C', 'overall', 'forecast_rmse'); bm = P('B', 'overall', 'forecast_rmse'); dm = P('D', 'overall', 'forecast_rmse')
w('### Encoder width alone did not produce a clearly distinguishable overall forecasting difference')
w('')
w(f"Cell C (GRU 128, latent 8) versus the baseline A: the mean forecast RMSE differs by {cm['mean_difference']:+.4f} ({pct(cm['relative_change_of_mean'])}; bootstrap 95% interval [{cm['bootstrap_ci95_mean_difference'][0]:+.4f}, {cm['bootstrap_ci95_mean_difference'][1]:+.4f}], which only just excludes zero), "
  f"the median difference is {cm['median_difference']:+.4f}, C is better than A on {fr(cm) * 100:.0f}% of trajectories, and the Holm-adjusted Wilcoxon p-value is {pv(cm['wilcoxon_p_holm'])} (not significant). "
  f"The small mean difference comes mostly from the sinusoid family ({P('C', 'sinusoid', 'forecast_rmse')['mean_difference']:+.3f}, p={pv(P('C', 'sinusoid', 'forecast_rmse')['wilcoxon_p_holm'])}), while ellipse is better "
  f"({P('C', 'ellipse', 'forecast_rmse')['mean_difference']:+.3f}, p={pv(P('C', 'ellipse', 'forecast_rmse')['wilcoxon_p_holm'])}); circle, spiral and figure-eight show no distinguishable difference. "
  f"The size of this difference is also well inside the run-to-run variation seen between identically configured runs (section 8). In short, widening the GRU on its own did not clearly change forecasting in this single-seed experiment.")
w('')
w('### Latent dimension and the combined cell')
w('')
w(f"- Latent-only cell B versus A: overall forecast {pct(bm['relative_change_of_mean'])} (p={pv(bm['wilcoxon_p_holm'])}), better than A on {fr(bm) * 100:.0f}% of trajectories. The difference is significant for circle ({P('B', 'circle', 'forecast_rmse')['mean_difference']:+.3f}) and sinusoid ({P('B', 'sinusoid', 'forecast_rmse')['mean_difference']:+.3f}); for spiral, ellipse and figure-eight it is not distinguishable from zero.")
w(f"- Combined cell D versus A: overall forecast {pct(dm['relative_change_of_mean'])} (p={pv(dm['wilcoxon_p_holm'])}), better on {fr(dm) * 100:.0f}% of trajectories. D has a higher mean than A on all five families, significantly so for circle, spiral, ellipse and sinusoid; figure-eight ({P('D', 'figure_eight', 'forecast_rmse')['mean_difference']:+.3f}) is not distinguishable.")
w(f"- The overall main effects are {fac('overall', 'forecast_rmse', LATENT)['mean']:+.3f} (latent) and {fac('overall', 'forecast_rmse', GRU)['mean']:+.3f} (GRU width). The GRU main effect is an average over both latent sizes; it is carried mainly by D versus B ({J['paired']['D_minus_B/overall/forecast_rmse']['mean_difference']:+.3f}), not by C versus A ({cm['mean_difference']:+.3f}). Describing both factors as independently worsening forecasting would therefore overstate the evidence.")
w(f"- **Interaction.** The overall forecast interaction is {fac('overall', 'forecast_rmse', INTER)['mean']:+.3f} (p={pv(fac('overall', 'forecast_rmse', INTER)['wilcoxon_p_holm'])}): raising the latent size changes forecast error more at GRU 128 (D minus C = {J['paired']['D_minus_C/overall/forecast_rmse']['mean_difference']:+.3f}) than at GRU 64 (B minus A = {bm['mean_difference']:+.3f}), so the two changes reinforce each other overall. "
  f"The sign is the opposite for sinusoid ({fac('sinusoid', 'forecast_rmse', INTER)['mean']:+.3f}), where they offset, so the pattern differs between families and an overall average should not be over-interpreted.")
w('')
w('## 5. The forecasting difference depends on the horizon')
w('')
w('The 7 to 10 window is not uniform. Mean per-trajectory RMSE inside sub-windows (derived from the saved predictions):')
w('')
w('| Window | A | B | C | D | B / C / D better than A on |')
w('|---|---:|---:|---:|---:|---|')
for wd in WIN:
    row = [W[(m, wd, 'overall')].mean() for m in CELLS]; bt = [(W[(m, wd, 'overall')] < W[('A', wd, 'overall')]).mean() for m in 'BCD']
    w(f"| {wd[0]} < t <= {wd[1]} | " + ' | '.join(f'{x:.4f}' for x in row) + f" | {bt[0] * 100:.0f}% / {bt[1] * 100:.0f}% / {bt[2] * 100:.0f}% of trajectories |")
w('')
w('Per family (mean per-trajectory RMSE in the window):')
w('')
w('| Family | Window | A | B | C | D |')
w('|---|---|---:|---:|---:|---:|')
for g in FAMS:
    for wd in WIN:
        w(f"| {g} | {wd[0]} < t <= {wd[1]} | " + ' | '.join(f"{W[(m, wd, g)].mean():.3f}" for m in CELLS) + ' |')
w('')
w(f"Using the across-trajectory RMSE at each time (`error_vs_time` in the JSON, Figure 3): the larger cells are better than A just after t = 7 and then cross above it: B first exceeds A at t = {CROSS['B'][0]:.2f} (stays above from t = {CROSS['B'][1]:.2f}), C at t = {CROSS['C'][0]:.2f} (from {CROSS['C'][1]:.2f}), D at t = {CROSS['D'][0]:.2f} (from {CROSS['D'][1]:.2f}). "
  f"At t = 10 the across-trajectory RMSE is A {et['A']['overall'][-1]:.3f}, B {et['B']['overall'][-1]:.3f}, C {et['C']['overall'][-1]:.3f}, D {et['D']['overall'][-1]:.3f}. "
  'The full-window forecast difference in section 3 is therefore a **late-horizon effect (faster error growth)**; a shorter forecast window would give the opposite sign for B and D overall. Any statement about "worse forecasting" has to name the horizon.')
w('')
w('## 6. Training histories and checkpoint selection')
w('')
w('Checkpoints are chosen by validation -ELBO only (negative log-likelihood of the 30 observed noisy points plus KL). Forecast RMSE is logged each epoch but never used. Intermediate checkpoints were not saved, so the table below uses the **validation** split only; it cannot be extended to the test set.')
w('')
w('| Cell | Selected epoch | Val. forecast RMSE at selected epoch (pooled) | Lowest val. forecast RMSE over training (epoch) | Gap | Train NLL / val NLL at selected epoch | Mean val. -ELBO, epochs 251-275 / 276-300 |')
w('|---|---:|---:|---|---:|---|---|')
for m in CELLS:
    a = au[m]; w(f"| {LAB[m]} | {a['selected_epoch']} | {a['val_forecast_rmse_at_selected']:.4f} | {a['min_val_forecast_rmse']:.4f} ({a['min_val_forecast_epoch']}) | {pct(a['forecast_gap_selected_vs_min'])} | {a['train_nll_at_selected']:.1f} / {a['val_nll_at_selected']:.1f} | {TR[m][0]:.1f} / {TR[m][1]:.1f} |")
w('')
mins = {m: au[m]['min_val_forecast_rmse'] for m in CELLS}
w(f"- Selection by validation -ELBO picks epochs 294-299 of 300, so the cells were still (slowly) improving on the training objective when the cosine schedule ended. Training had not fully converged.")
w(f"- Selection could matter modestly: selecting by lowest validation forecast RMSE instead would lower C's validation forecast error by {au['C']['forecast_gap_selected_vs_min'] * 100:.1f}%, but it would not change the picture for B and D (their lowest validation forecast errors, {mins['B']:.4f} and {mins['D']:.4f}, remain well above A's {mins['A']:.4f}). Selection is therefore **not** a sufficient explanation of the forecasting differences. (This is a post-hoc diagnostic on validation data, not a proposed selection rule.)")
w(f"- Validation and test agree on the ordering of the cells: pooled test forecast RMSE A {J['pooled_rmse']['A']['overall']['forecast_rmse']:.3f}, C {J['pooled_rmse']['C']['overall']['forecast_rmse']:.3f}, B {J['pooled_rmse']['B']['overall']['forecast_rmse']:.3f}, D {J['pooled_rmse']['D']['overall']['forecast_rmse']:.3f}; validation A {Mi['A']['selected_val_forecast_rmse_pooled']:.3f}, C {Mi['C']['selected_val_forecast_rmse_pooled']:.3f}, B {Mi['B']['selected_val_forecast_rmse_pooled']:.3f}, D {Mi['D']['selected_val_forecast_rmse_pooled']:.3f} (A and C swap places; they are close).")
w('')
w('## 7. Is this overfitting?')
w('')
w(f'- **Classical overfitting (validation loss rising while training loss falls) was not observed.** Validation NLL and validation context RMSE fall throughout training (Figure 4), and training and validation NLL at the selected epoch differ by at most {GAPMAX:.1f} nats for every cell (table above). Training NLL uses a sampled latent state and validation uses the posterior mean, so the two are not strictly like-for-like.')
w('- That does **not** rule out other explanations. The training loss only supervises the 30 observed points at t <= 7 (verified in `batch_objective`); no loss term involves t > 7. A model can therefore fit the observed interval very well while its extrapolation is left unconstrained. This is a **plausible hypothesis**, not a tested one: the `full_clean` supervision option that would test it has not been run.')
w('')
w('## 8. Two different kinds of uncertainty')
w('')
w('- **Test-trajectory uncertainty** (what every interval and p-value in this report measures): small, because 750 trajectories are compared pairwise.')
w('- **Training-seed uncertainty: not measured here.** Each cell is one training run. Reference scale from existing results:')
w(f"  - The same architecture trained twice with different data streams differs in mean forecast RMSE by {pct(rA['relative_difference'])} (cell A versus the earlier baseline: {rA['mean_new']:.4f} versus {rA['mean_old']:.4f}) and {pct(rD['relative_difference'])} (cell D versus the earlier medium run: {rD['mean_new']:.4f} versus {rD['mean_old']:.4f}). Two pairs only, so this is an indication of scale, not a variance estimate, and it does not say which run is the more representative.")
w(f"  - In the five-seed two-shape study (a different experiment) the forecast RMSE of one architecture ranges from {min(sv):.3f} to {max(sv):.3f} (mean {SEED['mean']:.3f}, SD {SEED['std']:.3f}, coefficient of variation {SEED['std'] / SEED['mean'] * 100:.0f}%).")
w(f"  - The forecast differences in this ablation (B {pct(J['relative_change_vs_A_of_mean']['B']['overall']['forecast_rmse'])}, C {pct(J['relative_change_vs_A_of_mean']['C']['overall']['forecast_rmse'])}, D {pct(J['relative_change_vs_A_of_mean']['D']['overall']['forecast_rmse'])} versus A) are of the same order as this run-to-run variation. A very small p-value means the two *fixed trained models* differ on the test set; it does not show that the architecture change causes the difference in new training runs.")
w(f"- Reconstruction improvements are much larger ({RMIN} to {RMAX}) and consistent across all five families, but they are also single-seed.")
w('')
w('## 9. Interpretation, separated by strength of evidence')
w('')
w('**Observed (single seed, this protocol)**')
w(f"1. Reconstruction of withheld context points improves for every larger cell in every family (overall {pct(RC['B']['recon_rmse'])}, {pct(RC['C']['recon_rmse'])}, {pct(RC['D']['recon_rmse'])} for B, C, D versus A; cell A is worse than linear interpolation, B, C and D are better).")
w(f"2. Over the full 7 to 10 window the mean forecast error is higher for B ({pct(RC['B']['forecast_rmse'])}) and D ({pct(RC['D']['forecast_rmse'])}) than for A; for C it is {pct(RC['C']['forecast_rmse'])} and the difference is not significant overall.")
w('3. The latent-dimension change is associated with the larger forecast change; the encoder-width change alone is not clearly distinguishable from the baseline overall.')
w('4. The forecast differences appear late in the horizon: larger cells are better than A in the first part of the window and worse after about t = 8 to 8.6.')
w('5. The sinusoid family contributes most of the overall forecast difference for B, C and D; circle contributes for B and D; figure-eight shows no distinguishable difference for any cell.')
w('')
w('**Plausible explanations (not tested)**')
w('- The training objective supervises only observed points at t <= 7, so extrapolation is unconstrained and may depend on details of the learned latent dynamics that differ between sizes.')
w('- Larger latent spaces may produce faster error growth in the learned ODE (consistent with the late-horizon pattern).')
w('- Differences in optimisation (the cells were still improving at epoch 300) or in initialisation.')
w('')
w('**Not established**')
w('- That capacity worsens forecasting in general, or that the effect would reappear with other seeds.')
w('- Why forecasting degrades; any causal mechanism.')
w('- That latent dimension and GRU width have separable, independent effects (the interaction is significant and family-dependent).')
w('- That any of these models beats a stronger baseline than hold-last or constant velocity (none was tested).')
w('')
w('## 10. Limitations')
w('- One seed per cell; no multi-seed confirmation of any ablation result.')
w('- Single synthetic dataset, one observation protocol (30 points, noise 0.05, forecast window 7 to 10).')
w('- Only naive forecast baselines (hold-last, constant velocity); no parametric or sequence-model baseline.')
w('- Only validation-split diagnostics are available for epochs other than the selected one.')
w('- `torchdiffeq` fixed-step `rk4` is the 3/8-rule fourth-order method; predictions at query times between solver steps are linearly interpolated. Measured effect on RMSE is about 2e-4 (see the audit), negligible here.')
w('')
w('## 11. Recommended next steps (not executed)')
w('1. Multi-seed confirmation of at least cells A and D (seeds 123, 456, 789, 2026, with `--seed` and `--data-seed` set per seed); B and C as well if affordable. About 81 minutes and 0.3 MB per run.')
w('2. Report effects as mean +/- SD across seeds, and count in how many seeds each sign reproduces.')
w('3. If degradation reproduces: test the observed-only objective hypothesis (`--supervision full_clean`, a labelled diagnostic) before proposing any fix.')
w('4. Add stronger baselines (per-family parametric fit extrapolation; a sequence-model baseline) before claiming any forecasting benefit.')
w('')
w('## 12. How to present this to a research supervisor')
w('')
w(f'Suggested wording: "In an exploratory, single-seed 2x2 ablation on the five-family dataset (latent 8 or 16, GRU 64 or 128; identical data streams), larger encoders reconstruct withheld points much better ({RMIN} to {RMAX}). Over the full 7 to 10 forecast window the largest model and the latent-only variant forecast worse on average ({FD} and {FB}), whereas widening the GRU alone made no clearly distinguishable overall difference ({FC}, not significant). The effect appears only after about t = 8; early in the window the larger models are better. These differences are of the same order as the seed-to-seed variation seen in our earlier five-seed study, so we have not shown that capacity changes forecasting; the next step is multi-seed confirmation."')
w('')
w('Do **not** say: that capacity "hurts" forecasting in general; that the controlled run shows the "true" architectural effect; that this is overfitting; that latent and GRU effects are independent; that the work replicates the paper or has produced an improved forecasting model.')
w('')
w('## 13. Files and figures')
w('')
w('Files in `results/ablation_2x2/`: `ablation_2x2_results.json` (summary statistics per model and family, paired contrasts, factorial effects, error versus time; it does **not** store per-trajectory RMSE), `predictions_test.npz` (test truth, observations, and predictions of all seven models), `checkpoint_selection_audit.md/.json`, `tables/` (5 Markdown tables), `figures/` (6 figures), `ABLATION_2X2_REPORT.original_uncorrected.md` (backup).')
w('')
w('| Figure | Content |')
w('|---|---|')
w('| `fig1_cells_by_family.png` | Mean forecast and reconstruction RMSE per family for cells A-D with baselines (error bars: standard error over test trajectories) |')
w('| `fig2_interaction.png` | 2x2 interaction plots per family and overall, forecast and reconstruction |')
w('| `fig3_error_vs_time.png` | Across-trajectory RMSE versus time on the **test** split (before t = 7 it includes the observed timestamps) |')
w('| `fig4_training_histories.png` | Validation NLL, training NLL, validation context RMSE and validation forecast RMSE per epoch (validation split only) |')
w('| `fig5_percent_improved.png` | Share of the same test trajectories on which B, C, D beat A |')
w('| `fig6_examples.png` | First test trajectory of each family; illustration only, not selected as typical |')
w('')
w('Known cosmetic issues (figures were left unchanged to preserve the original results): in Figure 5 (right panel) the legend overlaps some bars; Figure 3 curves before t = 7 include the observed timestamps, so they are not forecast errors.')
w('')
w('## Appendix A. Corrections relative to the original report')
w('')
w('| # | Original statement | Corrected |')
w('|---|---|---|')
oc = {'A': 0.0796, 'B': 0.0533, 'C': 0.0563, 'D': 0.0404}; orch = {'B': -33.0, 'C': -29.3, 'D': -49.2}
w(f"| 1 | Overall reconstruction RMSE A/B/C/D = {oc['A']}/{oc['B']}/{oc['C']}/{oc['D']}; changes vs A {orch['B']}%/{orch['C']}%/{orch['D']}% | {res('A', 'overall', 'recon_rmse'):.4f}/{res('B', 'overall', 'recon_rmse'):.4f}/{res('C', 'overall', 'recon_rmse'):.4f}/{res('D', 'overall', 'recon_rmse'):.4f}; changes " + '/'.join(pct(J['relative_change_vs_A_of_mean'][m]['overall']['recon_rmse']) for m in 'BCD') + ' (forecast numbers in the original were correct) |')
w(f"| 2 | Holm correction \"within the 90 factorial tests\" | {len(J['factorial'])} factorial tests (and {len(J['paired'])} cell-contrast tests in a separate correction family) |")
w('| 3 | JSON "includes per-trajectory RMSE for all test trajectories" | It does not; it holds summaries, paired contrasts, factorial effects and error versus time |')
w('| 4 | "Both factors independently worsened forecast accuracy" | Latent-only B: significant on circle and sinusoid only; GRU-only C: not distinguishable overall (section 4); the GRU main effect is carried by D versus B |')
w('| 5 | Interaction described as "partially cancel but reinforce" | Overall forecast interaction is positive (effects reinforce); it offsets only for sinusoid |')
w('| 6 | "Latent main effect is negative for forecast" | Sign convention was confused: positive means higher error (worse) |')
w('| 7 | "latent-16 and latent-32 cells" | The 2x2 has no latent-32 cell (latent 8 and 16 only) |')
w('| 8 | "Robust to checkpoint-selection bias (validation -ELBO and test forecast RMSE are well-correlated)" | Unsupported as stated. What is shown: selection is unlikely to explain B and D, and validation and test order the cells similarly (section 6) |')
w('| 9 | Figure 3 described as validation and test; Figure 4 as including -ELBO | Figure 3 is test only; Figure 4 shows validation NLL, training NLL, validation context RMSE, validation forecast RMSE |')
w('| 10 | Controlled run shows the "true" architecture effect; uncontrolled runs favoured forecast "by chance" | Removed: the 21-22% differences show run-to-run variability, not which run is right (section 8) |')
w('| 11 | "Classic reconstruction-forecast trade-off confirmed" | Not established from one seed |')
w('| 12 | (chat summary) "95 tests pass" | The suite has 106 tests, all passing |')
w('| 13 | Omitted | Forecast difference is a late-horizon effect (section 5) |')
(R / 'ABLATION_2X2_REPORT.md').write_text('\n'.join(L) + '\n')
print('written', R / 'ABLATION_2X2_REPORT.md', len(L), 'lines')
