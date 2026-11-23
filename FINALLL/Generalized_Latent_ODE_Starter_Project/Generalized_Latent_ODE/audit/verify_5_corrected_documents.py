#!/usr/bin/env python3
"""Cross-checks the corrected documents against numbers recomputed independently from the RAW saved predictions.

Run from the project root:  python audit/verify_5_corrected_documents.py
The report generators read summary JSON files; this script does not. It recomputes per-trajectory RMSE from
predictions_test.npz files and compares them with the numbers as WRITTEN in the Markdown tables (parsed with regular expressions).
Read-only. Exits non-zero on any mismatch.
"""
import json, re, sys
import numpy as np
from pathlib import Path

FAMS = ['circle', 'spiral', 'ellipse', 'figure_eight', 'sinusoid']; fails = []; n_checked = 0
def check(name, written, actual, tol):
    global n_checked; n_checked += 1
    if abs(written - actual) > tol: fails.append(f'{name}: written {written} vs recomputed {actual:.6f}')
nums = lambda cell: [float(x) for x in re.findall(r'[-+]?\d+\.\d+(?:e[-+]?\d+)?|[-+]?\d+(?:e[-+]?\d+)?', cell)]
def table_rows(text, start, end):
    seg = text[text.index(start): text.index(end, text.index(start) + 1)] if end else text[text.index(start):]
    return [[c.strip() for c in l.strip().strip('|').split('|')] for l in seg.splitlines() if l.startswith('|') and not set(l) <= set('|-: ')]

def per_traj(pred, truth, qt, mask):
    d2 = ((pred - truth) ** 2).sum(-1); fut = qt > 7; miss = (~mask) & ~fut[None]
    return np.sqrt((d2 * miss).sum(1) / (2 * miss.sum(1))), np.sqrt(d2[:, fut].mean(1) / 2), d2

# ------------------------------------------------------------------ ablation report
Z = np.load('results/ablation_2x2/predictions_test.npz'); qt, truth, fam = Z['query_t'], Z['truth'], Z['family']
idx = np.abs(Z['obs_t'][:, :, None] - qt[None, None, :]).argmin(-1); mask = np.zeros(truth.shape[:2], bool); np.put_along_axis(mask, idx, True, 1)
R = {m: per_traj(Z['pred_' + m], truth, qt, mask) for m in ['A', 'B', 'C', 'D', 'orig_baseline', 'orig_medium']}
sel = {f: fam == i for i, f in enumerate(FAMS)}; sel['overall'] = np.ones(len(fam), bool)
txt = Path('results/ablation_2x2/ABLATION_2X2_REPORT.md').read_text()
# section 3 overall table
rows = table_rows(txt, '| Cell | Reconstruction | vs A', '\nReference baselines')
for r in rows[1:]:
    m = r[0][0]; rec, fc = R[m][0], R[m][1]
    mean_r, med_r, sd_r = nums(r[1]); mean_f, med_f, sd_f = nums(r[4])
    for nm, w_, a_ in (('recon mean', mean_r, rec.mean()), ('recon median', med_r, np.median(rec)), ('recon sd', sd_r, rec.std(ddof=1)),
                      ('forecast mean', mean_f, fc.mean()), ('forecast median', med_f, np.median(fc)), ('forecast sd', sd_f, fc.std(ddof=1))):
        check(f'ablation overall {m} {nm}', w_, a_, 6e-5)
    if m != 'A':
        check(f'ablation {m} recon change %', float(re.findall(r'[-+]\d+\.\d+', r[2])[0]), (rec.mean() / R['A'][0].mean() - 1) * 100, .06)
        check(f'ablation {m} forecast change %', float(re.findall(r'[-+]\d+\.\d+', r[5])[0]), (fc.mean() / R['A'][1].mean() - 1) * 100, .06)
        check(f'ablation {m} recon better-than-A %', float(r[3].rstrip('%')), (rec < R['A'][0]).mean() * 100, .51)
        check(f'ablation {m} forecast better-than-A %', float(r[6].rstrip('%')), (fc < R['A'][1]).mean() * 100, .51)
# per-family tables
for title, start, end, k in (('forecast', '### Per family: forecast RMSE', '### Per family: reconstruction RMSE', 1), ('recon', '### Per family: reconstruction RMSE', '\nReconstruction: B, C and D', 0)):
    for r in table_rows(txt, start, end)[1:]:
        g = r[0]
        check(f'ablation {title} {g} A', float(r[1]), R['A'][k][sel[g]].mean(), 6e-5)
        for j, m in enumerate('BCD', start=2):
            check(f'ablation {title} {g} {m}', nums(r[j])[0], R[m][k][sel[g]].mean(), 6e-5)
            check(f'ablation {title} {g} {m} change %', float(re.findall(r'\(([-+]\d+\.\d+)%', r[j])[0]), (R[m][k][sel[g]].mean() / R['A'][k][sel[g]].mean() - 1) * 100, .06)
# factorial overall rows
def contr(k, s):
    a, b, c, d = (R[m][k][s] for m in 'ABCD'); return [x.mean() for x in (.5 * ((b - a) + (d - c)), .5 * ((c - a) + (d - b)), (d - c) - (b - a))]
for r in table_rows(txt, '| Group | Metric | Latent 8 to 16', '\n### Encoder width alone'):
    if r[0] in ('Group',): continue
    k = 1 if r[1] == 'forecast' else 0; vals = contr(k, sel[r[0]])
    for j, v in enumerate(vals, start=2): check(f'ablation factorial {r[0]} {r[1]} col{j}', float(re.findall(r'[-+]\d+\.\d+', r[j])[0]), v, 6e-5)
# window table
for r in table_rows(txt, '| Window | A | B | C | D |', '\nPer family (mean per-trajectory RMSE in the window)')[1:]:
    lo, hi = [float(x) for x in re.findall(r'(\d+) < t <= (\d+)', r[0])[0]]; s = (qt > lo) & (qt <= hi)
    for j, m in enumerate('ABCD', start=1):
        d2 = ((Z['pred_' + m] - truth) ** 2).sum(-1)[:, s]; check(f'ablation window {lo}-{hi} {m}', float(r[j]), np.sqrt(d2.sum(1) / (2 * s.sum())).mean(), 6e-5)
# appendix row 1 and run-to-run percentages
check('ablation appendix recon A', float(re.findall(r'\| (\d\.\d{4})/', txt.split('| 1 |')[1])[0]), R['A'][0].mean(), 6e-5)
check('ablation run-to-run A vs earlier baseline %', float(re.search(r'by (\+\d+\.\d)% \(cell A versus', txt).group(1)), (R['A'][1].mean() / R['orig_baseline'][1].mean() - 1) * 100, .06)
check('ablation run-to-run D vs earlier medium %', float(re.findall(r'and (\+\d+\.\d)% \(cell D', txt)[0]), (R['D'][1].mean() / R['orig_medium'][1].mean() - 1) * 100, .06)

# ------------------------------------------------------------------ five-shape report
Z5 = np.load('results/five_shapes/predictions_test.npz'); qt5, tr5, fam5 = Z5['query_t'], Z5['truth'], Z5['family']
idx = np.abs(Z5['obs_t'][:, :, None] - qt5[None, None, :]).argmin(-1); mk = np.zeros(tr5.shape[:2], bool); np.put_along_axis(mk, idx, True, 1)
P3, P5 = per_traj(Z5['pred_three'], tr5, qt5, mk), per_traj(Z5['pred_five'], tr5, qt5, mk)
hold = np.repeat(Z5['obs'][:, -1:, :], len(qt5), axis=1)                        # hold-last observation (all t > 7 are after the last observation)
cv = per_traj(Z5['pred_const_vel'], tr5, qt5, mk)
lin = np.stack([np.stack([np.interp(qt5, Z5['obs_t'][i], Z5['obs'][i, :, c]) for c in range(2)], -1) for i in range(len(fam5))])
HL, LI = per_traj(hold, tr5, qt5, mk)[1], per_traj(lin, tr5, qt5, mk)[0]
s5 = {f: fam5 == i for i, f in enumerate(FAMS)}; s5['overall'] = np.ones(len(fam5), bool)
t5 = Path('reports/five_shapes_report.md').read_text()
for r in table_rows(t5, '| Family | Status | Recon. RMSE | Linear interp.', '\nOn the two unseen families')[1:]:
    g = r[0]; check(f'five zero-shot {g} recon', nums(r[2])[0], P3[0][s5[g]].mean(), 6e-5); check(f'five zero-shot {g} linear', float(r[3]), LI[s5[g]].mean(), 6e-5)
    check(f'five zero-shot {g} forecast', nums(r[4])[0], P3[1][s5[g]].mean(), 6e-5); check(f'five zero-shot {g} hold-last', float(r[5]), HL[s5[g]].mean(), 6e-5); check(f'five zero-shot {g} const-vel', float(r[6]), cv[1][s5[g]].mean(), 6e-5)
for r in table_rows(t5, '| Family | Recon. RMSE mean (median) | Linear interp.', '\nThe model forecasts every family')[1:]:
    g = r[0]; check(f'five model {g} recon', nums(r[1])[0], P5[0][s5[g]].mean(), 6e-5); check(f'five model {g} forecast', nums(r[3])[0], P5[1][s5[g]].mean(), 6e-5)
for r in table_rows(t5, '| Family | Status for the three-shape model | Forecast: three', '\n- On the three original')[1:]:
    g = r[0]; d = P5[1][s5[g]] - P3[1][s5[g]]
    check(f'five paired {g} three', float(r[2]), P3[1][s5[g]].mean(), 6e-5); check(f'five paired {g} five', float(r[3]), P5[1][s5[g]].mean(), 6e-5)
    check(f'five paired {g} diff', float(re.findall(r'[-+]\d+\.\d{4}', r[4])[0]), d.mean(), 6e-5); check(f'five paired {g} five-better %', float(r[5].rstrip('%')), (d < 0).mean() * 100, .51)

# ------------------------------------------------------------------ README tables
rd = Path('README.md').read_text()
for r in table_rows(rd, '| Family | Three-shape model (frozen) | Five-shape model | Hold-last baseline |', '\n* **No zero-shot')[1:]:
    g = r[0].split(' ')[0]; check(f'README five-family {g} three', float(r[1]), P3[1][s5[g]].mean(), 6e-4); check(f'README five-family {g} five', float(r[2]), P5[1][s5[g]].mean(), 6e-4); check(f'README five-family {g} hold-last', float(r[3]), HL[s5[g]].mean(), 6e-4)
for r in table_rows(rd, '| Cell (latent / GRU) | Parameters |', '\nLinear interpolation scores')[1:]:
    m = r[0][0]; check(f'README ablation {m} recon', float(r[2]), R[m][0].mean(), 6e-5); check(f'README ablation {m} forecast', float(r[3]), R[m][1].mean(), 6e-5)
# README GeoLife: written numbers vs the saved summary JSON and the claim "fine-tuned lower than scratch at 30 observations only"
G = json.load(open('results/geolife/main_evaluation/geolife_trained_summary.json'))
check('README GeoLife forecast scratch', float(re.findall(r'scratch (\d+), fine-tuned (\d+)', rd)[0][0]), G['forecast_n30']['A']['geolife_scratch']['median'], .51)
check('README GeoLife forecast fine-tuned', float(re.findall(r'scratch (\d+), fine-tuned (\d+)', rd)[0][1]), G['forecast_n30']['A']['geolife_finetuned_from_three_shapes']['median'], .51)
t5g = Path('results/geolife/tables/table5_synthetic_vs_geolife_trained.md').read_text().splitlines()
den = {int(c[0]): c for c in ([x.strip() for x in l.strip('|').split('|')] for l in t5g if re.match(r'\| (10|20|30|40|50) \|', l))}
lower = [n for n in den if float(den[n][7]) < float(den[n][6])]; n_checked += 1
if lower != [30]: fails.append(f'README claim "fine-tuned lower than scratch at 30 observations only" not supported: lower at {lower}')

print(f'{n_checked} numeric/qualitative checks against raw saved predictions; {len(fails)} mismatches')
for f in fails: print('  MISMATCH', f)
sys.exit(1 if fails else 0)
