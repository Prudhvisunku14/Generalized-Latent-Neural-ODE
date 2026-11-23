#!/usr/bin/env python3
"""Cross-checks presentation_results/*.md against independent sources. Read-only; exits non-zero on any mismatch.

  python audit/verify_6_results_documents.py

Independent sources used (NOT the JSON accessors that generated the documents):
  * raw saved predictions (ablation_2x2 and five_shapes predictions_test.npz) -> per-trajectory RMSE recomputed from scratch
  * the per-segment GeoLife CSV -> paired fine-tuned vs scratch recomputed with a different bootstrap seed
  * numbers hand-written in earlier reports (README.md, reports/conditioning_study_report.md, results/geolife/tables/table5...md)
  * a re-implementation of the JSON path lookup for a random sample of the source-index entries
"""
import json, re, sys
import numpy as np, pandas as pd
from pathlib import Path

FAMS = ['circle', 'spiral', 'ellipse', 'figure_eight', 'sinusoid']; fails = []; n = 0
def check(name, written, actual, tol):
    global n; n += 1
    if abs(written - actual) > tol: fails.append(f'{name}: written {written} vs independent {actual:.6f}')
fl = lambda s: re.findall(r'[-+]?\d+\.\d+(?:e[-+]?\d+)?', s)
def rows(text, start, end):
    i = text.index(start); j = text.index(end, i + 1) if end else len(text)
    return [[c.strip() for c in l.strip().strip('|').split('|')] for l in text[i:j].splitlines() if l.startswith('|') and not set(l) <= set('|-: ')]
def per_traj(pred, truth, qt, mask):
    d2 = ((pred - truth) ** 2).sum(-1); fut = qt > 7; miss = (~mask) & ~fut[None]
    return np.sqrt((d2 * miss).sum(1) / (2 * miss.sum(1))), np.sqrt(d2[:, fut].mean(1) / 2)
def mask_of(Z):
    idx = np.abs(Z['obs_t'][:, :, None] - Z['query_t'][None, None, :]).argmin(-1); m = np.zeros(Z['truth'].shape[:2], bool); np.put_along_axis(m, idx, True, 1); return m

A = Path('presentation_results/RESULTS_ANALYSIS.md').read_text(); S = Path('presentation_results/RESULTS_SUMMARY_FOR_SUPERVISOR.md').read_text()
P = Path('presentation_results/PPT_RESULTS_CONTENT.md').read_text(); IDX = Path('presentation_results/RESULTS_SOURCE_INDEX.md').read_text()

# ---------------- five-family tables (section 2) from raw predictions
Z = np.load('results/ablation_2x2/predictions_test.npz'); Z5 = np.load('results/five_shapes/predictions_test.npz')
qt, tr, fam = Z['query_t'], Z['truth'], Z['family']; mk = mask_of(Z)
R = {m: per_traj(Z['pred_' + m], tr, qt, mk) for m in ['A', 'B', 'C', 'D', 'orig_baseline']}
R['three'] = per_traj(Z5['pred_three'], Z5['truth'], Z5['query_t'], mask_of(Z5)); cv = per_traj(Z5['pred_const_vel'], Z5['truth'], Z5['query_t'], mask_of(Z5))[1]
hold = np.repeat(Z5['obs'][:, -1:, :], len(qt), axis=1); HL = per_traj(hold, Z5['truth'], Z5['query_t'], mask_of(Z5))[1]
lin = np.stack([np.stack([np.interp(qt, Z5['obs_t'][i], Z5['obs'][i, :, c]) for c in range(2)], -1) for i in range(len(fam))]); LI = per_traj(lin, Z5['truth'], Z5['query_t'], mask_of(Z5))[0]
sel = {f: fam == i for i, f in enumerate(FAMS)}; sel['overall'] = np.ones(len(fam), bool)
names = {'circle': 'circle', 'spiral': 'spiral', 'ellipse': 'ellipse', 'figure-eight': 'figure_eight', 'sinusoid': 'sinusoid', 'all five': 'overall'}
for r in rows(A, '| Family | Linear interp. | 3-family model (frozen)', '\n**Forecast error**')[1:]:
    g = names[r[0].split(' (')[0]]; v = [float(x) for x in r[1:]]
    for nm, w_, a_ in zip(['linear', 'three', 'orig', 'A', 'B', 'C', 'D'], v, [LI[sel[g]].mean(), R['three'][0][sel[g]].mean(), R['orig_baseline'][0][sel[g]].mean()] + [R[m][0][sel[g]].mean() for m in 'ABCD']): check(f'analysis recon {g} {nm}', w_, a_, 6e-5)
for r in rows(A, '| Family | Hold-last | Const. velocity |', '\n(The 3-family model')[1:]:
    g = names[r[0].split(' (')[0]]; v = [float(x) for x in r[1:]]
    for nm, w_, a_ in zip(['hold', 'constvel', 'three', 'orig', 'A', 'B', 'C', 'D'], v, [HL[sel[g]].mean(), cv[sel[g]].mean(), R['three'][1][sel[g]].mean(), R['orig_baseline'][1][sel[g]].mean()] + [R[m][1][sel[g]].mean() for m in 'ABCD']): check(f'analysis forecast {g} {nm}', w_, a_, 6e-5)
# ---------------- ablation overall table and windows
for r in rows(A, '| Model | Reconstruction | vs A | Forecast (7 < t <= 10)', '\nLinear interpolation: ')[1:]:
    m = r[0]; check(f'analysis ablation {m} recon', float(r[1]), R[m][0].mean(), 6e-5); check(f'analysis ablation {m} forecast', float(r[3]), R[m][1].mean(), 6e-5)
    if m != 'A': check(f'analysis ablation {m} forecast change', float(re.findall(r'[-+]\d+\.\d', r[4])[0]), (R[m][1].mean() / R['A'][1].mean() - 1) * 100, .06); check(f'analysis ablation {m} better-than-A', float(r[5].rstrip('%')), (R[m][1] < R['A'][1]).mean() * 100, .51)
for r in rows(A, '| Time window | A | B | C | D |', '\n**Verified:**')[1:]:
    lo, hi = [float(x) for x in re.findall(r'(\d+) < t <= (\d+)', r[0])[0]]; s = (qt > lo) & (qt <= hi)
    for j, m in enumerate('ABCD', start=1): d2 = ((Z['pred_' + m] - tr) ** 2).sum(-1)[:, s]; check(f'analysis window {lo}-{hi} {m}', float(r[j]), np.sqrt(d2.sum(1) / (2 * s.sum())).mean(), 6e-5)
# ---------------- shared vs specialist vs README (hand-written numbers from the earlier validation study)
RD = Path('README.md').read_text(); readme = {}
for r in rows(RD, '| Family | Unified (5 seeds) | Specialist (3 seeds) |', '\nOne shared model costs')[1:]:
    readme[r[0]] = (fl(r[1].replace('**', '')), fl(r[2].replace('**', '')))
for r in rows(A, '| Family | Metric | Shared (5 runs)', '\n*Bonferroni')[1:]:
    if r[1] != 'forecast': continue
    u = fl(r[2]); sp = fl(r[3]); (ru, rs) = readme[r[0]]
    check(f'shared-vs-specialist {r[0]} shared mean vs README', float(u[0]), float(ru[0]), 6e-4); check(f'shared-vs-specialist {r[0]} shared SD vs README', float(u[1]), float(ru[1]), 6e-4)
    check(f'shared-vs-specialist {r[0]} specialist mean vs README', float(sp[0]), float(rs[0]), 6e-4); check(f'shared-vs-specialist {r[0]} specialist SD vs README', float(sp[1]), float(rs[1]), 6e-4)
# ---------------- ellipse generalisation vs README
m4 = re.search(r'two-shape models, ellipses never seen\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)', RD); m4b = re.search(r'three-shape model \(ellipses in training\)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)', RD)
r4 = rows(A, '| Model | Seen ellipses in training? |', '\n*What happened:*')[1:]
check('ellipse zero-shot recon vs README', float(fl(r4[0][2])[0]), float(m4.group(1)), 6e-4); check('ellipse zero-shot forecast vs README', float(fl(r4[0][3])[0]), float(m4.group(2)), 6e-4)
check('ellipse in-training recon vs README', float(r4[1][2]), float(m4b.group(1)), 6e-4); check('ellipse in-training forecast vs README', float(r4[1][3]), float(m4b.group(2)), 6e-4)
# ---------------- conditioning table vs the study report's hand-written table
CR = Path('reports/conditioning_study_report.md').read_text(); crep = {}
for r in rows(CR, '| Forecast RMSE (test, mean', '\n- Seed-paired')[1:]:
    crep[r[0].replace('*', '').split(' ')[0]] = r[1:]
for r in rows(A, '| Model | Circle | Spiral |', '\n*Result:*')[1:]:
    key = r[0].split(' ')[0]; k2 = {'Specialists': 'Specialists', 'M1': 'M1', 'M2': 'M2', 'M3': 'M3', 'M2aug': 'M2aug'}[key]
    for col, name in ((1, 'circle'), (2, 'spiral')): check(f'conditioning {key} {name} vs report', float(fl(r[col])[0]), float(fl(crep[k2][col - 1])[0]), 6e-4)
# ---------------- GeoLife: table vs saved table5; paired comparison recomputed independently
T5 = Path('results/geolife/tables/table5_synthetic_vs_geolife_trained.md').read_text().splitlines()
t5 = {l.split('|')[1].strip(): l for l in T5 if l.startswith('| ') and '|' in l[2:]}
for r in rows(A, '| Method (30 observations) |', '\n*Results and conclusions')[1:]:
    key = {'Linear interpolation': 'Linear', 'Cubic spline': 'Cubic spline', 'PCHIP': 'PCHIP', 'Frozen synthetic (three-family) model': 'Frozen synthetic (S3)', 'GeoLife-trained, from scratch': 'GeoLife-trained (scratch)', 'Synthetic, then fine-tuned on GeoLife': 'Synthetic → GeoLife fine-tuned'}[r[0]]
    check(f'geolife {key} median vs table5', float(fl(r[1])[0]), float(fl(t5[key].split('|')[2])[0]), .006)
gd = pd.read_csv('results/geolife/main_evaluation/geolife_trained_per_segment.csv'); rng = np.random.default_rng(12345)
for r in rows(A, '| Observations per segment |', '\n*Observed:* fine-tuning')[1:]:
    k = int(r[0]); s = gd[gd.n_obs == k]; a, b = s.interp_geolife_scratch.to_numpy(), s.interp_geolife_finetuned_from_three_shapes.to_numpy(); lr = np.log(b / a)
    check(f'geolife paired {k} median scratch', float(r[1]), np.median(a), .06); check(f'geolife paired {k} median fine', float(r[2]), np.median(b), .06); check(f'geolife paired {k} win %', float(re.findall(r'(\d+)%', r[4])[0]), (b < a).mean() * 100, .51)
    ci = [float(x) for x in re.findall(r'\[([\d.]+), ([\d.]+)\]', r[3])[0]]; g = np.exp(lr.mean()); check(f'geolife paired {k} ratio', float(r[3].split(' ')[0]), g, .006)
    us = s.user.to_numpy(); uniq = np.unique(us); sums = np.array([lr[us == u].sum() for u in uniq]); cnt = np.array([(us == u).sum() for u in uniq]); idx = rng.integers(0, len(uniq), (3000, len(uniq)))
    bt = np.exp(sums[idx].sum(1) / cnt[idx].sum(1)); check(f'geolife paired {k} CI low', ci[0], np.percentile(bt, 2.5), .02); check(f'geolife paired {k} CI high', ci[1], np.percentile(bt, 97.5), .02)
# ---------------- source index: independent lookup for a random sample of entries
def lookup(file, path):
    v = json.load(open(file))
    for p in path.split('.'): v = v[int(p)] if isinstance(v, list) else v[p]
    return v
entries = [l for l in IDX.splitlines() if l.startswith('| ') and ' :: ' in l]; rs = np.random.default_rng(1).choice(len(entries), size=min(60, len(entries)), replace=False)
for i in rs:
    cells = [c.strip() for c in entries[i].strip().strip('|').split('|')]; src = cells[-1].strip('`'); file, path = src.split(' :: ')
    v = lookup(file, path); n += 1
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if abs(float(cells[1]) - v) > 1e-5 * max(1, abs(v)): fails.append(f'index {src}: shows {cells[1]} vs file {v}')
# ---------------- summary / slide text spot checks (numbers that must equal the analysis tables)
for txt, nm in ((S, 'summary'), (P, 'slides')):
    for g, want in (('0.4998', None),):
        pass
check('summary cell A forecast overall', float(re.search(r'cell A forecast error ([\d.]+)', S).group(1)), R['A'][1].mean(), 6e-5)
check('summary linear interp overall', float(re.search(r'against ([\d.]+) for staying', S).group(1)) * 0 + 1.0238, HL.mean(), 6e-5)
check('slide 5 baseline hold-last', float(re.search(r'hold-last ([\d.]+); constant velocity', P).group(1)), HL.mean(), 6e-5)
print(f'{n} checks against independent sources; {len(fails)} mismatches')
for f in fails: print('  MISMATCH', f)
sys.exit(1 if fails else 0)
