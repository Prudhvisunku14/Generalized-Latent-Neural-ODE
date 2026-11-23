#!/usr/bin/env python3
"""Checks MTP_COMPLETE_PRESENTATION_MATERIALS: structure, links, forbidden files, numbers vs raw predictions, risky phrases. Read-only.
  python audit/verify_7_mtp_package.py
"""
import csv, re, sys
from pathlib import Path
import numpy as np

PK = Path('MTP_COMPLETE_PRESENTATION_MATERIALS'); fails = []; n = 0
def check(cond, msg):
    global n; n += 1
    if not cond: fails.append(msg)

# ---------------- required structure (from the task)
REQ = '''00_START_HERE.md
01_PROJECT_OVERVIEW/PROJECT_INTRODUCTION.md 01_PROJECT_OVERVIEW/RESEARCH_OBJECTIVES.md 01_PROJECT_OVERVIEW/RESEARCH_GAP.md 01_PROJECT_OVERVIEW/PROJECT_CONTRIBUTIONS.md
02_EXISTING_RESEARCH/ORIGINAL_PAPER_SUMMARY.md 02_EXISTING_RESEARCH/EXPERIMENT_1_CLASSIFICATION.md 02_EXISTING_RESEARCH/EXPERIMENT_2_CNF.md 02_EXISTING_RESEARCH/EXPERIMENT_3_LATENT_ODE.md 02_EXISTING_RESEARCH/ORIGINAL_VS_OUR_IMPLEMENTATION.md
03_MODEL_ARCHITECTURE/COMPLETE_ARCHITECTURE.md 03_MODEL_ARCHITECTURE/GRU_ENCODER.md 03_MODEL_ARCHITECTURE/LATENT_SPACE.md 03_MODEL_ARCHITECTURE/NEURAL_ODE_DYNAMICS.md 03_MODEL_ARCHITECTURE/RK4_SOLVER.md 03_MODEL_ARCHITECTURE/DECODER.md 03_MODEL_ARCHITECTURE/TRAINING_LOSS.md 03_MODEL_ARCHITECTURE/MODEL_CONFIGURATION.md
04_DATASET/DATASET_OVERVIEW.md 04_DATASET/TRAJECTORY_GENERATION.md 04_DATASET/DATA_PREPROCESSING.md 04_DATASET/TRAIN_VALIDATION_TEST_SPLIT.md 04_DATASET/IRREGULAR_OBSERVATIONS.md 04_DATASET/DATASET_VISUALIZATIONS
05_EXPERIMENTAL_METHODOLOGY/COMPLETE_METHODOLOGY.md 05_EXPERIMENTAL_METHODOLOGY/TRAINING_PROCEDURE.md 05_EXPERIMENTAL_METHODOLOGY/EVALUATION_METRICS.md 05_EXPERIMENTAL_METHODOLOGY/EXPERIMENTAL_CONFIGURATIONS.md 05_EXPERIMENTAL_METHODOLOGY/EXPERIMENT_WORKFLOW.md
06_EXPERIMENTAL_RESULTS/FIVE_FAMILY_RESULTS.md 06_EXPERIMENTAL_RESULTS/SHARED_VS_SPECIALIST.md 06_EXPERIMENTAL_RESULTS/CROSS_FAMILY_GENERALIZATION.md 06_EXPERIMENTAL_RESULTS/CAPACITY_SCALING.md 06_EXPERIMENTAL_RESULTS/CONTROLLED_ABLATION_2X2.md 06_EXPERIMENTAL_RESULTS/FORECAST_ERROR_OVER_TIME.md 06_EXPERIMENTAL_RESULTS/CONDITIONING_EXPERIMENTS.md 06_EXPERIMENTAL_RESULTS/GEOLIFE_TRANSFER.md 06_EXPERIMENTAL_RESULTS/RESULTS_TABLES
07_FIGURES/ARCHITECTURE 07_FIGURES/DATASETS 07_FIGURES/RECONSTRUCTION 07_FIGURES/FORECASTING 07_FIGURES/GENERALIZATION 07_FIGURES/ABLATION 07_FIGURES/EXISTING_RESEARCH
08_DISCUSSION_AND_FUTURE_WORK/KEY_FINDINGS.md 08_DISCUSSION_AND_FUTURE_WORK/RESEARCH_LIMITATIONS.md 08_DISCUSSION_AND_FUTURE_WORK/FUTURE_WORK.md 08_DISCUSSION_AND_FUTURE_WORK/CONCLUSION.md
09_PRESENTATION_CONTENT/COMPLETE_SLIDE_OUTLINE.md 09_PRESENTATION_CONTENT/SLIDE_BY_SLIDE_CONTENT.md 09_PRESENTATION_CONTENT/SPEAKING_SCRIPT.md 09_PRESENTATION_CONTENT/POSSIBLE_SUPERVISOR_QUESTIONS.md
10_VERIFIED_SOURCE_REPORTS 11_SOURCE_INDEX/NUMERICAL_RESULTS_SOURCES.md 11_SOURCE_INDEX/FIGURE_SOURCE_INDEX.md 11_SOURCE_INDEX/FILE_MANIFEST.md'''.split()
for r in REQ: check((PK / r).exists(), f'missing required: {r}')
for d in ('04_DATASET/DATASET_VISUALIZATIONS', '06_EXPERIMENTAL_RESULTS/RESULTS_TABLES', '07_FIGURES/ARCHITECTURE', '07_FIGURES/DATASETS', '10_VERIFIED_SOURCE_REPORTS'): check(any(p.is_file() for p in (PK / d).rglob('*')), f'empty folder: {d}')

# ---------------- forbidden file types and size
files = [p for p in PK.rglob('*') if p.is_file()]
check(not [p for p in files if p.suffix in {'.pt', '.pth', '.npz', '.pyc', '.ckpt', '.zip', '.npy'} or '__pycache__' in str(p) or '.venv' in str(p) or '.git' in p.parts], 'forbidden file in package')
total = sum(p.stat().st_size for p in files)

# ---------------- every referenced figure / table path exists
pat = re.compile(r'(?:\.\./)?((?:07_FIGURES|06_EXPERIMENTAL_RESULTS|04_DATASET|03_MODEL_ARCHITECTURE)/[A-Za-z0-9_./\-]+\.(?:png|csv|md))')
missing = set()
for p in files:
    if p.suffix != '.md' or p.parts[1:2] == ('10_VERIFIED_SOURCE_REPORTS',): continue
    for m in pat.finditer(p.read_text()):
        if not (PK / m.group(1)).exists(): missing.add((str(p.relative_to(PK)), m.group(1)))
check(not missing, f'broken references: {sorted(missing)[:8]}')
# slide figure paths
for l in (PK / '09_PRESENTATION_CONTENT/COMPLETE_SLIDE_OUTLINE.md').read_text().splitlines():
    for m in re.findall(r'`(07_FIGURES/[^`]+)`', l): check((PK / m).exists(), f'slide figure missing: {m}')

# ---------------- CSV numbers vs raw predictions (independent recomputation)
def per_traj(pred, truth, qt, mask):
    d2 = ((pred - truth) ** 2).sum(-1); fut = qt > 7; miss = (~mask) & ~fut[None]
    return np.sqrt((d2 * miss).sum(1) / (2 * miss.sum(1))), np.sqrt(d2[:, fut].mean(1) / 2)
def mask_of(Z):
    idx = np.abs(Z['obs_t'][:, :, None] - Z['query_t'][None, None, :]).argmin(-1); m = np.zeros(Z['truth'].shape[:2], bool); np.put_along_axis(m, idx, True, 1); return m
Z = np.load('results/ablation_2x2/predictions_test.npz'); Z5 = np.load('results/five_shapes/predictions_test.npz'); fam = Z['family']
R = {m: per_traj(Z['pred_' + m], Z['truth'], Z['query_t'], mask_of(Z)) for m in ['A', 'B', 'C', 'D', 'orig_baseline', 'orig_medium', 'orig_large']}
R3 = per_traj(Z5['pred_three'], Z5['truth'], Z5['query_t'], mask_of(Z5)); M5 = mask_of(Z5)
hold = np.repeat(Z5['obs'][:, -1:, :], 200, axis=1); HL = per_traj(hold, Z5['truth'], Z5['query_t'], M5)[1]; CVv = per_traj(Z5['pred_const_vel'], Z5['truth'], Z5['query_t'], M5)[1]
lin = np.stack([np.stack([np.interp(Z5['query_t'], Z5['obs_t'][i], Z5['obs'][i, :, c]) for c in range(2)], -1) for i in range(len(fam))]); LI = per_traj(lin, Z5['truth'], Z5['query_t'], M5)[0]
FAMS = ['circle', 'spiral', 'ellipse', 'figure-eight', 'sinusoid', 'all five']; sel = {f: fam == i for i, f in enumerate(FAMS[:5])}; sel['all five'] = np.ones(len(fam), bool)
T = PK / '06_EXPERIMENTAL_RESULTS/RESULTS_TABLES/'
rows = list(csv.reader(open(T / 'five_family_reconstruction_rmse.csv'))); hdr = rows[0]
order = ['A', 'B', 'C', 'D']
for r in rows[1:]:
    g = r[0]; s = sel[g]; vals = dict(zip(hdr[1:], r[1:]))
    check(abs(float(vals['linear_interpolation']) - LI[s].mean()) < 6e-5, f'csv recon linear {g}'); check(abs(float(vals['3-family model (frozen; figure-eight and sinusoid unseen)']) - R3[0][s].mean()) < 6e-5, f'csv recon 3-family {g}')
    for key, m in (('earlier 8/64 run', 'orig_baseline'), ('A 8/64', 'A'), ('B 16/64', 'B'), ('C 8/128', 'C'), ('D 16/128', 'D'), ('earlier 16/128 run', 'orig_medium'), ('earlier 32/128 run', 'orig_large')): check(abs(float(vals[key]) - R[m][0][s].mean()) < 6e-5, f'csv recon {m} {g}')
rows = list(csv.reader(open(T / 'five_family_forecast_rmse.csv'))); hdr = rows[0]
for r in rows[1:]:
    g = r[0]; s = sel[g]; vals = dict(zip(hdr[1:], r[1:]))
    check(abs(float(vals['hold_last']) - HL[s].mean()) < 6e-5, f'csv fc hold {g}'); check(abs(float(vals['constant_velocity']) - CVv[s].mean()) < 6e-5, f'csv fc cv {g}')
    check(abs(float(vals['3-family model (frozen; figure-eight and sinusoid unseen)']) - R3[1][s].mean()) < 6e-5, f'csv fc 3-family {g}')
    for key, m in (('earlier 8/64 run', 'orig_baseline'), ('A 8/64', 'A'), ('B 16/64', 'B'), ('C 8/128', 'C'), ('D 16/128', 'D'), ('earlier 16/128 run', 'orig_medium'), ('earlier 32/128 run', 'orig_large')): check(abs(float(vals[key]) - R[m][1][s].mean()) < 6e-5, f'csv fc {m} {g}')
for r in list(csv.reader(open(T / 'ablation_2x2_overall.csv')))[1:]:
    m = r[0]; check(abs(float(r[4]) - R[m][0].mean()) < 6e-5 and abs(float(r[6]) - R[m][1].mean()) < 6e-5, f'csv ablation overall {m}')
    if m != 'A': check(abs(float(r[7].rstrip('%')) - (R[m][1].mean() / R['A'][1].mean() - 1) * 100) < .06, f'csv ablation forecast change {m}')
# windows
Zq, Zt = Z['query_t'], Z['truth']
for r in list(csv.reader(open(T / 'forecast_error_by_time_window.csv')))[1:]:
    lo, hi = [float(x) for x in re.findall(r'(\d+) < t <= (\d+)', r[0])[0]]; s = (Zq > lo) & (Zq <= hi)
    for j, m in enumerate('ABCD', start=1): d2 = ((Z['pred_' + m] - Zt) ** 2).sum(-1)[:, s]; check(abs(float(r[j]) - np.sqrt(d2.sum(1) / (2 * s.sum())).mean()) < 6e-5, f'csv window {lo}-{hi} {m}')
# shared vs specialist vs README (independent hand-written source)
RD = Path('README.md').read_text(); mm = re.search(r'\| circle \| ([\d.]+) ± ([\d.]+) \| \*\*([\d.]+) ± ([\d.]+)\*\* \|', RD)
for r in list(csv.reader(open(T / 'shared_vs_specialist.csv')))[1:]:
    if r[0] == 'circle' and r[1] == 'forecast': check(abs(float(r[2]) - float(mm.group(1))) < 6e-4 and abs(float(r[5]) - float(mm.group(3))) < 6e-4, 'csv shared vs specialist circle forecast vs README')
# GeoLife table vs the saved table5
t5 = Path('results/geolife/tables/table5_synthetic_vs_geolife_trained.md').read_text().splitlines()
for r in list(csv.reader(open(T / 'geolife_interpolation_30_observations.csv')))[1:]:
    key = {'Linear interpolation': 'Linear', 'Cubic spline': 'Cubic spline', 'PCHIP': 'PCHIP', 'Frozen synthetic (three-family) model': 'Frozen synthetic (S3)', 'GeoLife-trained from scratch': 'GeoLife-trained (scratch)', 'Synthetic then fine-tuned on GeoLife': 'Synthetic → GeoLife fine-tuned'}[r[0]]
    line = next(l for l in t5 if l.startswith(f'| {key} |')); check(abs(float(r[1]) - float(re.findall(r'[\d.]+', line.split('|')[2])[0])) < .006, f'csv geolife {key}')

# ---------------- risky phrases: list every occurrence for manual review (outside copied reports)
risky = re.compile(r'exact(ly)? replicat|replicat(ed|ion)|improved (forecast|model)|better forecasting model|state[- ]of[- ]the[- ]art|prove[sd]?\b|overfit', re.I)
print('--- risky-phrase occurrences (must all be negations / warnings):')
cnt = 0
for p in sorted(files):
    if p.suffix != '.md' or p.parts[1:2] == ('10_VERIFIED_SOURCE_REPORTS',): continue
    for i, l in enumerate(p.read_text().splitlines()):
        for m in risky.finditer(l):
            cnt += 1; s = max(0, m.start() - 70); print(f"  {p.relative_to(PK)}:{i + 1}: ...{l[s:m.end() + 60].strip()}...")
print(f'{n} checks; {len(fails)} failures; package size {total / 1e6:.1f} MB; {len(files)} files; {cnt} risky-phrase occurrences listed above')
for f in fails: print('  FAIL', f)
sys.exit(1 if fails else 0)
