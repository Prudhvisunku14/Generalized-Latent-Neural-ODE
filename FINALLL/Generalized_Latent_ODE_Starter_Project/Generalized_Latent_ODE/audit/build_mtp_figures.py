#!/usr/bin/env python3
"""New figures for MTP_COMPLETE_PRESENTATION_MATERIALS (architecture, methodology flowchart, dataset).
Drawn from the actual model code dimensions and from SAVED dataset / prediction files only; equation-based panels are labelled as illustrations.
Run from the project root: python audit/build_mtp_figures.py
"""
import sys
sys.path.insert(0, 'audit')
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from results_common import BG, INK, INK2, GRID, SER, style, FAMS, fname

PK = Path('MTP_COMPLETE_PRESENTATION_MATERIALS'); AR = PK / '07_FIGURES/ARCHITECTURE'; DS = PK / '07_FIGURES/DATASETS'
for d in (AR, DS): d.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'figure.facecolor': BG, 'savefig.facecolor': BG})


def box(ax, x, y, w, h, text, fc, fs=10, ec=None, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec=ec or '#8d8c86', lw=1.2))
    lines = text.split('\n')
    ax.text(x + w / 2, y + h / 2, '\n'.join(lines), ha='center', va='center', fontsize=fs, color=INK, fontweight='normal', linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, label=None, off=(0, .12)):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=16, lw=1.6, color=INK2))
    if label: ax.text((x0 + x1) / 2 + off[0], (y0 + y1) / 2 + off[1], label, ha='center', va='bottom', fontsize=8.5, color=INK2)


# ======================================================================================= architecture pipeline
fig, ax = plt.subplots(figsize=(17, 7.2)); ax.set_xlim(0, 16.6); ax.set_ylim(0, 7.2); ax.axis('off')
C = ['#dcebfb', '#fde3d7', '#d5f1e7', '#fbeec4', '#f9dce7', '#e5e1f7']
y = 4.0; h = 2.0; w = 2.5; xs = [.2, 2.95, 5.7, 8.45, 11.2, 13.95]
box(ax, xs[0], y, w, h, 'OBSERVED\nTRAJECTORY\n30 noisy points (x, y)\nirregular times, t <= 7', '#f1efe9', fs=9)
box(ax, xs[1], y, w, h, 'GRU ENCODER\nreads points newest\nto oldest; features\n[x, y, t/T, dt/T]\nGRU: 4 -> 64 units', C[0], fs=9)
box(ax, xs[2], y, w, h, 'POSTERIOR\nLinear 64 -> 2L\nmean mu and\nlog-variance\n(clamped to [-8, 5])', C[1], fs=9)
box(ax, xs[3], y, w, h, 'INITIAL STATE z0\n(dimension L = 8)\ntrain: mu + sigma*eps\ntest: z0 = mu', C[2], fs=9)
box(ax, xs[4], y, w, h, 'NEURAL ODE\ndz/dt = f(z)\nMLP L -> 64 -> 64 -> L\ntanh; time-invariant', C[3], fs=9)
box(ax, xs[5], y, 2.45, h, 'DECODER\nz(t) -> (x, y)\nMLP L -> 64 -> 64 -> 2\ntanh', C[4], fs=9)
for k in range(5): arrow(ax, xs[k] + w, y + h / 2, xs[k + 1], y + h / 2)
box(ax, 7.2, 1.95, 6.2, 1.35, 'RK4 SOLVER: torchdiffeq "rk4" (3/8 rule), step 0.1\nintegrates z0 forward over t in [0, 10]; values at query times\nbetween solver steps are linearly interpolated', C[5], fs=9)
ax.annotate('', xy=(12.45, y - .02), xytext=(12.45, 3.32), arrowprops=dict(arrowstyle='<->', color=INK2, lw=1.5))
box(ax, 13.95, 1.95, 2.45, 1.35, 'PREDICTED PATH\nreconstruction t <= 7\nforecast 7 < t <= 10', '#f1efe9', fs=9)
arrow(ax, 15.15, y, 15.15, 3.32)
box(ax, .2, .25, 7.4, 1.3, 'TRAINING LOSS = negative ELBO\nGaussian NLL (sigma = 0.05) on the 30 OBSERVED points only\n+ KL( q(z0 | observations) || N(0, I) ), weight 1', '#fff7d6', fs=9)
box(ax, 8.0, .25, 8.4, 1.3, 'NO loss term involves t > 7: the future is never supervised in training.\nFuture points are used only for evaluation. Checkpoint = lowest VALIDATION\nnegative ELBO (test results are never used for selection).', '#fde3d7', fs=9)
ax.text(8.3, 6.85, 'Latent Neural ODE used in this project (baseline "Model D": latent 8, GRU 64; 24,602 parameters)', ha='center', fontsize=14, color=INK)
ax.text(8.3, 6.45, 'Parameters: GRU 13,440 | posterior layer 1,040 | ODE vector field 5,256 | decoder 4,866. Other variants change only L and the GRU width (MODEL_CONFIGURATION.md).', ha='center', fontsize=9.5, color=INK2)
ax.text(.2, 3.65, 'T = 10 (time is divided by T before entering the GRU)', fontsize=8.5, color=INK2, ha='left')
fig.savefig(AR / 'fig_architecture_pipeline.png', dpi=200, bbox_inches='tight'); plt.close(fig)

# ======================================================================================= methodology flowchart
fig, ax = plt.subplots(figsize=(15, 7.4)); ax.set_xlim(0, 15); ax.set_ylim(0, 7.4); ax.axis('off')
steps = [('1  DATASET\nGENERATION', '5 families x 1,000\n200 time points on [0, 10]\nnoise SD 0.05, seed 42'), ('2  SPLIT', 'per family 700 / 150 / 150\ntrain / validation / test\nno id shared between splits'),
         ('3  NORMALISE', 'mean and scale from the\nTRAINING split only\n(five-family scale 0.794)'), ('4  OBSERVATIONS', 'each batch: 30 random times\nt <= 7, first at t = 0\nstored noisy values'),
         ('5  ENCODE', 'GRU (backwards) ->\nmu, log-variance ->\ninitial latent state z0'), ('6  EVOLVE + DECODE', 'RK4 integrates dz/dt = f(z)\ndecoder maps z(t) to (x, y)\nat the query times'),
         ('7  TRAIN', 'negative ELBO, Adam 3e-3\ncosine schedule, 300 epochs\nbatch 32, grad clip 1'), ('8  SELECT', 'keep the epoch with the\nlowest VALIDATION\nnegative ELBO'),
         ('9  EVALUATE', 'test split, identical 30\nobservations for all models\nposterior mean z0'), ('10  COMPARE', 'reconstruction RMSE (t <= 7)\nforecast RMSE (7 < t <= 10)\nvs simple baselines')]
pos = [(.3, 5.2), (3.3, 5.2), (6.3, 5.2), (9.3, 5.2), (12.3, 5.2), (12.3, 2.7), (9.3, 2.7), (6.3, 2.7), (3.3, 2.7), (.3, 2.7)]
for i, ((t, d), (x, yy)) in enumerate(zip(steps, pos)):
    box(ax, x, yy, 2.4, 1.7, f'{t}\n\n{d}', C[i % 5] if i < 5 else C[(i + 1) % 5], fs=8.6)
    if i < 4: arrow(ax, x + 2.4, yy + .85, pos[i + 1][0], pos[i + 1][1] + .85)
    elif i == 4: arrow(ax, x + 1.2, yy, pos[5][0] + 1.2, pos[5][1] + 1.7)
    elif i < 9: arrow(ax, x, yy + .85, pos[i + 1][0] + 2.4, pos[i + 1][1] + .85)
box(ax, .3, .35, 14.4, 1.3, 'STATISTICS: paired comparisons on identical test trajectories (bootstrap 95% intervals, Wilcoxon tests with Holm correction); multi-seed Welch tests for shared vs specialist.\nCROSS-FAMILY: models evaluated on families absent from their training data (zero-shot). REAL DATA: GeoLife GPS segments from held-out users.', '#f1efe9', fs=9.2)
ax.text(7.5, 7.05, 'Experimental workflow (training and evaluation)', ha='center', fontsize=14, color=INK)
fig.savefig(AR / 'fig_methodology_flowchart.png', dpi=200, bbox_inches='tight'); plt.close(fig)

# ======================================================================================= dataset figures
tz = np.load('data/processed/five_shapes/test.npz'); xy, fam, tt = tz['xy'], tz['family'], tz['time']
FC = dict(zip(FAMS, SER))
# 1. gallery: real test trajectories
fig, axes = plt.subplots(1, 5, figsize=(17, 3.9))
for i, (ax, f) in enumerate(zip(axes, FAMS)):
    idx = np.flatnonzero(fam == i)[:8]
    for k in idx: ax.plot(xy[k, :, 0], xy[k, :, 1], lw=1.4, alpha=.8, color=FC[f])
    ax.set_aspect('equal', adjustable='datalim'); ax.set_title(fname(f), fontsize=12); ax.grid(alpha=.2); ax.set_facecolor(BG)
fig.suptitle('Figure 1. The five trajectory families: first 8 clean test trajectories of each family (from data/processed/five_shapes/test.npz; t from 0 to 10)', fontsize=12, y=1.02)
fig.savefig(DS / 'fig1_five_families_dataset_samples.png', dpi=180, bbox_inches='tight'); plt.close(fig)

# 2. parameter variations: equation-based illustration (same formulas as src/glo/data.py and src/glo/five_shapes.py)
t = np.linspace(0, 10, 200)
def circ(R, om=1.1, ph=0.3, c=(0, 0)): a = om * t + ph; return c[0] + R * np.cos(a), c[1] + R * np.sin(a)
def spir(g, R=1.0, om=1.1, ph=0.3): a = om * t + ph; r = R * np.exp(g * t); return r * np.cos(a), r * np.sin(a)
def elli(ratio, a=1.4, al=0.6, om=1.1, ph=0.3):
    b = a * ratio; th = om * t + ph; xx, yy = a * np.cos(th), b * np.sin(th); return np.cos(al) * xx - np.sin(al) * yy, np.sin(al) * xx + np.cos(al) * yy
def fig8(B, A=1.2, om=1.1, ph=0.3): th = om * t + ph; return A * np.sin(th), B * np.sin(2 * th)
def sinu(om, A=.8, v=.25, ph=0.3, cx=0): x0 = cx - 5 * v; return x0 + v * t, A * np.sin(om * t + ph)
sweeps = [('circle\nradius R = 0.6, 1.0, 1.4', [circ(.6), circ(1.0), circ(1.4)]), ('spiral\ngrowth g = -0.10, +0.06, +0.11', [spir(-.10), spir(.06), spir(.11)]),
          ('ellipse\naxis ratio b/a = 0.40, 0.60, 0.78', [elli(.4), elli(.6), elli(.78)]), ('figure-eight\nvertical amplitude B = 0.45, 0.70, 1.0', [fig8(.45), fig8(.7), fig8(1.0)]),
          ('sinusoid\nangular frequency = 0.75, 1.10, 1.45', [sinu(.75), sinu(1.1), sinu(1.45)])]
fig, axes = plt.subplots(1, 5, figsize=(17, 3.9))
for ax, (ttl, curves), f in zip(axes, sweeps, FAMS):
    for j, (cx_, cy_) in enumerate(curves): ax.plot(cx_, cy_, lw=2, color=SER[j] if False else ['#2a78d6', '#eb6834', '#1baf7a'][j], label=['low', 'middle', 'high'][j])
    ax.set_aspect('equal', adjustable='datalim'); ax.set_title(ttl, fontsize=9.5); ax.grid(alpha=.2)
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle('Figure 2. ILLUSTRATION (drawn from the generation equations, not experimental data): effect of one key parameter in each family; other parameters fixed', fontsize=11.5, y=1.03)
fig.savefig(DS / 'fig2_parameter_variations_illustration.png', dpi=180, bbox_inches='tight'); plt.close(fig)

# 3. regular vs irregular sampling (irregular = actual observation times drawn for test trajectory 0 in the saved evaluation)
Z = np.load('results/ablation_2x2/predictions_test.npz'); obs_t0 = Z['obs_t'][0]; gaps = np.diff(Z['obs_t'], axis=1).ravel()
reg_t = np.linspace(0, 7, 30)
fig, axes = plt.subplots(3, 1, figsize=(12, 5.6), gridspec_kw={'height_ratios': [1, 1, 1.5]})
for ax, tms, ttl, col in ((axes[0], reg_t, 'REGULAR sampling (for contrast; NOT used): 30 equally spaced times', SER[2]), (axes[1], obs_t0, 'IRREGULAR sampling (used): the 30 observation times actually drawn for test trajectory 0', SER[1])):
    ax.vlines(tms, 0, 1, color=col, lw=2); ax.set_xlim(0, 10); ax.set_ylim(0, 1.3); ax.axvline(7, color=INK, ls=':'); ax.set_yticks([]); ax.set_title(ttl, fontsize=10, loc='left'); ax.set_facecolor(BG)
axes[2].hist(gaps, bins=40, color=SER[0]); axes[2].set_xlabel('gap between consecutive observation times (time units), all 750 test trajectories'); axes[2].set_ylabel('count'); style(axes[2])
axes[2].set_title(f'Gap sizes in the saved evaluation: min {gaps.min():.2f}, mean {gaps.mean():.2f}, max {gaps.max():.2f} (times are multiples of the 0.05 grid; first observation always at t = 0)', fontsize=9.5, loc='left')
fig.suptitle('Figure 3. Regular versus irregular observation times (observations only before t = 7)', fontsize=12, y=1.0); fig.tight_layout(); fig.savefig(DS / 'fig3_regular_vs_irregular_sampling.png', dpi=180, bbox_inches='tight'); plt.close(fig)

# 4. observed vs forecasting interval (saved test trajectory of the spiral family, ablation predictions file)
k = int(np.flatnonzero(Z['family'] == 1)[0]); truth, obs, obs_t, qt = Z['truth'][k], Z['obs'][k], Z['obs_t'][k], Z['query_t']
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={'width_ratios': [1.1, 1.5, 1.5]})
ax = axes[0]; past = qt <= 7
ax.plot(*truth[past].T, color='#b9b8b2', lw=5, label='true path, t <= 7'); ax.plot(*truth[~past].T, color='#b9b8b2', lw=5, ls=(0, (1.5, 1)), label='true path, 7 < t <= 10 (never shown to the model)')
ax.scatter(*obs.T, s=18, color=SER[1], zorder=3, label='30 noisy observations'); ax.set_aspect('equal', adjustable='datalim'); ax.legend(fontsize=7.5, frameon=False, loc='lower left'); ax.grid(alpha=.2); ax.set_title('xy plane', fontsize=11)
for ax, c, nm in ((axes[1], 0, 'x(t)'), (axes[2], 1, 'y(t)')):
    ax.plot(qt, truth[:, c], color='#7d7c76', lw=2); ax.scatter(obs_t, obs[:, c], s=18, color=SER[1], zorder=3)
    ax.axvspan(0, 7, color='#dcebfb', alpha=.6); ax.axvspan(7, 10, color='#fde3d7', alpha=.7); ax.axvline(7, color=INK, ls=':')
    ax.text(3.5, ax.get_ylim()[1], 'OBSERVED INTERVAL\n(reconstruction is scored here)', ha='center', va='top', fontsize=8.5); ax.text(8.5, ax.get_ylim()[1], 'FORECAST\ninterval', ha='center', va='top', fontsize=8.5)
    ax.set_xlabel('time t'); ax.set_ylabel(nm); style(ax, 'both')
fig.suptitle('Figure 4. Observed interval vs forecasting interval (saved test spiral; truth and observations only, no model output)', fontsize=12, y=1.03)
fig.savefig(DS / 'fig4_observed_vs_forecast_interval.png', dpi=180, bbox_inches='tight'); plt.close(fig)

# 5. dataset organisation
fig, ax = plt.subplots(figsize=(14, 5.6)); ax.set_xlim(0, 14); ax.set_ylim(0, 5.6); ax.axis('off')
box(ax, .2, 2.2, 2.6, 1.4, 'FIVE-FAMILY DATASET\n5,000 trajectories\n1,000 per family', '#f1efe9')
for j, f in enumerate(FAMS): box(ax, 3.4, 4.6 - j * .95, 2.3, .8, f'{fname(f)}: 1,000\n700 train / 150 val / 150 test', ['#dcebfb', '#fde3d7', '#d5f1e7', '#fbeec4', '#e5e1f7'][j], fs=8.8); arrow(ax, 2.8, 2.9, 3.4, 5.0 - j * .95)
box(ax, 6.6, 3.9, 2.6, 1.3, 'TRAIN\n3,500 trajectories\n(700 per family)', '#d5f1e7'); box(ax, 6.6, 2.2, 2.6, 1.3, 'VALIDATION\n750 trajectories\n(150 per family)', '#fbeec4'); box(ax, 6.6, .5, 2.6, 1.3, 'TEST\n750 trajectories\n(150 per family)', '#fde3d7')
ax.plot([6.1, 6.1], [.9, 5.4], color=INK2, lw=1.6); ax.plot([5.8, 6.1], [2.9, 2.9], color=INK2, lw=1.6)
for yy in (4.5, 2.8, 1.1): arrow(ax, 6.1, 2.9, 6.6, yy + .35)
box(ax, 9.9, 3.9, 3.9, 1.3, 'fit the model weights; normalisation\nmean/scale computed here only', '#f1efe9', fs=9); box(ax, 9.9, 2.2, 3.9, 1.3, 'choose the checkpoint (lowest\nvalidation negative ELBO); fixed\nobservation draws every epoch', '#f1efe9', fs=9); box(ax, 9.9, .5, 3.9, 1.3, 'final scores only; identical 30 observations\nfor every model (observation seed 123);\nnever used for selection', '#f1efe9', fs=9)
for yy in (4.5, 2.8, 1.1): arrow(ax, 9.2, yy + .35, 9.9, yy + .35)
ax.text(7, 5.45, 'Figure 5. Training, validation and test organisation (counts from data/processed/five_shapes/metadata.json; the three families of the earlier three-family data are inside this set, unchanged)', ha='center', fontsize=10)
fig.savefig(DS / 'fig5_dataset_split_organisation.png', dpi=180, bbox_inches='tight'); plt.close(fig)
print('MTP figures written to', AR, 'and', DS)
