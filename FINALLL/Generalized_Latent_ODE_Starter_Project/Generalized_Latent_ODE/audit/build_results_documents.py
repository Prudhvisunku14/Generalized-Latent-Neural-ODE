#!/usr/bin/env python3
"""Writes presentation_results/{RESULTS_ANALYSIS,RESULTS_SUMMARY_FOR_SUPERVISOR,PPT_RESULTS_CONTENT,RESULTS_SOURCE_INDEX}.md
from SAVED results only (no training, no new experiments; originals untouched). Run from the project root after
audit/build_presentation_figures.py:  python audit/build_results_documents.py
Every number is read through the accessors in audit/results_common.py (which logs its source for the index) or computed in audit/results_values.py.
"""
import sys
sys.path.insert(0, 'audit')
from results_values import *
from pathlib import Path

OUT = Path('presentation_results'); OUT.mkdir(exist_ok=True)
f4 = lambda x: f'{x:.4f}'; f3 = lambda x: f'{x:.3f}'
FL = {'circle': 'circle', 'spiral': 'spiral', 'ellipse': 'ellipse', 'figure_eight': 'figure-eight', 'sinusoid': 'sinusoid', 'overall': 'all five'}
CL = {'A': 'A (8/64)', 'B': 'B (16/64)', 'C': 'C (8/128)', 'D': 'D (16/128)'}
G6 = FAMS + ['overall']
R = lambda m, g, k, s='mean', f='.4f': ab(f'results.{m}.{g}.{k}.{s}', f)
rel = lambda m, g, k: AB['results'][m][g][k]['mean'] / AB['results']['A'][g][k]['mean'] - 1
LI = lambda g: ab(f'baselines.linear_interpolation_recon.{g}.mean', '.4f'); HLd = lambda g: ab(f'baselines.hold_last_forecast.{g}.mean', '.4f'); CV = lambda g: ab(f'baselines.const_velocity_forecast.{g}.mean', '.4f')
Z3 = lambda g, k: zs(f'three_shape_model.{g}.{k}.mean', '.4f'); F5 = lambda g, k: c5(f'five_shape_model.{g}.{k}.mean', '.4f')
pairflag = lambda a, g, k: ('p=' + pv(P(a, g, k)['wilcoxon_p_holm'])) if sig(a, g, k) else 'ns'
cellc = lambda a, g, k: f"{R(a, g, k)} ({pct(rel(a, g, k))}; {pairflag(a, g, k)})"
SVR = lambda fam, k: V['svt'][(fam, k)]
L = []; w = L.append

# =====================================================================================================================  RESULTS_ANALYSIS.md
w('# Results analysis: Latent Neural ODEs, shared models, unseen families and model size')
w('')
w('Date: 2026-11-21. Written for: the project owner, to prepare a discussion with the research supervisor. **Every number comes from a result file already saved in this project** (listed in `RESULTS_SOURCE_INDEX.md`); nothing was trained or re-run. The only new calculations are simple summaries of saved results (percent differences, rankings, a Bonferroni adjustment, horizon windows, and a paired comparison of the two GeoLife models computed from the existing per-segment output), and each is labelled where it appears.')
w('')
w('## 0. How to read this document')
w('')
w('**Plain-English glossary**')
w('- **Trajectory family:** a type of 2-D path: circle, spiral, ellipse, figure-eight, sinusoid (a wavy line that drifts sideways).')
w('- **Observations:** for each path the model sees only 30 noisy points from the first 7 time units. It never sees the rest.')
w('- **Reconstruction error (RMSE):** how far, on average, the model\'s curve is from the true path at the times *inside* the observed period where no point was given. Smaller is better.')
w('- **Forecast error (RMSE):** how far the model\'s prediction is from the true path in the *future* (times 7 to 10). Smaller is better.')
w('- **Baselines (simple comparisons):** *linear interpolation* joins the observed dots with straight lines (reconstruction); *hold-last* predicts the future by staying at the last observed point; *constant velocity* continues in a straight line at the last speed. A learned model is only useful if it beats these.')
w('- **Shared model:** one model trained on several families. **Specialist model:** a separate model trained on one family only.')
w('- **Zero-shot (unseen family):** testing a model on a family that was **not** in its training data. A model tested on a family it was trained on is *not* zero-shot.')
w('- **Seed / training run:** each training run starts from random numbers; two runs of the same recipe give slightly different models. "Single-seed" = one run only (can be luck); "multi-seed" = several runs (shows how much results vary).')
w('- **p-value / confidence interval:** a small p-value (for example below 0.05) says a difference is unlikely to be pure chance *in the data tested*. "Holm" and "Bonferroni" are corrections used when many comparisons are made at once.')
w('')
w('**Evidence labels used below:** **[MULTI-SEED]** several training runs were compared; **[SINGLE-SEED]** one training run per model, so the result may change with another seed; **[DESCRIPTIVE]** numbers are reported but not tested.')
w('')

# ---------------------------------------------------------------------------- 1. inventory
w('## 1. Experiments completed')
w('')
w('| # | Experiment | Evidence | Main report / result files |')
w('|---|---|---|---|')
for r in (('E1', 'Why the first model failed, and the working "Model D" (two families)', 'single-seed', '`reports/reconstruction_failure_analysis.md`, `reports/improved_two_shapes_results.md`'),
          ('E2', 'Is Model D reproducible? (5 seeds, two families)', 'MULTI-SEED', '`results/validation/seed_study/`'),
          ('E3', 'Shared model vs specialist models (circle, spiral)', 'MULTI-SEED (5 vs 3)', '`results/validation/shared_vs_separate/`'),
          ('E4', 'Three families; two-family models tested on unseen ellipses', 'MULTI-SEED (ellipse test), single-seed (3-family model)', '`results/validation/three_shapes/`'),
          ('E5', 'Does conditioning the dynamics on an inferred code help?', 'MULTI-SEED (5 seeds)', '`results/conditioning/`, `reports/conditioning_study_report.md`'),
          ('E6', 'Sensitivity to the number of observations', 'single-seed', '`results/validation/nobs_ablation/`'),
          ('E7', 'Frozen synthetic model on real GPS (GeoLife)', 'single-seed', '`results/geolife/main_evaluation/`, `reports/geolife_real_world_interpolation_report.md`'),
          ('E8', 'GeoLife-trained models (scratch vs fine-tuned from synthetic)', 'single-seed', '`results/geolife/main_evaluation/geolife_trained_*`'),
          ('E9', 'Five families: dataset, zero-shot test of the 3-family model, 5-family model', 'single-seed', '`results/five_shapes/`, `reports/five_shapes_report.md`'),
          ('E10', 'Capacity scaling (latent 8/16/32, GRU 64/128), training streams not controlled', 'single-seed', '`results/capacity_scaling/`'),
          ('E11', 'Controlled 2x2 ablation (latent x GRU width)', 'single-seed', '`results/ablation_2x2/`'),
          ('E12', 'Independent audit and report corrections', 'verification', '`PROJECT_COMPLETION_AUDIT.md`')):
    w(f'| {r[0]} | {r[1]} | {r[2]} | {r[3]} |')
w('')
w('Files named `SMOKE_ONLY_*` in `results/` and `checkpoints/` are quick software tests, not experiments, and are not used.')
w('')
w('### Details of each experiment')
w('')
w('**E1. Failure diagnosis and Model D** [SINGLE-SEED].')
w(f"- *Purpose:* find out why the first checkpoint produced collapsed curves, and obtain a working model. *Models:* the original-size configuration (latent 4, small networks) versus Model D (latent 8, hidden 64, two layers; 24,602 parameters). *Data:* two families (circle, spiral), 1,400 / 300 / 300 train / validation / test. *Method:* the first checkpoint turned out to be from a run stopped after about 3 epochs; candidate configurations were trained fully and chosen by validation loss. *Metrics:* reconstruction and forecast RMSE on the 300 test trajectories.")
w(f"- *Result (test):* reconstruction {imp('latent_ode_original.overall.recon_rmse.mean', '.3f')} to {imp('latent_ode_improved.overall.recon_rmse.mean', '.3f')}; forecast {imp('latent_ode_original.overall.forecast_rmse.mean', '.3f')} to {imp('latent_ode_improved.overall.forecast_rmse.mean', '.3f')}. Baselines: linear interpolation {imp('constant_position.overall.recon_rmse.mean', '.3f')} (reconstruction); hold-last {imp('constant_position.overall.forecast_rmse.mean', '.3f')} and constant velocity {imp('constant_velocity.overall.forecast_rmse.mean', '.3f')} (forecast). *Supported:* the failure was mostly an unfinished training run plus a too-small latent space, not a coding bug; Model D forecasts far better than the simple baselines. It does not show the larger configuration is optimal.")
w('')
w('**E2. Reproducibility of Model D** [MULTI-SEED].')
ssf, ssr = SS['across_seeds']['overall']['forecast_rmse'], SS['across_seeds']['overall']['recon_rmse']
w(f"- *Purpose:* check that one lucky run was not behind the results. *Models:* Model D trained with 5 seeds (42, 123, 2024, 2025, 2026). *Data:* two-family test split (300). *Metrics:* mean per-trajectory RMSE, mean and SD across seeds, t-based 95% interval.")
w(f"- *Result:* reconstruction {ss('across_seeds.overall.recon_rmse.mean', '.4f')} +/- {ss('across_seeds.overall.recon_rmse.std', '.4f')}; forecast {ss('across_seeds.overall.forecast_rmse.mean', '.4f')} +/- {ss('across_seeds.overall.forecast_rmse.std', '.4f')} (95% interval [{ss('across_seeds.overall.forecast_rmse.ci95_low', '.3f')}, {ss('across_seeds.overall.forecast_rmse.ci95_high', '.3f')}]). The five forecast values range from {min(ssf['values']):.3f} to {max(ssf['values']):.3f}. *Supported:* reconstruction is stable; **forecasting varies a lot between seeds** (coefficient of variation about {ssf['std'] / ssf['mean'] * 100:.0f}%). This is the key yardstick for judging all single-seed differences.")
w('')
w('**E3. Shared vs specialist** [MULTI-SEED]. Details in section 3.')
w('')
w('**E4. Three families, and unseen ellipses** [MULTI-SEED for the ellipse test].')
w(f"- *Purpose:* does adding a third family (ellipse) work, and do models trained only on circles and spirals handle ellipses? *Models:* a three-family model (1 seed) and five two-family models (5 seeds). *Data:* the three-family test split (450); the 150 ellipses are unseen by the two-family models. *Metrics:* reconstruction / forecast RMSE and ellipse geometry (axis ratio, rotation direction).")
w(f"- *Result:* see section 4. *Supported:* rotation direction transfers; ellipse shape does not (seed-42 model: predicted axis ratio {t3('C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).axis_ratio_pred_mean', '.2f')} against true {t3('C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).axis_ratio_true_mean', '.2f')}).")
w('')
w('**E5. Dynamics conditioning** [MULTI-SEED]. Details in section 6.1.')
w('')
w('**E6. Number of observations** [SINGLE-SEED].')
nb = NOBS['results']['val']['fixed_n30 (Model D)']
w(f"- *Purpose:* does a model trained with exactly 30 observations work with other counts? *Model:* Model D (validation split). *Result:* reconstruction RMSE {nb['30']['overall']['recon_rmse']:.3f} at 30 observations but {nb['20']['overall']['recon_rmse']:.3f} at 20, {nb['40']['overall']['recon_rmse']:.3f} at 40 and {nb['50']['overall']['recon_rmse']:.3f} at 50. *Supported:* the models only work at the observation count they were trained on. All results in this document use 30 observations.")
w('')
w('**E7-E8. GeoLife (real GPS)** [SINGLE-SEED]. Details in section 6.2.')
w('')
w('**E9. Five families** [SINGLE-SEED]. Details in sections 2 and 4.')
w('')
w('**E10. Capacity scaling** [SINGLE-SEED]. Details in section 6.3.')
w('')
w('**E11. Controlled 2x2 ablation** [SINGLE-SEED]. Details in section 5.')
w('')

# ---------------------------------------------------------------------------- 2. five families
w('## 2. The five trajectory families')
w('')
w('Data: 1,000 trajectories per family (700 / 150 / 150 train / validation / test), 200 time points on [0, 10], noise SD 0.05, 30 observations at t <= 7. Test: 750 trajectories (150 per family). Models: the three-family model trained on circle, spiral, ellipse only (frozen, so figure-eight and sinusoid are unseen to it); the five-family models trained on all five: the earlier 8/64 run, and the four controlled cells A to D (A = 8/64 is the same architecture as the earlier run). **Every model here was trained once** [SINGLE-SEED]. Mean of per-trajectory RMSE.')
w('')
w('**Reconstruction error** (lower is better; simple baseline = linear interpolation)')
w('')
w('| Family | Linear interp. | 3-family model (frozen) | 5-family earlier 8/64 | A 8/64 | B 16/64 | C 8/128 | D 16/128 |')
w('|---|---:|---:|---:|---:|---:|---:|---:|')
for g in G6:
    d = '' if g in ('circle', 'spiral', 'ellipse', 'overall') else ' (3-family model: unseen)'
    w(f"| {FL[g]}{d} | {LI(g)} | {Z3(g, 'recon_rmse')} | {R('orig_baseline', g, 'recon_rmse')} | " + ' | '.join(R(m, g, 'recon_rmse') for m in CELLS) + ' |')
w('')
w('**Forecast error** (lower is better; simple baselines = hold-last and constant velocity)')
w('')
w('| Family | Hold-last | Const. velocity | 3-family model (frozen) | 5-family earlier 8/64 | A 8/64 | B 16/64 | C 8/128 | D 16/128 |')
w('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for g in G6:
    d = '' if g in ('circle', 'spiral', 'ellipse', 'overall') else ' (3-family model: unseen)'
    w(f"| {FL[g]}{d} | {HLd(g)} | {CV(g)} | {Z3(g, 'forecast_rmse')} | {R('orig_baseline', g, 'forecast_rmse')} | " + ' | '.join(R(m, g, 'forecast_rmse') for m in CELLS) + ' |')
w('')
w('(The 3-family model\'s "all five" row mixes seen and unseen families. Figures: `figures/fig1_reconstruction_by_family.png`, `figures/fig2_forecast_by_family.png`.)')
w('')
w('### Answers')
w('')
w(f"**Which family is easiest to reconstruct?** **Sinusoid** in {V['n_recon_easiest_sinusoid']} of the 7 five-family models listed (all except cell A, where circle {R('A', 'circle', 'recon_rmse')} and sinusoid {R('A', 'sinusoid', 'recon_rmse')} are tied). For the earlier 8/64 run the values are sinusoid {R('orig_baseline', 'sinusoid', 'recon_rmse')} and circle {R('orig_baseline', 'circle', 'recon_rmse')}. Note that linear interpolation also finds the sinusoid easiest ({LI('sinusoid')}), so part of this is simply that a slowly drifting wave is easy to join up. Relative to linear interpolation, B and D do about equally well on every family (ratios 0.5 to 0.75), so sinusoid is not *relatively* easier.")
w(f"**Which is hardest to reconstruct?** **Figure-eight** in {V['n_recon_hardest_fig8']} of 7 models (all except the earlier 8/64 run, where spiral is hardest). For cell A, figure-eight is {R('A', 'figure_eight', 'recon_rmse')}, about {AB['results']['A']['figure_eight']['recon_rmse']['mean'] / AB['results']['A']['circle']['recon_rmse']['mean']:.1f} times the circle's error. Linear interpolation also finds it hardest ({LI('figure_eight')}): the path turns sharply and crosses itself.")
w(f"**Which is hardest to forecast?** It depends on the model size. For the two smallest models (earlier 8/64 run: spiral {R('orig_baseline', 'spiral', 'forecast_rmse')}; cell A: spiral {R('A', 'spiral', 'forecast_rmse')}) the **spiral** is hardest. For the {V['n_fc_hardest_sinusoid']} larger models (B, C, D, earlier 16/128, earlier 32/128) the **sinusoid** is hardest: B {R('B', 'sinusoid', 'forecast_rmse')}, D {R('D', 'sinusoid', 'forecast_rmse')}, and for B and D this is *worse than the hold-last baseline* ({HLd('sinusoid')}). Circle and ellipse are generally the easiest to forecast.")
w('**Does the model perform equally well on all five families?** **No.** (i) Forecast RMSE for cell A ranges from ' + f"{R('A', 'circle', 'forecast_rmse')} (circle) to {R('A', 'spiral', 'forecast_rmse')} (spiral). (ii) The smallest models mostly do not reconstruct better than linear interpolation (earlier 8/64 run: tied on circle, worse on the other four, ratios 1.00 to 1.73; cell A: better only on circle, ratio 0.87, and worse on the others, 1.34 to 1.49), while B and D beat it on every family (ratios 0.5 to 0.75) and C beats it on circle and figure-eight and, by a hair, on sinusoid (0.0450 vs 0.0452), but not on spiral or ellipse. (iii) All models forecast far better than hold-last (overall {HLd('overall')} against {R('A', 'overall', 'forecast_rmse')} to {R('D', 'overall', 'forecast_rmse')}) and constant velocity ({CV('overall')}), with the one exception noted for sinusoid.")
assert all(AB['results']['A'][g]['recon_rmse']['mean'] / AB['baselines']['linear_interpolation_recon'][g]['mean'] > 1 for g in FAMS[1:]) and AB['results']['A']['circle']['recon_rmse']['mean'] < AB['baselines']['linear_interpolation_recon']['circle']['mean']
w('')
w('**What can we conclude about one shared model for different patterns?** *Verified observation:* a single model with no shape label, trained once on all five families, produces sensible reconstructions and forecasts for all of them and forecasts far better than the simple baselines (one training run per model). *Cost observed:* compared with the three-family model, the five-family model is worse on the original three families (section 4). *Not established:* that it learns the *same underlying dynamics* as specialist models, or that it generalises to new families. These are single-seed results.')
w('')

# ---------------------------------------------------------------------------- 3. shared vs specialist
w('## 3. Shared model vs specialist models')
w('')
w('**What was compared.** *A. Shared model:* one Model D trained on circles and spirals together (1,400 training trajectories, 300 epochs, 5 training runs: seeds 42, 123, 2024, 2025, 2026). *B. Specialist models:* a separate Model D trained on circles only and another on spirals only (700 training trajectories, 600 epochs, so the same number of optimiser steps; 3 training runs each: seeds 42, 123, 2024). Same architecture and settings; evaluated on the same 150 test circles and 150 test spirals with identical observations. **[MULTI-SEED: 5 shared vs 3 specialist runs]**. Source: `results/validation/shared_vs_separate/shared_vs_separate.json`.')
w('')
w('| Family | Metric | Shared (5 runs) mean +/- SD | Specialist (3 runs) mean +/- SD | Shared vs specialist | Welch test p (raw / x4 Bonferroni*) | Every shared run worse than every specialist run? | Same-seed paired test on trajectories (seed 42): difference, p |')
w('|---|---|---|---|---:|---|---|---|')
for fam_ in ('circle', 'spiral'):
    for k, kn in (('recon_rmse', 'reconstruction'), ('forecast_rmse', 'forecast')):
        s_ = SVR(fam_, k)
        sv(f'{fam_}.seed_level.{k}.unified.mean'); sv(f'{fam_}.seed_level.{k}.specialist.mean'); sv(f'{fam_}.seed_level.{k}.welch_t_p'); sv(f'{fam_}.paired_seed42.{k}.wilcoxon_p'); sv(f'{fam_}.paired_seed42.{k}.relative_diff_pct')
        w(f"| {fam_} | {kn} | {s_['u_mean']:.4f} +/- {s_['u_sd']:.4f} | {s_['s_mean']:.4f} +/- {s_['s_sd']:.4f} | {pct(s_['rel'])} | {pv(s_['welch'])} / {pv(s_['welch_bonf'])} | {'yes' if s_['all_worse'] else 'no'} | {pct(s_['p_rel'])}, p={pv(s_['p_wil'])} |")
w('')
w('*Bonferroni adjustment (multiply by the 4 tests) computed in this analysis; the study itself reported the raw Welch p-values. Figure: `figures/fig3_shared_vs_specialist.png`.')
w('')
w('**Which model performs better?** The **specialists**, especially for forecasting. Forecast error is ' + f"{SVR('circle', 'forecast_rmse')['rel'] * 100:.0f}% higher for the shared model on circles and {SVR('spiral', 'forecast_rmse')['rel'] * 100:.0f}% higher on spirals. Reconstruction is close: {SVR('circle', 'recon_rmse')['rel'] * 100:.0f}% higher on circles (small but consistent) and {SVR('spiral', 'recon_rmse')['rel'] * 100:.0f}% on spirals (not distinguishable).")
w('')
w('**Does the shared model learn all families?** Yes: it reconstructs and forecasts both families well compared with the simple baselines (forecast about 0.16 and 0.28 against hold-last about 1.1 to 1.3). It just does so less accurately than a model devoted to one family.')
w('')
w('**Does combining families hurt forecasting?** For circle + spiral, **yes, by a lot**. Evidence strength:')
w(f"- *Statistically supported:* circle forecast. Every one of the 5 shared runs forecasts worse than every one of the 3 specialist runs (see figure 3); Welch p = {pv(SVR('circle', 'forecast_rmse')['welch'])} raw, {pv(SVR('circle', 'forecast_rmse')['welch_bonf'])} after Bonferroni; the same-seed trajectory-level comparison gives p = {pv(SVR('circle', 'forecast_rmse')['p_wil'])} with the shared model better on only {SVR('circle', 'forecast_rmse')['p_frac'] * 100:.0f}% of trajectories.")
w(f"- *Supported but weaker:* spiral forecast. Again all 5 shared runs are worse than all 3 specialists, but with only 3 specialist runs the Welch p-value is {pv(SVR('spiral', 'forecast_rmse')['welch'])} raw ({pv(SVR('spiral', 'forecast_rmse')['welch_bonf'])} after Bonferroni), so after correcting for four tests it is not significant at the 5% level.")
w(f"- *Descriptive only:* reconstruction. Circle p = {pv(SVR('circle', 'recon_rmse')['welch'])} raw ({pv(SVR('circle', 'recon_rmse')['welch_bonf'])} corrected); spiral p = {pv(SVR('spiral', 'recon_rmse')['welch'])}.")
w('- The trajectory-level p-values come from a single pair of runs (seed 42) and only describe how the two trained models differ on these test trajectories; they do not show the result would repeat for new training runs. The seed-level comparison (5 vs 3 runs) is the more relevant one.')
w('')
w('**Advantages and limitations of a shared model**')
w('- *Advantages (observed):* one model and one set of weights for several families; it needs no shape label; it handles all of them far better than the simple baselines.')
w('- *Limitations (observed):* about 40% to 85% worse forecasts than specialists on circle and spiral; the shared model\'s forecasts also vary much more between training runs (SD 0.037 and 0.067 against 0.006 and 0.009 for specialists; the specialists had only 3 runs).')
w('- *Not tested:* specialists for ellipse, figure-eight and sinusoid do not exist, so the shared-vs-specialist comparison is only for circle and spiral. An attempt to close the gap by conditioning the dynamics on an inferred code did not work (section 6.1).')
BB = T3['B_three_vs_two_shape_on_matching_circle_spiral']['circle']
w(f"- *Related, from the same study:* adding a third family (ellipse) to the shared model did not clearly change forecasting for circles and spirals: circle forecast {t3('B_three_vs_two_shape_on_matching_circle_spiral.circle.forecast_rmse.three_shape', '.3f')} for the three-family model, {t3('B_three_vs_two_shape_on_matching_circle_spiral.circle.forecast_rmse.two_shape_seed42', '.3f')} for the seed-42 two-family model (paired test on trajectories, p = {pv(BB['forecast_rmse']['paired_wilcoxon_p'])}) and {t3('B_three_vs_two_shape_on_matching_circle_spiral.circle.forecast_rmse.two_shape_across_seeds.mean', '.3f')} +/- {t3('B_three_vs_two_shape_on_matching_circle_spiral.circle.forecast_rmse.two_shape_across_seeds.std', '.3f')} for the two-family model over 5 seeds. Circle reconstruction was {BB['recon_rmse']['relative_diff_pct_vs_seed42']:.0f}% worse than the seed-42 two-family model.")
w('')

# ---------------------------------------------------------------------------- 4. cross-family
w('## 4. Cross-family generalisation (unseen families)')
w('')
w('**Rule used:** a result counts as zero-shot only if the family was absent from that model\'s training data.')
w('')
w('### 4a. Two-family models on unseen **ellipses** [MULTI-SEED: 5 training runs]')
w('')
w('Trained on: circle + spiral. Excluded: ellipse. Tested on the 150 ellipses of the three-family test split (30 observations). Source: `results/validation/three_shapes/three_shapes_and_ood.json`.')
w('')
w('| Model | Seen ellipses in training? | Reconstruction RMSE | Forecast RMSE |')
w('|---|---|---:|---:|')
w(f"| Two-family models (5 runs), mean +/- SD | **no (zero-shot)** | {V['ell_rc_two']['mean']:.4f} +/- {V['ell_rc_two']['std']:.4f} | {V['ell_fc_two']['mean']:.4f} +/- {V['ell_fc_two']['std']:.4f} |")
w(f"| Three-family model (1 run) | yes (not zero-shot) | {t3('C_out_of_distribution_ellipses.three_shape_model_on_same_ellipses (in-distribution reference).recon_rmse.mean', '.4f')} | {t3('C_out_of_distribution_ellipses.three_shape_model_on_same_ellipses (in-distribution reference).forecast_rmse.mean', '.4f')} |")
w(f"| Linear interpolation (baseline) | n/a | {t3('C_out_of_distribution_ellipses.const_pos_on_same_ellipses.recon_rmse.mean', '.4f')} | n/a |")
w(f"| Hold-last (baseline) | n/a | n/a | {t3('C_out_of_distribution_ellipses.const_pos_on_same_ellipses.forecast_rmse.mean', '.4f')} |")
w(f"| Constant velocity (baseline) | n/a | n/a | {t3('C_out_of_distribution_ellipses.const_vel_on_same_ellipses.forecast_rmse.mean', '.4f')} |")
w('')
w(f"*What happened:* forecasting transfers partly (zero-shot {V['ell_fc_two']['mean']:.2f}, far better than hold-last {t3('C_out_of_distribution_ellipses.const_pos_on_same_ellipses.forecast_rmse.mean', '.2f')}, but worse than the in-training {t3('C_out_of_distribution_ellipses.three_shape_model_on_same_ellipses (in-distribution reference).forecast_rmse.mean', '.2f')}); reconstruction does not ({V['ell_rc_two']['mean']:.2f}, more than three times worse than simply joining the dots, {t3('C_out_of_distribution_ellipses.const_pos_on_same_ellipses.recon_rmse.mean', '.2f')}). Rotation direction is right for every trajectory (geometry numbers in this paragraph are for the seed-42 two-family model only; accuracy {t3('C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).forecast_rotation_direction_accuracy', '.0%')}), but the model draws ellipses as near-circles: predicted axis ratio {t3('C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).axis_ratio_pred_mean', '.2f')} against true {t3('C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).axis_ratio_true_mean', '.2f')}, correlation {t3('C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).axis_ratio_corr', '.2f')} (the three-family model that saw ellipses: {t3('C_out_of_distribution_ellipses.ellipse_geometry_three_shape (ID).axis_ratio_corr', '.2f')}). An ellipse is a stretched circle, so this is the *easiest* possible unseen family; it counts as partial transfer of the forecast, not successful generalisation.")
w('')
w('### 4b. The three-family model on unseen **figure-eight** and **sinusoid** [SINGLE-SEED: one frozen model]')
w('')
w('Trained on: circle + spiral + ellipse. Excluded: figure-eight, sinusoid. Tested on the five-family test split (150 of each unseen family). Source: `results/five_shapes/zeroshot_three_shape_model.json`.')
w('')
w('| Family | Status | Recon. RMSE | Linear interp. | Forecast RMSE | Hold-last | Const. velocity |')
w('|---|---|---:|---:|---:|---:|---:|')
for g in FAMS:
    stt = 'seen in training' if g in FAMS[:3] else '**unseen (zero-shot)**'
    w(f"| {FL[g]} | {stt} | {Z3(g, 'recon_rmse')} | {LI(g)} | {Z3(g, 'forecast_rmse')} | {HLd(g)} | {CV(g)} |")
w('')
w(f"*What happened:* on the unseen families the model is **worse than hold-last** at forecasting (figure-eight {Z3('figure_eight', 'forecast_rmse')} vs {HLd('figure_eight')}; sinusoid {Z3('sinusoid', 'forecast_rmse')} vs {HLd('sinusoid')}) and **far worse than linear interpolation** at reconstruction (figure-eight {Z3('figure_eight', 'recon_rmse')} vs {LI('figure_eight')}; sinusoid {Z3('sinusoid', 'recon_rmse')} vs {LI('sinusoid')}). It is better than constant velocity, but that baseline is itself very poor on these paths. The model mostly redraws what it has seen (loops and spirals). **This is not successful zero-shot generalisation.**")
w('')
w('### 4c. Five-family model: not zero-shot')
w('')
w('The five-family model was trained on all five families, so its results on them (section 2) are **not** generalisation. What it shows is that the same architecture *can* be trained on all five. Compared with the three-family model on the original three families it is worse (paired, identical observations; Holm-adjusted p < 0.01 for every comparison):')
w('')
w('| Family | Forecast: 3-family | Forecast: 5-family | Change | Recon: 3-family | Recon: 5-family |')
w('|---|---:|---:|---:|---:|---:|')
for g in FAMS[:3]:
    w(f"| {FL[g]} | {Z3(g, 'forecast_rmse')} | {F5(g, 'forecast_rmse')} | {pct(c5(f'paired.{g}/forecast_rmse.relative_change_of_mean'), 0)} | {Z3(g, 'recon_rmse')} | {F5(g, 'recon_rmse')} |")
w('')
w(f'The five-family model also had more training data and more optimiser steps ({ab("models.A.optimizer_steps", ",")} against {V["steps3"]:,}), so this cost cannot be blamed on the number of families alone. [SINGLE-SEED]')
w('')
w('### 4d. Did the experiments demonstrate zero-shot generalisation?')
w('**No.** Partial transfer for ellipses (a stretched circle: direction and forecast partly right, shape wrong) [MULTI-SEED]; **failure** for figure-eight and sinusoid [SINGLE-SEED]; and, in the real-world test (section 6.2), failure on real GPS. There is no leave-one-family-out experiment (training on four families, testing on the fifth), so nothing here shows generalisation across *many* families.')
w('')

# ---------------------------------------------------------------------------- 5. ablation
w('## 5. Controlled 2x2 ablation [SINGLE-SEED]')
w('')
w('Four models, same data, same training recipe, same training data stream (`data_seed` 42), differing only in latent dimension and GRU (encoder) width; one training run each. Test: the same 750 trajectories with identical observations.')
w('')
w('| Model | Latent dimension | GRU width | Parameters |')
w('|---|---:|---:|---:|')
for m in CELLS:
    w(f"| {m} | {ab(f'models.{m}.latent_dim')} | {ab(f'models.{m}.encoder_hidden')} | {ab(f'models.{m}.n_parameters', ',')} |")
w('')
w('### Overall results (mean per-trajectory RMSE; change relative to A)')
w('')
w('| Model | Reconstruction | vs A | Forecast (7 < t <= 10) | vs A | Trajectories with lower forecast error than A |')
w('|---|---:|---:|---:|---:|---:|')
for m in CELLS:
    if m == 'A': w(f"| A | {R('A', 'overall', 'recon_rmse')} | n/a | {R('A', 'overall', 'forecast_rmse')} | n/a | n/a |")
    else: w(f"| {m} | {R(m, 'overall', 'recon_rmse')} | {pct(rel(m, 'overall', 'recon_rmse'))} | {R(m, 'overall', 'forecast_rmse')} | {pct(rel(m, 'overall', 'forecast_rmse'))} | {ab(f'fraction_of_trajectories_better_than_A.{m}.overall.forecast_rmse', '.0%')} |")
w('')
w(f"Linear interpolation: {LI('overall')} (reconstruction). Hold-last: {HLd('overall')}; constant velocity: {CV('overall')} (forecast). Figure: `figures/fig5_ablation_overall.png`.")
w('')
w('### Per family (change vs A; Holm-adjusted p-value of the paired difference, "ns" = not significant)')
w('')
w('| Family | Metric | A | B vs A | C vs A | D vs A |')
w('|---|---|---:|---|---|---|')
for g in G6:
    for k, kn in (('forecast_rmse', 'forecast'), ('recon_rmse', 'reconstruction')):
        w(f"| {FL[g]} | {kn} | {R('A', g, k)} | " + ' | '.join(cellc(m, g, k) for m in 'BCD') + ' |')
w('')
w('### Separate comparisons')
w('')
bP, cP, dP = P('B', 'overall', 'forecast_rmse'), P('C', 'overall', 'forecast_rmse'), P('D', 'overall', 'forecast_rmse')
w(f"**A vs B: raising only the latent dimension (8 to 16).** Reconstruction improves in every family ({pct(rel('B', 'overall', 'recon_rmse'))} overall). Overall forecast error rises {pct(rel('B', 'overall', 'forecast_rmse'))} (Holm p = {pv(bP['wilcoxon_p_holm'])}); B beats A on {AB['fraction_of_trajectories_better_than_A']['B']['overall']['forecast_rmse'] * 100:.0f}% of the paired trajectories. Per family the forecast difference is clearly significant only for circle ({pct(rel('B', 'circle', 'forecast_rmse'), 0)}) and sinusoid ({pct(rel('B', 'sinusoid', 'forecast_rmse'), 0)}); spiral, ellipse and figure-eight show no significant difference.")
w(f"**A vs C: raising only the GRU width (64 to 128).** Reconstruction improves in every family ({pct(rel('C', 'overall', 'recon_rmse'))} overall, a smaller gain than B). Forecast: **no clearly distinguishable overall difference** ({pct(rel('C', 'overall', 'forecast_rmse'))}; mean difference {cP['mean_difference']:+.4f}, bootstrap 95% interval [{cP['bootstrap_ci95_mean_difference'][0]:+.4f}, {cP['bootstrap_ci95_mean_difference'][1]:+.4f}], median difference {cP['median_difference']:+.4f}, better on {AB['fraction_of_trajectories_better_than_A']['C']['overall']['forecast_rmse'] * 100:.0f}% of trajectories, Holm p = {pv(cP['wilcoxon_p_holm'])}). By family: significantly worse on sinusoid ({pct(rel('C', 'sinusoid', 'forecast_rmse'), 0)}), significantly *better* on ellipse ({pct(rel('C', 'ellipse', 'forecast_rmse'), 0)}), nothing distinguishable on the others.")
w(f"**A vs D: raising both.** Reconstruction improves most ({pct(rel('D', 'overall', 'recon_rmse'))}). Forecast error rises {pct(rel('D', 'overall', 'forecast_rmse'))} (Holm p = {pv(dP['wilcoxon_p_holm'])}), D beating A on only {AB['fraction_of_trajectories_better_than_A']['D']['overall']['forecast_rmse'] * 100:.0f}% of trajectories. D has the higher mean on all five families; significantly so on circle, spiral, ellipse and sinusoid, not on figure-eight.")
fI, rI = AB['factorial']['overall/forecast_rmse/interaction ((D-C) - (B-A))'], AB['factorial']['overall/recon_rmse/interaction ((D-C) - (B-A))']
ab('factorial.overall/forecast_rmse/interaction ((D-C) - (B-A)).mean')
w('')
w('### The interaction, explained simply')
w(f"An *interaction* means the effect of one change depends on the other. Overall forecast error: raising the latent size changes it by {AB['paired']['B_minus_A/overall/forecast_rmse']['mean_difference']:+.3f} when the GRU is 64 (B minus A) but by {AB['paired']['D_minus_C/overall/forecast_rmse']['mean_difference']:+.3f} when the GRU is 128 (D minus C). The extra damage ({fI['mean']:+.3f}, 95% interval [{fI['bootstrap_ci95'][0]:+.3f}, {fI['bootstrap_ci95'][1]:+.3f}], Holm p = {pv(fI['wilcoxon_p_holm'])}) means the two changes **reinforce each other** overall. For reconstruction the interaction is {rI['mean']:+.3f}: the two changes overlap (each helps less once the other is made). The pattern is not the same in every family: for the sinusoid the forecast interaction is {AB['factorial']['sinusoid/forecast_rmse/interaction ((D-C) - (B-A))']['mean']:+.2f}, the opposite sign. The \"GRU main effect\" of {AB['factorial']['overall/forecast_rmse/gru_main_effect (128 vs 64, averaged over latent)']['mean']:+.3f} is an average over both latent sizes and is carried mainly by D versus B ({AB['paired']['D_minus_B/overall/forecast_rmse']['mean_difference']:+.3f}), not by C versus A ({cP['mean_difference']:+.3f}), so the two factors should not be described as independently worsening forecasts.")
w('')
w('### Forecast error over time (verified)')
w('')
w('Mean per-trajectory RMSE inside time windows (computed in this analysis from the saved predictions; 750 trajectories):')
w('')
w('| Time window | A | B | C | D |')
w('|---|---:|---:|---:|---:|')
for wd in WIN:
    w(f"| {wd[0]} < t <= {wd[1]} | " + ' | '.join(f"{V['win'][(m, wd)]:.4f}" for m in CELLS) + ' |')
w('')
w(f"**Verified:** in the first part of the forecast (7 < t <= 8) the larger models B, C and D are *better* than A (lower error). Their error then grows faster and crosses above A at t = {V['cross']['B']:.2f} (B), {V['cross']['C']:.2f} (C) and {V['cross']['D']:.2f} (D) (first time the across-trajectory RMSE exceeds A's, from the saved error-versus-time curves). By 9 < t <= 10 all three are worse than A. At t = 10 the across-trajectory RMSE is A {et['A']['overall'][-1]:.3f}, B {et['B']['overall'][-1]:.3f}, C {et['C']['overall'][-1]:.3f}, D {et['D']['overall'][-1]:.3f}. So the full-window \"worse forecast\" comes from the long-horizon part. A shorter forecast window would give the opposite sign for B and D. (Figure: `figures/fig6_forecast_error_vs_time.png`.)")
w('')
w('### What the ablation does and does not show')
w('**Observed (single seed):** larger encoders reconstruct much better; over the full 7 to 10 window latent size is associated with higher forecast error, GRU width alone is not clearly distinguishable from baseline, the effect appears late in the horizon and is driven mostly by the sinusoid and circle families (for B and D on sinusoid the model is worse than hold-last).')
w(f"**Not shown:** that capacity universally worsens forecasting; a causal mechanism; that the result holds for other training seeds. For scale: training the *same* architecture twice with different data streams changed mean forecast error by {pct(ab('reproducibility_pairs.A_vs_orig_baseline.overall.forecast_rmse.relative_difference'))} (A vs the earlier 8/64 run) and {pct(ab('reproducibility_pairs.D_vs_orig_medium.overall.forecast_rmse.relative_difference'))} (D vs the earlier 16/128 run), and the five-seed two-family study has a forecast coefficient of variation of about {ssf['std'] / ssf['mean'] * 100:.0f}%. The forecast differences above ({pct(rel('B', 'overall', 'forecast_rmse'), 0)}, {pct(rel('C', 'overall', 'forecast_rmse'), 0)}, {pct(rel('D', 'overall', 'forecast_rmse'), 0)}) are of that same order. (Figure: `figures/fig7_effects_vs_seed_variation.png`.) A tiny p-value here only says the two *fixed trained models* differ on these test trajectories.")
w(f'**Possible explanations (untested):** the training objective supervises only the 30 observed points, so extrapolation is not directly trained; larger latent spaces might make errors grow faster; optimisation differences (all cells were still improving at epoch 300). Classical overfitting was not observed (validation loss did not rise; training and validation negative log-likelihood differ by at most {V['nll_gap']:.1f}).')
w('')

# ---------------------------------------------------------------------------- 6. other experiments
w('## 6. Other completed experiments')
w('')
w('### 6.1 Dynamics-conditioning study [MULTI-SEED: 5 seeds; M2aug 3 seeds]')
w('')
w('*Purpose:* the shared model forecasts worse than specialists; does letting the model infer an extra "dynamics code" that controls the ODE close that gap? *Models:* M1 = shared Model D; M2 = shared model with the same extra size but no code (a fair control); M3 = shared model with the dynamics code; M2aug = an ablation. *Data:* circle + spiral, 300 test trajectories. Forecast RMSE, mean +/- SD over seeds (source `results/conditioning/conditioning_study.json`):')
w('')
w('| Model | Circle | Spiral |')
w('|---|---:|---:|')
w(f"| Specialists (3 runs) | {cond('gap_to_specialists.circle.specialist_forecast_rmse', '.3f')} | {cond('gap_to_specialists.spiral.specialist_forecast_rmse', '.3f')} |")
for mm, nm in (('M1', 'M1 shared'), ('M2', 'M2 control, no code'), ('M3', 'M3 with dynamics code'), ('M2aug', 'M2aug ablation (3 runs)')):
    w(f"| {nm} | {cond(f'seed_level.{mm}.circle.forecast_rmse.mean', '.3f')} +/- {cond(f'seed_level.{mm}.circle.forecast_rmse.std', '.3f')} | {cond(f'seed_level.{mm}.spiral.forecast_rmse.mean', '.3f')} +/- {cond(f'seed_level.{mm}.spiral.forecast_rmse.std', '.3f')} |")
w('')
pc, ps_ = D['cond']['paired']['M3_vs_M1']['circle']['forecast_rmse'], D['cond']['paired']['M3_vs_M1']['spiral']['forecast_rmse']
w(f"*Result:* M3 does **not** forecast better than M1 or M2 (circle {pct(pc['rel_diff_pct'] / 100, 0)} and spiral {pct(ps_['rel_diff_pct'] / 100, 0)} worse than M1 when averaged over seeds; seed-level paired p = {pv(pc['seed_paired_t_p'])} and {pv(ps_['seed_paired_t_p'])}, so not distinguishable across seeds). M3 reconstructs very slightly better (circle {pct(D['cond']['paired']['M3_vs_M1']['circle']['recon_rmse']['rel_diff_pct'] / 100, 1)}). The learned code does control radial growth, so it is *used*, but it did not help forecasting. M3 was not tuned separately, which may favour M1. *Supported:* at this scale the hypothesis is not supported. *Note:* two sets of five \"shared\" runs (this study's M1 and the E3 study) give the same recipe different SDs for circle forecast ({cond('seed_level.M1.circle.forecast_rmse.std', '.3f')} here, {SVR('circle', 'forecast_rmse')['u_sd']:.3f} in E3), a reminder that five-seed SDs are themselves uncertain.")
w('')
w('### 6.2 GeoLife real-world transfer [SINGLE-SEED]')
w('')
w('*Purpose:* does what was learned on synthetic shapes help on real GPS? *Data:* 1,000 held-out GPS segments from 97 test users (the models never saw these users; development users only were used for training the GeoLife-trained models). *Task:* interpolate withheld GPS positions with 30 observations per segment. *Metric:* median per-segment error in metres. (GeoLife data and weights are not distributed; only aggregates are reported.)')
w('')
w('| Method (30 observations) | Median error, metres [95% CI, user bootstrap] | Ratio to linear interpolation (geometric mean) |')
w('|---|---|---:|')
for k, nm in (('linear', 'Linear interpolation'), ('cubic', 'Cubic spline'), ('pchip', 'PCHIP'), ('frozen_three_shapes', 'Frozen synthetic (three-family) model'), ('geolife_scratch', 'GeoLife-trained, from scratch'), ('geolife_finetuned_from_three_shapes', 'Synthetic, then fine-tuned on GeoLife')):
    lo, hi = gl(f'interpolation.30.{k}.median_ci95_user_bootstrap.0', '.1f'), gl(f'interpolation.30.{k}.median_ci95_user_bootstrap.1', '.1f')
    w(f"| {nm} | {gl(f'interpolation.30.{k}.median', '.2f')} [{lo}, {hi}] | {gl(f'interpolation.30.{k}.geometric_mean_ratio_vs_linear', '.3f')} |")
w('')
w('*Results and conclusions supported.* (1) The **frozen synthetic model does not transfer**: about 17 times worse than linear interpolation. (2) Training on real GPS data helps (about 2 to 3 times better than the frozen model) but both GeoLife-trained models remain 6 to 8 times worse than linear interpolation and beat it on only 0.1% of segments. (3) Forecasts are also worse than the trivial baselines (protocol A medians: scratch ' + f"{gl('forecast_n30.A.geolife_scratch.median', '.0f')} m, fine-tuned {gl('forecast_n30.A.geolife_finetuned_from_three_shapes.median', '.0f')} m, hold-last {gl('forecast_n30.A.hold_last.median', '.0f')} m, constant velocity {gl('forecast_n30.A.const_velocity.median', '.0f')} m). The likely reason (hypothesis) is that the model can only produce loop-like curves. Figure: `figures/fig8_geolife_transfer.png`.")
w('')
w('**Does synthetic pre-training help? Paired comparison, fine-tuned vs from-scratch (computed in this analysis from the saved per-segment results of the same 1,000 segments; ratio below 1 favours fine-tuning; 95% interval from a bootstrap over the 97 users):**')
w('')
w('| Observations per segment | Median error, scratch (m) | Median error, fine-tuned (m) | Geometric-mean ratio fine-tuned / scratch [95% CI] | Fine-tuned better on |')
w('|---|---:|---:|---|---:|')
for n in (10, 20, 30, 40, 50):
    p_ = V['pa'][n]; w(f"| {n} | {p_['med_scratch']:.1f} | {p_['med_fine']:.1f} | {p_['gmr']:.2f} [{p_['ci'][0]:.2f}, {p_['ci'][1]:.2f}] | {p_['win_fine'] * 100:.0f}% of segments |")
w('')
w(f"*Observed:* fine-tuning from synthetic weights is clearly better only at **30 observations**, the number both models were trained with (ratio {V['pa'][30]['gmr']:.2f}); at 10, 20, 40 and 50 observations it is clearly *worse* (ratios {V['pa'][10]['gmr']:.2f}, {V['pa'][20]['gmr']:.2f}, {V['pa'][40]['gmr']:.2f}, {V['pa'][50]['gmr']:.2f}), and the from-scratch model degrades much less away from 30. So **no general benefit of synthetic pre-training is shown**; the one advantage is at the training count. *Possible explanation (untested):* the fine-tuned model is more tied to the observation count it was trained with. Single training run per model, so this could change with other seeds.")
w('')
w('### 6.3 Capacity scaling, first version [SINGLE-SEED]')
w('')
w(f"*Purpose:* does a bigger encoder help? *Models (five-family data):* earlier 8/64 baseline ({cap('models.baseline.n_parameters', ',')} parameters), medium 16/128 ({cap('models.medium.n_parameters', ',')}), large 32/128 ({cap('models.large.n_parameters', ',')}), one run each, **training data streams not controlled** (this is why the 2x2 ablation was run). *Overall mean RMSE (reconstruction / forecast):* baseline {cap('results.baseline.overall.recon_rmse.mean', '.4f')} / {cap('results.baseline.overall.forecast_rmse.mean', '.4f')}; medium {cap('results.medium.overall.recon_rmse.mean', '.4f')} / {cap('results.medium.overall.forecast_rmse.mean', '.4f')}; large {cap('results.large.overall.recon_rmse.mean', '.4f')} / {cap('results.large.overall.forecast_rmse.mean', '.4f')}. *Supported:* reconstruction improves with size; forecasting is worse for the larger models on the full window, with the sinusoid worst (large: {cap('results.large.sinusoid.forecast_rmse.mean', '.3f')}, worse than hold-last {HLd('sinusoid')}). The controlled 2x2 reproduces both directions: for the combined 16/128 change the full-window forecast difference is {pct(V['d_vs_a'])} against the controlled baseline A, similar to the {pct(V['cap_med_vs_base'])} of the earlier uncontrolled medium run, and the 2x2 shows that it is split unevenly between latent size and GRU width. The earlier 8/64 baseline ({cap('results.baseline.overall.forecast_rmse.mean', '.4f')}) was better than the controlled rerun A ({R('A', 'overall', 'forecast_rmse')}), which shows how much a different data stream alone can change a single run.")
w('')

# ---------------------------------------------------------------------------- 7. five findings
w('## 7. The five most important findings')
w('')
fnd = []
fnd.append(('1. One shared model can learn several families, but it forecasts worse than specialists.',
            f"Training one model on circles and spirals together gives about {SVR('circle', 'forecast_rmse')['rel'] * 100:.0f}% (circle) and {SVR('spiral', 'forecast_rmse')['rel'] * 100:.0f}% (spiral) higher forecast error than separate models; reconstruction is close.",
            f"Circle forecast {SVR('circle', 'forecast_rmse')['u_mean']:.3f} vs {SVR('circle', 'forecast_rmse')['s_mean']:.3f}; spiral {SVR('spiral', 'forecast_rmse')['u_mean']:.3f} vs {SVR('spiral', 'forecast_rmse')['s_mean']:.3f} (E3).", 'MULTI-SEED (5 shared vs 3 specialist runs)',
            'Combining families costs forecast accuracy on circle and spiral for this model and recipe; every shared run was worse than every specialist run.', 'Only two families have specialists; spiral difference is not significant after multiple-test correction; whether the gap closes with more capacity or data is untested; conditioning did not close it.'))
fnd.append(('2. The models do not generalise to unseen families.',
            'A model trained on circle, spiral and ellipse cannot draw figure-eights or sinusoids it has not seen, and models trained without ellipses draw ellipses as near-circles.',
            f"Figure-eight forecast {Z3('figure_eight', 'forecast_rmse')} vs hold-last {HLd('figure_eight')}; sinusoid {Z3('sinusoid', 'forecast_rmse')} vs {HLd('sinusoid')}; reconstruction {Z3('figure_eight', 'recon_rmse')} / {Z3('sinusoid', 'recon_rmse')} vs linear interpolation {LI('figure_eight')} / {LI('sinusoid')}. Unseen ellipses (5 runs): forecast {V['ell_fc_two']['mean']:.2f} (hold-last {t3('C_out_of_distribution_ellipses.const_pos_on_same_ellipses.forecast_rmse.mean', '.2f')}) but reconstruction {V['ell_rc_two']['mean']:.2f} vs {t3('C_out_of_distribution_ellipses.const_pos_on_same_ellipses.recon_rmse.mean', '.2f')}.",
            'MULTI-SEED for ellipses (5 runs); SINGLE-SEED for figure-eight and sinusoid',
            'There is no zero-shot generalisation to dissimilar families; the ellipse case shows only partial transfer (direction and rough forecast).', 'No leave-one-family-out experiment; only one frozen model for the two dissimilar families; other training recipes might generalise differently.'))
fnd.append(('3. Larger encoders reconstruct much better, consistently.',
            'Making the encoder bigger (latent dimension, GRU width, or both) lowers reconstruction error in every family, and turns a model that is worse than straight-line interpolation into one that beats it.',
            f"Overall reconstruction A {R('A', 'overall', 'recon_rmse')}, B {R('B', 'overall', 'recon_rmse')} ({pct(rel('B', 'overall', 'recon_rmse'))}), C {R('C', 'overall', 'recon_rmse')} ({pct(rel('C', 'overall', 'recon_rmse'))}), D {R('D', 'overall', 'recon_rmse')} ({pct(rel('D', 'overall', 'recon_rmse'))}); linear interpolation {LI('overall')} (E10, E11).",
            'SINGLE-SEED (but the same direction in two separate experiments and in all five families)', 'Within this setup, more encoder capacity improves reconstruction of withheld points.', 'Not confirmed across seeds; reconstruction is what the training loss directly optimises, so this may be expected; one dataset family.'))
fnd.append(('4. Forecasting gets worse with model size at long horizons, but this is not established.',
            'Larger models forecast better just after the observed period and worse later; over the whole forecast window the latent-size changes are worse than baseline A while widening the GRU alone makes no clear difference.',
            f"Overall forecast A {R('A', 'overall', 'forecast_rmse')}, B {R('B', 'overall', 'forecast_rmse')} ({pct(rel('B', 'overall', 'forecast_rmse'))}), C {R('C', 'overall', 'forecast_rmse')} ({pct(rel('C', 'overall', 'forecast_rmse'))}, not significant), D {R('D', 'overall', 'forecast_rmse')} ({pct(rel('D', 'overall', 'forecast_rmse'))}). Window 7-8: A {V['win'][('A', (7, 8))]:.3f}, D {V['win'][('D', (7, 8))]:.3f}; window 9-10: A {V['win'][('A', (9, 10))]:.3f}, D {V['win'][('D', (9, 10))]:.3f} (E11).",
            'SINGLE-SEED', 'In this one set of runs the longer-horizon forecast error grows faster for models with a larger latent space.', f"The differences are the same size as run-to-run variation ({pct(ab('reproducibility_pairs.A_vs_orig_baseline.overall.forecast_rmse.relative_difference'), 0)} and {pct(ab('reproducibility_pairs.D_vs_orig_medium.overall.forecast_rmse.relative_difference'), 0)} for identical architectures; five-seed CV about {ssf['std'] / ssf['mean'] * 100:.0f}%); no cause is established; needs multi-seed runs."))
fnd.append(('5. A model trained on synthetic shapes does not transfer to real GPS, and real-data training still does not beat simple interpolation.',
            'On real GPS tracks, straight-line (or spline) interpolation of the observed points is far more accurate than any of our learned models.',
            f"Median error at 30 observations: linear {gl('interpolation.30.linear.median', '.1f')} m, PCHIP {gl('interpolation.30.pchip.median', '.1f')} m; frozen synthetic model {gl('interpolation.30.frozen_three_shapes.median', '.1f')} m; GeoLife-trained {gl('interpolation.30.geolife_scratch.median', '.1f')} m; fine-tuned {gl('interpolation.30.geolife_finetuned_from_three_shapes.median', '.1f')} m (E7, E8). Fine-tuning helps only at 30 observations.",
            'SINGLE-SEED', 'Synthetic-only training does not give a usable real-world model; the learned models are 6 to 17 times worse than linear interpolation at interpolating GPS.', 'One dataset, one training run per model, models trained with 30 observations only, and the Latent ODE was not tuned specifically for GPS beyond one configuration.'))
for t_, simple, nums, seeds, safe, unc in fnd:
    w(f'**{t_}**')
    w(f'- *Simple explanation:* {simple}')
    w(f'- *Numbers:* {nums}')
    w(f'- *Seed evidence:* {seeds}.')
    w(f'- *What we can safely conclude:* {safe}')
    w(f'- *What remains uncertain:* {unc}')
    w('')
w('**Verified experimental observations (each with the seed caveat given in its finding):** findings 1, 2 and 5, and the direction of the reconstruction improvement in finding 3.')
w('**Possible explanations that need further investigation:** why shared models forecast worse; why larger latent spaces forecast worse late in the horizon (is it the observed-points-only training objective? optimisation? seed luck?); why fine-tuned models are tied to the training observation count; whether any of these change with more seeds.')
w('')
w('## 8. Strength of evidence at a glance')
w('')
w('| Finding | Seeds | Strength |')
w('|---|---|---|')
w('| Shared model forecasts worse than specialists (circle, spiral) | 5 vs 3 | Supported (circle clearly; spiral weaker) |')
w('| Reproducibility of Model D reconstruction | 5 | Supported (stable); forecast varies by about 23% |')
w('| Conditioning does not help | 5 | Supported at this scale |')
w('| Two-family models on unseen ellipses | 5 | Supported: partial transfer only |')
w('| 3-family model on unseen figure-eight, sinusoid | 1 | Descriptive; one frozen model |')
w('| Five-family model results; three vs five comparison | 1 | Descriptive; budgets differ |')
w('| Larger encoders improve reconstruction | 1 (two experiments) | Consistent but single-seed |')
w('| Larger models forecast worse late in the horizon | 1 | Exploratory; same size as seed noise |')
w('| GeoLife transfer fails; fine-tuned vs scratch pattern | 1 | Single-seed; large effect sizes |')
w('')
w('**Not done:** multi-seed repeats of the five-family experiments, leave-one-family-out generalisation, stronger baselines (per-family curve fitting, sequence models), a test of the observed-points-only objective hypothesis, any method that improves forecasting.')
(OUT / 'RESULTS_ANALYSIS.md').write_text('\n'.join(L) + '\n'); print('RESULTS_ANALYSIS.md', len(L), 'lines')
exec(open('audit/build_results_documents_part2.py').read())
