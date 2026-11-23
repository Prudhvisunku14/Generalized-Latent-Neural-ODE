#!/usr/bin/env python3
"""Writes reports/five_shapes_report.md from SAVED results only (no model is run, nothing is trained or modified).

Sources: results/five_shapes/{compare_three_vs_five.json, zeroshot_three_shape_model.json, dataset_checks.json},
data/processed/five_shapes/metadata.json, results/capacity_scaling/capacity_scaling_results.json (training facts of the five-shape
model), results/validation/three_shapes/three_shapes_and_ood.json (stored three-shape evaluation). Run from the project root.
"""
import json, math
from pathlib import Path

C = json.load(open('results/five_shapes/compare_three_vs_five.json')); Zs = json.load(open('results/five_shapes/zeroshot_three_shape_model.json'))
D = json.load(open('results/five_shapes/dataset_checks.json')); M = json.load(open('data/processed/five_shapes/metadata.json'))
CAP = json.load(open('results/capacity_scaling/capacity_scaling_results.json'))['models']; ST = json.load(open('results/validation/three_shapes/three_shapes_and_ood.json'))
FAMS = ['circle', 'spiral', 'ellipse', 'figure_eight', 'sinusoid']; ORIG = FAMS[:3]; G = FAMS + ['overall']
fm = lambda x: f'{x:.4f}'; pct = lambda x: f'{x * 100:+.0f}%'; pv = lambda p: f'{p:.2g}'
r3 = lambda g, k, s='mean': C['three_shape_model'][g][k][s]; r5 = lambda g, k, s='mean': C['five_shape_model'][g][k][s]
bl = lambda b, g, k: C[b][g][k]['mean']; pr = lambda g, k: C['paired'][f'{g}/{k}']
five, three = CAP['baseline'], CAP['three_shape']
ST_in = ST['A_three_shape_model_in_distribution']['model']
steps = lambda n: math.ceil(n / 32)

# guards for the qualitative statements made in the text
assert all(r5(g, 'forecast_rmse') < bl('const_pos', g, 'forecast_rmse') and r5(g, 'forecast_rmse') < bl('const_vel', g, 'forecast_rmse') for g in FAMS), 'five-shape beats naive forecast baselines'
assert all(r3(g, 'forecast_rmse') < r5(g, 'forecast_rmse') for g in ORIG) and all(r5(g, 'forecast_rmse') < r3(g, 'forecast_rmse') for g in FAMS[3:]), 'direction of the three/five comparison'
assert all(pr(g, 'forecast_rmse')['wilcoxon_p_holm'] < .05 and pr(g, 'recon_rmse')['wilcoxon_p_holm'] < .05 for g in FAMS), 'all paired comparisons significant'
assert all(r3(g, 'forecast_rmse') < bl('const_vel', g, 'forecast_rmse') for g in FAMS[3:]), 'zero-shot beats constant velocity'
assert r3('figure_eight', 'forecast_rmse') > bl('const_pos', 'figure_eight', 'forecast_rmse') and r3('sinusoid', 'forecast_rmse') > bl('const_pos', 'sinusoid', 'forecast_rmse'), 'zero-shot worse than hold-last'
assert all(r3(g, 'recon_rmse') > bl('const_pos', g, 'recon_rmse') for g in FAMS[3:]), 'zero-shot reconstruction worse than linear interpolation'
worse_recon = [g for g in FAMS if r5(g, 'recon_rmse') > bl('const_pos', g, 'recon_rmse')]
assert worse_recon == FAMS, worse_recon
CIRC_REL = r5('circle', 'recon_rmse') / bl('const_pos', 'circle', 'recon_rmse') - 1
assert 0 < CIRC_REL < 0.01
OTHERS_REL = [r5(g, 'recon_rmse') / bl('const_pos', g, 'recon_rmse') - 1 for g in FAMS[1:]]
assert D['failures'] == [] and D['n_unique_ids'] == 5000

L = []; w = L.append
w('# Five trajectory families: dataset, zero-shot test of the three-shape model, and a five-shape model')
w('')
w('Date: 2026-11-21. Written for: the project owner and supervisor. All numbers come from results already saved in this repository (listed in section 10); nothing was trained or re-run for this report.')
w('')
w('## 1. Summary')
w('')
w('We added two trajectory families, **figure-eight** and **sinusoid**, to circle, spiral and ellipse (1,000 trajectories each, same time grid, noise and observation protocol), then asked two questions:')
w('')
w(f"1. **Does the frozen three-shape model handle the two families it never saw (zero-shot)?** No. On figure-eight and sinusoid its forecasts ({fm(r3('figure_eight', 'forecast_rmse'))} and {fm(r3('sinusoid', 'forecast_rmse'))}) are worse than simply holding the last observation ({fm(bl('const_pos', 'figure_eight', 'forecast_rmse'))} and {fm(bl('const_pos', 'sinusoid', 'forecast_rmse'))}), and its reconstruction ({fm(r3('figure_eight', 'recon_rmse'))} and {fm(r3('sinusoid', 'recon_rmse'))}) is far worse than linear interpolation ({fm(bl('const_pos', 'figure_eight', 'recon_rmse'))} and {fm(bl('const_pos', 'sinusoid', 'recon_rmse'))}).")
w(f"2. **Can one model with the same architecture and settings be trained on all five families?** Yes. It forecasts all five far better than the naive forecast baselines ({fm(min(r5(g, 'forecast_rmse') for g in FAMS))} to {fm(max(r5(g, 'forecast_rmse') for g in FAMS))} against hold-last {fm(min(bl('const_pos', g, 'forecast_rmse') for g in FAMS))} to {fm(max(bl('const_pos', g, 'forecast_rmse') for g in FAMS))}). But, compared with the three-shape model, it is **worse on the three original families** (forecast {pct(pr('circle', 'forecast_rmse')['relative_change_of_mean'])} on circle, {pct(pr('spiral', 'forecast_rmse')['relative_change_of_mean'])} on spiral, {pct(pr('ellipse', 'forecast_rmse')['relative_change_of_mean'])} on ellipse), and its reconstruction of withheld points is no better than plain linear interpolation on any family (essentially tied on circle, {CIRC_REL * 100:+.1f}%, and worse by {min(OTHERS_REL) * 100:.0f}% to {max(OTHERS_REL) * 100:.0f}% on the other four).")
w('')
w('What this does **not** show: the five-shape model was trained on all five families, so its results on them are **not zero-shot generalisation**. All results are from one training run (seed 42). No claim of novelty, of improved forecasting, or of generalisation to arbitrary trajectories is made. The project is an adapted re-implementation and experimental extension of Latent Neural ODEs, not a replication of Chen et al. (2018).')
w('')
w('## 2. The five families')
w('')
w('Time-ordered 2-D trajectories on 200 equally spaced times in [0, 10]; Gaussian observation noise SD 0.05 (clean trajectories are kept as targets). The first three families are the project\'s existing generators (unchanged); the two new ones are:')
w('')
fe, si = M['new_family_parameter_ranges']['figure_eight'], M['new_family_parameter_ranges']['sinusoid']
w('| Family | Equations (theta = omega t + phase) | Parameter ranges (uniform) |')
w('|---|---|---|')
w(f"| figure-eight | x = cx + A sin(theta), y = cy + B sin(2 theta) | A in {fe['A_horizontal']}, B in {fe['B_vertical']}, abs(omega) in {fe['abs_omega']}, phase in [-pi, pi], centre in {fe['centre']}, direction = sign of omega (+/-, p = 1/2) |")
w(f"| sinusoid | x = x0 + v t, y = cy + A sin(omega t + phase), x0 = cx - 5v | A in {si['amplitude']}, omega in {si['omega']}, abs(v) in {si['abs_velocity']}, phase in [-pi, pi], centre in {si['centre']}, direction = sign of v |")
w('')
w('Circle, spiral and ellipse keep their original generators and parameter ranges (`src/glo/data.py`).')
w('')
w('## 3. Dataset construction and validation')
w('')
pc = D['per_family_split_counts']
w(f"- **Counts:** {M['n_per_family']} per family, {D['n_trajectories']:,} in total ({D['n_unique_ids']:,} unique ids). Stratified split by family: train {D['split_counts']['train']:,} / validation {D['split_counts']['val']} / test {D['split_counts']['test']} (700 / 150 / 150 per family in every split; no trajectory id appears in more than one split).")
w(f"- **Original families preserved:** the circle, spiral and ellipse trajectories, their noisy observations, observation masks and split membership are array-for-array identical to `three_shapes` in all three splits (checked in `dataset_checks.json`: every field `true`).")
w(f"- **Observation protocol:** {M['default_n_obs']} observed points per trajectory, all at t <= {M['forecast_start']:g} (first at t = 0), randomly spaced; no observation after t = 7 reaches the encoder. Forecast window: 7 < t <= 10. No shape label is given to any model.")
w(f"- **Shape checks:** figure-eight: all {D['figure_eight']['n']} cover a full eight with alternating centre crossings (largest vertical offset at the centre crossing {D['figure_eight']['max_abs_y_offset_at_centre_over_B']:.1e} of B; median y-to-x zero-crossing ratio {D['figure_eight']['y_to_x_zero_crossing_ratio_median']:.1f}). Sinusoid: all {D['sinusoid']['n']} have monotone x (maximum deviation from linear {D['sinusoid']['x_linear_max_residual']:.1f}) and the y half-range matches the amplitude (ratio {D['sinusoid']['y_half_range_over_amplitude_min_median_max'][0]:.4f} to {D['sinusoid']['y_half_range_over_amplitude_min_median_max'][2]:.4f}). Failures recorded: {len(D['failures'])}.")
w(f"- **Scale:** median RMS radius about the own centroid is {min(v['rms_radius_about_own_centroid_median'] for v in D['coordinate_scale_per_family'].values()):.2f} to {max(v['rms_radius_about_own_centroid_median'] for v in D['coordinate_scale_per_family'].values()):.2f} across the five families, so no family is trivially different in size. Normalisation constants are computed from the five-shape training split only (mean [{M['normalization_mean'][0]:.4f}, {M['normalization_mean'][1]:.4f}], scale {M['normalization_scale']:.4f}).")
w('- The splits and leakage checks were independently re-verified in the project audit (`PROJECT_COMPLETION_AUDIT.md`, `audit/verify_1_datasets.py`).')
w('')
w('## 4. Models and evaluation protocol')
w('')
w('| Model | Trained on | Parameters | Epochs / best epoch | Best validation -ELBO | Optimiser steps | Training time |')
w('|---|---|---:|---|---:|---:|---:|')
w(f"| Three-shape (frozen, packaged) | circle, spiral, ellipse (2,100 trajectories) | {three['n_parameters']:,} | {three['epochs_run']} / {three['best_epoch']} | {three['best_val_neg_elbo']:.2f} (three-shape validation data) | {steps(2100) * 300:,} | {three['training_minutes']:.0f} min |")
w(f"| Five-shape | all five (3,500 trajectories) | {five['n_parameters']:,} | {five['epochs_run']} / {five['best_epoch']} | {five['best_val_neg_elbo']:.2f} (five-shape validation data) | {steps(3500) * 300:,} | {five['training_minutes']:.0f} min |")
w('')
w('Both use the same architecture and training configuration (`configs/model_d.yaml`: latent 8, GRU 64, tanh MLPs 64x2, rk4 step 0.1, Adam 3e-3 with cosine schedule, batch 32, 300 epochs, KL weight 1, fixed sigma 0.05, seed 42). The two validation -ELBO values are on different validation sets and must not be compared.')
w('')
w('**Evaluation:** five-shape test split (750 trajectories, 150 per family); every model and baseline receives the **identical** 30 observations (same times and noise, asserted); posterior-mean initial state; each model is normalised with its own training-set constants; metrics are per-trajectory RMSE in data units, then averaged. *Reconstruction RMSE* uses the withheld context points (t <= 7); *forecast RMSE* uses all points with 7 < t <= 10. Baselines: linear interpolation (reconstruction), hold-last observation and constant velocity (forecast). Paired comparisons use the per-trajectory difference on identical trajectories with a bootstrap 95% interval and a Wilcoxon test; Holm correction over the ' + f"{len(C['paired'])} tests (five families x two metrics)." + ' Test-set intervals describe test-trajectory sampling only, not training-seed variability.')
w('')
w('## 5. Zero-shot test: the frozen three-shape model on figure-eight and sinusoid')
w('')
w(f"The three-shape model (checksum verified; its stored three-shape results were reproduced exactly, maximum deviation {Zs['reproduces_stored_three_shape_results']['max_abs_deviation']:.1f}) was applied unchanged. Mean (median) of per-trajectory RMSE:")
w('')
w('| Family | Status | Recon. RMSE | Linear interp. | Forecast RMSE | Hold-last | Const. velocity |')
w('|---|---|---|---:|---|---:|---:|')
for g in G:
    stt = 'seen in training' if g in ORIG else ('**unseen (zero-shot)**' if g in FAMS else 'all five')
    z = Zs['three_shape_model'][g]
    w(f"| {g} | {stt} | {z['recon_rmse']['mean']:.4f} ({z['recon_rmse']['median']:.4f}) | {Zs['const_pos'][g]['recon_rmse']['mean']:.4f} | {z['forecast_rmse']['mean']:.4f} ({z['forecast_rmse']['median']:.4f}) | {Zs['const_pos'][g]['forecast_rmse']['mean']:.4f} | {Zs['const_vel'][g]['forecast_rmse']['mean']:.4f} |")
w('')
w('On the two unseen families the model is worse than the hold-last baseline for forecasting and worse than linear interpolation for reconstruction (by a large margin for reconstruction). It is better than constant-velocity extrapolation, but that baseline is itself very poor on these trajectories (forecast RMSE 2.6 and 1.4). The model does not generalise to the unseen families. (`overall` mixes seen and unseen families and should not be read as a generalisation score.) This is one frozen model from one training seed.')
w('')
w('## 6. The five-shape model')
w('')
w('| Family | Recon. RMSE mean (median) | Linear interp. | Forecast RMSE mean (median) | Hold-last | Const. velocity |')
w('|---|---|---:|---|---:|---:|')
for g in G:
    w(f"| {g} | {r5(g, 'recon_rmse'):.4f} ({r5(g, 'recon_rmse', 'median'):.4f}) | {bl('const_pos', g, 'recon_rmse'):.4f} | {r5(g, 'forecast_rmse'):.4f} ({r5(g, 'forecast_rmse', 'median'):.4f}) | {bl('const_pos', g, 'forecast_rmse'):.4f} | {bl('const_vel', g, 'forecast_rmse'):.4f} |")
w('')
w(f"The model forecasts every family far better than the naive baselines. Its **reconstruction** of withheld context points is not better than plain linear interpolation on any family: essentially tied on circle ({fm(r5('circle', 'recon_rmse'))} vs {fm(bl('const_pos', 'circle', 'recon_rmse'))}, {CIRC_REL * 100:+.1f}%) and worse on spiral, ellipse, figure_eight and sinusoid. (The later, larger models in the capacity and ablation studies do beat linear interpolation; see section 9.)")
w('')
w('## 7. Three-shape versus five-shape model (paired, identical observations)')
w('')
w('Difference = five-shape minus three-shape (positive = the five-shape model has higher error). Holm-adjusted p-values.')
w('')
w('| Family | Status for the three-shape model | Forecast: three | Forecast: five | Difference [95% CI] | Five better on | Holm p | Recon: three | Recon: five | Holm p (recon.) |')
w('|---|---|---:|---:|---|---:|---:|---:|---:|---:|')
for g in FAMS:
    p, q = pr(g, 'forecast_rmse'), pr(g, 'recon_rmse')
    w(f"| {g} | {'trained on' if g in ORIG else '**unseen**'} | {fm(p['three_mean'])} | {fm(p['five_mean'])} | {p['mean_difference_five_minus_three']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] | {p['five_better_fraction'] * 100:.0f}% | {pv(p['wilcoxon_p_holm'])} | {fm(q['three_mean'])} | {fm(q['five_mean'])} | {pv(q['wilcoxon_p_holm'])} |")
w('')
w('- On the three original families the five-shape model is **worse** (higher forecast and reconstruction error; better on only 15% to 27% of trajectories for forecasting). Sharing one model across five families cost accuracy on the original three, consistent with the earlier unified-versus-specialist result for two families.')
w('- On the two new families the five-shape model is far better than the three-shape model, which is expected because it was trained on them.')
w('- **Caveat for this comparison:** the five-shape model also had a different training set size and more optimiser steps (section 8, item 4), so the cost on the original families cannot be attributed to the number of families alone.')
w('')
w('## 8. Known issues and caveats')
w('')
w('1. **Figure `error_vs_time_three_vs_five.png`:** the third curve is labelled "const. velocity" along the whole time axis, but before t = 7 it is the linear-interpolation baseline (constant velocity applies only after the last observation), and it passes through the observed timestamps, which is why its error is so low there. The numbers in the tables are not affected. The figure was left unchanged to preserve the original results.')
w('2. **Figure `per_family_rmse_three_vs_five.png`:** the x-axis labels of figure_eight and sinusoid overlap. Cosmetic.')
w(f"3. **Observation draws differ between test splits.** Observations are drawn in dataset order, so the same three-shape model evaluated on the three-shape test split (stored results) and on the five-shape test split sees different observation subsets. Forecast RMSE for circle is {ST_in['circle']['forecast_rmse']['mean']:.4f} on the stored three-shape evaluation and {r3('circle', 'forecast_rmse'):.4f} here (spiral {ST_in['spiral']['forecast_rmse']['mean']:.4f} vs {r3('spiral', 'forecast_rmse'):.4f}; ellipse {ST_in['ellipse']['forecast_rmse']['mean']:.4f} vs {r3('ellipse', 'forecast_rmse'):.4f}). The two tables are each internally consistent but are not interchangeable; this report uses only the five-shape test-split evaluation for the three-versus-five comparison.")
w(f"4. **Training budget differs.** Both models ran 300 epochs, but with batch size 32 the three-shape model took {steps(2100)} optimiser steps per epoch ({steps(2100) * 300:,} in total) and the five-shape model {steps(3500)} ({steps(3500) * 300:,}). The normalisation scale also differs (five-shape {M['normalization_scale']:.4f}).")
w('5. **One training seed** for each model. The only multi-seed evidence in the project is the earlier five-seed two-shape study, which shows forecast RMSE of one architecture varying by about 23% (coefficient of variation) between seeds. Differences between single runs should be read with that in mind.')
w('6. Metrics are averages of per-trajectory RMSE; pooled RMSE is a different aggregate and is stored separately in the capacity and ablation results.')
w('')
w('## 9. Related later experiments (separate reports)')
w('')
w('Using the five-shape model as the baseline, two single-seed follow-ups varied encoder size: `results/capacity_scaling/CAPACITY_SCALING_REPORT.md` (latent 8/16/32, GRU 64/128; training streams not controlled) and `results/ablation_2x2/ABLATION_2X2_REPORT.md` (a controlled 2x2 of latent 8/16 by GRU 64/128). In both, larger encoders reconstructed withheld points better; the full-horizon forecast was higher for some larger variants, but the differences are of the same order as seed-to-seed variation and have not been confirmed across seeds. See those reports for the exact, qualified statements.')
w('')
w('## 10. What can and cannot be claimed, and sources')
w('')
w('**Can be claimed (from these results):**')
w('- The five-family dataset is valid, leakage-free, and contains the three-shape data unchanged.')
w('- A model trained on circle, spiral and ellipse is worse than hold-last at forecasting and worse than linear interpolation at reconstructing figure-eight and sinusoid trajectories (it is better only than constant-velocity extrapolation, a very poor baseline here); one frozen model, one seed.')
w('- A model with the same architecture can be trained on all five families without labels and forecasts them far better than naive baselines, at a cost in accuracy on the original three families (one seed each, budgets not matched).')
w('')
w('**Cannot be claimed:** generalisation to unseen families by the five-shape model (it saw all five); that adding families causes the accuracy loss on the original ones (confounded with training size and steps); any improved forecasting method; novelty; results that hold across seeds.')
w('')
w('**Plain-English answers.** *Can the three-family model draw figure-eights and waves it has never seen?* No: it does worse than just repeating the last observation. *Can one model learn all five?* Yes, and it forecasts them well compared with simple baselines, but it is a bit worse than the three-family model on the original three. *Is that proof that more families hurt?* No, because the training size and number of steps also changed and only one run was made.')
w('')
w('**Sources (all existing):** `results/five_shapes/compare_three_vs_five.json`, `zeroshot_three_shape_model.json`, `dataset_checks.json`, `tables/`, `figures/`, `predictions_test.npz`; `data/processed/five_shapes/metadata.json`; `results/capacity_scaling/capacity_scaling_results.json` (training facts); `results/validation/three_shapes/three_shapes_and_ood.json` (stored three-shape evaluation); `src/glo/five_shapes.py`, `scripts/30_generate_five_shapes.py`, `scripts/31_evaluate_five_shapes.py`; `tests/test_five_shapes.py`. Report generated by `audit/build_five_shapes_report.py`.')
Path('reports/five_shapes_report.md').write_text('\n'.join(L) + '\n'); print('written reports/five_shapes_report.md', len(L), 'lines')
