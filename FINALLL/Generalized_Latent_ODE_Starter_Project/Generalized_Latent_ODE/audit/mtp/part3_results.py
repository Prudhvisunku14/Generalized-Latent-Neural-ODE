# section 06 (results documents and tables); executed from build_mtp_package.py
T = '06_EXPERIMENTAL_RESULTS/RESULTS_TABLES/'
ALL7 = [('orig_baseline', 'earlier 8/64 run'), ('A', 'A 8/64'), ('B', 'B 16/64'), ('C', 'C 8/128'), ('D', 'D 16/128'), ('orig_medium', 'earlier 16/128 run'), ('orig_large', 'earlier 32/128 run')]

# ------------------------------------------------------------------ tables (CSV + Markdown strings reused in the documents)
rc_rows = [[FL[g], LI(g), Z3(g, 'recon_rmse')] + [R(m, g, 'recon_rmse') for m, _ in ALL7] for g in G6]
fc_rows = [[FL[g], HLd(g), CV(g), Z3(g, 'forecast_rmse')] + [R(m, g, 'forecast_rmse') for m, _ in ALL7] for g in G6]
h_rc = ['family', 'linear_interpolation', '3-family model (frozen; figure-eight and sinusoid unseen)'] + [n for _, n in ALL7]
h_fc = ['family', 'hold_last', 'constant_velocity', '3-family model (frozen; figure-eight and sinusoid unseen)'] + [n for _, n in ALL7]
CSV(T + 'five_family_reconstruction_rmse.csv', h_rc, rc_rows); CSV(T + 'five_family_forecast_rmse.csv', h_fc, fc_rows)
TB_RC, TB_FC = MD(h_rc, rc_rows), MD(h_fc, fc_rows)

sv_rows = []
for fam_ in ('circle', 'spiral'):
    for k, kn in (('recon_rmse', 'reconstruction'), ('forecast_rmse', 'forecast')):
        s_ = SVR(fam_, k); sv(f'{fam_}.seed_level.{k}.unified.mean'); sv(f'{fam_}.seed_level.{k}.specialist.mean'); sv(f'{fam_}.seed_level.{k}.welch_t_p')
        sv_rows.append([fam_, kn, f"{s_['u_mean']:.4f}", f"{s_['u_sd']:.4f}", s_['u_n'], f"{s_['s_mean']:.4f}", f"{s_['s_sd']:.4f}", s_['s_n'], f"{s_['rel'] * 100:+.1f}%", pv(s_['welch']), pv(s_['welch_bonf']), 'yes' if s_['all_worse'] else 'no',
                        f"{s_['p_rel'] * 100:+.1f}%", pv(s_['p_wil'])])
h_sv = ['family', 'metric', 'shared_mean', 'shared_sd', 'shared_runs', 'specialist_mean', 'specialist_sd', 'specialist_runs', 'shared_vs_specialist', 'welch_p_raw', 'welch_p_bonferroni_x4', 'every_shared_run_worse_than_every_specialist_run', 'seed42_paired_difference', 'seed42_paired_wilcoxon_p']
CSV(T + 'shared_vs_specialist.csv', h_sv, sv_rows); TB_SV = MD(h_sv, sv_rows)

og = 'C_out_of_distribution_ellipses'; ID = f'{og}.three_shape_model_on_same_ellipses (in-distribution reference)'
xr = [['ellipse', 'two-family models (5 runs), zero-shot', f"{V['ell_rc_two']['mean']:.4f} +/- {V['ell_rc_two']['std']:.4f}", f"{V['ell_fc_two']['mean']:.4f} +/- {V['ell_fc_two']['std']:.4f}", 'multi-seed (5)'],
      ['ellipse', 'three-family model (trained on ellipses; NOT zero-shot)', t3(ID + '.recon_rmse.mean', '.4f'), t3(ID + '.forecast_rmse.mean', '.4f'), 'single run'],
      ['ellipse', 'baselines: linear interp. / hold-last / const. velocity', t3(f'{og}.const_pos_on_same_ellipses.recon_rmse.mean', '.4f'), f"{t3(f'{og}.const_pos_on_same_ellipses.forecast_rmse.mean', '.4f')} / {t3(f'{og}.const_vel_on_same_ellipses.forecast_rmse.mean', '.4f')}", '']]
for g in FAMS[3:]:
    xr.append([FL[g], '3-family model (frozen), zero-shot', Z3(g, 'recon_rmse'), Z3(g, 'forecast_rmse'), 'single run']); xr.append([FL[g], 'baselines: linear interp. / hold-last / const. velocity', LI(g), f'{HLd(g)} / {CV(g)}', ''])
    xr.append([FL[g], '5-family model (trained on it; NOT zero-shot)', F5(g, 'recon_rmse'), F5(g, 'forecast_rmse'), 'single run'])
h_x = ['unseen family', 'model', 'reconstruction RMSE', 'forecast RMSE', 'evidence']
CSV(T + 'cross_family_generalisation.csv', h_x, xr); TB_X = MD(h_x, xr)

cap_rows = [[nm, cap(f'models.{k}.n_parameters', ','), cap(f'results.{k}.overall.recon_rmse.mean', '.4f'), cap(f'results.{k}.overall.forecast_rmse.mean', '.4f')] + [cap(f'results.{k}.{g}.forecast_rmse.mean', '.4f') for g in FAMS] for k, nm in
            (('baseline', 'baseline 8/64'), ('medium', 'medium 16/128'), ('large', 'large 32/128'), ('three_shape', 'three-family (frozen)'))]
h_cap = ['model', 'parameters', 'overall recon', 'overall forecast'] + [f'forecast {FL[g]}' for g in FAMS]
CSV(T + 'capacity_scaling.csv', h_cap, cap_rows); TB_CAP = MD(h_cap, cap_rows)

ab_rows = [[m, ab(f'models.{m}.latent_dim'), ab(f'models.{m}.encoder_hidden'), ab(f'models.{m}.n_parameters', ','), R(m, 'overall', 'recon_rmse'), '' if m == 'A' else pct(rel(m, 'overall', 'recon_rmse')), R(m, 'overall', 'forecast_rmse'),
           '' if m == 'A' else pct(rel(m, 'overall', 'forecast_rmse')), '' if m == 'A' else f"{AB['fraction_of_trajectories_better_than_A'][m]['overall']['forecast_rmse'] * 100:.0f}%"] for m in CELLS]
h_ab = ['model', 'latent', 'GRU', 'parameters', 'recon RMSE', 'recon vs A', 'forecast RMSE', 'forecast vs A', 'trajectories with lower forecast error than A']
CSV(T + 'ablation_2x2_overall.csv', h_ab, ab_rows); TB_AB = MD(h_ab, ab_rows)
pf_rows = [[FL[g], kn, R('A', g, k)] + [cellc(m, g, k) for m in 'BCD'] for g in G6 for k, kn in (('forecast_rmse', 'forecast'), ('recon_rmse', 'reconstruction'))]
h_pf = ['family', 'metric', 'A', 'B vs A (change; Holm p)', 'C vs A', 'D vs A']
CSV(T + 'ablation_2x2_per_family.csv', h_pf, pf_rows); TB_PF = MD(h_pf, pf_rows)
LAT, GRU, INT = 'latent_main_effect (16 vs 8, averaged over GRU)', 'gru_main_effect (128 vs 64, averaged over latent)', 'interaction ((D-C) - (B-A))'
fx_rows = []
for g in G6:
    for k, kn in (('forecast_rmse', 'forecast'), ('recon_rmse', 'reconstruction')):
        row = [FL[g], kn]
        for nm in (LAT, GRU, INT):
            f_ = AB['factorial'][f'{g}/{k}/{nm}']; row.append(f"{f_['mean']:+.4f} [{f_['bootstrap_ci95'][0]:+.3f}, {f_['bootstrap_ci95'][1]:+.3f}] (Holm p={pv(f_['wilcoxon_p_holm'])})")
        fx_rows.append(row)
h_fx = ['family', 'metric', 'latent 8 to 16', 'GRU 64 to 128', 'interaction (D-C)-(B-A)']
CSV(T + 'ablation_2x2_factorial_effects.csv', h_fx, fx_rows); TB_FX = MD(h_fx, fx_rows)
wn_rows = [[f'{wd[0]} < t <= {wd[1]}'] + [f"{V['win'][(m, wd)]:.4f}" for m in CELLS] for wd in WIN]; h_wn = ['window', 'A', 'B', 'C', 'D']
CSV(T + 'forecast_error_by_time_window.csv', h_wn, wn_rows); TB_WN = MD(h_wn, wn_rows)
CSV(T + 'forecast_error_vs_time_overall.csv', ['t'] + [f'RMSE across 750 test trajectories, model {m}' for m in CELLS], [[f'{t_:.4f}'] + [f"{et[m]['overall'][i]:.5f}" for m in CELLS] for i, t_ in enumerate(qte)])

cn_rows = [[nm] + [f"{cond(f'seed_level.{mm}.{fam_}.{k}.mean', '.4f')} +/- {cond(f'seed_level.{mm}.{fam_}.{k}.std', '.4f')}" for fam_ in ('circle', 'spiral') for k in ('recon_rmse', 'forecast_rmse')] for mm, nm in
           (('M1', 'M1 shared (24,602 params)'), ('M2', 'M2 control, no code (25,049)'), ('M3', 'M3 dynamics-conditioned (25,118)'), ('M2aug', 'M2aug ablation, 3 runs (25,894)'))]
cn_rows.insert(0, ['specialists (3 runs)', '', cond('gap_to_specialists.circle.specialist_forecast_rmse', '.4f'), '', cond('gap_to_specialists.spiral.specialist_forecast_rmse', '.4f')])
h_cn = ['model', 'circle recon', 'circle forecast', 'spiral recon', 'spiral forecast']
CSV(T + 'conditioning_experiments.csv', h_cn, cn_rows); TB_CN = MD(h_cn, cn_rows)

METH = [('linear', 'Linear interpolation'), ('cubic', 'Cubic spline'), ('pchip', 'PCHIP'), ('frozen_three_shapes', 'Frozen synthetic (three-family) model'), ('geolife_scratch', 'GeoLife-trained from scratch'), ('geolife_finetuned_from_three_shapes', 'Synthetic then fine-tuned on GeoLife')]
gl_rows = [[nm, gl(f'interpolation.30.{k}.median', '.2f'), f"[{gl(f'interpolation.30.{k}.median_ci95_user_bootstrap.0', '.1f')}, {gl(f'interpolation.30.{k}.median_ci95_user_bootstrap.1', '.1f')}]", gl(f'interpolation.30.{k}.mean', '.2f'), gl(f'interpolation.30.{k}.geometric_mean_ratio_vs_linear', '.3f')] for k, nm in METH]
h_gl = ['method (30 observations)', 'median error (m)', '95% CI (user bootstrap)', 'mean error (m)', 'geometric-mean ratio vs linear']
CSV(T + 'geolife_interpolation_30_observations.csv', h_gl, gl_rows); TB_GL = MD(h_gl, gl_rows)
dens_rows = [[n] + [gl(f'interpolation.{n}.{k}.median', '.2f') for k, _ in METH] for n in (10, 20, 30, 40, 50)]; h_dn = ['observations'] + [nm for _, nm in METH]
CSV(T + 'geolife_median_error_by_observation_count.csv', h_dn, dens_rows); TB_DN = MD(h_dn, dens_rows)
pa_rows = [[n, f"{V['pa'][n]['med_scratch']:.1f}", f"{V['pa'][n]['med_fine']:.1f}", f"{V['pa'][n]['gmr']:.2f}", f"[{V['pa'][n]['ci'][0]:.2f}, {V['pa'][n]['ci'][1]:.2f}]", f"{V['pa'][n]['win_fine'] * 100:.0f}%"] for n in (10, 20, 30, 40, 50)]
h_pa = ['observations', 'median scratch (m)', 'median fine-tuned (m)', 'geometric-mean ratio fine-tuned / scratch', '95% CI (user bootstrap)', 'fine-tuned better on']
CSV(T + 'geolife_finetuned_vs_scratch_paired.csv', h_pa, pa_rows); TB_PA = MD(h_pa, pa_rows)
ds_rows = [['circle', '1,000', '700 / 150 / 150', 'x = cx + R cos(theta); y = cy + R sin(theta)'], ['spiral', '1,000', '700 / 150 / 150', 'radius R exp(g t)'], ['ellipse', '1,000', '700 / 150 / 150', 'semi-axes a, b = a*U(0.40, 0.78), tilt alpha'],
           ['figure-eight', '1,000', '700 / 150 / 150', 'x = cx + A sin(theta); y = cy + B sin(2 theta)'], ['sinusoid', '1,000', '700 / 150 / 150', 'x = x0 + v t; y = cy + A sin(omega t + phi)']]
CSV(T + 'dataset_statistics.csv', ['family', 'trajectories', 'train / val / test', 'equation'], ds_rows)

W('06_EXPERIMENTAL_RESULTS/RESULTS_TABLES/README.md', '''
# Results tables

Each table exists as CSV (this folder) and as Markdown inside the corresponding results document. All numbers are read from saved result files (`11_SOURCE_INDEX/NUMERICAL_RESULTS_SOURCES.md`). `original_tables/` holds the tables written by the original scripts, copied unchanged.

| File | Content |
|---|---|
| five_family_reconstruction_rmse.csv, five_family_forecast_rmse.csv | per-family results of all five-family models and baselines |
| shared_vs_specialist.csv | two-family shared vs specialist (5 vs 3 runs) |
| cross_family_generalisation.csv | unseen-family results |
| capacity_scaling.csv | capacity-scaling study |
| ablation_2x2_overall.csv, ablation_2x2_per_family.csv, ablation_2x2_factorial_effects.csv | controlled 2x2 ablation |
| forecast_error_by_time_window.csv, forecast_error_vs_time_overall.csv | error versus horizon |
| conditioning_experiments.csv | conditioning study |
| geolife_*.csv | GeoLife transfer, density study, paired fine-tuned vs scratch |
| model_parameter_counts.csv | parameter breakdown per configuration |
| dataset_statistics.csv | dataset summary |
''')

# ------------------------------------------------------------------ result documents
def exp_doc(fname_, title, objective, method, models, dataset, evidence, results, figs, simple, supported, limits):
    W('06_EXPERIMENTAL_RESULTS/' + fname_, f'''
# {title}

**Evidence level:** {evidence}

## Objective
{objective}

## Method
{method}

## Models
{models}

## Dataset
{dataset}

## Exact numerical results
{results}

## Figures
{figs}

## Simple explanation
{simple}

## What the results support
{supported}

## Limitations
{limits}
''')

exp_doc('FIVE_FAMILY_RESULTS.md', 'Experiment A: five-family modelling', 'Check whether one shared Latent ODE with no shape label can reconstruct and forecast all five trajectory families, and see which families are easy or hard.',
        'Train models on the five-family training split (3,500 trajectories); evaluate on the 750 test trajectories with identical 30 observations (observation seed 123); report per-family mean per-trajectory RMSE for reconstruction (unobserved points, t <= 7) and forecast (7 < t <= 10); compare with simple baselines.',
        'Five-family Model D (8/64): the earlier run and the controlled rerun A; larger variants B, C, D and the earlier 16/128 and 32/128 runs (details in the ablation and capacity documents); the frozen three-family model (trained on circle, spiral, ellipse only) for reference.',
        'Five-family test split: 150 test trajectories per family (750 in total), 30 observations each.', '**SINGLE-SEED** (one training run per model).',
        f'''**Reconstruction RMSE (lower is better)**

{TB_RC}

**Forecast RMSE (lower is better)**

{TB_FC}

*The three-family model has not seen figure-eight or sinusoid; its "all five" row mixes seen and unseen families.*

**Answers.** Easiest to reconstruct: **sinusoid** in {V['n_recon_easiest_sinusoid']} of the 7 five-family models (cell A: circle and sinusoid tied at {R('A', 'circle', 'recon_rmse')} / {R('A', 'sinusoid', 'recon_rmse')}); hardest: **figure-eight** in {V['n_recon_hardest_fig8']} of 7. Hardest to forecast: spiral for the two smallest models ({R('orig_baseline', 'spiral', 'forecast_rmse')}, {R('A', 'spiral', 'forecast_rmse')}), **sinusoid** for the {V['n_fc_hardest_sinusoid']} larger ones (B {R('B', 'sinusoid', 'forecast_rmse')}, D {R('D', 'sinusoid', 'forecast_rmse')}, both worse than hold-last {HLd('sinusoid')}). The model does **not** perform equally on all five families.''',
        '`07_FIGURES/RECONSTRUCTION/fig1_reconstruction_by_family.png`, `07_FIGURES/FORECASTING/fig2_forecast_by_family.png`',
        'One shared model, trained once on all five path types and never told which type it is looking at, produces sensible curves for all of them. Its forecasts beat the simple baselines by a wide margin. Its gap-filling (reconstruction) is no better than joining the dots when the model is small, and better when it is larger.',
        '- One model without shape labels works on all five families; forecasts are far better than hold-last and constant velocity overall.\n- Performance differs by family (see the tables). The smallest models do not beat linear interpolation at reconstruction on most families; B and D beat it on every family.\n- Compared with the three-family model, the five-family model is worse on the original three families (circle forecast 0.1401 -> 0.2594; spiral 0.2908 -> 0.4270; ellipse 0.1885 -> 0.2609), but training size and optimiser steps also differ (19,800 vs 33,000), so this cannot be attributed to the number of families alone.',
        '- Single seed. - Models trained on all five families, so their results are not zero-shot generalisation. - Only naive baselines were used. - Different observation draws between the three-family and five-family test splits make a model\'s numbers non-interchangeable across the two splits.')

exp_doc('SHARED_VS_SPECIALIST.md', 'Experiment B: shared model vs specialist models', 'Measure what one shared model costs compared with separate models for each family, and whether the shared model still learns all families.',
        'Same architecture and training recipe for both. Shared: trained on circle + spiral together (1,400 trajectories, 300 epochs). Specialists: trained on one family each (700 trajectories, 600 epochs = same number of optimiser steps). Evaluated on the same 150 test circles and 150 test spirals with identical observations. Seed-level comparison with Welch tests; trajectory-level paired test for seed 42.',
        'Shared Model D (5 training runs: seeds 42, 123, 2024, 2025, 2026) vs circle-only and spiral-only specialists (3 training runs each: seeds 42, 123, 2024).', 'Two-family data (circle + spiral); 300 test trajectories (150 per family).', '**MULTI-SEED** (5 shared runs vs 3 specialist runs).',
        f'''{TB_SV}

*Bonferroni adjustment (multiply by the 4 tests) was computed in this analysis; the original study reported the raw Welch p-values.*

- **Forecast:** the shared model is {SVR('circle', 'forecast_rmse')['rel'] * 100:.0f}% worse on circles and {SVR('spiral', 'forecast_rmse')['rel'] * 100:.0f}% worse on spirals. In both families every one of the 5 shared runs is worse than every one of the 3 specialist runs.
- **Reconstruction:** {SVR('circle', 'recon_rmse')['rel'] * 100:.0f}% worse on circles (small but consistent), {SVR('spiral', 'recon_rmse')['rel'] * 100:.0f}% on spirals (not distinguishable).
- **Variability across runs:** SD of forecast RMSE: shared {SVR('circle', 'forecast_rmse')['u_sd']:.3f} / {SVR('spiral', 'forecast_rmse')['u_sd']:.3f}; specialists {SVR('circle', 'forecast_rmse')['s_sd']:.3f} / {SVR('spiral', 'forecast_rmse')['s_sd']:.3f}.
- **Adding a third family (ellipse) to the shared model** did not clearly change circle forecasting (three-family model {t3('B_three_vs_two_shape_on_matching_circle_spiral.circle.forecast_rmse.three_shape', '.3f')} vs two-family seed-42 model {t3('B_three_vs_two_shape_on_matching_circle_spiral.circle.forecast_rmse.two_shape_seed42', '.3f')}, paired p = {pv(T3['B_three_vs_two_shape_on_matching_circle_spiral']['circle']['forecast_rmse']['paired_wilcoxon_p'])}) but circle reconstruction was {T3['B_three_vs_two_shape_on_matching_circle_spiral']['circle']['recon_rmse']['relative_diff_pct_vs_seed42']:.0f}% worse.''',
        '`07_FIGURES/FORECASTING/fig3_shared_vs_specialist.png` (new, shows every run), `07_FIGURES/FORECASTING/shared_vs_separate_original.png` (original study figure)',
        'Teaching one model several path types is convenient (one model, no labels) but it costs accuracy: specialists forecast better. The shared model does still learn both families, just less precisely, and its different training runs disagree more with each other.',
        f"- *Statistically supported:* circle forecast (Welch p = {pv(SVR('circle', 'forecast_rmse')['welch'])}, {pv(SVR('circle', 'forecast_rmse')['welch_bonf'])} after Bonferroni).\n- *Supported but weaker:* spiral forecast (p = {pv(SVR('spiral', 'forecast_rmse')['welch'])}, {pv(SVR('spiral', 'forecast_rmse')['welch_bonf'])} after correction: not significant at 5%; all 5 shared runs are nevertheless worse than all 3 specialists).\n- *Descriptive:* reconstruction.\n- The same-seed trajectory-level tests describe only two trained models (seed 42) and do not show that the result repeats for new training runs.",
        '- Only two families have specialists; none for ellipse, figure-eight or sinusoid. - Only three specialist runs. - Whether more capacity or data would close the gap was not tested; conditioning the dynamics did not (see CONDITIONING_EXPERIMENTS.md). - Two different sets of five shared runs (this study and the conditioning study) give different spreads for circle forecast (SD 0.037 vs 0.007), so five-seed SDs are themselves uncertain.')

exp_doc('CROSS_FAMILY_GENERALIZATION.md', 'Experiment C: cross-family generalisation (unseen families)', 'Test whether a model trained on some trajectory families can handle families that were absent from its training data (zero-shot).',
        'Models are evaluated on families excluded from their training data, with 30 observations and identical observation draws for the models and the baselines. A result on a family the model was trained on is **not** zero-shot.',
        '(i) Five two-family models (trained on circle + spiral; excluded: ellipse); (ii) one frozen three-family model (trained on circle + spiral + ellipse; excluded: figure-eight, sinusoid); (iii) the five-family model, shown only for reference because it was trained on all five.',
        'Three-family test split (ellipses) and five-family test split (figure-eight and sinusoid), 150 trajectories per family.', '**MULTI-SEED** for the ellipse test (5 two-family runs); **SINGLE-SEED** for figure-eight and sinusoid (one frozen model).',
        f'''{TB_X}

- **Ellipses (5 runs):** forecasting transfers partly ({V['ell_fc_two']['mean']:.2f} vs hold-last {t3(f'{og}.const_pos_on_same_ellipses.forecast_rmse.mean', '.2f')}, but worse than the in-training {t3(ID + '.forecast_rmse.mean', '.2f')}); reconstruction does not ({V['ell_rc_two']['mean']:.2f}, more than three times worse than joining the dots, {t3(f'{og}.const_pos_on_same_ellipses.recon_rmse.mean', '.2f')}). Rotation direction is correct for every ellipse (seed-42 model), but the shape is drawn as a near-circle: predicted axis ratio {t3(f'{og}.ellipse_geometry_two_shape_seed42 (OOD).axis_ratio_pred_mean', '.2f')} vs true {t3(f'{og}.ellipse_geometry_two_shape_seed42 (OOD).axis_ratio_true_mean', '.2f')} (correlation {t3(f'{og}.ellipse_geometry_two_shape_seed42 (OOD).axis_ratio_corr', '.2f')}; the model that saw ellipses: {t3(f'{og}.ellipse_geometry_three_shape (ID).axis_ratio_corr', '.2f')}).
- **Figure-eight and sinusoid (one frozen model):** worse than **hold-last** at forecasting (figure-eight {Z3('figure_eight', 'forecast_rmse')} vs {HLd('figure_eight')}; sinusoid {Z3('sinusoid', 'forecast_rmse')} vs {HLd('sinusoid')}) and far worse than **linear interpolation** at reconstruction ({Z3('figure_eight', 'recon_rmse')} vs {LI('figure_eight')}; {Z3('sinusoid', 'recon_rmse')} vs {LI('sinusoid')}). It is better than constant velocity, but that baseline is very poor on these paths ({CV('figure_eight')}, {CV('sinusoid')}).
- **Five-family model:** saw all five families, so its numbers (forecast {F5('figure_eight', 'forecast_rmse')} and {F5('sinusoid', 'forecast_rmse')}) are not generalisation.''',
        '`07_FIGURES/GENERALIZATION/fig4_cross_family_generalisation.png` (new), `07_FIGURES/GENERALIZATION/ood_ellipses.png`, `three_vs_two_and_ood_bars.png`, `per_family_rmse_three_vs_five.png` (original figures; the last has overlapping x labels)',
        'A model that has only seen loops and spirals does not know how to draw figure-eights or waves, and it is worse than the simplest baselines on them. For ellipses (stretched circles) it gets the direction and a rough forecast right but draws them as near-circles.',
        '- **No successful zero-shot generalisation was demonstrated.** Partial transfer for ellipses (the easiest unseen family); failure for dissimilar families (and, in the real-world test, for GPS data).\n- Adding families to training costs accuracy on the original ones (single run, budgets differ).',
        '- No leave-one-family-out experiment (train on four families, test on the fifth). - One frozen model and one seed for figure-eight and sinusoid. - Ellipse geometry numbers are for the seed-42 model only. - Other training recipes might generalise differently.')

exp_doc('CAPACITY_SCALING.md', 'Experiment D: capacity scaling (first version, training streams not controlled)', 'Ask whether a larger encoder (latent dimension and GRU width) changes reconstruction and forecasting on the five-family data.',
        'Train three sizes with the identical recipe (`model_d.yaml`, seed 42, 300 epochs); only latent dimension and GRU width change (ODE and decoder widths stay 64). Evaluate on the same 750 test trajectories with identical observations; paired bootstrap and Wilcoxon (Holm) statistics.',
        f"Baseline 8/64 ({cap('models.baseline.n_parameters', ',')} parameters), medium 16/128 ({cap('models.medium.n_parameters', ',')}), large 32/128 ({cap('models.large.n_parameters', ',')}); the frozen three-family model for reference.", 'Five-family test split (750 trajectories).', '**SINGLE-SEED. Training data streams were not controlled** (`data_seed` unset: batch order and training observation draws differ between architectures). This is why the controlled 2x2 ablation was run.',
        f'''{TB_CAP}

Reconstruction improves with size in every family (overall {cap('results.baseline.overall.recon_rmse.mean', '.4f')} -> {cap('results.medium.overall.recon_rmse.mean', '.4f')} -> {cap('results.large.overall.recon_rmse.mean', '.4f')}). Forecasting on the full 7-10 window is worse for the larger models ({cap('results.baseline.overall.forecast_rmse.mean', '.4f')} -> {cap('results.medium.overall.forecast_rmse.mean', '.4f')} -> {cap('results.large.overall.forecast_rmse.mean', '.4f')}), with sinusoid the extreme case (large: {cap('results.large.sinusoid.forecast_rmse.mean', '.3f')}, worse than hold-last {HLd('sinusoid')}). Medium vs baseline forecast: {pct(V['cap_med_vs_base'])}; large vs baseline: {pct(V['cap_large_vs_base'])}.''',
        '`07_FIGURES/ABLATION/capacity_fig1_capacity_vs_rmse.png`, `capacity_fig2_family_comparison.png`, `capacity_fig4_tradeoff.png`',
        'Bigger encoders fit the observed points much better. Over the whole forecast window they did worse in this one set of runs.',
        '- Reconstruction improves with encoder size (single seed, consistent in all five families).\n- Full-window forecasting was worse for the larger models in this experiment. The controlled 2x2 ablation shows a similar combined effect (D vs A ' + pct(V['d_vs_a']) + ') and splits it between latent size and GRU width.',
        '- Single seed with uncontrolled data stream: the same 8/64 architecture rerun under a controlled stream (cell A) forecasts 22% worse than this earlier run (0.3998 vs 0.3265), showing how much a different data stream alone can change one run. - Latent dimension and GRU width were changed together. - The cause of the forecast differences is not established. - An earlier version of the written capacity report contained unsupported statements and was corrected (see its correction notice).')

exp_doc('CONTROLLED_ABLATION_2X2.md', 'Experiment E: controlled 2x2 ablation (latent dimension x GRU width)', 'Separate the effects of latent dimension and encoder (GRU) width on reconstruction and forecasting under identical data streams, and audit checkpoint selection.',
        'Four models trained with the identical recipe (300 epochs, seed 42, `data_seed` 42 so batch order and training observation draws are identical across models; 33,000 optimiser steps each); checkpoints selected by validation negative ELBO only; evaluated on the same 750 test trajectories with identical observations. Paired bootstrap, Wilcoxon (Holm), factorial effects (latent main effect, GRU main effect, interaction).',
        '| Model | Latent dimension | GRU width |\n|---|---:|---:|\n| A | 8 | 64 |\n| B | 16 | 64 |\n| C | 8 | 128 |\n| D | 16 | 128 |', 'Five-family test split (750 trajectories, 150 per family).', '**SINGLE-SEED** (one training run per model). Test-trajectory uncertainty is quantified; **training-seed uncertainty is not**.',
        f'''{TB_AB}

Linear interpolation: {LI('overall')} (reconstruction). Hold-last {HLd('overall')}; constant velocity {CV('overall')} (forecast).

**Per family (change vs A; Holm-adjusted p of the paired difference; "ns" = not significant)**

{TB_PF}

**Factorial effects (positive = higher error; 95% bootstrap interval over test trajectories)**

{TB_FX}

**A vs B (latent dimension only):** reconstruction {pct(rel('B', 'overall', 'recon_rmse'))}; forecast {pct(rel('B', 'overall', 'forecast_rmse'))} (Holm p = {pv(bP['wilcoxon_p_holm'])}), significant only for circle ({pct(rel('B', 'circle', 'forecast_rmse'), 0)}) and sinusoid ({pct(rel('B', 'sinusoid', 'forecast_rmse'), 0)}).
**A vs C (GRU width only):** reconstruction {pct(rel('C', 'overall', 'recon_rmse'))}; forecast {pct(rel('C', 'overall', 'forecast_rmse'))} (mean difference {cP['mean_difference']:+.4f}, bootstrap 95% interval [{cP['bootstrap_ci95_mean_difference'][0]:+.4f}, {cP['bootstrap_ci95_mean_difference'][1]:+.4f}], median difference {cP['median_difference']:+.4f}, better on {AB['fraction_of_trajectories_better_than_A']['C']['overall']['forecast_rmse'] * 100:.0f}% of trajectories, Holm p = {pv(cP['wilcoxon_p_holm'])}): **no clearly distinguishable overall forecasting difference**; worse on sinusoid ({pct(rel('C', 'sinusoid', 'forecast_rmse'), 0)}), better on ellipse ({pct(rel('C', 'ellipse', 'forecast_rmse'), 0)}).
**A vs D (both):** reconstruction {pct(rel('D', 'overall', 'recon_rmse'))}; forecast {pct(rel('D', 'overall', 'forecast_rmse'))} (Holm p = {pv(dP['wilcoxon_p_holm'])}), higher than A on all five families (significant on four; not figure-eight).
**Interaction:** overall forecast interaction {fI['mean']:+.3f} (Holm p = {pv(fI['wilcoxon_p_holm'])}): raising latent size changes forecast error by {AB['paired']['B_minus_A/overall/forecast_rmse']['mean_difference']:+.3f} with GRU 64 but by {AB['paired']['D_minus_C/overall/forecast_rmse']['mean_difference']:+.3f} with GRU 128, so the two changes **reinforce** each other overall (the sign is opposite for sinusoid). The "GRU main effect" is carried by D versus B, not by C versus A.

**Checkpoint selection (validation split only):** selected epochs {', '.join(str(AB['models'][m]['selected_epoch']) for m in CELLS)} of 300; selecting by lowest validation forecast RMSE instead would change C by {sel('runs.cell C (8/128).forecast_gap_selected_vs_min', '.1%')} but would not change the picture for B and D. Classical overfitting (rising validation loss) was not observed. Controls verified: all four cells ran 300 epochs, identical training arguments except latent/GRU, `data_seed` 42, 33,000 steps, identical test observations.''',
        '`07_FIGURES/ABLATION/fig5_ablation_overall.png`, `fig7_effects_vs_seed_variation.png` (new); `ablation_fig1_cells_by_family.png`, `ablation_fig2_interaction.png`, `ablation_fig4_training_histories.png`, `ablation_fig5_percent_improved.png`, `ablation_fig6_examples.png` (original)',
        'We changed the encoder in two ways, separately and together. Bigger encoders fill the gaps much better. For forecasting, making the latent space larger was associated with worse full-window results, but making only the GRU wider made no clear difference. So we must not say "both factors worsen forecasting".',
        '- Reconstruction improves with each change (supported, single seed, all five families).\n- Full-window forecasting: worse for B (+30%) and D (+48%); C not distinguishable overall.\n- The effect is not additive (significant interaction) and differs by family.\n- Nothing here proves a cause.',
        f"- **Single seed per model.** Retraining the same architecture under a different data stream changed forecast error by {pct(ab('reproducibility_pairs.A_vs_orig_baseline.overall.forecast_rmse.relative_difference'))} (A vs the earlier 8/64 run) and {pct(ab('reproducibility_pairs.D_vs_orig_medium.overall.forecast_rmse.relative_difference'))} (D vs the earlier 16/128 run), and the five-seed two-family study has a forecast coefficient of variation of about {ssf['std'] / ssf['mean'] * 100:.0f}%; the forecast differences above are of that order.\n- Initial weights and reparameterisation noise differ between architectures (unavoidable).\n- All cells were still improving at epoch 300.\n- Do not claim that capacity universally worsens forecasting, or that the training objective is the cause (untested).")

exp_doc('FORECAST_ERROR_OVER_TIME.md', 'Experiment F: forecasting error over the prediction horizon', 'See how forecast error changes as the prediction moves further beyond the last observation, and how the four 2x2 models differ.',
        'From the saved test predictions of the four ablation models: (a) RMSE across the 750 test trajectories at every query time (saved in the results file); (b) mean per-trajectory RMSE inside windows 7-8, 8-9 and 9-10 (computed from the saved predictions).', 'Models A, B, C, D of the controlled 2x2 ablation.', 'Five-family test split (750 trajectories).', '**SINGLE-SEED.**',
        f'''{TB_WN}

- **Early horizon (7 < t <= 8):** B, C and D have **lower** error than A ({V['win'][('B', (7, 8))]:.3f}, {V['win'][('C', (7, 8))]:.3f}, {V['win'][('D', (7, 8))]:.3f} vs {V['win'][('A', (7, 8))]:.3f}).
- **Crossing:** the across-trajectory RMSE of B, C and D first exceeds that of A at t = {V['cross']['B']:.2f} (B), {V['cross']['C']:.2f} (C), {V['cross']['D']:.2f} (D).
- **Late horizon (9 < t <= 10):** all three larger models are worse ({V['win'][('B', (9, 10))]:.3f}, {V['win'][('C', (9, 10))]:.3f}, {V['win'][('D', (9, 10))]:.3f} vs {V['win'][('A', (9, 10))]:.3f}). At t = 10 the across-trajectory RMSE is A {et['A']['overall'][-1]:.3f}, B {et['B']['overall'][-1]:.3f}, C {et['C']['overall'][-1]:.3f}, D {et['D']['overall'][-1]:.3f}.
- Before t = 7 (observed period) the larger models are also better (that part includes the observed times).''',
        '`07_FIGURES/FORECASTING/fig6_forecast_error_vs_time.png` (new), `07_FIGURES/FORECASTING/ablation_fig3_error_vs_time.png` (original, per family)',
        'Errors always grow as we look further into the future. The bigger models start out more accurate right after the last observation but their errors grow faster, so they overtake the smallest model around t = 8 to 8.6 and end up worse by t = 10. A shorter forecast window would give the opposite conclusion for B and D.',
        'The full-window "worse forecasting" of the latent-size changes comes from the long-horizon part of the window (faster error growth).', 'Single seed; the cause of the faster error growth is not established (training only supervises observed points; that is a hypothesis, not a test result). Different aggregation (across trajectories at each time vs mean per-trajectory over a window) is used for the crossing times and the window table; both are stated.')

exp_doc('CONDITIONING_EXPERIMENTS.md', 'Experiment G: dynamics-conditioning experiments', 'The shared model forecasts worse than specialists. Does letting the encoder infer an extra "dynamics code" that controls the ODE close that gap?',
        'The encoder outputs, besides the initial state, a 4-dimensional code c = tanh(W_c h) (no KL term). The ODE uses dz/dt = f(z, c), dc/dt = 0 (constant per trajectory). A parameter-matched control without the code and an ablation with extra freely evolving latent dimensions are trained under identical recipes; `data_seed` = seed so all models at a given seed see identical training data; 5 seeds (M2aug 3). Statistics: seed-level mean +/- SD, seed-paired t-tests, trajectory-level paired bootstrap / Wilcoxon on seed-averaged errors.',
        'M1 shared Model D (24,602 parameters); M2 unconditioned, parameter-matched (dynamics width 67; 25,049); M3 dynamics-conditioned (code c in R^4; 25,118); M2aug (latent 12; 25,894).', 'Two-family data (circle + spiral); 300 test trajectories.', '**MULTI-SEED** (5 seeds; M2aug 3 seeds).',
        f'''**Mean +/- SD over seeds**

{TB_CN}

Seed-paired M3 minus M1 forecast: circle {pct(D['cond']['paired']['M3_vs_M1']['circle']['forecast_rmse']['rel_diff_pct'] / 100, 0)} and spiral {pct(D['cond']['paired']['M3_vs_M1']['spiral']['forecast_rmse']['rel_diff_pct'] / 100, 0)} when averaged over seeds (seed-level paired p = {pv(D['cond']['paired']['M3_vs_M1']['circle']['forecast_rmse']['seed_paired_t_p'])} and {pv(D['cond']['paired']['M3_vs_M1']['spiral']['forecast_rmse']['seed_paired_t_p'])}: not distinguishable across seeds). M3 reconstructs very slightly better (circle {pct(D['cond']['paired']['M3_vs_M1']['circle']['recon_rmse']['rel_diff_pct'] / 100, 1)}). The learned code does control radial growth (probe and sensitivity analyses), so it is *used*.''',
        '`07_FIGURES/ARCHITECTURE/conditioning_fig1_architecture_m3.png`, `07_FIGURES/FORECASTING/conditioning_fig5_model_comparison.png`, `conditioning_fig6_error_vs_horizon.png`, `conditioning_fig9_best_median_worst_M3.png`; reconstruction examples in `07_FIGURES/RECONSTRUCTION/`',
        'We tried to give the model an extra "dial" that tells the dynamics what kind of motion this is. The dial was used, but forecasts did not improve; they were, on average, a little worse.',
        '**The hypothesis (conditioning closes the shared-vs-specialist gap) is not supported at this scale.** Conditioning an ODE on an inferred code is established in the literature (Neural ODE Processes, GOKU-nets, CoDA, LEADS); no novelty is claimed.',
        '- M3 received no tuning of its own (hyper-parameters were tuned for M1), which may favour M1. - Two families only. - Specialists had 3 runs. - M2 and M3 forecasts vary a lot between seeds (SD 0.07 to 0.11).')

exp_doc('GEOLIFE_TRANSFER.md', 'Experiment H: transfer from synthetic shapes to real GPS (GeoLife)', 'Synthetic data are clean and known; real movement is not. Does a model trained on synthetic geometry say anything useful about real GPS trajectories, and does training on real data (from scratch or starting from the synthetic weights) help?',
        'GeoLife GPS Trajectories 1.3 (Microsoft Research; 182 users). Segments are windows of 200 records, each mapped to the model\'s time grid [0, 10]; 30 observations (t <= 7); the task is to interpolate withheld recorded GPS positions, scored in metres at the recorded timestamps. Thresholds were fixed from a dataset audit before any model was run. Users are split: development users (pilot, normalisation strategy, and later training and validation of the GeoLife-trained models) and held-out test users; the test set has 1,000 segments from 97 users. Confidence intervals use a bootstrap over users.',
        'Linear interpolation, cubic spline, PCHIP (classical baselines); the frozen synthetic three-family model; a Latent ODE of the same architecture trained from scratch on real GPS windows; the synthetic weights fine-tuned on real GPS windows.', 'GeoLife (real GPS; mostly Beijing; 200-record dense windows). **Data and trained weights are not redistributed** (Microsoft Research non-commercial licence); only aggregate statistics are used here.', '**SINGLE-SEED** (one training run per GeoLife-trained model).',
        f'''{TB_GL}

**Median error by number of observations (m)**

{TB_DN}

**Paired comparison, fine-tuned vs from-scratch (computed in this analysis from the saved per-segment results; ratio < 1 favours fine-tuning)**

{TB_PA}

- The frozen synthetic model is about {gl('interpolation.30.frozen_three_shapes.geometric_mean_ratio_vs_linear', '.0f')} times worse than linear interpolation; the two GeoLife-trained models are {gl('interpolation.30.geolife_finetuned_from_three_shapes.geometric_mean_ratio_vs_linear', '.1f')} to {gl('interpolation.30.geolife_scratch.geometric_mean_ratio_vs_linear', '.1f')} times worse and beat linear interpolation on only 0.1% of segments.
- **Forecasting (protocol A, median metres):** scratch {gl('forecast_n30.A.geolife_scratch.median', '.0f')}, fine-tuned {gl('forecast_n30.A.geolife_finetuned_from_three_shapes.median', '.0f')} vs hold-last {gl('forecast_n30.A.hold_last.median', '.0f')} and constant velocity {gl('forecast_n30.A.const_velocity.median', '.0f')}: worse than both trivial baselines.
- **Synthetic pre-training vs scratch:** fine-tuning is clearly better only at **30 observations** (ratio {V['pa'][30]['gmr']:.2f}), the count the models were trained with; it is clearly worse at 10, 20, 40 and 50 observations ({V['pa'][10]['gmr']:.2f}, {V['pa'][20]['gmr']:.2f}, {V['pa'][40]['gmr']:.2f}, {V['pa'][50]['gmr']:.2f}). No general benefit of synthetic pre-training is shown.''',
        '`07_FIGURES/GENERALIZATION/fig8_geolife_transfer.png` (new), plus aggregate original figures `geolife_fig5_error_distribution.png`, `geolife_fig6_error_vs_observation_count.png`, `geolife_fig7_characteristics.png`, `geolife_fig8_error_vs_gap.png`, `geolife_fig9_synthetic_vs_geolife_trained.png` (figures that draw individual users\' paths are deliberately **not** included)',
        'Real GPS tracks are smooth and irregular; simply joining the observed dots is much more accurate than anything our learned models produce. Training on real data helps, but not enough.',
        '- The synthetic-trained model does **not** transfer to real GPS.\n- Real-data training improves on the frozen model (about 2 to 3 times) but remains 6 to 8 times worse than linear interpolation.\n- Possible reason (hypothesis): the learned dynamics produce loop-like curves.',
        '- One dataset, one training run per model, models trained with 30 observations only. - The fine-tuned-vs-scratch pattern (better at 30, worse elsewhere) is single-run and its explanation (fine-tuned model more tied to the training observation count) is untested. - The GeoLife-trained weights are not distributed (licence).')
