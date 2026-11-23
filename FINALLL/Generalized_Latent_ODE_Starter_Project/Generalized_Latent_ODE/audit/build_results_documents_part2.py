# executed from build_results_documents.py (shares its namespace): summary for the supervisor, slide content, source index
S = []; s_ = S.append
s_('# Results summary for the supervisor (simple English)')
s_('')
s_('Date: 2026-11-21. Everything below is taken from results already saved in the project (see `RESULTS_SOURCE_INDEX.md`). Nothing was re-run. **Almost all results are from a single training run per model unless marked "5 runs".** A single run can be lucky or unlucky, so single-run differences of 20% or less should not be treated as findings.')
s_('')
s_('## The project in five sentences')
s_('1. We use a *Latent Neural ODE*: a neural network that looks at a few noisy points of a 2-D path, turns them into a hidden starting state, evolves that state forward in continuous time with a learned rule, and draws the path.')
s_('2. It is an adapted re-implementation and extension of Chen et al. (2018), **not** an exact replication and **not** a new or improved forecasting model.')
s_('3. We tested it on five path types (circle, spiral, ellipse, figure-eight, sinusoid) with one shared model and no shape label.')
s_('4. We asked: does one model for many path types lose accuracy; does it generalise to path types it never saw; and how does model size matter?')
s_('5. Every model sees only 30 noisy points from times 0 to 7 and must (a) fill in the gaps inside that period ("reconstruction") and (b) predict times 7 to 10 ("forecast").')
s_('')
s_('## What we found, step by step')
s_('')
s_(f"**Step 1. One shared model works for all five path types.** It forecasts far better than the simple baselines (cell A forecast error {R('A', 'overall', 'forecast_rmse')} against {HLd('overall')} for staying at the last point). But the smallest models reconstruct no better than joining the dots with straight lines (cell A {R('A', 'overall', 'recon_rmse')} against {LI('overall')}). *(single run)*")
s_('')
s_(f"**Step 2. A shared model is less accurate than specialist models.** For circles and spirals, one model trained on both has {SVR('circle', 'forecast_rmse')['rel'] * 100:.0f}% (circle) and {SVR('spiral', 'forecast_rmse')['rel'] * 100:.0f}% (spiral) higher forecast error than separate models. *(5 shared runs vs 3 specialist runs; clear for circles, weaker for spirals.)*")
s_('')
s_(f"**Step 3. Unseen path types are not handled.** The three-family model, tested on figure-eights and sinusoids it never saw, forecasts worse than just staying at the last point ({Z3('figure_eight', 'forecast_rmse')} vs {HLd('figure_eight')} and {Z3('sinusoid', 'forecast_rmse')} vs {HLd('sinusoid')}). Models trained without ellipses draw ellipses as near-circles (forecast {V['ell_fc_two']['mean']:.2f}, reconstruction {V['ell_rc_two']['mean']:.2f}, against {t3('C_out_of_distribution_ellipses.const_pos_on_same_ellipses.recon_rmse.mean', '.2f')} for joining the dots). *(ellipse: 5 runs; figure-eight and sinusoid: 1 run)*")
s_('')
s_(f"**Step 4. Bigger encoders reconstruct better (single-run result).** In the controlled 2x2 experiment reconstruction error falls by {-rel('B', 'overall', 'recon_rmse') * 100:.0f}% (latent 16), {-rel('C', 'overall', 'recon_rmse') * 100:.0f}% (GRU 128) and {-rel('D', 'overall', 'recon_rmse') * 100:.0f}% (both), and it is lower in every family.")
s_('')
s_(f"**Step 5. Forecasting with bigger encoders is mixed.** Over the whole forecast window, latent 16 forecasts {rel('B', 'overall', 'forecast_rmse') * 100:.0f}% worse and both together {rel('D', 'overall', 'forecast_rmse') * 100:.0f}% worse, but widening the GRU alone is not clearly different ({rel('C', 'overall', 'forecast_rmse') * 100:.0f}% higher, not significant). The larger models are actually **better** just after t = 7 and become worse after about t = 8 to 8.6. *(single run; the differences are as large as the variation you get by simply retraining the same architecture, about 21 to 22%, so this is a hint, not a finding.)*")
s_('')
s_(f"**Step 6. Real GPS data defeats the synthetic-trained model.** Straight-line interpolation has a median error of {gl('interpolation.30.linear.median', '.1f')} m; our synthetic model {gl('interpolation.30.frozen_three_shapes.median', '.0f')} m. Training on real GPS data brings it to {gl('interpolation.30.geolife_scratch.median', '.0f')} m (scratch) or {gl('interpolation.30.geolife_finetuned_from_three_shapes.median', '.0f')} m (fine-tuned), still about 6 to 8 times worse than the straight line. *(single run)*")
s_('')
s_('## Answers to the ten questions')
s_('')
orig_med_rc = AB['results']['orig_medium']['overall']['recon_rmse']['mean']
s_(f"1. **What experiments have we completed?** Failure diagnosis and Model D; 5-seed reproducibility; shared vs specialist; three families and unseen ellipses; dynamics conditioning (5 seeds); observation-count sensitivity; GeoLife transfer and GeoLife training; the five-family dataset with zero-shot test; capacity scaling; the controlled 2x2 ablation; and an independent audit.")
s_(f"2. **Most important numbers.** Shared vs specialist forecast: circle {SVR('circle', 'forecast_rmse')['u_mean']:.3f} vs {SVR('circle', 'forecast_rmse')['s_mean']:.3f}, spiral {SVR('spiral', 'forecast_rmse')['u_mean']:.3f} vs {SVR('spiral', 'forecast_rmse')['s_mean']:.3f}. Unseen families (3-family model): figure-eight {Z3('figure_eight', 'forecast_rmse')}, sinusoid {Z3('sinusoid', 'forecast_rmse')} vs hold-last {HLd('figure_eight')}, {HLd('sinusoid')}. Ablation (A, B, C, D): reconstruction {R('A', 'overall', 'recon_rmse')}, {R('B', 'overall', 'recon_rmse')}, {R('C', 'overall', 'recon_rmse')}, {R('D', 'overall', 'recon_rmse')}; forecast {R('A', 'overall', 'forecast_rmse')}, {R('B', 'overall', 'forecast_rmse')}, {R('C', 'overall', 'forecast_rmse')}, {R('D', 'overall', 'forecast_rmse')}. GeoLife (30 obs): linear {gl('interpolation.30.linear.median', '.1f')} m vs synthetic model {gl('interpolation.30.frozen_three_shapes.median', '.1f')} m.")
s_(f"3. **Best at reconstruction?** On the five-family data the larger encoders: cell D {R('D', 'overall', 'recon_rmse')} (and the earlier 16/128 run {orig_med_rc:.4f}, practically the same recipe), against {R('A', 'overall', 'recon_rmse')} for the smallest model and {LI('overall')} for linear interpolation. On circle and spiral data the specialists are slightly better than the shared model (circle {SVR('circle', 'recon_rmse')['s_mean']:.4f} vs {SVR('circle', 'recon_rmse')['u_mean']:.4f}). (One training run each, except the specialist comparison.)")
s_(f"4. **Best at forecasting?** The smallest models: on the five-family data the earlier 8/64 run ({R('orig_baseline', 'overall', 'forecast_rmse')}) and its controlled rerun A ({R('A', 'overall', 'forecast_rmse')}) are better over the full window than the larger ones; on circle and spiral the specialists ({SVR('circle', 'forecast_rmse')['s_mean']:.3f}, {SVR('spiral', 'forecast_rmse')['s_mean']:.3f}). The larger models are better only in the first time unit after t = 7. The runs are single, so the ranking among the smaller variants (8/64 against 8/128) is not firm.")
s_('5. **Is the shared model better than specialists?** No. It is worse at forecasting (clearly for circle) and slightly worse at reconstruction; its advantage is convenience (one model, no labels).')
s_('6. **Does it generalise to unseen families?** No for figure-eight and sinusoid (worse than simple baselines); only partially for ellipses (direction and rough forecast right, shape wrong, reconstruction poor); no for real GPS.')
s_('7. **What does the controlled ablation demonstrate?** That bigger encoders reconstruct better, and that in one set of runs latent size is associated with worse late-horizon forecasts while GRU width alone is not clearly different. It does **not** demonstrate that capacity hurts forecasting in general or why.')
s_('8. **Have we developed a better forecasting model?** **No.** We re-implemented and analysed the Latent ODE; no new method has been proposed or tested, and the larger variants did not forecast better over the full window.')
s_('9. **What is strong enough to present?** The verified implementation and data; the shared-vs-specialist cost; the failure to generalise to unseen families and to real GPS; reconstruction gains from larger encoders; the ablation as an exploratory single-run result with its caveats.')
s_('10. **Most important remaining questions.** Do the ablation effects repeat across seeds? Why do shared models forecast worse (and can the gap be closed)? Does a model trained on four families handle the fifth? Is the observed-points-only training objective limiting forecasts? How do stronger baselines (fit a curve per family, a sequence model) compare?')
s_('')
s_('## Key tables')
s_('')
s_('**Five-family models: mean error (one training run each).**')
s_('')
s_('| Family | Linear interp. (recon) | A recon | D recon | Hold-last (forecast) | A forecast | D forecast |')
s_('|---|---:|---:|---:|---:|---:|---:|')
for g in G6: s_(f"| {FL[g]} | {LI(g)} | {R('A', g, 'recon_rmse')} | {R('D', g, 'recon_rmse')} | {HLd(g)} | {R('A', g, 'forecast_rmse')} | {R('D', g, 'forecast_rmse')} |")
s_('')
s_('**Shared vs specialist (circle + spiral data).**')
s_('')
s_('| Family | Forecast: shared (5 runs) | Forecast: specialist (3 runs) | Difference | Statistical evidence |')
s_('|---|---:|---:|---:|---|')
for fam_ in ('circle', 'spiral'):
    q = SVR(fam_, 'forecast_rmse'); s_(f"| {fam_} | {q['u_mean']:.3f} +/- {q['u_sd']:.3f} | {q['s_mean']:.3f} +/- {q['s_sd']:.3f} | {pct(q['rel'], 0)} | all 5 shared runs worse than all 3 specialist runs; Welch p = {pv(q['welch'])} ({pv(q['welch_bonf'])} after Bonferroni) |")
s_('')
s_('**Unseen families (three-family model, one run) and ellipses (two-family models, 5 runs).**')
s_('')
s_('| Unseen family | Forecast | Hold-last | Reconstruction | Linear interp. |')
s_('|---|---:|---:|---:|---:|')
for g in ('figure_eight', 'sinusoid'): s_(f"| {FL[g]} | {Z3(g, 'forecast_rmse')} | {HLd(g)} | {Z3(g, 'recon_rmse')} | {LI(g)} |")
s_(f"| ellipse (5 runs, mean) | {V['ell_fc_two']['mean']:.4f} | {t3('C_out_of_distribution_ellipses.const_pos_on_same_ellipses.forecast_rmse.mean', '.4f')} | {V['ell_rc_two']['mean']:.4f} | {t3('C_out_of_distribution_ellipses.const_pos_on_same_ellipses.recon_rmse.mean', '.4f')} |")
s_('')
s_('**Controlled 2x2 ablation (one run each).**')
s_('')
s_('| Model | Latent / GRU | Reconstruction | Forecast | Forecast vs A |')
s_('|---|---|---:|---:|---:|')
for m in CELLS: s_(f"| {m} | {ab(f'models.{m}.latent_dim')} / {ab(f'models.{m}.encoder_hidden')} | {R(m, 'overall', 'recon_rmse')} | {R(m, 'overall', 'forecast_rmse')} | {'n/a' if m == 'A' else pct(rel(m, 'overall', 'forecast_rmse'))} |")
s_('')
s_('## What to say, and what not to say')
s_('**Safe to say:** the model works; one shared model costs forecast accuracy for circle and spiral; it does not generalise to unseen path types or to real GPS; bigger encoders reconstruct better; the full-window forecast difference between model sizes is exploratory, appears late in the horizon, and is the same size as seed-to-seed noise.')
s_('**Do not say:** that capacity hurts forecasting in general; that we found why; that we replicated the paper; that we built a better forecasting model; that the five-family model shows generalisation (it saw all five families).')
(OUT / 'RESULTS_SUMMARY_FOR_SUPERVISOR.md').write_text('\n'.join(S) + '\n'); print('RESULTS_SUMMARY_FOR_SUPERVISOR.md', len(S), 'lines')

# =====================================================================================================================  PPT_RESULTS_CONTENT.md
T = []; t_ = T.append
t_('# Presentation content: results slides and speaking script')
t_('')
t_('Prepared for the meeting with the supervisor ("Ma\'am"). 13 slides; figures are in `presentation_results/figures/` (new, drawn from saved data) or the existing project figures named below. Numbers are exact (4 decimals in tables, rounded in speech). **Say "single run" whenever a result comes from one training run.**')
t_('')
def slide(n, title, bullets, fig, numbers, meaning, script, limit):
    t_(f'## Slide {n}: {title}')
    t_('')
    for b in bullets: t_(f'- {b}')
    t_('')
    t_(f'**Figure / table:** {fig}')
    t_('')
    t_(f'**Exact numbers:** {numbers}')
    t_('')
    t_(f'**What it means (plain English):** {meaning}')
    t_('')
    t_(f'**Speaking script:** "{script}"')
    t_('')
    t_(f'**Limitation to state:** {limit}')
    t_('')
slide(1, 'What we built and what we asked',
      ['A Latent Neural ODE: noisy points in, hidden start state, learned continuous-time rule, curve out.', 'An adapted re-implementation and extension of Chen et al. (2018), not an exact replication and not a new method.', 'Five path types (circle, spiral, ellipse, figure-eight, sinusoid), one shared model, no shape label.', 'Each model sees 30 noisy points from t = 0 to 7, then must reconstruct the gaps and forecast t = 7 to 10.'],
      'Existing figure `results/five_shapes/figures/five_shapes_gallery.png` (the five families).',
      f"{ab('models.A.n_parameters', ',')} parameters in the baseline model; 1,000 trajectories per family; test set 750 trajectories (150 per family).",
      'We are testing how well one learned model copes with different kinds of motion, and how much model size matters.',
      'Our project builds on the Neural ODE paper. We did not copy its experiment: we used our own synthetic paths, five types in all. The model sees only thirty noisy points from the first seven time units. We measure two things: how well it fills in the gaps, and how well it predicts the future from seven to ten.',
      'This is a re-implementation and extension, not a replication, and we have not developed a new method.')
slide(2, 'How we measure success: simple baselines',
      ['Reconstruction error = distance from the true path at unobserved times inside t <= 7 (lower is better).', 'Forecast error = distance from the true path for 7 < t <= 10 (lower is better).', 'Baselines: linear interpolation (join the dots), hold-last (stay at the last point), constant velocity (keep moving at the last speed).', 'A learned model is only useful if it beats these.'],
      'Table (below).',
      f"All-five-families baselines: linear interpolation {LI('overall')} (reconstruction); hold-last {HLd('overall')}; constant velocity {CV('overall')} (forecast).",
      'Straight lines are already quite good for filling gaps, so a neural model must do better than 0.067 to be worth it; for forecasts, staying still gives about 1.0.',
      'Before showing results I need to explain the yardsticks. Our error measure is the average distance from the true path. For gaps inside the observed period, the simple comparison is joining the dots with straight lines. For the future, the simple comparisons are staying at the last point, or continuing at the last speed.',
      'These baselines are deliberately simple; stronger classical baselines (a curve fitted per family) were not tested.')
t_('| Baseline | Task | Mean error, all five families |')
t_('|---|---|---:|')
t_(f"| Linear interpolation | reconstruction | {LI('overall')} |")
t_(f"| Hold-last | forecast | {HLd('overall')} |")
t_(f"| Constant velocity | forecast | {CV('overall')} |")
t_('')
slide(3, 'Result 1: reconstruction by family',
      ['Larger encoders (B, C, D) reconstruct better than the smallest model A in every family.', f"A is worse than joining the dots on four of five families; B and D beat it on all five.", f"Sinusoid is the easiest family to reconstruct (lowest in {V['n_recon_easiest_sinusoid']} of 7 models); figure-eight is the hardest ({V['n_recon_hardest_fig8']} of 7).", 'All numbers are single training runs.'],
      '`figures/fig1_reconstruction_by_family.png`',
      f"Overall: A {R('A', 'overall', 'recon_rmse')}, B {R('B', 'overall', 'recon_rmse')}, C {R('C', 'overall', 'recon_rmse')}, D {R('D', 'overall', 'recon_rmse')}; linear interpolation {LI('overall')}. Figure-eight: A {R('A', 'figure_eight', 'recon_rmse')}, D {R('D', 'figure_eight', 'recon_rmse')}.",
      'The bigger encoders fit the observed part much more tightly. The figure-eight is hard because it turns sharply and crosses itself.',
      'On this slide, lower bars are better and the black dash is straight-line interpolation. The smallest model, in blue, is worse than the dash in most families. Models B and D, in orange and yellow, are clearly below it in every family; C, in green, is about level with it. The wave-shaped sinusoid is the easiest family; the figure-eight is the hardest.',
      'Single training run per model; reconstruction is what the training loss optimises directly, so this improvement is expected.')
slide(4, 'Result 2: forecasting by family',
      ['All models forecast far better than hold-last overall.', 'The hardest family to forecast depends on size: spiral for the smallest models, sinusoid for the larger ones.', f"For B and D the sinusoid forecast is worse than the hold-last baseline ({R('B', 'sinusoid', 'forecast_rmse')} and {R('D', 'sinusoid', 'forecast_rmse')} vs {HLd('sinusoid')}).", 'Circle and ellipse are generally the easiest.'],
      '`figures/fig2_forecast_by_family.png`',
      f"Overall forecast: A {R('A', 'overall', 'forecast_rmse')}, B {R('B', 'overall', 'forecast_rmse')}, C {R('C', 'overall', 'forecast_rmse')}, D {R('D', 'overall', 'forecast_rmse')}; hold-last {HLd('overall')}. Circle A {R('A', 'circle', 'forecast_rmse')}; spiral A {R('A', 'spiral', 'forecast_rmse')}.",
      'The model does not perform equally on all families: forecasting a wave drifting sideways is hard for the bigger models, and spirals are hard for the small ones.',
      'For forecasting, the models are well below the staying-still baseline in almost every case. The one clear exception is the sinusoid, where two of the larger models do worse than just staying at the last point.',
      'Single run per model; the constant-velocity baseline (2.0 to 2.6) is off the chart.')
slide(5, 'Result 3: one shared model vs specialist models (circle + spiral)',
      [f"Shared model (5 runs) vs specialists (3 runs): circle forecast {SVR('circle', 'forecast_rmse')['u_mean']:.3f} vs {SVR('circle', 'forecast_rmse')['s_mean']:.3f} ({pct(SVR('circle', 'forecast_rmse')['rel'], 0)}); spiral {SVR('spiral', 'forecast_rmse')['u_mean']:.3f} vs {SVR('spiral', 'forecast_rmse')['s_mean']:.3f} ({pct(SVR('spiral', 'forecast_rmse')['rel'], 0)}).", 'Every shared run was worse than every specialist run.', f"Reconstruction is close ({pct(SVR('circle', 'recon_rmse')['rel'], 0)} and {pct(SVR('spiral', 'recon_rmse')['rel'], 0)}).", 'Shared forecasts also vary more from run to run.'],
      '`figures/fig3_shared_vs_specialist.png`',
      f"Welch p for circle forecast = {pv(SVR('circle', 'forecast_rmse')['welch'])} ({pv(SVR('circle', 'forecast_rmse')['welch_bonf'])} after Bonferroni); spiral forecast p = {pv(SVR('spiral', 'forecast_rmse')['welch'])} ({pv(SVR('spiral', 'forecast_rmse')['welch_bonf'])} after Bonferroni, not significant). SD across runs: shared {SVR('circle', 'forecast_rmse')['u_sd']:.3f} / {SVR('spiral', 'forecast_rmse')['u_sd']:.3f}; specialists {SVR('circle', 'forecast_rmse')['s_sd']:.3f} / {SVR('spiral', 'forecast_rmse')['s_sd']:.3f}.",
      'Using one model for several families is convenient but costs forecast accuracy: specialists are better.',
      'This is the main multi-run experiment. Each dot is one training run. The blue shared model is clearly worse at forecasting than the orange specialists, and its runs vary more. For circles this holds even after correcting for multiple tests; for spirals the difference is in the same direction but weaker.',
      'Only two families have specialists, and only three specialist runs exist.')
slide(6, 'Result 4: unseen families: no zero-shot generalisation',
      [f"Three-family model on unseen figure-eight / sinusoid forecasts worse than hold-last ({Z3('figure_eight', 'forecast_rmse')} vs {HLd('figure_eight')}; {Z3('sinusoid', 'forecast_rmse')} vs {HLd('sinusoid')}).", f"Reconstruction is far worse than joining the dots ({Z3('figure_eight', 'recon_rmse')} vs {LI('figure_eight')}; {Z3('sinusoid', 'recon_rmse')} vs {LI('sinusoid')}).", 'The five-family model has seen all five, so its results are not zero-shot.', 'On the three original families the five-family model is worse than the three-family one (forecast ' + ', '.join(pct(c5(f'paired.{g}/forecast_rmse.relative_change_of_mean'), 0) for g in FAMS[:3]) + ').'],
      '`figures/fig4_cross_family_generalisation.png`',
      f"Five-family model forecast: circle {F5('circle', 'forecast_rmse')}, spiral {F5('spiral', 'forecast_rmse')}, ellipse {F5('ellipse', 'forecast_rmse')}, figure-eight {F5('figure_eight', 'forecast_rmse')}, sinusoid {F5('sinusoid', 'forecast_rmse')}. Training steps {V['steps3']:,} (3-family) vs {ab('models.A.optimizer_steps', ',')} (5-family).",
      'A model trained on loops and spirals cannot suddenly draw figure-eights or waves. And adding two families to the training set cost some accuracy on the old ones, though training size and steps also changed.',
      'In the shaded rows the blue filled dot is the three-family model on families it never saw. It is far from the baselines in the wrong direction. The hollow dot is the five-family model, which did see those families, so it is not a generalisation test.',
      'One frozen model, one seed; no leave-one-family-out experiment.')
slide(7, 'Result 5: unseen ellipses: partial transfer (5 runs)',
      [f"Models trained on circle + spiral only, tested on ellipses: forecast {V['ell_fc_two']['mean']:.3f} +/- {V['ell_fc_two']['std']:.3f}, much better than hold-last ({t3('C_out_of_distribution_ellipses.const_pos_on_same_ellipses.forecast_rmse.mean', '.2f')}).", f"Reconstruction {V['ell_rc_two']['mean']:.3f}, more than 3 times worse than joining the dots ({t3('C_out_of_distribution_ellipses.const_pos_on_same_ellipses.recon_rmse.mean', '.3f')}).", 'Rotation direction correct for 100% of ellipses and shape wrong (seed-42 model): predicted axis ratio ' + f"{t3('C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).axis_ratio_pred_mean', '.2f')} vs true {t3('C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).axis_ratio_true_mean', '.2f')}.", 'An ellipse is a stretched circle, so this is the easiest unseen family.'],
      'Table (below).', f"In-training reference (three-family model): reconstruction {t3('C_out_of_distribution_ellipses.three_shape_model_on_same_ellipses (in-distribution reference).recon_rmse.mean', '.3f')}, forecast {t3('C_out_of_distribution_ellipses.three_shape_model_on_same_ellipses (in-distribution reference).forecast_rmse.mean', '.3f')}.",
      'The model keeps the rotation and roughly the size but turns ellipses into near-circles.',
      'Ellipses are the friendliest unseen family. The models still get the direction right and forecast roughly, but they round the shape toward a circle and fail at filling gaps accurately. This is partial transfer, not success.',
      'Based on 5 two-family runs; ellipses are geometrically close to circles.')
t_('| Model on ellipses | Reconstruction | Forecast |')
t_('|---|---:|---:|')
t_(f"| Two-family models (5 runs), unseen | {V['ell_rc_two']['mean']:.4f} | {V['ell_fc_two']['mean']:.4f} |")
t_(f"| Three-family model, seen in training | {t3('C_out_of_distribution_ellipses.three_shape_model_on_same_ellipses (in-distribution reference).recon_rmse.mean', '.4f')} | {t3('C_out_of_distribution_ellipses.three_shape_model_on_same_ellipses (in-distribution reference).forecast_rmse.mean', '.4f')} |")
t_(f"| Linear interpolation / hold-last | {t3('C_out_of_distribution_ellipses.const_pos_on_same_ellipses.recon_rmse.mean', '.4f')} | {t3('C_out_of_distribution_ellipses.const_pos_on_same_ellipses.forecast_rmse.mean', '.4f')} |")
t_('')
slide(8, 'Result 6: controlled 2x2 ablation: reconstruction and forecast',
      ['Four models: latent 8 or 16, GRU 64 or 128; same data, recipe and data stream; one run each.', f"Reconstruction improves: B {pct(rel('B', 'overall', 'recon_rmse'), 0)}, C {pct(rel('C', 'overall', 'recon_rmse'), 0)}, D {pct(rel('D', 'overall', 'recon_rmse'), 0)} vs A.", f"Full-window forecast: B {pct(rel('B', 'overall', 'forecast_rmse'), 0)}, D {pct(rel('D', 'overall', 'forecast_rmse'), 0)}; C {pct(rel('C', 'overall', 'forecast_rmse'), 0)} (not significant).", 'Widening the GRU alone made no clearly distinguishable forecast difference.'],
      '`figures/fig5_ablation_overall.png`',
      f"Parameters: A {ab('models.A.n_parameters', ',')}, B {ab('models.B.n_parameters', ',')}, C {ab('models.C.n_parameters', ',')}, D {ab('models.D.n_parameters', ',')}. C vs A forecast: mean difference {cP['mean_difference']:+.4f}, Holm p = {pv(cP['wilcoxon_p_holm'])}; better on {AB['fraction_of_trajectories_better_than_A']['C']['overall']['forecast_rmse'] * 100:.0f}% of trajectories. Interaction (forecast) {fI['mean']:+.3f}, p = {pv(fI['wilcoxon_p_holm'])}.",
      'Bigger encoders fit the observed part better. For forecasting, increasing latent size is associated with worse full-window results; widening the GRU alone is not clearly different.',
      'This is our most controlled experiment: four models differing in only two settings. Reconstruction improves with size. For forecasting, only the latent-size changes look worse. I want to stress that each bar is a single training run.',
      'Single seed: effects of +6% to +48% are the same size as run-to-run variation (about 21 to 22% for the same architecture).')
slide(9, 'Result 7: forecast error over time',
      ['Larger models are better just after t = 7, worse later.', f"They cross above model A at t = {V['cross']['B']:.2f} (B), {V['cross']['C']:.2f} (C), {V['cross']['D']:.2f} (D).", f"Window 7-8: A {V['win'][('A', (7, 8))]:.3f}, B {V['win'][('B', (7, 8))]:.3f}, C {V['win'][('C', (7, 8))]:.3f}, D {V['win'][('D', (7, 8))]:.3f}; window 9-10: A {V['win'][('A', (9, 10))]:.3f}, B {V['win'][('B', (9, 10))]:.3f}, C {V['win'][('C', (9, 10))]:.3f}, D {V['win'][('D', (9, 10))]:.3f}.", 'So "worse forecasting" depends on the horizon.'],
      '`figures/fig6_forecast_error_vs_time.png`',
      f"RMSE across the 750 test trajectories at t = 10: A {et['A']['overall'][-1]:.3f}, B {et['B']['overall'][-1]:.3f}, C {et['C']['overall'][-1]:.3f}, D {et['D']['overall'][-1]:.3f}.",
      'Errors always grow with time, but they grow faster for the models with larger latent space.',
      'This plot is why I am careful with the word "worse". Right after the forecast starts, the bigger models are better. After about t equals eight they overtake model A and keep growing. A shorter forecast window would give the opposite conclusion.',
      'Single run; the cause (for example the training objective only sees observed points) is a hypothesis we have not tested.')
slide(10, 'Why we are careful: effect sizes vs run-to-run variation',
      [f"Same architecture, two training runs: forecast error differs by {pct(ab('reproducibility_pairs.A_vs_orig_baseline.overall.forecast_rmse.relative_difference'), 0)} and {pct(ab('reproducibility_pairs.D_vs_orig_medium.overall.forecast_rmse.relative_difference'), 0)}.", f"Five seeds, same recipe (two-family data): forecast {min(ssf['values']):.3f} to {max(ssf['values']):.3f}, CV about {ssf['std'] / ssf['mean'] * 100:.0f}%.", f"Ablation forecast effects: {pct(rel('B', 'overall', 'forecast_rmse'), 0)}, {pct(rel('C', 'overall', 'forecast_rmse'), 0)}, {pct(rel('D', 'overall', 'forecast_rmse'), 0)}.", 'So single-run forecast differences of this size are not findings yet.'],
      '`figures/fig7_effects_vs_seed_variation.png`',
      f"Two-family five-seed mean {ssf['mean']:.4f} +/- {ssf['std']:.4f} (95% CI [{ssf['ci95_low']:.3f}, {ssf['ci95_high']:.3f}]).",
      'Retraining the same model gives noticeably different forecasts, so a difference between two single runs may just be luck.',
      'This slide puts the ablation effects on the same scale as the variation we see when we simply retrain the same recipe. They are comparable. That is why I call the ablation exploratory and why multi-seed runs are the next step.',
      'The three groups come from different experiments and are placed on one axis only to show scale.')
slide(11, 'Real-world test: GPS data (GeoLife)',
      [f"Linear interpolation {gl('interpolation.30.linear.median', '.1f')} m vs frozen synthetic model {gl('interpolation.30.frozen_three_shapes.median', '.1f')} m: it does not transfer.", f"Trained on real GPS: scratch {gl('interpolation.30.geolife_scratch.median', '.1f')} m, fine-tuned {gl('interpolation.30.geolife_finetuned_from_three_shapes.median', '.1f')} m: better, but still 6 to 8 times worse than the straight line.", f"Fine-tuning from synthetic weights beats scratch only at 30 observations (ratio {V['pa'][30]['gmr']:.2f}); worse at 10, 20, 40, 50.", 'Forecasts are worse than hold-last and constant velocity.'],
      '`figures/fig8_geolife_transfer.png`',
      f"Median error (30 observations, 1,000 segments, 97 users): cubic {gl('interpolation.30.cubic.median', '.2f')} m, PCHIP {gl('interpolation.30.pchip.median', '.2f')} m. Fine-tuned / scratch ratios: 10 obs {V['pa'][10]['gmr']:.2f}, 20 {V['pa'][20]['gmr']:.2f}, 30 {V['pa'][30]['gmr']:.2f}, 40 {V['pa'][40]['gmr']:.2f}, 50 {V['pa'][50]['gmr']:.2f}.",
      'Real movement is not made of the loops our synthetic data teaches; simple interpolation wins easily.',
      'We also asked whether a model trained on synthetic shapes helps with real GPS tracks. It does not: straight lines are much better. Training on real data helps but does not close the gap. Synthetic pre-training helped only at the thirty observations used in training.',
      'One dataset (mostly Beijing), one run per model; the fine-tuned versus scratch comparison was computed from the saved per-segment results.')
slide(12, 'What our experiments show, and do not show',
      ['Shown: the implementation and data are valid; shared models cost forecast accuracy; no zero-shot generalisation; larger encoders reconstruct better.', 'Exploratory only: forecasts vs model size (single run, late-horizon, same size as run-to-run noise).', 'Not shown: a better forecasting model, a cause for the forecast differences, results across seeds for the five-family work.', 'The project is an adapted re-implementation and experimental extension, not a replication or a new method.'],
      'Table (below).', 'See table.',
      'We have a solid, verified analysis framework and several clear negative results; the main positive result is reconstruction from larger encoders.',
      'To summarise: we can clearly say what does not work: sharing a model costs forecast accuracy, and unseen families and real GPS are not handled. We cannot yet say why forecasting gets worse with size; that needs multiple seeds.',
      'All claims are limited to this synthetic data and these training runs.')
t_('| Question | Answer | Evidence |')
t_('|---|---|---|')
t_('| Shared better than specialist? | No (forecast) | 5 vs 3 runs |')
t_('| Generalises to unseen families? | No / partial for ellipses | 5 runs (ellipse), 1 run (others) |')
t_('| Larger encoders improve reconstruction? | Yes | single run, consistent |')
t_('| Larger encoders worsen forecasts? | Exploratory, late horizon only | single run |')
t_('| Better forecasting model developed? | No | n/a |')
t_('')
slide(13, 'Next steps',
      ['Repeat the 2x2 ablation with several seeds (at least A and D, four more seeds each).', 'Add stronger baselines: a curve fitted per family, and a sequence-model baseline.', 'Leave-one-family-out generalisation tests with several seeds.', 'Test whether training only on observed points limits forecasts (a labelled diagnostic), before proposing any improvement.'],
      'Plan list (no figure).', 'Each repeat is about 80 minutes of CPU training per run, about 0.3 MB per checkpoint.',
      'These are the experiments that would turn our exploratory observations into conclusions.',
      'The most important next step is repeating the controlled ablation with several seeds, because that tells us whether the forecast differences are real. After that, stronger baselines and leave-one-family-out tests.',
      'None of these experiments has been run.')
(OUT / 'PPT_RESULTS_CONTENT.md').write_text('\n'.join(T) + '\n'); print('PPT_RESULTS_CONTENT.md', len(T), 'lines')

# =====================================================================================================================  RESULTS_SOURCE_INDEX.md
I = []; i_ = I.append
i_('# Source index: where every number and figure comes from')
i_('')
i_('Generated automatically by `audit/build_results_documents.py`: every value read through an accessor is logged with its file and key path, so this index lists exactly the values used in the documents. Paths are relative to the project root. "key path" = nested keys in the JSON file joined by dots.')
i_('')
i_('## A. Figures')
i_('')
i_('| Figure (in `presentation_results/figures/`) | Drawn from (saved data only) | Existing original figure with similar content |')
i_('|---|---|---|')
for r in (('fig1_reconstruction_by_family.png', '`results/ablation_2x2/ablation_2x2_results.json` (results.A..D, baselines.linear_interpolation_recon)', '`results/ablation_2x2/figures/fig1_cells_by_family.png`'),
          ('fig2_forecast_by_family.png', '`results/ablation_2x2/ablation_2x2_results.json` (results.A..D, baselines.hold_last_forecast)', '`results/ablation_2x2/figures/fig1_cells_by_family.png`'),
          ('fig3_shared_vs_specialist.png', '`results/validation/shared_vs_separate/shared_vs_separate.json` (circle/spiral.seed_level.*.unified|specialist, values)', '`results/validation/shared_vs_separate/shared_vs_separate.png`'),
          ('fig4_cross_family_generalisation.png', '`results/five_shapes/zeroshot_three_shape_model.json`, `results/five_shapes/compare_three_vs_five.json`', '`results/five_shapes/figures/per_family_rmse_three_vs_five.png`'),
          ('fig5_ablation_overall.png', '`results/ablation_2x2/ablation_2x2_results.json` (results.*.overall)', '`results/ablation_2x2/figures/fig2_interaction.png`'),
          ('fig6_forecast_error_vs_time.png', '`results/ablation_2x2/ablation_2x2_results.json` (error_vs_time)', '`results/ablation_2x2/figures/fig3_error_vs_time.png`'),
          ('fig7_effects_vs_seed_variation.png', '`ablation_2x2_results.json` (results, reproducibility_pairs); `results/validation/seed_study/seed_study.json` (across_seeds.overall.forecast_rmse.values)', 'none'),
          ('fig8_geolife_transfer.png', '`results/geolife/main_evaluation/geolife_trained_summary.json` (interpolation.30.*.median, median_ci95_user_bootstrap)', '`results/geolife/figures/fig3_method_comparison.png`, `fig9_synthetic_vs_geolife_trained.png`')):
    i_(f'| `{r[0]}` | {r[1]} | {r[2]} |')
i_('')
i_('## B. Values computed in this analysis from saved results (not read directly from a result file)')
i_('')
for r in ('Percent differences and ratios between two saved means (for example "+84.9%" shared vs specialist = shared mean / specialist mean - 1, from `shared_vs_separate.json` seed-level means).',
          'Bonferroni-adjusted p = min(1, 4 x Welch p), over the four seed-level tests (circle/spiral x reconstruction/forecast) in `shared_vs_separate.json`.',
          '"Every shared run worse than every specialist run": compares min of the shared per-run values with max of the specialist per-run values in the same file.',
          'Family rankings (easiest/hardest) across the 7 five-family models: argmin / argmax over `results.<model>.<family>.*.mean` in `ablation_2x2_results.json`.',
          'Window means (7-8, 8-9, 9-10): mean over the 750 trajectories of per-trajectory RMSE inside the window, recomputed from `results/ablation_2x2/predictions_test.npz` (checked against the JSON in `audit/verify_3_ablation_statistics.py`).',
          'Crossing times: first query time after t = 7 at which the across-trajectory RMSE of B, C or D exceeds that of A in `ablation_2x2_results.json` (error_vs_time).',
          'GeoLife fine-tuned vs scratch paired comparison: per segment ratio of `interp_geolife_finetuned_from_three_shapes` to `interp_geolife_scratch` in `results/geolife/main_evaluation/geolife_trained_per_segment.csv` (1,000 segments per observation count); geometric mean of the ratios; 95% interval from a bootstrap over the 97 users (5,000 resamples, seed 0); share of segments where fine-tuned is lower. Per-segment data are not copied anywhere.',
          'Per-family ranking relative to baselines: ratios of saved means to the saved linear-interpolation / hold-last baselines.'):
    i_(f'- {r}')
i_('')
i_('## C. Cross-checks')
i_('')
i_('`audit/verify_5_corrected_documents.py` re-derives the five-family and ablation tables from the raw saved predictions; `audit/verify_6_results_documents.py` checks the tables of these new documents against raw predictions and against the numbers printed in the earlier reports (see the end of this file for the last run).')
i_('')
i_('## D. Numerical claims and their sources')
i_('')
i_('| Claim | Value | Source file :: key path |')
i_('|---|---:|---|')
seen = set()
for claim, val, src in INDEX:
    key = (src,)
    if key in seen: continue
    seen.add(key)
    vs = f'{val:.6g}' if isinstance(val, (float, np.floating)) else str(val)
    i_(f"| {claim.replace('|', '/')} | {vs} | `{src.replace('|', '/')}` |")
(OUT / 'RESULTS_SOURCE_INDEX.md').write_text('\n'.join(I) + '\n'); print('RESULTS_SOURCE_INDEX.md', len(I), 'lines; distinct sourced values:', len(seen))
