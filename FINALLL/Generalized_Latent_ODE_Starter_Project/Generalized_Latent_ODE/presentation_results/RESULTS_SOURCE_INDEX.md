# Source index: where every number and figure comes from

Generated automatically by `audit/build_results_documents.py`: every value read through an accessor is logged with its file and key path, so this index lists exactly the values used in the documents. Paths are relative to the project root. "key path" = nested keys in the JSON file joined by dots.

## A. Figures

| Figure (in `presentation_results/figures/`) | Drawn from (saved data only) | Existing original figure with similar content |
|---|---|---|
| `fig1_reconstruction_by_family.png` | `results/ablation_2x2/ablation_2x2_results.json` (results.A..D, baselines.linear_interpolation_recon) | `results/ablation_2x2/figures/fig1_cells_by_family.png` |
| `fig2_forecast_by_family.png` | `results/ablation_2x2/ablation_2x2_results.json` (results.A..D, baselines.hold_last_forecast) | `results/ablation_2x2/figures/fig1_cells_by_family.png` |
| `fig3_shared_vs_specialist.png` | `results/validation/shared_vs_separate/shared_vs_separate.json` (circle/spiral.seed_level.*.unified|specialist, values) | `results/validation/shared_vs_separate/shared_vs_separate.png` |
| `fig4_cross_family_generalisation.png` | `results/five_shapes/zeroshot_three_shape_model.json`, `results/five_shapes/compare_three_vs_five.json` | `results/five_shapes/figures/per_family_rmse_three_vs_five.png` |
| `fig5_ablation_overall.png` | `results/ablation_2x2/ablation_2x2_results.json` (results.*.overall) | `results/ablation_2x2/figures/fig2_interaction.png` |
| `fig6_forecast_error_vs_time.png` | `results/ablation_2x2/ablation_2x2_results.json` (error_vs_time) | `results/ablation_2x2/figures/fig3_error_vs_time.png` |
| `fig7_effects_vs_seed_variation.png` | `ablation_2x2_results.json` (results, reproducibility_pairs); `results/validation/seed_study/seed_study.json` (across_seeds.overall.forecast_rmse.values) | none |
| `fig8_geolife_transfer.png` | `results/geolife/main_evaluation/geolife_trained_summary.json` (interpolation.30.*.median, median_ci95_user_bootstrap) | `results/geolife/figures/fig3_method_comparison.png`, `fig9_synthetic_vs_geolife_trained.png` |

## B. Values computed in this analysis from saved results (not read directly from a result file)

- Percent differences and ratios between two saved means (for example "+84.9%" shared vs specialist = shared mean / specialist mean - 1, from `shared_vs_separate.json` seed-level means).
- Bonferroni-adjusted p = min(1, 4 x Welch p), over the four seed-level tests (circle/spiral x reconstruction/forecast) in `shared_vs_separate.json`.
- "Every shared run worse than every specialist run": compares min of the shared per-run values with max of the specialist per-run values in the same file.
- Family rankings (easiest/hardest) across the 7 five-family models: argmin / argmax over `results.<model>.<family>.*.mean` in `ablation_2x2_results.json`.
- Window means (7-8, 8-9, 9-10): mean over the 750 trajectories of per-trajectory RMSE inside the window, recomputed from `results/ablation_2x2/predictions_test.npz` (checked against the JSON in `audit/verify_3_ablation_statistics.py`).
- Crossing times: first query time after t = 7 at which the across-trajectory RMSE of B, C or D exceeds that of A in `ablation_2x2_results.json` (error_vs_time).
- GeoLife fine-tuned vs scratch paired comparison: per segment ratio of `interp_geolife_finetuned_from_three_shapes` to `interp_geolife_scratch` in `results/geolife/main_evaluation/geolife_trained_per_segment.csv` (1,000 segments per observation count); geometric mean of the ratios; 95% interval from a bootstrap over the 97 users (5,000 resamples, seed 0); share of segments where fine-tuned is lower. Per-segment data are not copied anywhere.
- Per-family ranking relative to baselines: ratios of saved means to the saved linear-interpolation / hold-last baselines.

## C. Cross-checks

`audit/verify_5_corrected_documents.py` re-derives the five-family and ablation tables from the raw saved predictions; `audit/verify_6_results_documents.py` checks the tables of these new documents against raw predictions and against the numbers printed in the earlier reports (see the end of this file for the last run).

## D. Numerical claims and their sources

| Claim | Value | Source file :: key path |
|---|---:|---|
| latent_ode_original.overall.recon_rmse.mean | 0.75564 | `results/improved_two_shapes/baseline_comparison.json :: latent_ode_original.overall.recon_rmse.mean` |
| latent_ode_improved.overall.recon_rmse.mean | 0.0556867 | `results/improved_two_shapes/baseline_comparison.json :: latent_ode_improved.overall.recon_rmse.mean` |
| latent_ode_original.overall.forecast_rmse.mean | 0.842842 | `results/improved_two_shapes/baseline_comparison.json :: latent_ode_original.overall.forecast_rmse.mean` |
| latent_ode_improved.overall.forecast_rmse.mean | 0.191941 | `results/improved_two_shapes/baseline_comparison.json :: latent_ode_improved.overall.forecast_rmse.mean` |
| constant_position.overall.recon_rmse.mean | 0.0661935 | `results/improved_two_shapes/baseline_comparison.json :: constant_position.overall.recon_rmse.mean` |
| constant_position.overall.forecast_rmse.mean | 1.20895 | `results/improved_two_shapes/baseline_comparison.json :: constant_position.overall.forecast_rmse.mean` |
| constant_velocity.overall.forecast_rmse.mean | 2.36562 | `results/improved_two_shapes/baseline_comparison.json :: constant_velocity.overall.forecast_rmse.mean` |
| across_seeds.overall.recon_rmse.mean | 0.0599329 | `results/validation/seed_study/seed_study.json :: across_seeds.overall.recon_rmse.mean` |
| across_seeds.overall.recon_rmse.std | 0.00685212 | `results/validation/seed_study/seed_study.json :: across_seeds.overall.recon_rmse.std` |
| across_seeds.overall.forecast_rmse.mean | 0.219355 | `results/validation/seed_study/seed_study.json :: across_seeds.overall.forecast_rmse.mean` |
| across_seeds.overall.forecast_rmse.std | 0.0501823 | `results/validation/seed_study/seed_study.json :: across_seeds.overall.forecast_rmse.std` |
| across_seeds.overall.forecast_rmse.ci95_low | 0.157055 | `results/validation/seed_study/seed_study.json :: across_seeds.overall.forecast_rmse.ci95_low` |
| across_seeds.overall.forecast_rmse.ci95_high | 0.281654 | `results/validation/seed_study/seed_study.json :: across_seeds.overall.forecast_rmse.ci95_high` |
| C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).axis_ratio_pred_mean | 0.923604 | `results/validation/three_shapes/three_shapes_and_ood.json :: C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).axis_ratio_pred_mean` |
| C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).axis_ratio_true_mean | 0.591632 | `results/validation/three_shapes/three_shapes_and_ood.json :: C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).axis_ratio_true_mean` |
| baselines.linear_interpolation_recon.circle.mean | 0.0699414 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.linear_interpolation_recon.circle.mean` |
| three_shape_model.circle.recon_rmse.mean | 0.0621759 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.circle.recon_rmse.mean` |
| results.orig_baseline.circle.recon_rmse.mean | 0.0700578 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.circle.recon_rmse.mean` |
| results.A.circle.recon_rmse.mean | 0.0606554 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.circle.recon_rmse.mean` |
| results.B.circle.recon_rmse.mean | 0.0415857 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.circle.recon_rmse.mean` |
| results.C.circle.recon_rmse.mean | 0.0513546 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.circle.recon_rmse.mean` |
| results.D.circle.recon_rmse.mean | 0.0369487 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.circle.recon_rmse.mean` |
| baselines.linear_interpolation_recon.spiral.mean | 0.0653297 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.linear_interpolation_recon.spiral.mean` |
| three_shape_model.spiral.recon_rmse.mean | 0.0932613 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.spiral.recon_rmse.mean` |
| results.orig_baseline.spiral.recon_rmse.mean | 0.113143 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.spiral.recon_rmse.mean` |
| results.A.spiral.recon_rmse.mean | 0.0883345 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.spiral.recon_rmse.mean` |
| results.B.spiral.recon_rmse.mean | 0.0465585 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.spiral.recon_rmse.mean` |
| results.C.spiral.recon_rmse.mean | 0.0710906 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.spiral.recon_rmse.mean` |
| results.D.spiral.recon_rmse.mean | 0.0434099 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.spiral.recon_rmse.mean` |
| baselines.linear_interpolation_recon.ellipse.mean | 0.0637371 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.linear_interpolation_recon.ellipse.mean` |
| three_shape_model.ellipse.recon_rmse.mean | 0.0887923 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.ellipse.recon_rmse.mean` |
| results.orig_baseline.ellipse.recon_rmse.mean | 0.0973143 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.ellipse.recon_rmse.mean` |
| results.A.ellipse.recon_rmse.mean | 0.0945919 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.ellipse.recon_rmse.mean` |
| results.B.ellipse.recon_rmse.mean | 0.0436283 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.ellipse.recon_rmse.mean` |
| results.C.ellipse.recon_rmse.mean | 0.0667816 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.ellipse.recon_rmse.mean` |
| results.D.ellipse.recon_rmse.mean | 0.0407961 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.ellipse.recon_rmse.mean` |
| baselines.linear_interpolation_recon.figure_eight.mean | 0.091708 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.linear_interpolation_recon.figure_eight.mean` |
| three_shape_model.figure_eight.recon_rmse.mean | 0.732054 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.figure_eight.recon_rmse.mean` |
| results.orig_baseline.figure_eight.recon_rmse.mean | 0.10858 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.figure_eight.recon_rmse.mean` |
| results.A.figure_eight.recon_rmse.mean | 0.136442 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.figure_eight.recon_rmse.mean` |
| results.B.figure_eight.recon_rmse.mean | 0.0654012 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.figure_eight.recon_rmse.mean` |
| results.C.figure_eight.recon_rmse.mean | 0.0753711 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.figure_eight.recon_rmse.mean` |
| results.D.figure_eight.recon_rmse.mean | 0.0567664 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.figure_eight.recon_rmse.mean` |
| baselines.linear_interpolation_recon.sinusoid.mean | 0.0452147 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.linear_interpolation_recon.sinusoid.mean` |
| three_shape_model.sinusoid.recon_rmse.mean | 0.344138 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.sinusoid.recon_rmse.mean` |
| results.orig_baseline.sinusoid.recon_rmse.mean | 0.064719 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.sinusoid.recon_rmse.mean` |
| results.A.sinusoid.recon_rmse.mean | 0.0607323 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.sinusoid.recon_rmse.mean` |
| results.B.sinusoid.recon_rmse.mean | 0.0335856 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.sinusoid.recon_rmse.mean` |
| results.C.sinusoid.recon_rmse.mean | 0.0450182 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.sinusoid.recon_rmse.mean` |
| results.D.sinusoid.recon_rmse.mean | 0.0282045 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.sinusoid.recon_rmse.mean` |
| baselines.linear_interpolation_recon.overall.mean | 0.0671862 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.linear_interpolation_recon.overall.mean` |
| three_shape_model.overall.recon_rmse.mean | 0.264084 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.overall.recon_rmse.mean` |
| results.orig_baseline.overall.recon_rmse.mean | 0.0907628 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.overall.recon_rmse.mean` |
| results.A.overall.recon_rmse.mean | 0.0881513 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.overall.recon_rmse.mean` |
| results.B.overall.recon_rmse.mean | 0.0461518 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.overall.recon_rmse.mean` |
| results.C.overall.recon_rmse.mean | 0.0619232 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.overall.recon_rmse.mean` |
| results.D.overall.recon_rmse.mean | 0.0412251 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.overall.recon_rmse.mean` |
| baselines.hold_last_forecast.circle.mean | 1.13701 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.hold_last_forecast.circle.mean` |
| baselines.const_velocity_forecast.circle.mean | 2.32739 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.const_velocity_forecast.circle.mean` |
| three_shape_model.circle.forecast_rmse.mean | 0.140127 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.circle.forecast_rmse.mean` |
| results.orig_baseline.circle.forecast_rmse.mean | 0.25942 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.circle.forecast_rmse.mean` |
| results.A.circle.forecast_rmse.mean | 0.267574 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.circle.forecast_rmse.mean` |
| results.B.circle.forecast_rmse.mean | 0.37823 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.circle.forecast_rmse.mean` |
| results.C.circle.forecast_rmse.mean | 0.288246 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.circle.forecast_rmse.mean` |
| results.D.circle.forecast_rmse.mean | 0.518226 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.circle.forecast_rmse.mean` |
| baselines.hold_last_forecast.spiral.mean | 1.27631 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.hold_last_forecast.spiral.mean` |
| baselines.const_velocity_forecast.spiral.mean | 2.38767 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.const_velocity_forecast.spiral.mean` |
| three_shape_model.spiral.forecast_rmse.mean | 0.290796 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.spiral.forecast_rmse.mean` |
| results.orig_baseline.spiral.forecast_rmse.mean | 0.427031 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.spiral.forecast_rmse.mean` |
| results.A.spiral.forecast_rmse.mean | 0.49147 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.spiral.forecast_rmse.mean` |
| results.B.spiral.forecast_rmse.mean | 0.535259 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.spiral.forecast_rmse.mean` |
| results.C.spiral.forecast_rmse.mean | 0.49866 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.spiral.forecast_rmse.mean` |
| results.D.spiral.forecast_rmse.mean | 0.660501 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.spiral.forecast_rmse.mean` |
| baselines.hold_last_forecast.ellipse.mean | 1.12001 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.hold_last_forecast.ellipse.mean` |
| baselines.const_velocity_forecast.ellipse.mean | 2.0143 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.const_velocity_forecast.ellipse.mean` |
| three_shape_model.ellipse.forecast_rmse.mean | 0.188535 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.ellipse.forecast_rmse.mean` |
| results.orig_baseline.ellipse.forecast_rmse.mean | 0.26087 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.ellipse.forecast_rmse.mean` |
| results.A.ellipse.forecast_rmse.mean | 0.391559 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.ellipse.forecast_rmse.mean` |
| results.B.ellipse.forecast_rmse.mean | 0.374921 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.ellipse.forecast_rmse.mean` |
| results.C.ellipse.forecast_rmse.mean | 0.331235 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.ellipse.forecast_rmse.mean` |
| results.D.ellipse.forecast_rmse.mean | 0.524001 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.ellipse.forecast_rmse.mean` |
| baselines.hold_last_forecast.figure_eight.mean | 0.960963 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.hold_last_forecast.figure_eight.mean` |
| baselines.const_velocity_forecast.figure_eight.mean | 2.61761 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.const_velocity_forecast.figure_eight.mean` |
| three_shape_model.figure_eight.forecast_rmse.mean | 1.11737 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.figure_eight.forecast_rmse.mean` |
| results.orig_baseline.figure_eight.forecast_rmse.mean | 0.299173 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.figure_eight.forecast_rmse.mean` |
| results.A.figure_eight.forecast_rmse.mean | 0.473974 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.figure_eight.forecast_rmse.mean` |
| results.B.figure_eight.forecast_rmse.mean | 0.45109 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.figure_eight.forecast_rmse.mean` |
| results.C.figure_eight.forecast_rmse.mean | 0.43099 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.figure_eight.forecast_rmse.mean` |
| results.D.figure_eight.forecast_rmse.mean | 0.484519 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.figure_eight.forecast_rmse.mean` |
| baselines.hold_last_forecast.sinusoid.mean | 0.624795 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.hold_last_forecast.sinusoid.mean` |
| baselines.const_velocity_forecast.sinusoid.mean | 1.3898 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.const_velocity_forecast.sinusoid.mean` |
| three_shape_model.sinusoid.forecast_rmse.mean | 0.973432 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.sinusoid.forecast_rmse.mean` |
| results.orig_baseline.sinusoid.forecast_rmse.mean | 0.386051 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.sinusoid.forecast_rmse.mean` |
| results.A.sinusoid.forecast_rmse.mean | 0.374383 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.sinusoid.forecast_rmse.mean` |
| results.B.sinusoid.forecast_rmse.mean | 0.860213 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.sinusoid.forecast_rmse.mean` |
| results.C.sinusoid.forecast_rmse.mean | 0.575063 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.sinusoid.forecast_rmse.mean` |
| results.D.sinusoid.forecast_rmse.mean | 0.780814 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.sinusoid.forecast_rmse.mean` |
| baselines.hold_last_forecast.overall.mean | 1.02382 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.hold_last_forecast.overall.mean` |
| baselines.const_velocity_forecast.overall.mean | 2.14735 | `results/ablation_2x2/ablation_2x2_results.json :: baselines.const_velocity_forecast.overall.mean` |
| three_shape_model.overall.forecast_rmse.mean | 0.542051 | `results/five_shapes/zeroshot_three_shape_model.json :: three_shape_model.overall.forecast_rmse.mean` |
| results.orig_baseline.overall.forecast_rmse.mean | 0.326509 | `results/ablation_2x2/ablation_2x2_results.json :: results.orig_baseline.overall.forecast_rmse.mean` |
| results.A.overall.forecast_rmse.mean | 0.399792 | `results/ablation_2x2/ablation_2x2_results.json :: results.A.overall.forecast_rmse.mean` |
| results.B.overall.forecast_rmse.mean | 0.519943 | `results/ablation_2x2/ablation_2x2_results.json :: results.B.overall.forecast_rmse.mean` |
| results.C.overall.forecast_rmse.mean | 0.424839 | `results/ablation_2x2/ablation_2x2_results.json :: results.C.overall.forecast_rmse.mean` |
| results.D.overall.forecast_rmse.mean | 0.593612 | `results/ablation_2x2/ablation_2x2_results.json :: results.D.overall.forecast_rmse.mean` |
| circle.seed_level.recon_rmse.unified.mean | 0.0531726 | `results/validation/shared_vs_separate/shared_vs_separate.json :: circle.seed_level.recon_rmse.unified.mean` |
| circle.seed_level.recon_rmse.specialist.mean | 0.0457315 | `results/validation/shared_vs_separate/shared_vs_separate.json :: circle.seed_level.recon_rmse.specialist.mean` |
| circle.seed_level.recon_rmse.welch_t_p | 0.0473354 | `results/validation/shared_vs_separate/shared_vs_separate.json :: circle.seed_level.recon_rmse.welch_t_p` |
| circle.paired_seed42.recon_rmse.wilcoxon_p | 0.000403706 | `results/validation/shared_vs_separate/shared_vs_separate.json :: circle.paired_seed42.recon_rmse.wilcoxon_p` |
| circle.paired_seed42.recon_rmse.relative_diff_pct | 12.2864 | `results/validation/shared_vs_separate/shared_vs_separate.json :: circle.paired_seed42.recon_rmse.relative_diff_pct` |
| circle.seed_level.forecast_rmse.unified.mean | 0.160566 | `results/validation/shared_vs_separate/shared_vs_separate.json :: circle.seed_level.forecast_rmse.unified.mean` |
| circle.seed_level.forecast_rmse.specialist.mean | 0.0868448 | `results/validation/shared_vs_separate/shared_vs_separate.json :: circle.seed_level.forecast_rmse.specialist.mean` |
| circle.seed_level.forecast_rmse.welch_t_p | 0.0098777 | `results/validation/shared_vs_separate/shared_vs_separate.json :: circle.seed_level.forecast_rmse.welch_t_p` |
| circle.paired_seed42.forecast_rmse.wilcoxon_p | 4.7879e-15 | `results/validation/shared_vs_separate/shared_vs_separate.json :: circle.paired_seed42.forecast_rmse.wilcoxon_p` |
| circle.paired_seed42.forecast_rmse.relative_diff_pct | 75.6165 | `results/validation/shared_vs_separate/shared_vs_separate.json :: circle.paired_seed42.forecast_rmse.relative_diff_pct` |
| spiral.seed_level.recon_rmse.unified.mean | 0.0680014 | `results/validation/shared_vs_separate/shared_vs_separate.json :: spiral.seed_level.recon_rmse.unified.mean` |
| spiral.seed_level.recon_rmse.specialist.mean | 0.0632334 | `results/validation/shared_vs_separate/shared_vs_separate.json :: spiral.seed_level.recon_rmse.specialist.mean` |
| spiral.seed_level.recon_rmse.welch_t_p | 0.356397 | `results/validation/shared_vs_separate/shared_vs_separate.json :: spiral.seed_level.recon_rmse.welch_t_p` |
| spiral.paired_seed42.recon_rmse.wilcoxon_p | 0.638356 | `results/validation/shared_vs_separate/shared_vs_separate.json :: spiral.paired_seed42.recon_rmse.wilcoxon_p` |
| spiral.paired_seed42.recon_rmse.relative_diff_pct | -2.80606 | `results/validation/shared_vs_separate/shared_vs_separate.json :: spiral.paired_seed42.recon_rmse.relative_diff_pct` |
| spiral.seed_level.forecast_rmse.unified.mean | 0.284081 | `results/validation/shared_vs_separate/shared_vs_separate.json :: spiral.seed_level.forecast_rmse.unified.mean` |
| spiral.seed_level.forecast_rmse.specialist.mean | 0.198305 | `results/validation/shared_vs_separate/shared_vs_separate.json :: spiral.seed_level.forecast_rmse.specialist.mean` |
| spiral.seed_level.forecast_rmse.welch_t_p | 0.0449903 | `results/validation/shared_vs_separate/shared_vs_separate.json :: spiral.seed_level.forecast_rmse.welch_t_p` |
| spiral.paired_seed42.forecast_rmse.wilcoxon_p | 7.92018e-07 | `results/validation/shared_vs_separate/shared_vs_separate.json :: spiral.paired_seed42.forecast_rmse.wilcoxon_p` |
| spiral.paired_seed42.forecast_rmse.relative_diff_pct | 26.8695 | `results/validation/shared_vs_separate/shared_vs_separate.json :: spiral.paired_seed42.forecast_rmse.relative_diff_pct` |
| B_three_vs_two_shape_on_matching_circle_spiral.circle.forecast_rmse.three_shape | 0.135818 | `results/validation/three_shapes/three_shapes_and_ood.json :: B_three_vs_two_shape_on_matching_circle_spiral.circle.forecast_rmse.three_shape` |
| B_three_vs_two_shape_on_matching_circle_spiral.circle.forecast_rmse.two_shape_seed42 | 0.141974 | `results/validation/three_shapes/three_shapes_and_ood.json :: B_three_vs_two_shape_on_matching_circle_spiral.circle.forecast_rmse.two_shape_seed42` |
| B_three_vs_two_shape_on_matching_circle_spiral.circle.forecast_rmse.two_shape_across_seeds.mean | 0.157088 | `results/validation/three_shapes/three_shapes_and_ood.json :: B_three_vs_two_shape_on_matching_circle_spiral.circle.forecast_rmse.two_shape_across_seeds.mean` |
| B_three_vs_two_shape_on_matching_circle_spiral.circle.forecast_rmse.two_shape_across_seeds.std | 0.0373541 | `results/validation/three_shapes/three_shapes_and_ood.json :: B_three_vs_two_shape_on_matching_circle_spiral.circle.forecast_rmse.two_shape_across_seeds.std` |
| C_out_of_distribution_ellipses.three_shape_model_on_same_ellipses (in-distribution reference).recon_rmse.mean | 0.0887434 | `results/validation/three_shapes/three_shapes_and_ood.json :: C_out_of_distribution_ellipses.three_shape_model_on_same_ellipses (in-distribution reference).recon_rmse.mean` |
| C_out_of_distribution_ellipses.three_shape_model_on_same_ellipses (in-distribution reference).forecast_rmse.mean | 0.184998 | `results/validation/three_shapes/three_shapes_and_ood.json :: C_out_of_distribution_ellipses.three_shape_model_on_same_ellipses (in-distribution reference).forecast_rmse.mean` |
| C_out_of_distribution_ellipses.const_pos_on_same_ellipses.recon_rmse.mean | 0.0625047 | `results/validation/three_shapes/three_shapes_and_ood.json :: C_out_of_distribution_ellipses.const_pos_on_same_ellipses.recon_rmse.mean` |
| C_out_of_distribution_ellipses.const_pos_on_same_ellipses.forecast_rmse.mean | 1.12126 | `results/validation/three_shapes/three_shapes_and_ood.json :: C_out_of_distribution_ellipses.const_pos_on_same_ellipses.forecast_rmse.mean` |
| C_out_of_distribution_ellipses.const_vel_on_same_ellipses.forecast_rmse.mean | 2.20821 | `results/validation/three_shapes/three_shapes_and_ood.json :: C_out_of_distribution_ellipses.const_vel_on_same_ellipses.forecast_rmse.mean` |
| C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).forecast_rotation_direction_accuracy | 1 | `results/validation/three_shapes/three_shapes_and_ood.json :: C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).forecast_rotation_direction_accuracy` |
| C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).axis_ratio_corr | 0.184563 | `results/validation/three_shapes/three_shapes_and_ood.json :: C_out_of_distribution_ellipses.ellipse_geometry_two_shape_seed42 (OOD).axis_ratio_corr` |
| C_out_of_distribution_ellipses.ellipse_geometry_three_shape (ID).axis_ratio_corr | 0.863904 | `results/validation/three_shapes/three_shapes_and_ood.json :: C_out_of_distribution_ellipses.ellipse_geometry_three_shape (ID).axis_ratio_corr` |
| five_shape_model.circle.forecast_rmse.mean | 0.25942 | `results/five_shapes/compare_three_vs_five.json :: five_shape_model.circle.forecast_rmse.mean` |
| paired.circle/forecast_rmse.relative_change_of_mean | 0.851324 | `results/five_shapes/compare_three_vs_five.json :: paired.circle/forecast_rmse.relative_change_of_mean` |
| five_shape_model.circle.recon_rmse.mean | 0.0700578 | `results/five_shapes/compare_three_vs_five.json :: five_shape_model.circle.recon_rmse.mean` |
| five_shape_model.spiral.forecast_rmse.mean | 0.427031 | `results/five_shapes/compare_three_vs_five.json :: five_shape_model.spiral.forecast_rmse.mean` |
| paired.spiral/forecast_rmse.relative_change_of_mean | 0.468493 | `results/five_shapes/compare_three_vs_five.json :: paired.spiral/forecast_rmse.relative_change_of_mean` |
| five_shape_model.spiral.recon_rmse.mean | 0.113143 | `results/five_shapes/compare_three_vs_five.json :: five_shape_model.spiral.recon_rmse.mean` |
| five_shape_model.ellipse.forecast_rmse.mean | 0.26087 | `results/five_shapes/compare_three_vs_five.json :: five_shape_model.ellipse.forecast_rmse.mean` |
| paired.ellipse/forecast_rmse.relative_change_of_mean | 0.383672 | `results/five_shapes/compare_three_vs_five.json :: paired.ellipse/forecast_rmse.relative_change_of_mean` |
| five_shape_model.ellipse.recon_rmse.mean | 0.0973143 | `results/five_shapes/compare_three_vs_five.json :: five_shape_model.ellipse.recon_rmse.mean` |
| models.A.optimizer_steps | 33000 | `results/ablation_2x2/ablation_2x2_results.json :: models.A.optimizer_steps` |
| models.A.latent_dim | 8 | `results/ablation_2x2/ablation_2x2_results.json :: models.A.latent_dim` |
| models.A.encoder_hidden | 64 | `results/ablation_2x2/ablation_2x2_results.json :: models.A.encoder_hidden` |
| models.A.n_parameters | 24602 | `results/ablation_2x2/ablation_2x2_results.json :: models.A.n_parameters` |
| models.B.latent_dim | 16 | `results/ablation_2x2/ablation_2x2_results.json :: models.B.latent_dim` |
| models.B.encoder_hidden | 64 | `results/ablation_2x2/ablation_2x2_results.json :: models.B.encoder_hidden` |
| models.B.n_parameters | 27186 | `results/ablation_2x2/ablation_2x2_results.json :: models.B.n_parameters` |
| models.C.latent_dim | 8 | `results/ablation_2x2/ablation_2x2_results.json :: models.C.latent_dim` |
| models.C.encoder_hidden | 128 | `results/ablation_2x2/ablation_2x2_results.json :: models.C.encoder_hidden` |
| models.C.n_parameters | 63642 | `results/ablation_2x2/ablation_2x2_results.json :: models.C.n_parameters` |
| models.D.latent_dim | 16 | `results/ablation_2x2/ablation_2x2_results.json :: models.D.latent_dim` |
| models.D.encoder_hidden | 128 | `results/ablation_2x2/ablation_2x2_results.json :: models.D.encoder_hidden` |
| models.D.n_parameters | 67250 | `results/ablation_2x2/ablation_2x2_results.json :: models.D.n_parameters` |
| fraction_of_trajectories_better_than_A.B.overall.forecast_rmse | 0.350667 | `results/ablation_2x2/ablation_2x2_results.json :: fraction_of_trajectories_better_than_A.B.overall.forecast_rmse` |
| fraction_of_trajectories_better_than_A.C.overall.forecast_rmse | 0.477333 | `results/ablation_2x2/ablation_2x2_results.json :: fraction_of_trajectories_better_than_A.C.overall.forecast_rmse` |
| fraction_of_trajectories_better_than_A.D.overall.forecast_rmse | 0.269333 | `results/ablation_2x2/ablation_2x2_results.json :: fraction_of_trajectories_better_than_A.D.overall.forecast_rmse` |
| factorial.overall/forecast_rmse/interaction ((D-C) - (B-A)).mean | 0.0486225 | `results/ablation_2x2/ablation_2x2_results.json :: factorial.overall/forecast_rmse/interaction ((D-C) - (B-A)).mean` |
| reproducibility_pairs.A_vs_orig_baseline.overall.forecast_rmse.relative_difference | 0.224444 | `results/ablation_2x2/ablation_2x2_results.json :: reproducibility_pairs.A_vs_orig_baseline.overall.forecast_rmse.relative_difference` |
| reproducibility_pairs.D_vs_orig_medium.overall.forecast_rmse.relative_difference | 0.206856 | `results/ablation_2x2/ablation_2x2_results.json :: reproducibility_pairs.D_vs_orig_medium.overall.forecast_rmse.relative_difference` |
| gap_to_specialists.circle.specialist_forecast_rmse | 0.0868448 | `results/conditioning/conditioning_study.json :: gap_to_specialists.circle.specialist_forecast_rmse` |
| gap_to_specialists.spiral.specialist_forecast_rmse | 0.198305 | `results/conditioning/conditioning_study.json :: gap_to_specialists.spiral.specialist_forecast_rmse` |
| seed_level.M1.circle.forecast_rmse.mean | 0.145087 | `results/conditioning/conditioning_study.json :: seed_level.M1.circle.forecast_rmse.mean` |
| seed_level.M1.circle.forecast_rmse.std | 0.0070338 | `results/conditioning/conditioning_study.json :: seed_level.M1.circle.forecast_rmse.std` |
| seed_level.M1.spiral.forecast_rmse.mean | 0.263523 | `results/conditioning/conditioning_study.json :: seed_level.M1.spiral.forecast_rmse.mean` |
| seed_level.M1.spiral.forecast_rmse.std | 0.0356318 | `results/conditioning/conditioning_study.json :: seed_level.M1.spiral.forecast_rmse.std` |
| seed_level.M2.circle.forecast_rmse.mean | 0.171908 | `results/conditioning/conditioning_study.json :: seed_level.M2.circle.forecast_rmse.mean` |
| seed_level.M2.circle.forecast_rmse.std | 0.0959773 | `results/conditioning/conditioning_study.json :: seed_level.M2.circle.forecast_rmse.std` |
| seed_level.M2.spiral.forecast_rmse.mean | 0.269946 | `results/conditioning/conditioning_study.json :: seed_level.M2.spiral.forecast_rmse.mean` |
| seed_level.M2.spiral.forecast_rmse.std | 0.112926 | `results/conditioning/conditioning_study.json :: seed_level.M2.spiral.forecast_rmse.std` |
| seed_level.M3.circle.forecast_rmse.mean | 0.175902 | `results/conditioning/conditioning_study.json :: seed_level.M3.circle.forecast_rmse.mean` |
| seed_level.M3.circle.forecast_rmse.std | 0.0664882 | `results/conditioning/conditioning_study.json :: seed_level.M3.circle.forecast_rmse.std` |
| seed_level.M3.spiral.forecast_rmse.mean | 0.298538 | `results/conditioning/conditioning_study.json :: seed_level.M3.spiral.forecast_rmse.mean` |
| seed_level.M3.spiral.forecast_rmse.std | 0.0680876 | `results/conditioning/conditioning_study.json :: seed_level.M3.spiral.forecast_rmse.std` |
| seed_level.M2aug.circle.forecast_rmse.mean | 0.300634 | `results/conditioning/conditioning_study.json :: seed_level.M2aug.circle.forecast_rmse.mean` |
| seed_level.M2aug.circle.forecast_rmse.std | 0.107892 | `results/conditioning/conditioning_study.json :: seed_level.M2aug.circle.forecast_rmse.std` |
| seed_level.M2aug.spiral.forecast_rmse.mean | 0.434983 | `results/conditioning/conditioning_study.json :: seed_level.M2aug.spiral.forecast_rmse.mean` |
| seed_level.M2aug.spiral.forecast_rmse.std | 0.0948737 | `results/conditioning/conditioning_study.json :: seed_level.M2aug.spiral.forecast_rmse.std` |
| interpolation.30.linear.median_ci95_user_bootstrap.0 | 5.79753 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.linear.median_ci95_user_bootstrap.0` |
| interpolation.30.linear.median_ci95_user_bootstrap.1 | 8.47659 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.linear.median_ci95_user_bootstrap.1` |
| interpolation.30.linear.median | 7.09944 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.linear.median` |
| interpolation.30.linear.geometric_mean_ratio_vs_linear | 1 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.linear.geometric_mean_ratio_vs_linear` |
| interpolation.30.cubic.median_ci95_user_bootstrap.0 | 4.64313 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.cubic.median_ci95_user_bootstrap.0` |
| interpolation.30.cubic.median_ci95_user_bootstrap.1 | 6.90234 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.cubic.median_ci95_user_bootstrap.1` |
| interpolation.30.cubic.median | 5.56143 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.cubic.median` |
| interpolation.30.cubic.geometric_mean_ratio_vs_linear | 0.799206 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.cubic.geometric_mean_ratio_vs_linear` |
| interpolation.30.pchip.median_ci95_user_bootstrap.0 | 4.53771 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.pchip.median_ci95_user_bootstrap.0` |
| interpolation.30.pchip.median_ci95_user_bootstrap.1 | 6.68928 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.pchip.median_ci95_user_bootstrap.1` |
| interpolation.30.pchip.median | 5.61723 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.pchip.median` |
| interpolation.30.pchip.geometric_mean_ratio_vs_linear | 0.777574 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.pchip.geometric_mean_ratio_vs_linear` |
| interpolation.30.frozen_three_shapes.median_ci95_user_bootstrap.0 | 110.938 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.frozen_three_shapes.median_ci95_user_bootstrap.0` |
| interpolation.30.frozen_three_shapes.median_ci95_user_bootstrap.1 | 147.94 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.frozen_three_shapes.median_ci95_user_bootstrap.1` |
| interpolation.30.frozen_three_shapes.median | 126.403 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.frozen_three_shapes.median` |
| interpolation.30.frozen_three_shapes.geometric_mean_ratio_vs_linear | 17.2783 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.frozen_three_shapes.geometric_mean_ratio_vs_linear` |
| interpolation.30.geolife_scratch.median_ci95_user_bootstrap.0 | 51.5761 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.geolife_scratch.median_ci95_user_bootstrap.0` |
| interpolation.30.geolife_scratch.median_ci95_user_bootstrap.1 | 71.6603 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.geolife_scratch.median_ci95_user_bootstrap.1` |
| interpolation.30.geolife_scratch.median | 60.6185 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.geolife_scratch.median` |
| interpolation.30.geolife_scratch.geometric_mean_ratio_vs_linear | 8.22152 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.geolife_scratch.geometric_mean_ratio_vs_linear` |
| interpolation.30.geolife_finetuned_from_three_shapes.median_ci95_user_bootstrap.0 | 42.2398 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.geolife_finetuned_from_three_shapes.median_ci95_user_bootstrap.0` |
| interpolation.30.geolife_finetuned_from_three_shapes.median_ci95_user_bootstrap.1 | 54.6273 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.geolife_finetuned_from_three_shapes.median_ci95_user_bootstrap.1` |
| interpolation.30.geolife_finetuned_from_three_shapes.median | 47.1083 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.geolife_finetuned_from_three_shapes.median` |
| interpolation.30.geolife_finetuned_from_three_shapes.geometric_mean_ratio_vs_linear | 6.41475 | `results/geolife/main_evaluation/geolife_trained_summary.json :: interpolation.30.geolife_finetuned_from_three_shapes.geometric_mean_ratio_vs_linear` |
| forecast_n30.A.geolife_scratch.median | 322.255 | `results/geolife/main_evaluation/geolife_trained_summary.json :: forecast_n30.A.geolife_scratch.median` |
| forecast_n30.A.geolife_finetuned_from_three_shapes.median | 370.578 | `results/geolife/main_evaluation/geolife_trained_summary.json :: forecast_n30.A.geolife_finetuned_from_three_shapes.median` |
| forecast_n30.A.hold_last.median | 228.92 | `results/geolife/main_evaluation/geolife_trained_summary.json :: forecast_n30.A.hold_last.median` |
| forecast_n30.A.const_velocity.median | 151.434 | `results/geolife/main_evaluation/geolife_trained_summary.json :: forecast_n30.A.const_velocity.median` |
| models.baseline.n_parameters | 24602 | `results/capacity_scaling/capacity_scaling_results.json :: models.baseline.n_parameters` |
| models.medium.n_parameters | 67250 | `results/capacity_scaling/capacity_scaling_results.json :: models.medium.n_parameters` |
| models.large.n_parameters | 74466 | `results/capacity_scaling/capacity_scaling_results.json :: models.large.n_parameters` |
| results.baseline.overall.recon_rmse.mean | 0.0907628 | `results/capacity_scaling/capacity_scaling_results.json :: results.baseline.overall.recon_rmse.mean` |
| results.baseline.overall.forecast_rmse.mean | 0.326509 | `results/capacity_scaling/capacity_scaling_results.json :: results.baseline.overall.forecast_rmse.mean` |
| results.medium.overall.recon_rmse.mean | 0.0398668 | `results/capacity_scaling/capacity_scaling_results.json :: results.medium.overall.recon_rmse.mean` |
| results.medium.overall.forecast_rmse.mean | 0.491866 | `results/capacity_scaling/capacity_scaling_results.json :: results.medium.overall.forecast_rmse.mean` |
| results.large.overall.recon_rmse.mean | 0.0432374 | `results/capacity_scaling/capacity_scaling_results.json :: results.large.overall.recon_rmse.mean` |
| results.large.overall.forecast_rmse.mean | 0.59749 | `results/capacity_scaling/capacity_scaling_results.json :: results.large.overall.forecast_rmse.mean` |
| results.large.sinusoid.forecast_rmse.mean | 1.20741 | `results/capacity_scaling/capacity_scaling_results.json :: results.large.sinusoid.forecast_rmse.mean` |
| five_shape_model.figure_eight.forecast_rmse.mean | 0.299173 | `results/five_shapes/compare_three_vs_five.json :: five_shape_model.figure_eight.forecast_rmse.mean` |
| five_shape_model.sinusoid.forecast_rmse.mean | 0.386051 | `results/five_shapes/compare_three_vs_five.json :: five_shape_model.sinusoid.forecast_rmse.mean` |
