# Experiments

Eight independent investigations. Each has its own README under [`experiments/`](../experiments/) listing the objective, data, model, training, evaluation, saved results, reproduction command and limitations. Nothing is duplicated: the READMEs point to the existing configs, scripts, results and reports.

| # | Experiment | Seeds | Question | Folder | Main results / report |
|---|---|---|---|---|---|
| 1 | Three-family unified model | 1 (5 in the earlier two-shape study) | Can one model reconstruct and forecast circles, spirals, ellipses? | [three_family](../experiments/three_family/) | `results/validation/three_shapes/` |
| 2 | Five-family unified model | 1 | Does it extend to figure-eight and sinusoid? | [five_family](../experiments/five_family/) | [`reports/five_shapes_report.md`](../reports/five_shapes_report.md) |
| 3 | Shared vs specialist | 5 vs 3 per family | Is a shared model as good as a specialist? | [shared_vs_specialist](../experiments/shared_vs_specialist/) | `results/validation/shared_vs_separate/` |
| 4 | Cross-family generalisation | 1 frozen model (5 two-shape runs for ellipses) | What happens on unseen families? | [cross_family_generalization](../experiments/cross_family_generalization/) | `results/five_shapes/zeroshot_three_shape_model.json` |
| 5 | Capacity scaling | 1 per size | How does model size change error? | [capacity_scaling](../experiments/capacity_scaling/) | `results/capacity_scaling/` |
| 6 | Controlled 2x2 ablation | **1 per cell** | Latent dimension vs GRU width | [ablation_2x2](../experiments/ablation_2x2/) | [`ABLATION_2X2_REPORT.md`](../results/ablation_2x2/ABLATION_2X2_REPORT.md) |
| 7 | Dynamics conditioning | 5 (3 for M2aug) | Does conditioning on a dynamics code help? (not supported) | [conditioning](../experiments/conditioning/) | [`reports/conditioning_study_report.md`](../reports/conditioning_study_report.md) |
| 8 | GeoLife transfer | 1 per model | Does synthetic training transfer to GPS? (no) | [geolife_transfer](../experiments/geolife_transfer/) | [`reports/geolife_real_world_interpolation_report.md`](../reports/geolife_real_world_interpolation_report.md) |

Earlier work (original-size model, improved two-shape model, validation studies, observation-count ablation) is documented in `reports/` ([`improved_two_shapes_results.md`](../reports/improved_two_shapes_results.md), [`validation_study_report.md`](../reports/validation_study_report.md), [`reconstruction_failure_analysis.md`](../reports/reconstruction_failure_analysis.md)). Independent verification of every experiment: [`PROJECT_COMPLETION_AUDIT.md`](../PROJECT_COMPLETION_AUDIT.md).
