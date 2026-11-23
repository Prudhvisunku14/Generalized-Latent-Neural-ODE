#!/usr/bin/env python3
"""Write one short README per experiment under experiments/<name>/ pointing at the existing configs, scripts, results and
reports (nothing is duplicated). Asserts that every referenced path exists. Numbers are those already recorded in
results/ and reports/ (see docs/RESULTS.md)."""
from pathlib import Path
root = Path(__file__).resolve().parents[1]
S1 = 'single training seed'
X = {
 'three_family': dict(title='Experiment 1: three-family unified model (circle, spiral, ellipse)',
  obj='Train one shared Latent Neural ODE on three trajectory families without a family label and measure reconstruction (t <= 7) and forecasting (7 < t <= 10).',
  data='`data/processed/three_shapes/` (1,000 trajectories per family; 200 time points on [0, 10]; 30 noisy irregular observations at t <= 7). Split 70/15/15 per family (450 test trajectories).',
  model='Model D: latent 8, GRU 64, tanh MLPs 64x2, RK4 step 0.1; 24,602 parameters. `configs/model_d.yaml`.',
  train='Adam 3e-3 (cosine to 1e-5), batch 32, 300 epochs, -ELBO (fixed sigma 0.05, beta 1), checkpoint = lowest validation -ELBO. Seed 42 (' + S1 + ').',
  evalm='Test split, 30 observations, observation seed 123, posterior-mean latent state; per-trajectory RMSE averaged.',
  saved='Overall test: reconstruction RMSE 0.0817, forecast RMSE 0.2110 (original units); per-family values in `saved_models/three_shapes_unified/metadata.json` and `results/validation/three_shapes/`.',
  repro='`python scripts/03_evaluate_model.py --help`; packaged model: `python scripts/predict_trajectory.py --model three_shapes_unified --input input_trajectories/example_circle.csv`.',
  lim='Single seed. Forecasting error grows with horizon. Unified models forecast worse than specialists (Experiment 3). Only works with about 30 observations.',
  paths=['configs/model_d.yaml', 'scripts/10_three_shapes_and_ood.py', 'saved_models/three_shapes_unified/metadata.json', 'results/validation/three_shapes']),
 'five_family': dict(title='Experiment 2: five-family unified model (circle, spiral, ellipse, figure-eight, sinusoid)',
  obj='Extend the shared model to five families and compare it with the three-family model.',
  data='`data/processed/five_shapes/` (1,000 per family; 3,500 / 750 / 750 train/val/test). Circle, spiral, ellipse are identical to the three-family data. Generator: `scripts/30_generate_five_shapes.py`, `src/glo/five_shapes.py`.',
  model='Model D (latent 8, GRU 64; 24,602 parameters), same recipe as Experiment 1; 33,000 optimiser steps (19,800 for the three-family run).',
  train='Seed 42 (' + S1 + '); selected epoch 297 of 300.',
  evalm='`python scripts/31_evaluate_five_shapes.py` (test split, shared observations for all models, baselines: hold-last, constant velocity).',
  saved='Overall test: reconstruction 0.0908, forecast 0.3265 (hold-last forecast 1.0238; linear-interpolation reconstruction 0.0672, so the learned reconstruction is not better than interpolation). Report: `reports/five_shapes_report.md`; data: `results/five_shapes/`.',
  repro='`python scripts/31_evaluate_five_shapes.py`; `python scripts/predict_trajectory.py --model five_shapes_unified --input input_trajectories/example_figure_eight.csv`.',
  lim='Single seed. On the three original families the five-family model forecasts worse than the three-family model (same data stream differences confound the comparison).',
  paths=['scripts/30_generate_five_shapes.py', 'scripts/31_evaluate_five_shapes.py', 'reports/five_shapes_report.md', 'results/five_shapes/compare_three_vs_five.json', 'saved_models/five_shapes_unified/metadata.json']),
 'shared_vs_specialist': dict(title='Experiment 3: shared (unified) versus specialist models',
  obj='Does one model for several families forecast as well as a model trained on one family?',
  data='Circles and spirals (`data/processed/two_shapes/`).',
  model='Model D for every model. Unified: 5 training seeds; specialists (circle-only, spiral-only): 3 seeds each.',
  train='600 epochs for specialists, 300 for unified (`scripts/09_shared_vs_separate.py`, `scripts/07_seed_study.py`).',
  evalm='Seed-level Welch tests on test forecast RMSE; Bonferroni x4 adjustment computed in `docs/RESULTS.md`.',
  saved='Circle forecast 0.161 (shared) vs 0.087 (specialist), +85% (Welch p 0.0099, 0.04 after Bonferroni); spiral 0.284 vs 0.198, +43% (p 0.045, 0.18 after Bonferroni). Every shared run is worse than every specialist run. Files: `results/validation/shared_vs_separate/`, `results/validation/seed_study/`.',
  repro='`python scripts/09_shared_vs_separate.py --help`; packaged specialists: `--model circle_specialist`, `--model spiral_specialist`.',
  lim='Only 5 vs 3 runs; shared and specialist runs differ in epochs; two families only.',
  paths=['scripts/09_shared_vs_separate.py', 'scripts/07_seed_study.py', 'results/validation/shared_vs_separate/shared_vs_separate.json', 'results/validation/seed_study/seed_study.json']),
 'cross_family_generalization': dict(title='Experiment 4: cross-family (zero-shot) generalisation',
  obj='How do models trained on some families behave on families they never saw?',
  data='Held-out ellipses (for the two-shape models) and held-out figure-eight / sinusoid (for the three-family model), from the five-family test split.',
  model='Frozen `three_shapes_unified` and `two_shapes_unified`; no fine-tuning.',
  train='None (zero-shot).',
  evalm='Same protocol as Experiment 2; baselines hold-last, constant velocity, linear interpolation.',
  saved='Three-family model, unseen figure-eight / sinusoid forecast RMSE 1.1174 / 0.9734 versus hold-last 0.9610 / 0.6248, and reconstruction far worse than linear interpolation. Two-shape models on unseen ellipses (5 runs): forecast about 0.32, reconstruction about 0.21 vs 0.0625 for linear interpolation. Files: `results/five_shapes/zeroshot_three_shape_model.json`, `results/validation/`.',
  repro='`python scripts/31_evaluate_five_shapes.py`; `python scripts/10_three_shapes_and_ood.py --help`.',
  lim='One frozen model per case; no leave-one-family-out experiment.',
  paths=['results/five_shapes/zeroshot_three_shape_model.json', 'scripts/10_three_shapes_and_ood.py']),
 'capacity_scaling': dict(title='Experiment 5: model capacity scaling',
  obj='How do reconstruction and forecasting change with model size on the five-family data?',
  data='`data/processed/five_shapes/`.',
  model='Baseline (latent 8, GRU 64; 24,602 parameters), medium (16/128; 67,250), large (32/128; 74,466). `configs/capacity_scaling.yaml`.',
  train='`bash scripts/run_capacity_scaling.sh` (seed 42, ' + S1 + '; the data stream was not controlled in this study).',
  evalm='`python scripts/32_evaluate_capacity_scaling.py` (test split, shared observations).',
  saved='`results/capacity_scaling/` including `CAPACITY_SCALING_REPORT.md` (regenerated from the JSON after an earlier draft contained unsupported paired statistics; see `PROJECT_COMPLETION_AUDIT.md`).',
  repro='`python scripts/32_evaluate_capacity_scaling.py`.',
  lim='Single seed; data stream not controlled; do not read this as "larger models always forecast worse".',
  paths=['configs/capacity_scaling.yaml', 'scripts/run_capacity_scaling.sh', 'scripts/32_evaluate_capacity_scaling.py', 'results/capacity_scaling/CAPACITY_SCALING_REPORT.md']),
 'ablation_2x2': dict(title='Experiment 6: controlled 2x2 ablation (latent dimension x GRU width)',
  obj='Separate the effect of latent dimension from encoder width with identical data order and seeds.',
  data='`data/processed/five_shapes/`; fixed `--seed 42 --data-seed 42` for all four cells.',
  model='| Model | Latent | GRU | Parameters |\n|---|---|---|---|\n| A | 8 | 64 | 24,602 |\n| B | 16 | 64 | 27,186 |\n| C | 8 | 128 | 63,642 |\n| D | 16 | 128 | 67,250 |',
  train='`bash scripts/run_ablation_2x2.sh` (four 4-thread jobs). **Single training seed per cell.**',
  evalm='`python scripts/33_evaluate_ablation_2x2.py` (paired bootstrap and Wilcoxon with Holm correction; factorial effects); `python scripts/34_audit_checkpoint_selection.py`.',
  saved='Reconstruction A/B/C/D 0.0882 / 0.0462 / 0.0619 / 0.0412 (linear interpolation 0.0672); forecast 0.3998 / 0.5199 / 0.4248 / 0.5936 (hold-last 1.0238). Forecast change vs A: B +30.1%, C +6.3% (Holm p 0.18, not significant), D +48.5%. Larger models are better for 7 < t <= 8 and cross above A at t = 8.09 (B), 8.64 (C), 7.94 (D). Corrected report: `results/ablation_2x2/ABLATION_2X2_REPORT.md` (original kept as `ABLATION_2X2_REPORT.original_uncorrected.md`).',
  repro='`python scripts/33_evaluate_ablation_2x2.py`; packaged cells: `--model ablation_2x2_A` ... `ablation_2x2_D`.',
  lim='**Single training seed.** The same architecture retrained with a different data stream differs by +22.4% and +20.7% in forecast RMSE, and the five-seed two-shape study has a forecast CV of about 23%: the forecast differences above are of that order. No causal mechanism is established.',
  paths=['configs/ablation_2x2.yaml', 'scripts/run_ablation_2x2.sh', 'scripts/33_evaluate_ablation_2x2.py', 'scripts/34_audit_checkpoint_selection.py', 'results/ablation_2x2/ABLATION_2X2_REPORT.md', 'results/ablation_2x2/ablation_2x2_results.json']),
 'conditioning': dict(title='Experiment 7: dynamics conditioning',
  obj='Does conditioning the ODE on an inferred dynamics code close the unified-versus-specialist gap?',
  data='Circles and spirals (`data/processed/two_shapes/`).',
  model='M1 unified baseline; M2 parameter-matched baseline; M2aug latent 12; M3 conditioned: a 64->4 tanh head gives a constant code c and dz/dt = f(z, c) (25,118 parameters). Configs in `configs/conditioning/`.',
  train='`bash scripts/run_conditioning_study.sh`; 5 seeds per model (3 for M2aug).',
  evalm='`python scripts/12_conditioning_study.py` (test split; forecast RMSE; probes of the code).',
  saved='`results/conditioning/` (table, figures, `conditioning_study.json`) and `reports/conditioning_study_report.md`. Conclusion: the hypothesis is **not supported** at this scale.',
  repro='`python scripts/12_conditioning_study.py --help`; packaged experimental model: `--model dynamics_conditioned`.',
  lim='Small scale, two families, 5 seeds.',
  paths=['configs/conditioning/m3_conditioned.yaml', 'scripts/12_conditioning_study.py', 'scripts/run_conditioning_study.sh', 'results/conditioning/conditioning_study.json', 'reports/conditioning_study_report.md']),
 'geolife_transfer': dict(title='Experiment 8: GeoLife transfer (synthetic to real GPS)',
  obj='Can a model trained on synthetic geometry be used on real GPS trajectories, with or without fine-tuning?',
  data='GeoLife 1.3 (Microsoft Research; **not distributed here**, see `data/geolife/raw/SOURCE.md`). 200-record segments, held-out users.',
  model='Frozen `three_shapes_unified`; GeoLife-trained from scratch; fine-tuned from the three-family weights. Linear, cubic and PCHIP interpolation as baselines. `configs/geolife.yaml`.',
  train='`scripts/22_geolife_training_data.py`, `scripts/23_geolife_trained_models.py` (one training run each).',
  evalm='`scripts/21_evaluate_geolife_transfer.py`, `scripts/23_geolife_trained_models.py evaluate`; user-cluster bootstrap.',
  saved='Median error with 30 observations (m): linear interpolation 7.10, frozen synthetic 126.4, from scratch 60.6, fine-tuned 47.1. The synthetic model does not transfer, and both trained models remain far worse than interpolation. Aggregate results and tables only: `results/geolife/`, `reports/geolife_real_world_interpolation_report.md`.',
  repro='Requires you to download GeoLife yourself and place it as described in the README. The trained GeoLife models are **not published** (licence question); they are kept locally.',
  lim='One dataset, one run per model, no raw or per-segment data published. The GeoLife report sections 9 and 11 (paired analysis) have not been written into the report.',
  paths=['configs/geolife.yaml', 'scripts/21_evaluate_geolife_transfer.py', 'scripts/23_geolife_trained_models.py', 'reports/geolife_real_world_interpolation_report.md', 'results/geolife/tables']),
}
for name, e in X.items():
    for p in e['paths']:
        assert (root / p).exists(), (name, p)
    d = root / 'experiments' / name; d.mkdir(parents=True, exist_ok=True)
    (d / 'README.md').write_text(
        f"# {e['title']}\n\n"
        f"## 1. Objective\n{e['obj']}\n\n## 2. Dataset\n{e['data']}\n\n## 3. Model configuration\n{e['model']}\n\n"
        f"## 4. Training procedure\n{e['train']}\n\n## 5. Evaluation methodology\n{e['evalm']}\n\n"
        f"## 6. Saved results\n{e['saved']}\n\n## 7. How to reproduce the evaluation\n{e['repro']}\n\n## 8. Limitations\n{e['lim']}\n\n"
        f"Back to the [experiment index](../README.md). Full numbers and sources: [`docs/RESULTS.md`](../../docs/RESULTS.md).\n")
    print('wrote', name)
