# Experiment 3: shared (unified) versus specialist models

## 1. Objective
Does one model for several families forecast as well as a model trained on one family?

## 2. Dataset
Circles and spirals (`data/processed/two_shapes/`).

## 3. Model configuration
Model D for every model. Unified: 5 training seeds; specialists (circle-only, spiral-only): 3 seeds each.

## 4. Training procedure
600 epochs for specialists, 300 for unified (`scripts/09_shared_vs_separate.py`, `scripts/07_seed_study.py`).

## 5. Evaluation methodology
Seed-level Welch tests on test forecast RMSE; Bonferroni x4 adjustment computed in `docs/RESULTS.md`.

## 6. Saved results
Circle forecast 0.161 (shared) vs 0.087 (specialist), +85% (Welch p 0.0099, 0.04 after Bonferroni); spiral 0.284 vs 0.198, +43% (p 0.045, 0.18 after Bonferroni). Every shared run is worse than every specialist run. Files: `results/validation/shared_vs_separate/`, `results/validation/seed_study/`.

## 7. How to reproduce the evaluation
`python scripts/09_shared_vs_separate.py --help`; packaged specialists: `--model circle_specialist`, `--model spiral_specialist`.

## 8. Limitations
Only 5 vs 3 runs; shared and specialist runs differ in epochs; two families only.

Back to the [experiment index](../README.md). Full numbers and sources: [`docs/RESULTS.md`](../../docs/RESULTS.md).
