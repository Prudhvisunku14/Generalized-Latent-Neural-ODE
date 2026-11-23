# Experiment 4: cross-family (zero-shot) generalisation

## 1. Objective
How do models trained on some families behave on families they never saw?

## 2. Dataset
Held-out ellipses (for the two-shape models) and held-out figure-eight / sinusoid (for the three-family model), from the five-family test split.

## 3. Model configuration
Frozen `three_shapes_unified` and `two_shapes_unified`; no fine-tuning.

## 4. Training procedure
None (zero-shot).

## 5. Evaluation methodology
Same protocol as Experiment 2; baselines hold-last, constant velocity, linear interpolation.

## 6. Saved results
Three-family model, unseen figure-eight / sinusoid forecast RMSE 1.1174 / 0.9734 versus hold-last 0.9610 / 0.6248, and reconstruction far worse than linear interpolation. Two-shape models on unseen ellipses (5 runs): forecast about 0.32, reconstruction about 0.21 vs 0.0625 for linear interpolation. Files: `results/five_shapes/zeroshot_three_shape_model.json`, `results/validation/`.

## 7. How to reproduce the evaluation
`python scripts/31_evaluate_five_shapes.py`; `python scripts/10_three_shapes_and_ood.py --help`.

## 8. Limitations
One frozen model per case; no leave-one-family-out experiment.

Back to the [experiment index](../README.md). Full numbers and sources: [`docs/RESULTS.md`](../../docs/RESULTS.md).
