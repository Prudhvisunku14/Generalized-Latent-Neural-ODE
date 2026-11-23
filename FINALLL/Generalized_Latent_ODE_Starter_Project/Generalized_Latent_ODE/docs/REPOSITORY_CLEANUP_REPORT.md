# Repository cleanup report

Date: 2026-11-21. Scope: prepare the project for publication on GitHub. No model was retrained, no result was changed, no checkpoint was altered.

## Safety measures
* Full backup outside the repository (everything except the 299 MB GeoLife archive, which was not touched): `/home/teaching/Desktop/FINALLL/_BACKUP_pre_github_cleanup_2026-11-21/` (`project_backup.tar.gz`, git status/log and diff of uncommitted tracked changes before cleanup).
* Baseline before changes: 106 tests passed. After: 119 tests passed (13 new).
* Remote `origin/main` was fetched; the local branch descends from it (fast-forward only; no force push).

## Decision: structure kept, not rewritten
The suggested layout (`src/generalized_latent_ode/…`, `models/`, `figures/…`) was **not** applied wholesale. The working code lives in `src/glo/`, scripts are numbered by phase, and every test, checksum record and report refers to these paths. Renaming would risk breaking verified reproduction for no scientific gain. Mapping:

| Suggested | Actual location |
|---|---|
| `src/generalized_latent_ode/` | `src/glo/` (model.py, solver.py, pipeline.py, data.py, five_shapes.py, evaluation.py, inference.py, …) |
| `models/` | `saved_models/` |
| `results/<experiment>/` | `results/` (five_shapes, capacity_scaling, ablation_2x2, conditioning, geolife, validation/…) |
| `docs/INFERENCE_GUIDE.md` | moved from `docs/PRETRAINED_INFERENCE_GUIDE.md` (`git mv`; references updated) |
| `scripts/train.py`, `evaluate.py`, … | `scripts/02_train_model.py`, `03_evaluate_model.py`, `31`–`34`, `predict_trajectory.py` |

## Added
* `saved_models/five_shapes_unified/` and `saved_models/ablation_2x2/{A,B,C,D}_*/` (byte-identical copies of existing checkpoints, `scripts/package_five_shapes_model.py`); `input_trajectories/example_figure_eight.csv`, `example_sinusoid.csv` (+ ground truth); `inference_outputs/five_shapes_*_example/`.
* `docs/` (Overview, Architecture, Datasets, Methodology, Experiments, Results, Reproducibility, Limitations, References, this report); `experiments/<name>/README.md` (8); `figures/architecture`, `figures/datasets` (7 figures that previously existed only in the local presentation bundle); `pyproject.toml`; `tests/test_five_family_models.py`.
* Previously untracked research artifacts now tracked: five-family data, checkpoints, logs, results and reports; capacity-scaling and 2×2 ablation (configs, scripts, checkpoints, results); `PROJECT_COMPLETION_AUDIT.md`; `audit/` (read-only verifiers and document generators); `presentation_results/`.
* Generated docs are produced by `audit/build_repo_docs.py` and `audit/build_experiment_readmes.py` from already verified sources.

## Removed
Only regenerable caches were deleted: all `__pycache__/` directories and `.pytest_cache/`. Nothing else was deleted.

## Kept locally, excluded from Git (`.gitignore`)
| Item | Why |
|---|---|
| `MTP_COMPLETE_PRESENTATION_MATERIALS/` and `.zip` (17 MB / 15 MB) | Duplicate presentation bundle (copies of reports and figures that are published elsewhere in the repository); kept for the presentation |
| `Geolife Trajectories 1.3.zip`, `User Guide-1.3.pdf` | Third-party dataset under a non-commercial licence; never redistributed |
| `data/geolife/*` (except `SOURCE.md`), per-segment/per-file GeoLife tables, GPS-path figures | Individual GPS records or derivatives |
| `saved_models/geolife_trained/`, `checkpoints/geolife/`, `logs/geolife/` | Weights trained on GeoLife; redistribution question open (see `saved_models/README.md`) |
| `*SMOKE_ONLY*`, `PROJECT_MANIFEST.json` | Engineering smoke artefacts / stale manifest (existing rules) |

## Retained although they look redundant (not deleted: uncertain or reproducibility-relevant)
* `backups/original_2026-11-15/` (4.5 MB, tracked): untouched originals from the first phase.
* `results/figures/*(copy).png`, `results/SMOKE_ONLY_*`: duplicate or smoke files that earlier reports may reference; safe to remove later after a reference check. **Needs your review.**
* `checkpoints/validation/` etc.: all seed runs are needed by the statistical comparisons; models are about 0.1 to 0.3 MB each, no Git LFS required.
* Absolute paths of the original machine remain only in run logs and `run_summary.json` provenance fields (no credentials). Scripts and docs use repository-relative paths.

## Checks
* Tests: `python -m pytest -q` → 119 passed.
* Secret scan over every file to be committed (token/key/password patterns, private-key headers, e-mail addresses): no findings (the only match is the pattern list inside `scripts/check_publication_readiness.py`).
* Largest committed file 10.5 MB (`data/processed/five_shapes/train.npz`); GitHub's 100 MB limit is not approached; no Git LFS was configured (not installed, not needed).
* Markdown links in README, docs, experiments and `saved_models/README.md` resolve to existing files.
