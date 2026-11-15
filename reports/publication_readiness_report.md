# Publication readiness report

Generated from `scripts/check_publication_readiness.py` (raw output: `results/publication_readiness.json`) and `python -m pytest -q tests`.

| Check | Result |
|---|---|
| Unit and integration tests | 71 passed |
| Packaged models (6): SHA-256 equals metadata and the original checkpoint; load; finite inference; weights unchanged by inference | all pass |
| Files committed / total size | 383 files, about 44 MB; no file near GitHub's 50 MB warning or 100 MB limit (largest about 8.7 MB) |
| Git LFS | not needed (each model file is about 106 KB) |
| Forbidden files (GeoLife archive, user guide, `.plt`, keys, `.env`, smoke-only files) | none |
| Secret-like strings | none |
| Required sources and documents present; every imported `glo` module committed | yes |
| Existing checkpoints | untouched (checksums verified by the GeoLife and inference scripts); nothing retrained |

## Deliberately not published
* GeoLife archive, user guide PDF, segment index, per-segment arrays and tables, pilot CSV/PNG files, and the four figures that draw individual users' GPS paths (Microsoft Research License Agreement: no redistribution of the data or derivative works).
* GeoLife-trained models and their logs: training was unfinished, and trained weights may count as derivative works. No results are claimed for them.
* `PROJECT_MANIFEST.json` (stale starter manifest listing smoke-only checkpoints) and all `*SMOKE_ONLY*` files.

## Warnings
* No software licence has been chosen for this code.
* Some text files contain the local path `/home/teaching/...` as provenance (not a credential).
* Results are single-dataset, small-scale; several unified-model results are single-seed (see README, Known limitations).
