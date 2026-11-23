# Reproducibility

Tested environment: Python 3.12, torch 2.5.1, torchdiffeq 0.2.5, numpy 1.26.4, scipy 1.13.1, pandas 2.2.2 (CPU).

## Install
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
# install PyTorch for your platform first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
pytest -q                                                 # 119 tests, about 25 s on CPU
```

## Seeds and determinism
* Training seed 42 for all single-run experiments; the 2x2 ablation additionally fixes `--data-seed 42` so all cells see the same data order.
* Evaluation: 30 irregular observations at t <= 7 (first always at t = 0), observation seed 123, identical for every model; posterior-mean latent state; 200 evaluation points on [0, 10].
* Inference is bit-reproducible on CPU with one thread.

## Commands
```bash
# data (deterministic; already included under data/processed/)
python scripts/01_generate_dataset.py --help
python scripts/30_generate_five_shapes.py

# training (about 80 min CPU per five-family run; checkpoints are never overwritten without --overwrite)
python scripts/02_train_model.py --config configs/model_d.yaml --experiment three_shapes
bash scripts/run_capacity_scaling.sh
bash scripts/run_ablation_2x2.sh

# evaluation of the checkpoints included in this repository
python scripts/31_evaluate_five_shapes.py
python scripts/32_evaluate_capacity_scaling.py
python scripts/33_evaluate_ablation_2x2.py
python scripts/34_audit_checkpoint_selection.py
python scripts/check_publication_readiness.py

# independent re-verification (read-only)
python audit/verify_1_datasets.py        # ... through verify_5_corrected_documents.py
```
Training scripts print `--help` for every option. GeoLife steps need the dataset you download yourself (see README); the GeoLife-trained models and per-segment tables are not distributed.

## What can and cannot be reproduced from the repository alone
| Item | From the repository alone? |
|---|---|
| Synthetic datasets, training, all synthetic evaluations | Yes |
| Packaged-model inference | Yes |
| GeoLife evaluations | Only after downloading GeoLife 1.3 yourself |
| Paper numbers (Chen et al., 2018), classification and CNF experiments | Not part of this repository |

Because most results are single-seed, re-running a training script with another seed is expected to change forecast errors by tens of percent (see [RESEARCH_LIMITATIONS](RESEARCH_LIMITATIONS.md)).
