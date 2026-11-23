#!/usr/bin/env python3
"""Assemble docs/ARCHITECTURE.md, DATASETS.md, METHODOLOGY.md, RESULTS.md, RESEARCH_LIMITATIONS.md from the already
verified presentation sources (MTP package / presentation_results). Text is not re-derived: sections are concatenated,
headings demoted, and slide-oriented wording / figure links adapted. Read-only on sources."""
import re
from pathlib import Path
root = Path(__file__).resolve().parents[1]
M = root / 'MTP_COMPLETE_PRESENTATION_MATERIALS'
docs = root / 'docs'


def demote(txt, by=1):
    return re.sub(r'^(#+) ', lambda m: '#' * (len(m.group(1)) + by) + ' ', txt, flags=re.M)


def clean(txt):
    txt = txt.replace('../07_FIGURES/ARCHITECTURE/', '../figures/architecture/')
    txt = re.sub(r'\n\(Figure file:[^\n]*\)\n', '\n', txt)
    txt = txt.replace('All results in the presentation are', 'All results are')
    txt = txt.replace('**Wording to use on slides:**', '**Recommended wording:**')
    txt = txt.replace(' (be explicit about these on the slides)', '')
    txt = txt.replace('`presentation_results/`, this package', '`presentation_results/`')
    return txt


def assemble(out, title, intro, files, base=M, strip_first_title=True):
    parts = [f'# {title}\n\n{intro}\n']
    for f in files:
        t = (base / f).read_text()
        parts.append(demote(clean(t)).strip() + '\n')
    (docs / out).write_text('\n'.join(parts))
    print('wrote', out)


assemble('ARCHITECTURE.md', 'Model architecture',
         'Variational Latent Neural ODE as implemented in `src/glo/model.py`, `solver.py` and `pipeline.py`. Parameter counts are measured by instantiating the model. Adapted from the verified project material; values are from the code, not from the original paper.',
         [f'03_MODEL_ARCHITECTURE/{n}.md' for n in ['COMPLETE_ARCHITECTURE', 'GRU_ENCODER', 'LATENT_SPACE', 'NEURAL_ODE_DYNAMICS', 'RK4_SOLVER', 'DECODER', 'TRAINING_LOSS', 'MODEL_CONFIGURATION']])
assemble('DATASETS.md', 'Datasets',
         'Synthetic trajectory families (generated deterministically by `scripts/01_generate_dataset.py` and `scripts/30_generate_five_shapes.py`) and the GeoLife evaluation data (not distributed).\n\n![five families](../figures/datasets/fig1_five_families_dataset_samples.png)\n',
         [f'04_DATASET/{n}.md' for n in ['DATASET_OVERVIEW', 'TRAJECTORY_GENERATION', 'IRREGULAR_OBSERVATIONS', 'TRAIN_VALIDATION_TEST_SPLIT', 'DATA_PREPROCESSING']])
assemble('METHODOLOGY.md', 'Methodology',
         'Training, validation, reconstruction and forecasting protocol. Test data are never used to select checkpoints or settings.\n\n![workflow](../figures/architecture/fig_methodology_flowchart.png)\n',
         [f'05_EXPERIMENTAL_METHODOLOGY/{n}.md' for n in ['COMPLETE_METHODOLOGY', 'TRAINING_PROCEDURE', 'EVALUATION_METRICS', 'EXPERIMENTAL_CONFIGURATIONS']])
assemble('RESEARCH_LIMITATIONS.md', 'Research limitations', 'What the evidence in this repository does and does not support.\n',
         ['08_DISCUSSION_AND_FUTURE_WORK/RESEARCH_LIMITATIONS.md', '08_DISCUSSION_AND_FUTURE_WORK/FUTURE_WORK.md'])

res = (root / 'presentation_results/RESULTS_ANALYSIS.md').read_text()
src = (root / 'presentation_results/RESULTS_SOURCE_INDEX.md').read_text()
res = res.replace('figures/fig', '../presentation_results/figures/fig')
src = src.replace('`presentation_results/figures/', '`presentation_results/figures/')
(docs / 'RESULTS.md').write_text(
    '# Results\n\nVerified numerical results of every experiment, with single-seed and multi-seed findings kept apart. '
    'Every number is traceable to a file under `results/`; the table of sources follows the analysis. '
    'Tables of per-experiment metrics: `results/*/tables/`. Per-segment GeoLife tables are not published (see [DATASETS.md](DATASETS.md)).\n\n'
    + demote(res).strip() + '\n\n---\n\n' + demote(src).strip() + '\n')
print('wrote RESULTS.md')
