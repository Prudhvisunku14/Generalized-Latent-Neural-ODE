#!/usr/bin/env python3
"""Pre-publication checks (evaluation only; trains and modifies nothing).

1. Every packaged model: SHA-256 of best_model.pt equals its metadata (and its original checkpoint where recorded), the model loads,
   and a deterministic forward/inference pass returns finite output of the right shape.
2. The set of files git would commit (tracked + untracked, minus ignored): sizes (GitHub warns at 50 MB, rejects at 100 MB),
   forbidden GeoLife / archive / key files, and secret-like strings.
3. Required source files and documents exist.
Writes results/publication_readiness.json and exits non-zero on any hard failure.
"""
import json, re, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, torch
from glo import inference as inf

torch.set_num_threads(1)
root = Path(__file__).resolve().parents[1]
saved = root / 'saved_models'
report = {'failures': [], 'warnings': []}


def fail(m): report['failures'].append(m)
def warn(m): report['warnings'].append(m)


# ------------------------------------------------------------------------------------------------------ 1. models
folders = {n: saved / rel for n, rel in inf.MODEL_REGISTRY.items()}
folders.update({f'geolife_trained/{d.name}': d for d in sorted((saved / 'geolife_trained').glob('*')) if d.is_dir()})
models = {}
rng = np.random.default_rng(0)
for name, folder in folders.items():
    r = {'folder': str(folder.relative_to(root))}
    try:
        meta = json.loads((folder / 'metadata.json').read_text())
        digest = inf.sha256_file(folder / 'best_model.pt')
        r['sha256'] = digest; r['size_bytes'] = (folder / 'best_model.pt').stat().st_size
        r['sha256_matches_metadata'] = digest == meta['sha256']
        orig = meta.get('original_checkpoint')
        if orig and (root / orig).exists():
            r['identical_to_original_checkpoint'] = inf.sha256_file(root / orig) == digest
        elif orig:
            r['identical_to_original_checkpoint'] = None
            warn(f'{name}: original checkpoint {orig} not present, only the metadata checksum was checked')
        lm = (inf.load_packaged_model(name, saved) if name in inf.MODEL_REGISTRY else inf.load_model_folder(folder))
        pre = lm.preprocessing
        t_obs = np.sort(np.concatenate([[0.0], rng.uniform(0.1, 7.0, 29)]))
        xy = rng.normal(0, 0.8, (30, 2)) * pre['normalization_scale'] + np.asarray(pre['normalization_mean'])
        before = inf.state_dict_digest(lm.model)
        out = inf.predict(lm, t_obs, xy, np.linspace(0, 10, 50))
        r.update(loads=True, n_parameters=lm.n_parameters, inference_output_shape=list(out['pred'].shape),
                 inference_finite=bool(np.isfinite(out['pred']).all()), weights_unchanged_by_inference=before == out['weights_digest'],
                 has_config_preprocessing_metadata=all((folder / f).exists() for f in ('model_config.json', 'preprocessing.json', 'metadata.json')))
        if not (r['sha256_matches_metadata'] and r['inference_finite'] and r['weights_unchanged_by_inference'] and r['inference_output_shape'] == [50, 2]
                and r.get('identical_to_original_checkpoint') is not False):
            fail(f'model {name}: a check failed: {r}')
    except Exception as e:                                         # noqa: BLE001  (report, do not hide)
        r.update(loads=False, error=repr(e)); fail(f'model {name}: {e!r}')
    models[name] = r
report['models'] = models

# -------------------------------------------------------------------------------------------- 2. what git would commit
files = sorted(set(subprocess.run(['git', 'ls-files', '-c', '-o', '--exclude-standard'], cwd=root, capture_output=True, text=True, check=True)
                   .stdout.splitlines()))
sizes = {f: (root / f).stat().st_size for f in files if (root / f).is_file()}
report['n_files'] = len(sizes); report['total_bytes'] = int(sum(sizes.values()))
report['largest_files'] = [{'file': f, 'bytes': s} for f, s in sorted(sizes.items(), key=lambda kv: -kv[1])[:8]]
for f, s in sizes.items():
    if s > 100 * 2 ** 20: fail(f'{f} is {s / 2 ** 20:.0f} MB (> 100 MB GitHub limit)')
    elif s > 50 * 2 ** 20: warn(f'{f} is {s / 2 ** 20:.0f} MB (> 50 MB GitHub warning)')
FORBIDDEN = re.compile(r'(\.plt$|\.zip$|geolife trajectories|user guide|\.pem$|\.key$|id_rsa|id_ed25519|\.env$|\.pyc$|SMOKE_ONLY)', re.I)
report['forbidden_files'] = [f for f in sizes if FORBIDDEN.search(f)]
if report['forbidden_files']:
    fail(f"forbidden files would be committed: {report['forbidden_files']}")
for f in sizes:
    if f.startswith('data/geolife/') and f != 'data/geolife/raw/SOURCE.md':
        fail(f'GeoLife-derived data would be committed: {f}')
SECRET = re.compile(r'(AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|'
                    r'(api[_-]?key|secret|passwd|password|token)\s*[:=]\s*[\'"][^\'"]{8,}[\'"])', re.I)
TEXT = {'.py', '.md', '.json', '.yaml', '.yml', '.txt', '.sh', '.csv', '.log', '.out', ''}
hits = []
for f in sizes:
    if Path(f).suffix.lower() in TEXT and sizes[f] < 5 * 2 ** 20:
        txt = (root / f).read_text(errors='ignore')
        for m in SECRET.finditer(txt):
            hits.append((f, m.group(0)[:20] + '...'))
report['secret_like_strings'] = hits
if hits:
    fail(f'secret-like strings: {hits}')
home = [f for f in sizes if Path(f).suffix.lower() in TEXT and '/home/teaching' in (root / f).read_text(errors='ignore')]
report['files_mentioning_local_home_path'] = home
if home:
    warn(f'{len(home)} text files mention the local path /home/teaching (harmless provenance, not a credential)')

# ----------------------------------------------------------------------------------------------- 3. required files
REQUIRED = ['README.md', 'requirements.txt', '.gitignore', 'src/glo/model.py', 'src/glo/inference.py', 'src/glo/geolife.py',
            'src/glo/geolife_evaluation.py', 'scripts/02_train_model.py', 'scripts/predict_trajectory.py', 'scripts/20_inspect_geolife.py',
            'scripts/21_evaluate_geolife_transfer.py', 'scripts/23_geolife_trained_models.py', 'saved_models/README.md',
            'docs/PRETRAINED_INFERENCE_GUIDE.md', 'reports/validation_study_report.md', 'reports/conditioning_study_report.md',
            'reports/geolife_real_world_interpolation_report.md', 'reports/geolife_dataset_audit.md', 'configs/model_d.yaml', 'configs/geolife.yaml',
            'data/processed/three_shapes/train.npz', 'tests/test_inference.py', 'tests/test_geolife.py', 'data/geolife/raw/SOURCE.md']
report['missing_required_files'] = [f for f in REQUIRED if f not in sizes]
if report['missing_required_files']:
    fail(f"required files missing from the commit: {report['missing_required_files']}")
imports = set()
for f in sizes:
    if f.endswith('.py'):
        imports |= set(re.findall(r'^(?:from|import) glo\.?(\w*)', (root / f).read_text(), re.M))
report['glo_modules_imported'] = sorted(i for i in imports if i)
report['glo_modules_not_committed'] = [m for m in report['glo_modules_imported'] if f'src/glo/{m}.py' not in sizes]
if report['glo_modules_not_committed']:
    fail(f"imported but not committed: {report['glo_modules_not_committed']}")

(root / 'results/publication_readiness.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k not in ('models', 'files_mentioning_local_home_path')}, indent=1))
print('\nmodel checks:')
for n, r in models.items():
    print(f"  {n:42s} loads={r.get('loads')} sha_ok={r.get('sha256_matches_metadata')} identical_to_original={r.get('identical_to_original_checkpoint')} "
          f"inference_finite={r.get('inference_finite')} params={r.get('n_parameters')} size={r.get('size_bytes')}")
print('\nFAILURES:', report['failures'] or 'none')
sys.exit(1 if report['failures'] else 0)
