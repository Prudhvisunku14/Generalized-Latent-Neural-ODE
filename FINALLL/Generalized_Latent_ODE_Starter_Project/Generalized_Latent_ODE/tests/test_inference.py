"""Pretrained-inference system: input validation, determinism, frozen weights, no leakage, and consistency
with the existing evaluation pipeline. Tests that need packaged models / real data skip if they are absent."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from glo import inference as inf

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / 'saved_models'
EXAMPLES = ROOT / 'input_trajectories'
need_models = pytest.mark.skipif(not (SAVED / 'three_shapes_unified' / 'best_model.pt').exists(), reason='saved_models not built')
need_examples = pytest.mark.skipif(not (EXAMPLES / 'example_circle.csv').exists(), reason='example inputs not built')
PRE = {'context_end': 7.0, 'time_horizon': 10.0, 'coordinate_abs_max_train': 4.7}


def write(tmp_path, text, name='obs.csv'):
    p = tmp_path / name
    p.write_text(text)
    return p


# --------------------------------------------------------------------------- input validation
def test_valid_csv_is_read_sorted_and_extra_columns_ignored(tmp_path):
    p = write(tmp_path, 'T, X ,y,family\n0.5,1,2,circle\n0,0,0,circle\n1.0,2,3,spiral\n')
    t, xy, warns = inf.read_observations(p, PRE, trained_n_obs=3)
    assert list(t) == [0.0, 0.5, 1.0] and xy.tolist() == [[0, 0], [1, 2], [2, 3]]      # sorted, pairs preserved
    assert any('Ignoring extra column' in w for w in warns) and any('not sorted' in w for w in warns)


@pytest.mark.parametrize('text, fragment', [
    ('t,x\n0,1\n1,2\n', "missing required column"),
    ('', 'empty'),
    ('t,x,y\n', 'no data rows'),
    ('t,x,y\n0,1,2\n1,abc,3\n', "line 3, column 'x'"),
    ('t,x,y\n0,1,2\n1,nan,3\n', 'not finite'),
    ('t,x,y\n0,1,2\n1,inf,3\n', 'not finite'),
    ('t,x,y\n0,1,2\n0,2,3\n1,0,0\n', 'duplicate'),
    ('t,x,y\n0,1,2\n', 'at least 2'),
    ('t,x,y\n-1,1,2\n1,2,3\n', 'negative'),
    ('t,x,y\n0,1,2\n6,1,1\n8,2,3\n', 'after the context end'),
])
def test_invalid_inputs_give_clear_errors(tmp_path, text, fragment):
    with pytest.raises(inf.InputError) as e:
        inf.read_observations(write(tmp_path, text), PRE)
    assert fragment in str(e.value)


def test_missing_file_and_warnings(tmp_path):
    with pytest.raises(inf.InputError, match='not found'):
        inf.read_observations(tmp_path / 'nope.csv', PRE)
    rows = ''.join(f'{t},{np.cos(t)},{np.sin(t)}\n' for t in np.linspace(1, 6, 12))
    _, _, w = inf.read_observations(write(tmp_path, 't,x,y\n' + rows), PRE, trained_n_obs=30)
    assert any('not t=0' in x for x in w) and any('trained with 30' in x for x in w)
    big = 't,x,y\n0,100,0\n1,0,0\n'
    assert any('training range' in x for x in inf.read_observations(write(tmp_path, big, 'b.csv'), PRE)[2])


def test_query_time_options():
    t, w = inf.make_query_times(200, 10.0, PRE)
    assert len(t) == 200 and t[0] == 0 and t[-1] == 10 and not w
    assert any('beyond' in x or 'exceeds the trained horizon' in x for x in inf.make_query_times(50, 12.0, PRE)[1])
    assert any('no forecast region' in x for x in inf.make_query_times(50, 6.0, PRE)[1])
    for bad in ((1, 10.0), (50, -1.0), (50, float('nan'))):
        with pytest.raises(inf.InputError):
            inf.make_query_times(*bad, PRE)


def test_region_labels_split_at_context_end():
    r = inf.region_labels(np.array([0., 6.99, 7.0, 7.01, 10.]), 7.0)
    assert list(r) == ['reconstruction', 'reconstruction', 'reconstruction', 'forecast', 'forecast']


# --------------------------------------------------------------------------- model loading
def test_unavailable_model_is_reported_not_substituted(tmp_path):
    with pytest.raises(inf.ModelUnavailableError, match='NOT substituted'):
        inf.load_packaged_model('three_shapes_unified', tmp_path)
    with pytest.raises(inf.ModelUnavailableError, match='Unknown model'):
        inf.load_packaged_model('ellipse_only', tmp_path)


@need_models
@pytest.mark.parametrize('name', list(inf.MODEL_REGISTRY))
def test_every_packaged_model_loads_and_checksums_match(name):
    if not (SAVED / inf.MODEL_REGISTRY[name] / 'best_model.pt').exists():
        pytest.skip(f'{name} not packaged')
    lm = inf.load_packaged_model(name, SAVED)
    assert lm.load_report['missing_keys'] == [] and lm.load_report['unexpected_keys'] == []
    assert lm.n_parameters == lm.metadata['n_parameters']
    orig = ROOT / lm.metadata['original_checkpoint']
    assert inf.sha256_file(orig) == lm.sha256                 # packaged weights are byte-identical to the original
    assert not lm.model.training and all(not p.requires_grad for p in lm.model.parameters())


# --------------------------------------------------------------------------- prediction properties
@pytest.fixture(scope='module')
def lm3():
    return inf.load_packaged_model('three_shapes_unified', SAVED)


@pytest.fixture(scope='module')
def example_obs(lm3):
    t, xy, _ = inf.read_observations(EXAMPLES / 'example_spiral.csv', lm3.preprocessing, 30)
    return t, xy


@need_models
@need_examples
def test_deterministic_frozen_and_no_gradients(lm3, example_obs):
    t, xy = example_obs
    q = np.linspace(0, 10, 200)
    digest = inf.state_dict_digest(lm3.model)
    a = inf.predict(lm3, t, xy, q)
    b = inf.predict(lm3, t, xy, q)
    assert np.array_equal(a['pred'], b['pred'])               # same input -> identical output
    assert inf.state_dict_digest(lm3.model) == digest == a['weights_digest']
    assert all(p.grad is None for p in lm3.model.parameters())
    assert a['pred'].shape == (200, 2) and np.isfinite(a['pred']).all()


@need_models
@need_examples
def test_output_times_do_not_change_predictions(lm3, example_obs):
    """Predictions at a time must not depend on which other times are requested (solver grid anchored at t=0..10)."""
    t, xy = example_obs
    full = inf.predict(lm3, t, xy, np.linspace(0, 10, 201))['pred']          # contains 1.0, 5.0, 7.0, 9.0
    sub = inf.predict(lm3, t, xy, np.array([1.0, 5.0, 7.0, 9.0]))['pred']
    assert np.allclose(full[[20, 100, 140, 180]], sub, atol=1e-6)
    short = inf.predict(lm3, t, xy, np.linspace(0, 8, 161))['pred']          # forecast-end 8: T for the encoder must stay 10
    assert np.allclose(short, full[:161], atol=1e-6)


@need_models
@need_examples
def test_observation_order_is_irrelevant_after_sorting(lm3, tmp_path, example_obs):
    t, xy = example_obs
    perm = np.random.default_rng(0).permutation(len(t))
    text = 't,x,y\n' + ''.join(f'{t[i]!r},{xy[i, 0]!r},{xy[i, 1]!r}\n' for i in perm)
    t2, xy2, w = inf.read_observations(write(tmp_path, text), lm3.preprocessing)
    assert np.array_equal(t2, t) and np.array_equal(xy2, xy) and any('not sorted' in x for x in w)


@need_models
def test_future_observations_are_rejected_not_used(lm3, tmp_path):
    with pytest.raises(inf.InputError, match='forecast interval'):
        inf.read_observations(write(tmp_path, 't,x,y\n0,1,0\n3,0,1\n7.5,-1,0\n'), lm3.preprocessing)


# --------------------------------------------------------------------------- consistency with the existing evaluation code
@need_models
@pytest.mark.parametrize('name', ['three_shapes_unified', 'two_shapes_unified', 'dynamics_conditioned', 'circle_specialist'])
def test_matches_existing_evaluation_pipeline(name):
    from glo.evaluation import collect
    if not (SAVED / inf.MODEL_REGISTRY[name] / 'best_model.pt').exists():
        pytest.skip('not packaged')
    lm = inf.load_packaged_model(name, SAVED)
    exp = lm.metadata['training_dataset']
    fams = None if len(lm.metadata['training_families']) > 2 or name != 'circle_specialist' else ['circle']
    ck = {'experiment': exp}
    res = collect(lm.model, ck, ROOT / 'data/processed', exp, split='test', families=fams, n_obs=30, n_query=200, seed=123)
    worst = 0.0
    for k in range(0, len(res['family']), max(1, len(res['family']) // 12)):
        out = inf.predict(lm, res['obs_t'][k].astype(np.float64), res['obs'][k].astype(np.float64), res['query_t'].astype(np.float64))
        worst = max(worst, float(np.abs(out['pred'] - res['pred'][k]).max()))
    assert worst < 1e-4, f'{name}: inference deviates from evaluation pipeline by {worst}'


# --------------------------------------------------------------------------- command line
def run_cli(*args):
    return subprocess.run([sys.executable, str(ROOT / 'scripts/predict_trajectory.py'), *map(str, args)],
                          capture_output=True, text=True, cwd=ROOT)


@need_models
@need_examples
def test_cli_outputs_and_ground_truth_never_influences_prediction(tmp_path):
    a, b = tmp_path / 'with_gt', tmp_path / 'without_gt'
    r1 = run_cli('--input', EXAMPLES / 'example_ellipse.csv', '--output', a, '--n-query', 150,
                 '--ground-truth', EXAMPLES / 'ground_truth/example_ellipse.csv')
    r2 = run_cli('--input', EXAMPLES / 'example_ellipse.csv', '--output', b, '--n-query', 150)
    assert r1.returncode == 0 and r2.returncode == 0, r1.stderr + r2.stderr
    for f in ('predicted_trajectory.csv', 'reconstruction_forecast.png', 'x_vs_time.png', 'y_vs_time.png', 'run_summary.json'):
        assert (a / f).exists() and (b / f).exists()
    assert (a / 'metrics.json').exists() and not (b / 'metrics.json').exists()        # metrics only when ground truth is given
    assert (a / 'predicted_trajectory.csv').read_bytes() == (b / 'predicted_trajectory.csv').read_bytes()
    m = json.loads((a / 'metrics.json').read_text())
    assert all(k in m for k in ('reconstruction_rmse', 'forecast_rmse', 'forecast_mae', 'forecast_ade', 'forecast_fde', 'chamfer_distance'))
    lines = (a / 'predicted_trajectory.csv').read_text().splitlines()
    assert lines[0] == 't,predicted_x,predicted_y,region' and len(lines) == 151
    assert {l.split(',')[3] for l in lines[1:]} == {'reconstruction', 'forecast'}


@need_models
@need_examples
def test_cli_reports_bad_input_and_unavailable_model(tmp_path):
    bad = write(tmp_path, 't,x,y\n0,1,2\n9,1,1\n')
    r = run_cli('--input', bad, '--output', tmp_path / 'o')
    assert r.returncode != 0 and 'after the context end' in r.stderr
    empty = tmp_path / 'empty_models'
    empty.mkdir()
    r = run_cli('--input', EXAMPLES / 'example_circle.csv', '--saved-models', empty, '--output', tmp_path / 'o2')
    assert r.returncode == 3 and 'NOT substituted' in r.stderr
    assert not (tmp_path / 'o2').exists()


@need_models
@need_examples
def test_cli_batch_mode(tmp_path):
    import shutil
    inp = tmp_path / 'inputs'; inp.mkdir()          # only the three examples (other user files may live in EXAMPLES)
    for stem in ('example_circle', 'example_spiral', 'example_ellipse'):
        shutil.copy(EXAMPLES / f'{stem}.csv', inp)
    r = run_cli('--input', inp, '--output', tmp_path / 'batch', '--ground-truth-dir', EXAMPLES / 'ground_truth', '--quiet')
    assert r.returncode == 0, r.stderr
    for stem in ('example_circle', 'example_spiral', 'example_ellipse'):
        assert (tmp_path / 'batch' / stem / 'predicted_trajectory.csv').exists()
        assert (tmp_path / 'batch' / stem / 'metrics.json').exists()
