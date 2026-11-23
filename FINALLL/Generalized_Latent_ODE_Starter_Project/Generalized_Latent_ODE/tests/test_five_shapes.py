import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from glo import five_shapes as F
from glo.data import FAMILY_TO_ID as OLD_IDS, generate_project_data, load_split, validate_dataset, sample_trajectory

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed'
T = np.linspace(0, 10, 200, dtype=np.float32)
needs_data = pytest.mark.skipif(not (DATA / 'five_shapes/test.npz').exists(), reason='five_shapes dataset not generated')


def test_original_family_ids_unchanged():
    assert {k: F.FAMILY_TO_ID[k] for k in OLD_IDS} == OLD_IDS
    assert F.PARAM_NAMES[:9] == list(__import__('glo.data', fromlist=['PARAM_NAMES']).PARAM_NAMES)


@pytest.mark.parametrize('family', F.NEW_FAMILIES)
def test_new_family_sampler_is_deterministic_and_finite(family):
    a = F.sample_new_trajectory(family, T, np.random.default_rng(3))
    b = F.sample_new_trajectory(family, T, np.random.default_rng(3))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert a[0].shape == (200, 2) and a[1].shape == (len(F.PARAM_NAMES),) and np.isfinite(a[0]).all()


def test_figure_eight_formula_and_shape_check():
    for seed in range(20):
        xy, p = F.sample_new_trajectory('figure_eight', T, np.random.default_rng(seed))
        theta = p[1] * T + p[2]
        assert np.allclose(xy[:, 0], p[3] + p[6] * np.sin(theta), atol=1e-5)
        assert np.allclose(xy[:, 1], p[4] + p[7] * np.sin(2 * theta), atol=1e-5)
        c = F.figure_eight_checks(xy, p, T)
        assert c['crossing_directions_alternate'] and c['covers_full_eight'] and c['max_abs_y_offset_at_centre_over_B'] < .1


def test_figure_eight_check_rejects_a_circle():
    xy, p = sample_trajectory('circle', T, np.random.default_rng(0))
    p11 = np.concatenate([p, [0, 0]])
    c = F.figure_eight_checks(xy, p11, T)
    assert not (c['crossing_directions_alternate'] and c['max_abs_y_offset_at_centre_over_B'] < .1)


def test_sinusoid_formula_and_shape_check():
    for seed in range(20):
        xy, p = F.sample_new_trajectory('sinusoid', T, np.random.default_rng(seed))
        assert np.allclose(xy[:, 0], p[10] + p[9] * T, atol=1e-5)
        assert np.allclose(xy[:, 1], p[4] + p[0] * np.sin(p[1] * T + p[2]), atol=1e-5)
        assert abs(p[10] + 5 * p[9] - p[3]) < 1e-5          # path centred on cx
        c = F.sinusoid_checks(xy, p, T)
        assert c['x_monotone'] and c['x_linear_max_residual'] < 1e-4


def test_parameter_ranges_respected():
    rng = np.random.default_rng(11)
    for _ in range(300):
        _, p = F.sample_new_trajectory('figure_eight', T, rng)
        assert .85 <= p[6] <= 1.5 and .45 <= p[7] <= 1.0 and .75 <= abs(p[1]) <= 1.45
        _, q = F.sample_new_trajectory('sinusoid', T, rng)
        assert .4 <= q[0] <= 1.0 and .75 <= q[1] <= 1.45 and .2 <= abs(q[9]) <= .3


def test_small_generation_reproducible_and_leak_free(tmp_path):
    a = F.generate_five_shapes(tmp_path / 'a', n_per_family=20, n_points=40, n_obs=9, seed=5)
    b = F.generate_five_shapes(tmp_path / 'b', n_per_family=20, n_points=40, n_obs=9, seed=5)
    assert a['split_counts'] == {'train': 70, 'val': 15, 'test': 15}
    assert validate_dataset(tmp_path / 'a', 'five_shapes') == a['split_counts']
    for s in ('train', 'val', 'test'):
        da, _ = load_split(tmp_path / 'a', 'five_shapes', s); db, _ = load_split(tmp_path / 'b', 'five_shapes', s)
        for k in ('xy', 'noisy_xy', 'ids', 'default_obs_mask', 'params'):
            assert np.array_equal(da[k], db[k])
        assert not da['default_obs_mask'][:, da['time'] > 7].any() and da['default_obs_mask'][:, 0].all()


def test_small_generation_matches_three_shape_generator(tmp_path):
    generate_project_data(tmp_path, n_per_family=20, n_points=40, n_obs=9, seed=5)
    F.generate_five_shapes(tmp_path, n_per_family=20, n_points=40, n_obs=9, seed=5, three_shapes_dir=tmp_path / 'three_shapes')
    for s in ('train', 'val', 'test'):
        d3, _ = load_split(tmp_path, 'three_shapes', s); d5, _ = load_split(tmp_path, 'five_shapes', s)
        pos = {k: i for i, k in enumerate(d5['ids'].tolist())}; j = [pos[k] for k in d3['ids'].tolist()]
        for k in ('xy', 'noisy_xy', 'default_obs_mask'):
            assert np.array_equal(d3[k], d5[k][j])


def test_refuses_to_overwrite(tmp_path):
    F.generate_five_shapes(tmp_path, n_per_family=10, n_points=40, n_obs=9)
    with pytest.raises(FileExistsError):
        F.generate_five_shapes(tmp_path, n_per_family=10, n_points=40, n_obs=9)


@needs_data
def test_real_dataset_counts_and_disjoint_splits():
    assert validate_dataset(DATA, 'five_shapes') == {'train': 3500, 'val': 750, 'test': 750}
    ids = {s: set(load_split(DATA, 'five_shapes', s)[0]['ids'].tolist()) for s in ('train', 'val', 'test')}
    assert sum(map(len, ids.values())) == 5000 and not (ids['train'] & ids['val'] or ids['train'] & ids['test'] or ids['val'] & ids['test'])
    for s, n in (('train', 700), ('val', 150), ('test', 150)):
        fam = load_split(DATA, 'five_shapes', s)[0]['family']
        assert np.bincount(fam, minlength=5).tolist() == [n] * 5


@needs_data
def test_real_dataset_keeps_three_shape_data_and_splits():
    for s in ('train', 'val', 'test'):
        d3, _ = load_split(DATA, 'three_shapes', s); d5, _ = load_split(DATA, 'five_shapes', s)
        pos = {k: i for i, k in enumerate(d5['ids'].tolist())}
        j = [pos[k] for k in d3['ids'].tolist()]                     # KeyError if a trajectory changed split
        for k in ('xy', 'noisy_xy', 'default_obs_mask', 'family'):
            assert np.array_equal(d3[k], d5[k][j])
        assert np.array_equal(d3['params'], d5['params'][j][:, :9])


@needs_data
def test_normalisation_from_five_shape_training_split_only():
    tr, meta = load_split(DATA, 'five_shapes', 'train')
    x = tr['xy'].astype(np.float64); m = x.mean((0, 1))
    assert np.allclose(meta['normalization_mean'], m) and np.isclose(meta['normalization_scale'], np.sqrt(((x - m) ** 2).mean()))


@needs_data
def test_train_script_accepts_five_shapes(tmp_path):
    ck = tmp_path / 'smoke.pt'
    r = subprocess.run([sys.executable, str(ROOT / 'scripts/02_train_model.py'), '--config', 'configs/model_d.yaml', '--experiment', 'five_shapes',
                        '--epochs', '1', '--max-train-batches', '2', '--max-val-batches', '1', '--checkpoint', str(ck), '--threads', '1'],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert 'train: 3500 val: 750' in r.stdout and ck.exists()
