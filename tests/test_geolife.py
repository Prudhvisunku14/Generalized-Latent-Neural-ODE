"""Tests for the GeoLife synthetic-to-real experiment (requirements 1-20 of the task, numbered below)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from glo import geolife as gl, geolife_evaluation as ge, inference as inf  # noqa: E402

HEADER = 'Geolife trajectory\nWGS 84\nAltitude is in Feet\nReserved 3\n0,2,255,My Track,0,0,2,8421376\n0\n'
SAVED = ROOT / 'saved_models'


def plt_bytes(rows):
    return (HEADER + '\n'.join(rows) + '\n').encode()


def synthetic_gps(n=200, dt=None, seed=0, lat0=39.98, lon0=116.31):
    """A smooth curved track (metres -> lat/lon) with irregular sampling, for tests that need a full segment."""
    rng = np.random.default_rng(seed)
    dt = rng.choice([1, 2, 5], size=n - 1) if dt is None else np.full(n - 1, dt)
    t = 1_224_730_000 + np.concatenate([[0], np.cumsum(dt)])
    s = (t - t[0]) * 3.0
    x, y = 200 * np.sin(s / 400), s * 0.5
    lat, lon = gl.aeqd_inverse(x, y, lat0, lon0)
    return t.astype(np.int64), lat, lon


@pytest.fixture(scope='module')
def lm():
    return inf.load_packaged_model('three_shapes_unified', SAVED)


# 1 ------------------------------------------------------------------------------------------- parsing
def test_01_plt_parsing():
    p = gl.parse_plt(plt_bytes(['39.984702,116.318417,0,492,39744.1201851852,2008-10-23,02:53:04',
                                '39.984683,116.31845,0,492,39744.1202546296,2008-10-23,02:53:10']))
    assert p.header_ok and p.n_lines == 2 and p.n_malformed == 0
    np.testing.assert_allclose(p.lat, [39.984702, 39.984683]); np.testing.assert_allclose(p.lon, [116.318417, 116.31845])


# 2 ------------------------------------------------------------------------------------------- timestamps
def test_02_timestamp_conversion():
    p = gl.parse_plt(plt_bytes(['39.9,116.3,0,492,39744.1201851852,2008-10-23,02:53:04']))
    assert p.t_unix[0] == 1224730384                      # 2008-10-23T02:53:04Z
    assert abs((39744.1201851852 - gl.EXCEL_EPOCH_UNIX_DAYS) * 86400 - p.t_unix[0]) < 1
    assert p.n_datenum_mismatch == 0
    bad = gl.parse_plt(plt_bytes(['39.9,116.3,0,492,39745.5,2008-10-23,02:53:04']))
    assert bad.n_datenum_mismatch == 1


# 3 ------------------------------------------------------------------------------------------- projection
def test_03_projection():
    lat0, lon0 = 39.98, 116.31
    rng = np.random.default_rng(1)
    lat = lat0 + rng.uniform(-0.05, 0.05, 50); lon = lon0 + rng.uniform(-0.05, 0.05, 50)
    x, y = gl.aeqd_forward(lat, lon, lat0, lon0)
    r = gl.gaussian_radius(lat0)
    np.testing.assert_allclose(np.hypot(x, y), gl.haversine(lat0, lon0, lat, lon, radius=r), rtol=1e-9, atol=1e-6)
    la2, lo2 = gl.aeqd_inverse(x, y, lat0, lon0)
    np.testing.assert_allclose(la2, lat, atol=1e-10); np.testing.assert_allclose(lo2, lon, atol=1e-10)
    xn, yn = gl.aeqd_forward(lat0 + 0.001, lon0, lat0, lon0)       # 0.001 deg north ~ 111 m, x = 0
    assert abs(xn) < 1e-6 and 110.5 < yn < 111.5
    xe, _ = gl.aeqd_forward(lat0, lon0 + 0.001, lat0, lon0)         # east positive
    assert xe == pytest.approx(r * np.radians(0.001) * np.cos(np.radians(lat0)), rel=1e-6)


# 4 ------------------------------------------------------------------------------------------- missing / duplicate
def test_04_duplicate_and_backward_timestamps():
    keep, dup, back = gl.clean_time_order(np.array([0, 1, 1, 2, 1, 3, 3, 5]))
    assert list(keep) == [0, 1, 3, 5, 7] and dup == 2 and back == 1
    p = gl.parse_plt(plt_bytes(['39.9,116.3,0,0,39744.1,2008-10-23,02:53:04', '39.9,116.3,0,0,39744.1,2008-10-23,02:53:04',
                                '', '39.91,116.3,0,0,39744.1,2008-10-23,02:53:09']))
    assert p.n_lines == 3                                       # blank line ignored


# 5 ------------------------------------------------------------------------------------------- invalid records
def test_05_invalid_records():
    p = gl.parse_plt(plt_bytes(['91.0,116.3,0,0,39744.1,2008-10-23,02:53:04',      # latitude out of range
                                '39.9,200.0,0,0,39744.1,2008-10-23,02:53:05',      # longitude out of range
                                '0,0,0,0,39744.1,2008-10-23,02:53:06',             # null island
                                'nan,116.3,0,0,39744.1,2008-10-23,02:53:07',       # non-finite
                                '39.9,116.3,0,0',                                  # wrong field count
                                'abc,116.3,0,0,39744.1,2008-10-23,02:53:08',       # non-numeric
                                '39.9,116.3,0,0,39744.1202,2008-10-23,02:53:09']))
    assert p.n_invalid_coord == 4 and p.n_malformed == 2 and len(p.lat) == 1


# 6 ------------------------------------------------------------------------------------------- segments
def test_06_continuous_segments():
    cfg = gl.SegmentConfig()
    t, lat, lon = synthetic_gps(450, dt=2)
    t = t.copy(); t[250:] += 600                                  # a 10-minute recording gap
    runs = gl.continuous_runs(t, lat, lon, cfg)
    assert runs == [(0, 250), (250, 450)]
    lat2 = lat.copy(); lat2[100] += 0.05                          # a 5.5 km spike (implausible jump) splits twice
    assert len(gl.continuous_runs(t[:250], lat2[:250], lon[:250], cfg)) == 3
    w = gl.candidate_windows(t, lat, lon, cfg)
    assert [(a, b) for a, b, _, _ in w] == [(0, 200), (250, 450)] and all(ok for *_, ok, _ in [(0, 0, x[2], x[3]) for x in w])
    ts, las, los = synthetic_gps(200, dt=1); las = np.full(200, 39.98) + np.random.default_rng(0).normal(0, 1e-5, 200)
    assert gl.window_checks(ts, las, np.full(200, 116.31), cfg) == (False, 'stationary')


# 7, 8 ---------------------------------------------------------------------------------------- task construction
def test_07_08_context_future_separation_and_30_observations():
    t, lat, lon = synthetic_gps()
    task = gl.build_task(t, lat, lon, 30, seed=5)
    assert len(task.obs_idx) == 30 and task.obs_idx[0] == 0 and len(np.unique(task.obs_idx)) == 30
    assert task.t_model[task.obs_idx].max() <= 7.0
    assert task.t_model[task.interp_idx].max() <= 7.0 and task.t_model[task.future_idx].min() > 7.0
    sets = [set(task.obs_idx), set(task.interp_idx), set(task.ctx_tail_idx), set(task.future_idx)]
    assert all(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1:])
    assert set().union(*sets) == set(range(len(t)))
    assert task.t_model[task.interp_idx].min() > 0 and task.t_model[task.interp_idx].max() < task.t_model[task.obs_idx[-1]]


# 9 ------------------------------------------------------------------------------------------- irregular time
def test_09_irregular_timestamps_preserved():
    t, lat, lon = synthetic_gps()
    task = gl.build_task(t, lat, lon, 30, seed=1)
    np.testing.assert_allclose(task.t_sec, t - t[0])
    np.testing.assert_allclose(task.t_model, 10 * task.t_sec / task.t_sec[-1])
    assert len(np.unique(np.round(np.diff(task.t_sec), 6))) > 1      # spacing NOT made uniform
    assert task.t_model[0] == 0 and task.t_model[-1] == 10


# 10 ------------------------------------------------------------------------------------------ normalisation
def test_10_normalisation_and_inverse():
    rng = np.random.default_rng(0); xy = rng.normal(0, 300, (30, 2)) + [1000, -500]
    for s in (gl.DEFAULT_STRATEGY, gl.NormStrategy('x', 'first_obs', 'obs_max', 1.3, 0.7, (0.5, -0.2))):
        tr = gl.fit_transform(xy, s)
        np.testing.assert_allclose(tr.to_metres(tr.to_model(xy)), xy, atol=1e-9)
    tr = gl.fit_transform(xy, gl.DEFAULT_STRATEGY); u = tr.to_model(xy)
    np.testing.assert_allclose(u.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(np.sqrt((np.linalg.norm(u, axis=1) ** 2).mean()), 1.0)


# 11 ------------------------------------------------------------------------------------------ leakage
def test_11_no_hidden_position_leakage(lm):
    t, lat, lon = synthetic_gps()
    task = gl.build_task(t, lat, lon, 30, seed=2)
    r1, p1 = ge.evaluate_interpolation(task, lm, gl.DEFAULT_STRATEGY, return_preds=True)
    hidden = np.setdiff1d(np.arange(len(t)), task.obs_idx)
    task.xy_m[hidden] += np.random.default_rng(0).normal(0, 500, (len(hidden), 2))   # corrupt every withheld record
    r2, p2 = ge.evaluate_interpolation(task, lm, gl.DEFAULT_STRATEGY, return_preds=True)
    for m in ge.INTERP_METHODS:
        np.testing.assert_array_equal(p1[m], p2[m])                 # predictions identical
    assert r1['scale_m_per_model_unit'] == r2['scale_m_per_model_unit']
    _, f1, _ = ge.evaluate_forecast(task, lm, gl.DEFAULT_STRATEGY, 'B', return_preds=True)
    task.xy_m[task.future_idx] = 0.0
    _, f2, _ = ge.evaluate_forecast(task, lm, gl.DEFAULT_STRATEGY, 'B', return_preds=True)
    np.testing.assert_array_equal(f1['latent_ode'], f2['latent_ode'])
    # protocol B uses no time after the last observation
    tb = ge.protocol_b_time(task); assert tb[task.obs_idx[-1]] == pytest.approx(7.0)


# 12, 13 -------------------------------------------------------------------------------------- baselines
def test_12_linear_interpolation():
    t = np.array([0., 1., 3., 4.]); xy = np.stack([2 * t + 1, -t], 1); q = np.array([0.5, 2.0, 3.7])
    np.testing.assert_allclose(ge.linear_interp(t, xy, q), np.stack([2 * q + 1, -q], 1))


def test_13_cubic_interpolation():
    t = np.array([0., 0.7, 1.5, 3., 4., 6.]); f = lambda s: np.stack([s ** 3 - 2 * s, 0.5 * s ** 2], 1)
    q = np.array([0.3, 2.2, 5.1])
    np.testing.assert_allclose(ge.cubic_interp(t, f(t), q), f(q), atol=1e-9)     # not-a-knot reproduces cubics
    np.testing.assert_allclose(ge.pchip_interp(t, f(t), t), f(t), atol=1e-12)    # interpolates the data
    with pytest.raises(ValueError):
        ge.cubic_interp(t, f(t), np.array([6.5]))                                 # no extrapolation as "interpolation"


# 14, 15, 17 ---------------------------------------------------------------------------------- frozen model
def test_14_frozen_model_loading(lm):
    assert lm.n_parameters == 24602 and not lm.model.training
    assert all(not p.requires_grad for p in lm.model.parameters())
    assert lm.sha256 == json.loads((SAVED / 'three_shapes_unified/metadata.json').read_text())['sha256']


def test_15_consistency_with_existing_inference(lm, tmp_path):
    t, lat, lon = synthetic_gps()
    task = gl.build_task(t, lat, lon, 30, seed=3)
    o, q = task.obs_idx, task.interp_idx
    pred_m, info = ge.latent_ode_predict(lm, task.t_model[o], task.xy_m[o], task.t_model[q], gl.DEFAULT_STRATEGY)
    tr = info['transform']
    direct = inf.predict(lm, task.t_model[o], tr.to_model(task.xy_m[o]), task.t_model[q])['pred']
    np.testing.assert_allclose(tr.to_model(pred_m), direct, atol=1e-9)
    # end to end through the existing command-line tool
    csv = tmp_path / 'obs.csv'; u = tr.to_model(task.xy_m[o])
    csv.write_text('t,x,y\n' + ''.join(f'{a:.12g},{b:.12g},{c:.12g}\n' for a, (b, c) in zip(task.t_model[o], u)))
    r = subprocess.run([sys.executable, str(ROOT / 'scripts/predict_trajectory.py'), '--input', str(csv), '--output', str(tmp_path / 'out'),
                        '--quiet', '--no-plots'], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    cli = np.loadtxt(tmp_path / 'out/predicted_trajectory.csv', delimiter=',', skiprows=1, usecols=(0, 1, 2))
    ref = inf.predict(lm, task.t_model[o], u, cli[:, 0])['pred']
    np.testing.assert_allclose(cli[:, 1:], ref, atol=2e-6)


def test_17_weights_unchanged_after_inference(lm):
    before = inf.state_dict_digest(lm.model); f_before = inf.sha256_file(SAVED / 'three_shapes_unified/best_model.pt')
    t, lat, lon = synthetic_gps()
    for s in range(3):
        task = gl.build_task(t, lat, lon, 30, seed=s)
        ge.evaluate_interpolation(task, lm, gl.DEFAULT_STRATEGY); ge.evaluate_forecast(task, lm, gl.DEFAULT_STRATEGY, 'A')
    assert inf.state_dict_digest(lm.model) == before
    assert inf.sha256_file(SAVED / 'three_shapes_unified/best_model.pt') == f_before


# 16 ------------------------------------------------------------------------------------------ reproducibility
def test_16_reproducible_sampling():
    t, lat, lon = synthetic_gps()
    a = gl.build_task(t, lat, lon, 30, seed=9).obs_idx; b = gl.build_task(t, lat, lon, 30, seed=9).obs_idx
    c = gl.build_task(t, lat, lon, 30, seed=10).obs_idx
    np.testing.assert_array_equal(a, b); assert not np.array_equal(a, c)


# 18 ------------------------------------------------------------------------------------------ metrics
def test_18_metrics_in_metres():
    true = np.zeros((4, 2)); pred = np.array([[3., 4.], [3., 4.], [0., 0.], [0., 0.]])
    m = ge.error_metrics(pred, true)
    assert m['ade'] == pytest.approx(2.5) and m['max_error'] == pytest.approx(5.0) and m['median_error'] == pytest.approx(2.5)
    assert m['rmse_xy'] == pytest.approx(np.sqrt((9 + 16) * 2 / 8)) and m['mae_xy'] == pytest.approx(14 / 8)
    assert m['fde'] == 0.0 and m['p90_error'] == pytest.approx(5.0)
    # pipeline definition: sqrt(sum of squared coordinate errors / (2N))
    d = pred - true; assert m['rmse_xy'] == pytest.approx(np.sqrt((d ** 2).sum() / (2 * len(d))))


# 19 ------------------------------------------------------------------------------------------ short trajectories
def test_19_short_trajectories():
    cfg = gl.SegmentConfig()
    t, lat, lon = synthetic_gps(150)
    assert gl.window_checks(t, lat, lon, cfg) == (False, 'too_short')
    assert gl.candidate_windows(t, lat, lon, cfg) == []
    t, lat, lon = synthetic_gps(25)
    with pytest.raises(ValueError):
        gl.build_task(t, lat, lon, 30, seed=0)
    assert gl.continuous_runs(t[:1], lat[:1], lon[:1], cfg) == [(0, 1)] and gl.continuous_runs(t[:0], lat[:0], lon[:0], cfg) == []
    p = gl.parse_plt(plt_bytes([]))
    assert len(p.lat) == 0 and p.n_lines == 0


# 20 ------------------------------------------------------------------------------------------ result files
RES = ROOT / 'results/geolife'


@pytest.mark.skipif(not (RES / 'pilot/trajectory_001/observations.csv').exists(),
                    reason='GeoLife-derived pilot files are not distributed (MSR-LA licence); run scripts/21 to create them')
def test_20_result_files_complete():
    for i in range(1, 11):
        d = RES / f'pilot/trajectory_{i:03d}'
        for f in ('observations.csv', 'withheld_ground_truth.csv', 'latent_ode_predictions.csv', 'linear_predictions.csv',
                  'cubic_predictions.csv', 'interpolation_comparison.png', 'x_vs_time.png', 'y_vs_time.png', 'metrics.json'):
            assert (d / f).exists() and (d / f).stat().st_size > 0, d / f
        m = json.loads((d / 'metrics.json').read_text())
        assert m['n_observations'] == 30 and set(ge.INTERP_METHODS) <= set(m['interpolation_metrics_m'])
    s = json.loads((RES / 'main_evaluation/summary.json').read_text())
    assert s['n_segments'] == 1000 and set(s['interpolation']) == set(ge.INTERP_METHODS)
    for f in ('fig1_geolife_examples', 'fig2_gps_with_observations', 'fig3_method_comparison', 'fig4_best_median_worst',
              'fig5_error_distribution', 'fig6_error_vs_observation_count', 'fig7_characteristics', 'fig8_error_vs_gap'):
        assert (RES / 'figures' / f'{f}.png').exists(), f


@pytest.mark.skipif(not (RES / 'splits_summary.json').exists(), reason='splits not prepared')
def test_user_splits_disjoint():
    sp = json.loads((RES / 'splits_summary.json').read_text())
    assert not set(sp['dev']['users']) & set(sp['test']['users'])
    meta = ROOT / 'data/geolife/processed/geolife_regular/metadata.json'
    if meta.exists():
        u = json.loads(meta.read_text())['users']
        assert not (set(u['train']) | set(u['val'])) & set(sp['test']['users']) and not set(u['train']) & set(u['val'])
