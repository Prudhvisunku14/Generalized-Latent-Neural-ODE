"""Smoke tests for the five-family and 2x2-ablation packaged models, and CLI path handling (no training)."""
import json, subprocess, sys
from pathlib import Path
import numpy as np, pytest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
from glo import inference as inf

NEW = ['five_shapes_unified', 'ablation_2x2_A', 'ablation_2x2_B', 'ablation_2x2_C', 'ablation_2x2_D']


@pytest.mark.parametrize('name', NEW)
def test_new_model_loads_and_predicts_finite_200_points(name):
    lm = inf.load_packaged_model(name, root / 'saved_models')
    t, xy, _ = inf.read_observations(root / 'input_trajectories/example_figure_eight.csv', lm.preprocessing)
    times, _ = inf.make_query_times(200, 10.0, lm.preprocessing)
    out = inf.predict(lm, t, xy, times)
    assert out['pred'].shape == (200, 2) and np.isfinite(out['pred']).all()
    assert (times > 7).sum() == 60                      # 60 forecast points, 140 reconstruction points


def test_ablation_architectures_match_design():
    want = {'A': (8, 64), 'B': (16, 64), 'C': (8, 128), 'D': (16, 128)}
    for k, (lat, gru) in want.items():
        d = next((root / 'saved_models/ablation_2x2').glob(f'{k}_*'))
        a = json.loads((d / 'model_config.json').read_text())['architecture']
        assert (a['latent_dim'], a['encoder_hidden']) == (lat, gru)


def test_cli_works_from_another_working_directory(tmp_path):
    r = subprocess.run([sys.executable, str(root / 'scripts/predict_trajectory.py'), '--model', 'five_shapes_unified',
                        '--input', str(root / 'input_trajectories/example_sinusoid.csv'), '--output', str(tmp_path / 'o'),
                        '--no-plots', '--quiet'], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / 'o/predicted_trajectory.csv').exists()


def test_missing_checkpoint_message_names_the_file(tmp_path):
    with pytest.raises(inf.ModelUnavailableError, match='best_model.pt'):
        inf.load_packaged_model('five_shapes_unified', tmp_path)
