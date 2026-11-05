"""Check the native configs distributed with the benchmark."""

import json
from pathlib import Path

import pytest
from betise.run_scenario_generation import validate_requested_generation_config
from betise.scenario_builder import build_type_scenario

DIRECTORY = Path(__file__).resolve().parents[1] / 'datasets' / 'zenodo-v2'
CONFIGS = sorted((DIRECTORY / 'configs').glob('*/*.json'))
STATIONARY_BASES = {'ar', 'ma', 'arma', 'white_noise', 'sarma',
                    'arch', 'garch', 'egarch', 'aparch'}


def test_native_only_folder_and_total():
    assert {p.name for p in DIRECTORY.iterdir()} == {'README.md', 'configs'}
    assert len(CONFIGS) == 120
    configs = [json.loads(p.read_text()) for p in CONFIGS]
    assert sum(c['num_series'] for c in configs) == 120000
    assert all(c['num_series'] == 1000 for c in configs)
    assert len({c['seed'] for c in configs}) == 120
    assert len({c['output_dir'] for c in configs}) == 120


@pytest.mark.parametrize('category', ['short', 'medium', 'long'])
def test_balance_and_paper_families(category):
    configs = [json.loads(p.read_text()) for p in CONFIGS if p.parent.name == category]
    assert len(configs) == 40
    stationary = sum(c['num_series'] for c in configs
                     if not c['features'] and set(c['base_components']) <= STATIONARY_BASES)
    assert stationary == 20000
    assert sum(c['num_series'] for c in configs) - stationary == 20000
    assert {'white_noise', 'ar', 'ma', 'arma', 'random_walk', 'random_walk_drift',
            'ari', 'ima', 'arima', 'single_seasonality', 'multiple_seasonality',
            'sarma', 'sarima', 'arch', 'garch', 'egarch', 'aparch'} <= {
                b for c in configs for b in c['base_components']}
    assert {'linear_trend', 'quadratic_trend', 'cubic_trend', 'exponential_trend',
            'damped_trend', 'mean_shift', 'variance_shift', 'trend_shift',
            'point_anomaly', 'collective_anomaly', 'contextual_anomaly'} == {
                f for c in configs for f in c['features']}
    assert any(c['base_components'] == ['random_walk'] and
               set(c['features']) == {'mean_shift', 'point_anomaly'} for c in configs)


@pytest.mark.parametrize('path', CONFIGS, ids=lambda p: p.stem)
def test_native_validation(path):
    config = json.loads(path.read_text())
    validate_requested_generation_config(config)
    # Exact lengths of 50--100 also use the library's short-series policy.
    build_type_scenario(config['base_components'], config['features'],
                        length_category=path.parent.name)
    if 'length' in config:
        assert path.parent.name == 'short'
        assert config['length'] == 100
        assert any(name in path.stem for name in
                   ('single_seasonality', 'multiple_seasonality', 'contextual_anomaly'))
    else:
        assert config['length_category'] == path.parent.name
