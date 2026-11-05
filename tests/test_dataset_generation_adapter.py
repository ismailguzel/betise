"""Check explicit short-period generation at the problematic length boundaries."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from betise.config import load_params_config
from betise.scenario_builder import build_type_scenario, iter_materialized_scenarios

DIRECTORY = Path(__file__).resolve().parents[1] / 'datasets' / 'zenodo-v2'
SPEC = importlib.util.spec_from_file_location('dataset_adapter', DIRECTORY / 'generate.py')
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


@pytest.mark.parametrize('length', [50, 71, 100])
@pytest.mark.parametrize('scenario_id', ['single_seasonality', 'multiple_seasonality', 'contextual_anomaly'])
def test_explicit_short_periods_at_boundaries(length, scenario_id, tmp_path, monkeypatch):
    scope = json.loads((DIRECTORY / 'scope.json').read_text())
    spec = next(s for s in scope['scenarios'] if s['id'] == scenario_id)
    numerical = scope['parameter_overrides_by_length']['short']['seasonality']
    overrides = {'seasonality': {base: numerical[base] for base in spec['base_components']}}
    original = load_params_config()
    scenario = build_type_scenario(spec['base_components'], spec['features'], length_category='short')
    recipes = list(iter_materialized_scenarios([scenario], categorical_mode='sampled',
                   seed=42, variants_per_type=1, length_category='short'))
    cfg = {'mode': 'requested', 'output_dir': str(tmp_path / 'data'), 'num_series': 1,
           'base_components': spec['base_components'], 'features': spec['features'],
           'length_category': 'short', 'seed': 42, 'categorical_mode': 'sampled',
           'variants_per_type': 32, 'shard_size': 10}
    job = {'config': cfg, 'params_overrides': overrides,
           'recipe_allocations': [{'recipe_id': recipes[0]['materialized_scenario_id'], 'num_series': 1}]}
    monkeypatch.setattr(adapter, 'LENGTH_PRESETS', {'short': (length, length)})
    summary = adapter.generate_job(job, None)
    assert summary['total_series'] == 1
    frame = pd.read_parquet(next((tmp_path / 'data').glob('*/*.parquet')))
    assert len(frame) == length
    assert np.isfinite(frame['data']).all()
    periods = json.loads(frame['seasonality_periods'].iloc[0])
    assert set(periods) == ({3, 7} if scenario_id == 'multiple_seasonality' else {7})
    assert all(length >= 6 * period for period in periods)
    assert json.loads(frame['generation_params_overrides'].iloc[0]) == overrides
    assert load_params_config() == original  # Published defaults remain untouched.
