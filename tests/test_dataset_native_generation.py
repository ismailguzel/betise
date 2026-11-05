"""Smoke-test benchmark configs with the unmodified native BeTiSe runner."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from betise.run_scenario_generation import run_generation

DIRECTORY = Path(__file__).resolve().parents[1] / 'datasets' / 'zenodo-v2' / 'configs'


@pytest.mark.parametrize('scenario', [
    'ar', 'single_seasonality', 'multiple_seasonality',
    'contextual_anomaly', 'stochastic_mean_point',
])
def test_native_short_generation(scenario, tmp_path):
    source = DIRECTORY / 'short' / f'short__{scenario}__batch01.json'
    config = json.loads(source.read_text())
    config['num_series'] = 2
    config['output_dir'] = str(tmp_path / 'output')
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    run_generation(path)
    shards = list((tmp_path / 'output').rglob('*.parquet'))
    assert shards
    frame = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    assert frame['series_id'].nunique() == 2
    assert np.isfinite(frame['data']).all()
    lengths = frame.groupby('series_id').size()
    if 'length' in config:
        assert (lengths == 100).all()
    else:
        assert lengths.between(50, 100).all()
