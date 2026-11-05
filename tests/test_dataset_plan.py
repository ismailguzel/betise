"""Verify the fixed release budget, categorical allocation and native configs."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / 'datasets' / 'zenodo-v2'
SPEC = importlib.util.spec_from_file_location('dataset_plan', DIRECTORY / 'build_plan.py')
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


@pytest.fixture
def scope(monkeypatch):
    value = json.loads((DIRECTORY / 'scope.json').read_text())
    monkeypatch.setattr(planner.betise, '__version__', value['betise_version'])
    return value


def test_fixed_120000_budget(scope):
    plan = planner.build_plan(scope)
    assert len(plan['rows']) == 120
    assert plan['total_series'] == 120000
    assert all(r['num_series'] == r['config']['num_series'] == 1000 for r in plan['rows'])
    for s in plan['summaries']:
        assert s['jobs'] == 40
        assert s['planned_stationary'] == s['planned_nonstationary'] == 20000
        assert s['total_series'] == 40000


def test_recipes_share_the_budget(scope):
    plan = planner.build_plan(scope)
    for row in plan['rows']:
        allocations = row['recipe_allocations']
        assert 1 <= len(allocations) <= 32
        assert sum(r['num_series'] for r in allocations) == 1000
        assert max(r['num_series'] for r in allocations) - min(r['num_series'] for r in allocations) <= 1
    point = next(r for r in plan['rows'] if r['scenario'] == 'point_anomaly' and r['length_category'] == 'medium')
    assert len(point['recipe_allocations']) == 7
    assert sum(r['num_series'] for r in point['recipe_allocations']) == 1000


def test_balance_batches_are_explicit(scope):
    plan = planner.build_plan(scope)
    extra = [r for r in plan['rows'] if r['role'] == 'balance_batch']
    assert len(extra) == 27
    assert all(r['expected_is_stationary'] == 1 and not r['features'] for r in extra)
    assert {r['scenario'] for r in extra} == {'ar', 'ma', 'arma', 'white_noise'}
    assert len([r for r in plan['rows'] if r['role'] == 'coverage']) == 93


def test_reproducible_seeds_and_configs(scope):
    plan = planner.build_plan(scope)
    assert plan == planner.build_plan(scope)
    assert len({r['seed'] for r in plan['rows']}) == 120
    assert len({r['id'] for r in plan['rows']}) == 120
    assert all(r['seed'] == r['config']['seed'] for r in plan['rows'])


def test_short_period_overrides(scope):
    plan = planner.build_plan(scope)
    rows = [r for r in plan['rows'] if r['params_overrides']]
    assert len(rows) == 3
    assert all(r['length_category'] == 'short' for r in rows)
    assert plan['excluded_scenarios'] == []
    for row in rows:
        periods = row['params_overrides']['seasonality']
        if 'single_seasonality' in periods:
            assert periods['single_seasonality']['period'] == [7]
        else:
            assert periods['multiple_seasonality']['periods'] == [3, 7]


def test_duplicate_canonical_scenario_is_rejected(scope):
    duplicate = deepcopy(scope['scenarios'][0])
    duplicate['id'] = 'another_name_for_ar'
    scope['scenarios'].append(duplicate)
    with pytest.raises(ValueError, match='Duplicate canonical scenario'):
        planner.build_plan(scope)


def test_wrong_quota_is_rejected(scope):
    scope['scenarios'][0]['production_batches'] += 1
    with pytest.raises(ValueError, match='binary quota'):
        planner.build_plan(scope)


def test_all_paper_families_and_combined_example_in_every_length(scope):
    plan = planner.build_plan(scope)
    alignment = scope['paper_alignment']
    for category in scope['length_categories']:
        rows = [r for r in plan['rows'] if r['length_category'] == category]
        assert set(alignment['required_base_components']) <= {b for r in rows for b in r['base_components']}
        assert set(alignment['required_overlays']) <= {f for r in rows for f in r['features']}
        assert any(r['base_components'] == ['random_walk'] and
                   set(r['features']) == {'mean_shift', 'point_anomaly'} for r in rows)


def test_missing_paper_family_is_rejected(scope):
    next(s for s in scope['scenarios'] if s['id'] == 'aparch')['production_batches'] = 0
    with pytest.raises(ValueError, match='Missing paper taxonomy'):
        planner.build_plan(scope)
