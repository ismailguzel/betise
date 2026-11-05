"""Build the fixed-budget 120,000-series plan without generating time series."""

import hashlib
import json

import betise
from betise.scenario_builder import build_type_scenario, iter_materialized_scenarios
from betise.run_scenario_generation import validate_requested_generation_config


def seed_for(base_seed, job_id):
    return int.from_bytes(hashlib.sha256(f"{base_seed}:{job_id}".encode()).digest()[:4], "big")


def build_plan(scope):
    if betise.__version__ != scope['betise_version']:
        raise ValueError('Use betise==' + scope['betise_version'])
    policy = scope['production_plan']
    per_job = policy['series_per_configuration']
    if per_job != 1000:
        raise ValueError('Production jobs must contain 1,000 series')
    rows, excluded, seen = [], [], set()
    for category, bounds in scope['length_categories'].items():
        for spec in scope['scenarios']:
            batches = spec['production_batches']
            if not isinstance(batches, int) or batches < 0:
                raise ValueError('Invalid production_batches')
            if not batches:
                excluded.append({'scenario': spec['id'], 'length_category': category,
                                 'reason': spec.get('selection_note', 'Not selected')})
                continue
            scenario = build_type_scenario(spec['base_components'], spec['features'],
                                           length_category=category)
            key = (category, scenario['scenario_id'])
            if key in seen:
                raise ValueError('Duplicate canonical scenario in scope')
            seen.add(key)
            for batch in range(batches):
                job_id = f"{category}__{spec['id']}__batch{batch + 1:02d}"
                seed = seed_for(scope['base_seed'], job_id)
                recipes = sorted(iter_materialized_scenarios(
                    [scenario], categorical_mode=policy['categorical_mode'],
                    variants_per_type=policy['variants_per_type'], seed=seed,
                    length_category=category), key=lambda r: r['materialized_scenario_id'])
                if not recipes:
                    raise ValueError('No recipes for ' + job_id)
                q, remainder = divmod(per_job, len(recipes))
                allocation = [{'recipe_id': recipe['materialized_scenario_id'],
                               'feature_overrides': recipe['feature_overrides'],
                               'num_series': q + (i < remainder)}
                              for i, recipe in enumerate(recipes)]
                config = {'mode': 'requested', 'output_dir': 'data/' + job_id,
                          'base_components': spec['base_components'], 'features': spec['features'],
                          'num_series': per_job, 'categorical_mode': policy['categorical_mode'],
                          'variants_per_type': policy['variants_per_type'],
                          'length_category': category, 'seed': seed, 'shard_size': scope['shard_size']}
                validate_requested_generation_config(config)
                blockers = []
                seasonal = scope.get('parameter_overrides_by_length', {}).get(category, {}).get('seasonality', {})
                applicable = {base: seasonal[base] for base in spec['base_components'] if base in seasonal}
                params_overrides = {'seasonality': applicable} if applicable else {}
                rows.append({'id': job_id, 'role': 'coverage' if batch == 0 else 'balance_batch',
                             'scenario': spec['id'], 'group': spec['group'],
                             'length_category': category, 'length_range': bounds,
                             'base_components': spec['base_components'], 'features': spec['features'],
                             'expected_is_stationary': spec['expected_is_stationary'],
                             'stationarity_status': 'generation-label convention',
                             'num_series': per_job, 'seed': seed, 'config': config,
                             'params_overrides': params_overrides,
                             'recipe_allocations': allocation, 'source': spec.get('source', ''),
                             'blockers': blockers})
    alignment = scope['paper_alignment']
    selected_bases = {b for row in rows for b in row['base_components']}
    selected_overlays = {f for row in rows for f in row['features']}
    missing_bases = set(alignment['required_base_components']) - selected_bases
    missing_overlays = set(alignment['required_overlays']) - selected_overlays
    if missing_bases or missing_overlays:
        raise ValueError(f"Missing paper taxonomy: bases={missing_bases}, overlays={missing_overlays}")
    example = alignment['required_combined_example']
    for category in scope['length_categories']:
        if not any(row['length_category'] == category and
                   row['base_components'] == example['base_components'] and
                   set(row['features']) == set(example['features']) for row in rows):
            raise ValueError('Missing section 4 combined example in ' + category)
    if len({r['id'] for r in rows}) != len(rows) or len({r['seed'] for r in rows}) != len(rows):
        raise ValueError('Duplicate job ID or seed')
    summaries = []
    for category in scope['length_categories']:
        members = [r for r in rows if r['length_category'] == category]
        stationary = sum(r['num_series'] for r in members if r['expected_is_stationary'] == 1)
        nonstationary = sum(r['num_series'] for r in members if r['expected_is_stationary'] == 0)
        target = scope['target_release']
        if (stationary != target['stationary_per_length_category'] or
                nonstationary != target['nonstationary_per_length_category']):
            raise ValueError('Allocation does not meet the binary quota for ' + category)
        summaries.append({'length_category': category, 'jobs': len(members),
                          'coverage_jobs': sum(r['role'] == 'coverage' for r in members),
                          'balance_jobs': sum(r['role'] == 'balance_batch' for r in members),
                          'planned_stationary': stationary, 'planned_nonstationary': nonstationary,
                          'total_series': stationary + nonstationary})
    total = sum(r['num_series'] for r in rows)
    if total != scope['target_release']['total_series']:
        raise ValueError('Total does not meet the release budget')
    return {'artifact_type': 'fixed_budget_requested_generation_plan',
            'status': scope['status'], 'betise_version': scope['betise_version'],
            'scope_sha256': hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest(),
            'paper_doi': scope['paper_doi'], 'summaries': summaries,
            'total_series': total, 'rows': rows, 'excluded_scenarios': excluded}
