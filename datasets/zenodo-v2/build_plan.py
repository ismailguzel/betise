"""Build the fixed-budget 120,000-series plan without generating time series."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

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
                                 'reason': spec['selection_note']})
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
                             'stationarity_status': spec['stationarity_review'],
                             'num_series': per_job, 'seed': seed, 'config': config,
                             'params_overrides': params_overrides,
                             'recipe_allocations': allocation, 'source': spec['source'],
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


def write_plan(plan, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / 'production-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    columns = ['id', 'role', 'scenario', 'group', 'length_category', 'length_range',
               'base_components', 'features', 'expected_is_stationary', 'stationarity_status',
               'num_series', 'seed', 'params_overrides', 'recipe_allocations', 'source', 'blockers']
    with (output / 'production-plan.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        for row in plan['rows']:
            writer.writerow({k: json.dumps(row[k], ensure_ascii=False)
                             if isinstance(row[k], (list, dict)) else row[k] for k in columns})
    lines = ['# Production plan — 120,000 series', '',
             'Generated by `build_plan.py` from `scope.json`. No time series are generated.', '',
             '**Status: `' + plan['status'] + '`**', '',
             'Each job produces exactly 1,000 series, shared across its selected categorical recipes.',
             'Numerical realizations vary; recipes do not multiply the budget.', '',
             '| Length | Jobs | Stationary | Non-stationary | Total |',
             '|---|---:|---:|---:|---:|']
    for s in plan['summaries']:
        lines.append('| {length_category} | {jobs} | {planned_stationary:,} | '
                     '{planned_nonstationary:,} | {total_series:,} |'.format(**s))
    lines += ['', '**Total: 120 jobs × 1,000 = 120,000 series.**', '',
              'The total follows the agreed release target. Equal counts per length and the exact',
              'scenario/batch allocation are v2 choices, not a recovered historical manifest.',
              'Stationarity counts are provisional expectations that must be checked in actual output.', '',
              '## Allocation per length', '',
              '| Scenario | Bases | Overlays | Jobs | Series | Expected stationary flag |',
              '|---|---|---|---:|---:|---:|']
    members = [r for r in plan['rows'] if r['length_category'] == 'medium']
    counts = Counter(r['scenario'] for r in members)
    used = set()
    for r in members:
        if r['scenario'] in used:
            continue
        used.add(r['scenario'])
        n = counts[r['scenario']]
        lines.append(f"| {r['scenario']} | {', '.join(r['base_components'])} | "
                     f"{', '.join(r['features']) or 'none'} | {n} | {n * 1000:,} | "
                     f"{r['expected_is_stationary']} |")
    lines += ['', 'The same broad allocation applies to all three lengths; short-series recipe rules differ.',
              'There are 31 distinct broad scenarios per length, plus nine fresh baseline balance batches.',
              'Repeated baseline batches are not new unique base/overlay combinations.', '',
              '## Coverage decisions', '',
              '- Retain all named model/property families, contextual anomalies and the section 4 stochastic-trend/break/point-anomaly scenario.',
              '- Omit separate seasonal-point, seasonal-collective and additional seasonal-contextual combinations',
              '  to keep 20 non-stationary jobs per length; these are explicit budget choices.',
              '- Sample up to 32 categorical recipes per job; the CSV/JSON records exact allocations.',
              '- No duplicated observations or forced label changes are planned for balancing.', '',
              '## Remaining work', '',
              '- Validate the explicit short-period overrides (single: 7; multiple: 3 and 7) on the server.',
              '- Audit the stationarity definition, parameter constraints and expected label mapping.',
              '- Run the new pilot on the intended server environment and validate counts, metadata, plots and event labels.',
              '- Finalize the scope/environment before enabling full production.', '']
    (output / 'PRODUCTION_PLAN.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', type=Path, default=Path(__file__).with_name('scope.json'))
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    plan = build_plan(json.loads(args.scope.read_text()))
    write_plan(plan, args.output)
    print(f"Wrote {len(plan['rows'])} jobs; planned series: {plan['total_series']:,}")


if __name__ == '__main__':
    main()
