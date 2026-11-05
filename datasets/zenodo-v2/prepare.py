"""Materialize native BeTiSe configs for the fixed-budget release plan."""

import argparse
import json
from pathlib import Path

from build_plan import build_plan
from betise.run_scenario_generation import validate_requested_generation_config
from betise.scenario_builder import build_type_scenario, iter_materialized_scenarios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['pilot', 'validation', 'candidate'], default='pilot')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    scope_path = Path(__file__).with_name('scope.json')
    scope = json.loads(scope_path.read_text())
    plan = build_plan(scope)
    root = args.output.resolve()
    if root.exists():
        raise SystemExit('Output exists; resume with run.py or choose a new directory.')
    jobs = []
    for row in plan['rows']:
        config = dict(row['config'])
        config['output_dir'] = str(root / 'data' / row['id'])
        config['num_series'] = scope['profiles'][args.profile]['series_per_job']
        if args.profile == 'candidate' and config['num_series'] != row['num_series']:
            raise SystemExit('Candidate profile conflicts with the release plan.')
        validate_requested_generation_config(config)
        scenario = build_type_scenario(config['base_components'], config['features'],
                                       length_category=config['length_category'])
        recipes = sorted(iter_materialized_scenarios(
            [scenario], categorical_mode=config['categorical_mode'], seed=config['seed'],
            variants_per_type=min(config['variants_per_type'], config['num_series']),
            length_category=config['length_category']), key=lambda r: r['materialized_scenario_id'])
        q, remainder = divmod(config['num_series'], len(recipes))
        allocations = [{'recipe_id': r['materialized_scenario_id'], 'num_series': q + (i < remainder)}
                       for i, r in enumerate(recipes)]
        jobs.append({'id': row['id'], 'role': row['role'], 'group': row['group'], 'config': config,
                     'config_path': str(root / 'configs' / (row['id'] + '.json')),
                     'expected_is_stationary': row['expected_is_stationary'],
                     'params_overrides': row['params_overrides'],
                     'recipe_allocations': allocations})
    root.mkdir(parents=True)
    (root / 'configs').mkdir()
    (root / 'scope.json').write_text(scope_path.read_text())
    (root / 'production-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    for job in jobs:
        Path(job['config_path']).write_text(json.dumps(job['config'], indent=2) + '\n')
    manifest = {'profile': args.profile, 'betise_version': scope['betise_version'],
                'scope_status': scope['status'], 'jobs': jobs,
                'excluded_scenarios': plan['excluded_scenarios'],
                'production_total_series': plan['total_series'],
                'total_series': sum(j['config']['num_series'] for j in jobs)}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"Prepared {len(jobs)} {args.profile} jobs, {manifest['total_series']:,} series")
    print(root / 'manifest.json')


if __name__ == '__main__':
    main()
