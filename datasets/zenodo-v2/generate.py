"""Dataset adapter for explicit numerical overrides with pinned BeTiSe 0.6.1."""

import json
from pathlib import Path
import random

import numpy as np
from betise.config import load_params_config
from betise.full_dataset_generation import generate_full_series
from betise.run_scenario_generation import LENGTH_PRESETS, run_generation
from betise.scenario_builder import build_type_scenario, iter_materialized_scenarios, to_generation_composition
from betise.scenario_output import _attach_scenario_context, _write_shard


def generate_job(job, config_path):
    """Use the native runner unless this dataset job declares numerical overrides."""
    overrides = job.get('params_overrides', {})
    if not overrides:
        return run_generation(config_path)
    cfg = job['config']
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    params = load_params_config(params=overrides)
    scenario = build_type_scenario(cfg['base_components'], cfg['features'],
                                   length_category=cfg['length_category'])
    recipes = sorted(iter_materialized_scenarios(
        [scenario], categorical_mode=cfg['categorical_mode'], seed=cfg['seed'],
        variants_per_type=min(cfg['variants_per_type'], cfg['num_series']),
        length_category=cfg['length_category']), key=lambda r: r['materialized_scenario_id'])
    expected = {r['recipe_id']: r['num_series'] for r in job['recipe_allocations']}
    if set(expected) != {r['materialized_scenario_id'] for r in recipes}:
        raise ValueError('Recipe selection differs from manifest')
    output = Path(cfg['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    frames, series_id, shard = [], 1, 0
    low, high = LENGTH_PRESETS[cfg['length_category']]
    for position, recipe in enumerate(recipes, start=1):
        composition = to_generation_composition(recipe)
        for realization in range(expected[recipe['materialized_scenario_id']]):
            length = int(np.random.randint(low, high + 1))
            frame = generate_full_series(composition=composition, full_cfg={'feature_defaults': {}},
                                         params_cfg=params, series_id=series_id, length=length)
            context = {'type_scenario_id': recipe['scenario_id'],
                       'materialized_scenario_id': recipe['materialized_scenario_id'],
                       'combination_size': recipe['combination_size'], 'recipe_index': position,
                       'realization_index': realization, 'seed': cfg['seed'],
                       'categorical_variant_ids': recipe['categorical_variant_ids'],
                       'feature_overrides': recipe['feature_overrides']}
            frame = _attach_scenario_context(frame, context)
            frame['generation_params_overrides'] = json.dumps(overrides, sort_keys=True)
            frames.append(frame)
            series_id += 1
            if len(frames) == cfg['shard_size']:
                _write_shard(frames, output, scenario['combination_size'], shard)
                frames, shard = [], shard + 1
    if frames:
        _write_shard(frames, output, scenario['combination_size'], shard)
        shard += 1
    if series_id - 1 != cfg['num_series']:
        raise ValueError('Generated count mismatch')
    summary = {**cfg, 'total_series': series_id - 1, 'shard_count': shard,
               'total_recipes': len(recipes), 'params_overrides': overrides,
               'adapter': 'dataset-specific numerical overrides; BeTiSe 0.6.1 generation engine'}
    (output / 'request_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary
