"""Build a small standalone reproduction archive; exclude generated datasets and credentials."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples-dir', type=Path, required=True, help='Validated smoke-pilot data directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    repo = source.parents[1]
    files = {}
    included = (
        'build_plan.py', 'prepare.py', 'run.py', 'generate.py', 'scope.json',
        'production-plan.json', 'production-plan.csv', 'PRODUCTION_PLAN.md',
        'PAPER_COVERAGE.md', 'SERVER_VALIDATION.md', 'server-validation.json',
        'server-requirements.txt', 'REPRODUCE.md',
    )
    for name in included:
        files['datasets/zenodo-v2/' + name] = (source / name).read_bytes()
    files['README.md'] = (source / 'REPRODUCE.md').read_bytes()
    files['requirements.txt'] = (source / 'server-requirements.txt').read_bytes()
    files['LICENSE'] = (repo / 'LICENSE').read_bytes()
    examples = []
    for job_id in ('short__ar__batch01', 'medium__stochastic_mean_point__batch01',
                   'long__single_seasonality__batch01'):
        directory = args.examples_dir / job_id
        if not (directory / 'validation.json').exists():
            raise ValueError('Sample job is not validated: ' + job_id)
        frame = pd.concat([pd.read_parquet(p) for p in sorted(directory.glob('*/*.parquet'))], ignore_index=True)
        frame['source_job_id'] = job_id
        frame['series_uid'] = job_id + ':' + frame['series_id'].astype(str)
        examples.append(frame)
    sample = pd.concat(examples, ignore_index=True)
    if sample['series_uid'].nunique() != 6:
        raise ValueError('Expected six example series')
    buffer = io.BytesIO()
    sample.to_parquet(buffer, index=False)
    files['examples/sample-series.parquet'] = buffer.getvalue()
    manifest = {'package_version': '1.0.0', 'betise_version': '0.6.1',
                'dataset_target_series': 120000, 'example_series': 6,
                'zenodo_version_doi': '10.5281/zenodo.18513505',
                'files': {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}}
    files['MANIFEST.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit('Refusing to overwrite an existing archive')
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            entry = zipfile.ZipInfo('betise-reproduction-v1.0.0/' + name, date_time=(2026, 11, 5, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    print(args.output)
    print('SHA256:', hashlib.sha256(args.output.read_bytes()).hexdigest())
    print('Bytes:', args.output.stat().st_size)


if __name__ == '__main__':
    main()
