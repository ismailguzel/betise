# Contributing to BeTiSe

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Getting started

```bash
git clone https://github.com/ismailguzel/betise.git
cd betise
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=betise --cov-report=term-missing
```

## Code style

This project uses [Black](https://black.readthedocs.io/) for formatting and [Ruff](https://docs.astral.sh/ruff/) for linting.

```bash
black .
ruff check .
```

## How to contribute

1. **Open an issue first** for significant changes so we can discuss the approach.
2. Fork the repository and create a feature branch: `git checkout -b feat/my-feature`
3. Write tests for your changes in `tests/`.
4. Make sure `pytest` and `ruff check .` both pass.
5. Open a pull request against `main`.

## Adding a new base series type

1. Add a `generate_<name>_series()` method to `betise/core/generator.py`.
2. Register the name in `_GENERATOR_MAP` inside `betise/dataset_generation.py`.
3. Add default parameters to `betise/config/params.json`.
4. Add at least one smoke test in `tests/test_generators.py`.

## Adding a new feature

1. Add a `generate_<feature>()` method to `betise/core/generator.py`.
2. Register the feature key in `apply_feature()` inside `betise/dataset_generation.py`.
3. Add at least one test in `tests/test_features.py`.

## Reporting bugs

Please include:
- Python and BeTiSe version (`python -c "import betise; print(betise.__version__)"`)
- A minimal reproducible example
- The full traceback

## License

By contributing you agree that your contributions will be licensed under the MIT License.
