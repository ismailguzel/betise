# CLAUDE.md — betise

Parametric synthetic time series generator (BeTiSe: **Be**nchmark **Ti**me **Se**ries).
This is the **library** repo — published on PyPI, developed independently of its consumers.

Human-facing docs already exist: `README.md`, `USAGE.md`, `APPENDIX.md`, `CONTRIBUTING.md`.
This file holds the operational context those don't cover.

## Public API — keep it small

```python
from betise import generate_dataframe
from betise.config import load_config

cfg = load_config(dataset={"base_series": "ar", "num_series": 3,
                           "length_range": [200, 200], "random_seed": 42,
                           "features": {"garch": {"enabled": True}}})
df, ctx = generate_dataframe(cfg)
```

`load_config()` deep-merges the `dataset` / `params` overrides over the packaged defaults in
`betise/config/*.json`. Those JSON files ship as package data — verify they are still inside
the wheel after any packaging change.

## Environment / commands

```bash
/Users/iguzel/miniforge3/envs/betise-dev/bin/python -m pytest tests/ -q   # 95 tests, ~10 s
ruff check betise/ tests/
black .
```

CI (`.github/workflows/python-package.yml`) runs pytest + ruff on Python 3.9 / 3.10 / 3.11.
Python 3.9 is still supported — do not use 3.10+ only syntax.

## Reviewing incoming PRs

External contributions arrive as **fork PRs** (e.g. `cemre-yazc:main`), often large and
notebook-heavy. Before merging:

1. `gh pr checks <n>` — CI green on all three Python versions
2. Merge locally into a throwaway branch and run the test suite
3. Count the real code diff separately from notebooks:
   `gh pr diff <n> | awk '/^diff --git/{f=$3} /^\+[^+]/{a[f]++} /^-[^-]/{d[f]++} END{for(k in a) print a[k], d[k], k}'`
4. **Run a downstream smoke test** — see next section
5. Only then `gh pr merge <n> --merge`

## Downstream consumer — check before breaking changes

`../hierarchical-ts-classification` (TDA research project) generates its datasets through
this library, driving it from JSON configs with an "all features off" dict plus per-class
overrides. A change to feature names or to which base series a feature accepts **breaks its
configs silently until generation time**.

Known live example: 0.4.0 made `contextual_anomaly` require a seasonal base series, which
broke all three of that project's configs. That was accepted deliberately, but it should
always be a conscious decision, flagged in the release notes — not a surprise.

Smoke-test a merge candidate against it without installing anything:

```bash
PYTHONPATH=/Users/iguzel/Desktop/betise /Users/iguzel/miniforge3/envs/betise-dev/bin/python - <<'EOF'
from betise import generate_dataframe
from betise.config import load_config
for base in ("ar", "arma", "white_noise", "random_walk", "arch", "sarima"):
    cfg = load_config(dataset={"base_series": base, "num_series": 2,
                               "length_range": [200, 200], "random_seed": 42})
    df, ctx = generate_dataframe(cfg)
    print(f"{base:12s} {df.shape} {ctx['label']}")
EOF
```

## Release process

Version lives in **two** places — bump both:
- `pyproject.toml` → `version = "X.Y.Z"`
- `betise/__init__.py` → the `__version__ = "X.Y.Z"` fallback

Then:

```bash
git commit -am "chore(release): bump version to X.Y.Z" && git push origin main
git tag -a vX.Y.Z -m "Release X.Y.Z: <summary>" && git push origin vX.Y.Z
rm -rf build betise.egg-info        # see below — stale build/ leaks deleted files
python -m build
python -m twine check dist/betise-X.Y.Z*
python -m twine upload dist/betise-X.Y.Z-py3-none-any.whl dist/betise-X.Y.Z.tar.gz
```

- **Always `rm -rf build betise.egg-info` first.** setuptools copies into `build/lib`
  and never prunes it, so files deleted or moved out of the package in this release
  are still sitting there and get packed into the wheel. This actually happened in
  0.5.0: the first build shipped both the old and new config plus scripts that had
  just been moved to `examples/`. Verify with
  `python -c "import zipfile;print(zipfile.ZipFile('dist/<wheel>').namelist())"`.

- PyPI token is already in `~/.pypirc` — never print, echo, or paste it.
- **Upload only the new version's files**, never `dist/*` — old builds (0.2.3, 0.3.0, …) are
  still sitting in `dist/`.
- A published version can never be replaced, so validate first: tests, `twine check`, and an
  install of the built wheel into a clean venv. **Ask before uploading.**
- Pre-1.0 versioning in use: breaking behaviour change → minor bump (0.3.0 → 0.4.0).
- Published so far: 0.2.0, 0.2.1, 0.2.2, 0.2.3, 0.3.0, 0.4.0, 0.5.0, 0.6.0.

## Conventions

- Never add `Co-Authored-By: Claude` or "Generated with Claude Code" to commits or PR bodies.
  AI agents must not appear as contributors on this repo.
- Never force-push or rewrite pushed history without explicit approval — this repo has an
  external contributor whose fork would diverge, and rewriting breaks merged-PR links.
