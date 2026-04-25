# Changelog

All notable changes to BeTiSe are documented here.

---

## [0.2.0] — 2026-04-25

### Changed — Package renamed
- Package renamed from `timeseries_dataset_generator` to `betise` for PyPI publication.
- Import path changes: `from betise import ...` / `from betise.config import load_config`
- Config files renamed from `*.config` to `*.json` for IDE compatibility.
- `setup.py` replaced by a single `pyproject.toml` (PEP 517/518 compliant).
- Version now managed via `importlib.metadata` — single source of truth in `pyproject.toml`.

### Fixed — Generator quality
- **SARMA/SARIMA burn-in**: Increased warmup to `max(8·s, 200)`, switched to `initial_state=np.zeros(k_states)` to eliminate diffuse Kalman prior transient. Added empirical std-ratio guard to reject residual transients.
- **Exponential trend**: Adjusted parameters (`b ∈ [1.8, 2.8]`, `t ∈ [0, 1]`) to prevent trend from dominating co-occurring features after z-normalisation.
- **Point / collective anomaly**: Removed post-anomaly z-normalisation that was shrinking spike amplitudes back to baseline level.
- **`is_spike` / `scale_factor` overrides**: Fixed `apply_feature` for `point_anomaly` to merge `feature_cfg` into `params_cfg` so per-call overrides take effect.
- **AR/MA/ARMA near-unit-root**: Reduced order range to `(1, 3)`, coefficient range to `(-0.75, 0.75)`, and added characteristic-root guard `|root| < 0.98`.

### Changed — Combination dataset
- `examples/data/combinations.csv`: cleaned from 840 → 573 valid combinations.
  - Removed stochastic bases × Variance Shift (HP filter makes effect invisible).
  - Removed stochastic bases × Trend Shift (indistinguishable from stochastic drift).
  - Removed volatility bases × Trend Shift (near-zero HP trend → no detectable effect).
  - Removed legacy `damped_trend` / `oscillatory_trend` entries (features no longer exist).

### Changed — Examples
- Added `examples/00_introduction.ipynb` — interactive getting-started notebook.
- Added `examples/07_feature_gallery.py` — two-section PDF: 15 base types + 12 features.
- Added `examples/08_combinations_gallery.py` — complete 573-plot PDF catalogue.
- Removed `examples/07_visual_report.py` (superseded by 08).

---

## [0.1.0] — 2025 (initial release as `timeseries_dataset_generator`)

- Initial release under the name `timeseries_dataset_generator`.
- 15 base process types: AR, MA, ARMA, White Noise, Random Walk, RW-Drift, ARI, IMA, ARIMA, SARMA, SARIMA, ARCH, GARCH, EGARCH, APARCH.
- 12 feature overlays: 4 trend types, 2 seasonality types, 3 anomaly types, 3 structural break types.
- Config-driven pipeline: `load_config()` + `generate_dataframe()` + `run()`.
- Parquet output with rich metadata columns.
- Published BeTiSe benchmark dataset on Zenodo (DOI: 10.5281/zenodo.18513505).
