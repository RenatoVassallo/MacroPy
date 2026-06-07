# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **`ThresholdVARSV`** — Threshold VAR with stochastic volatility, replicating
  Alessandri & Mumtaz (2019), *"Financial regimes and uncertainty shocks"*
  (Journal of Monetary Economics). Two endogenously-dated regimes (calm/crisis)
  selected by a threshold on a financial-distress indicator with an estimated
  delay, a common scalar stochastic-volatility factor that scales the whole
  covariance matrix, and volatility-in-mean effects. Implemented in `tvarsv.py`.
- **Gibbs sampler** with optional **multi-chain parallelism** (`joblib`) and a
  vectorized single-move volatility step (Jacquier-Polson-Rossi), random-walk
  Metropolis threshold and multinomial delay (Chen-Lee).
- **Generalized impulse responses** (Koop-Pesaran-Potter) reported by regime,
  vectorized over histories x Monte-Carlo paths with common random numbers and
  parallelized over posterior draws (`ThresholdVARSV.compute_irfs`).
- **Plotting helpers** in `plots_tvarsv.py`: `plot_regimes`, `plot_volatility`,
  `plot_irfs` (reproducing the paper's Figs. 1-3).
- New tutorial: `tutorials/tutorial_tvarsv.ipynb`.
- New dataset: `datasets/AlessandriMumtaz_Data.csv` (monthly US data, 1973-2014).
- `joblib` and `matplotlib` added as explicit dependencies.

## [0.1.6] - 2026-05-15

### Added
- **`BayesianPanelVAR`** — hierarchical Bayesian Panel VAR with unit-specific
  dynamics, pooled lag coefficients, common exogenous regressors, and support
  for unbalanced panels.
- **Panel priors** in `priors.py`: `HierarchicalPanelPrior`,
  `DiffusePanelExogenousPrior`, plus Banbura-style dummy-observation helpers.
- **Panel data utilities** in `data_handling.py`: `prepare_panel_data`,
  `prepare_panel_unit_data` for long-format panel inputs.
- **`PanelLocalProjections`** class and a richer set of LP utilities in `lp.py`:
  `smooth_lp_results`, `split_shock_signs`, `identify_boom_periods`,
  `cumulative_irf_ratio`.
- **BIS data API** in `get_macrodata.py`: `get_bis_data`, `get_bis_data_single`
  for cross-country macro series.
- **Panel coefficient plots** and other plotting helpers in `plots.py`.
- New tutorial: `tutorials/tutorial_bpvar.ipynb`.
- New demo dataset: `datasets/PVAR_Data.csv`.
- `MacroPy.__version__` now exposed via `importlib.metadata` (single source of
  truth in `pyproject.toml`).
- `LICENSE` file (MIT) and `CHANGELOG.md`.

### Changed
- Loosened dependency pins from `==` to `>=` floors so MacroPy can co-install
  with other scientific Python stacks.
- Modernized `pyproject.toml`: PEP 639 license metadata, project URLs,
  classifiers, explicit `setuptools.packages.find` for src-layout.
- `summary.py` reads its version string from package metadata instead of a
  hard-coded constant.
- `__init__.py` makes `get_macrodata` imports optional so that core estimators
  work without API client dependencies installed.

### Fixed
- Untracked stray `.DS_Store` files from `src/`.

## [0.1.5] - 2025-07

- Added return values to public methods, improved docstrings.
- Minnesota default hyperparameters.

## [0.1.4] - 2025

- FEVD for Bayesian VAR.
- Updated CVAR class and tutorial.

## [0.1.3] - 2025

- API tutorial and Smooth Local Projections improvements.

## [0.1.2] - 2025

- First public pre-release distributed as a wheel.
