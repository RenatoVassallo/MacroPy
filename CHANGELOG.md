# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **COVID treatment in `BayesianVAR`**: `covid_window` + `covid_mode`.
  `"lenza-primiceri"` implements Lenza & Primiceri (2022) volatility scaling —
  residuals in the window have variance `s_t^2 * Sigma`, estimation applies the
  exact GLS reweighting to the OLS moments and all three priors, and the scales
  are either user-fixed (`covid_scales`) or estimated by maximizing the analytic
  conjugate Normal-Wishart marginal likelihood (`covid_n_free` free scales plus
  geometric decay). `"dummies"` auto-builds one time dummy per window
  observation through the existing exogenous block.
- **Reproducibility**: `seed` argument in `BayesianVAR`. Each stochastic method
  draws from a deterministic per-stage generator, so `sample_posterior` and
  `forecast` are idempotent and runs with the same data and seed produce
  identical draws.
- **Headless tidy outputs**: `forecast_frame()` and
  `conditional_forecast_frame()` return DataFrames indexed by
  `(variable, horizon)` with `[date, mean, median, qXX...]` columns, without
  plotting. `conditional_forecast` now also accepts friendly
  `{variable_name: path}` condition dicts (partial horizons, `{horizon: value}`
  form, scalars), converted internally to the NaN matrix.
- **Hyperparameter selection**: `select_hyperparameters()` maximizes the
  Giannone-Lenza-Primiceri (2015) analytic marginal likelihood over
  `lamda1`/`lamda3`/`lamda4` (empirical Bayes) and rebuilds the prior;
  `log_marginal_likelihood()` exposes the objective. `lamda2` is excluded by
  construction (own/cross asymmetry breaks the Kronecker structure).
- New conjugate-NW machinery in `priors.py`: `nw_conjugate_moments`,
  `nw_log_marginal_likelihood` (validated against Monte Carlo integration),
  `ar1_residual_std`.
- **Unit tests** (`tests/test_bvar_production.py`): posterior recovery on a
  synthetic VAR, identical draws under a fixed seed, dict conditions honored
  exactly, the COVID scale factor absorbing injected 2020 outliers without
  moving pre-2020 coefficients, tidy-frame layout, and a Monte-Carlo check of
  the marginal-likelihood formula. `pytest` added to the dev group.

### Changed
- Tutorials refreshed for the new API: `tutorial_bvar.ipynb` now uses `seed`,
  the `{variable: path}` conditional-forecast dict, and adds sections on tidy
  forecast frames and GLP hyperparameter selection; `tutorial_bvar_pandemic.ipynb`
  now uses the built-in `covid_mode="dummies"` and adds a section on the
  recommended Lenza-Primiceri volatility scaling, comparing all three
  treatments (estimated scales on the Peru data: about 5-8x ordinary volatility
  in 2020Q2-2021Q2).

### Fixed
- **Forecast routines verified against the Canova-Ferroni `BVAR_` toolbox**
  (github.com/naffe15/BVAR_), draw by draw, via a line-by-line port of
  `forecasts.m` / `cforecasts.m`. Three deviations found and corrected:
  - `forecast()` appended the *shocked* values into the lag history used for
    the "no-shock" path, so `mean_forecasts` was not the deterministic
    forecast from horizon 2 onward (and contaminated the conditional-forecast
    baseline). The two paths now iterate on separate lag histories and match
    the reference exactly (0 gap over all posterior draws).
  - `conditional_forecast` used only the minimum-norm Waggoner-Zha shocks, so
    conditional bands reflected parameter uncertainty alone and were roughly
    2.5-3x too narrow for unconstrained variables. The full Waggoner-Zha
    distribution (null-space shock draws, seeded) is now the default; pass
    `shock_uncertainty=False` for the previous mean-only behavior. Imposed
    conditions still hold exactly in every draw.
  - With `irf_1std=0` the conditioning solved the minimum-norm problem in
    unit-shock units, which is not the Gaussian conditional expectation.
    Conditioning now always operates in 1-s.d. structural units internally,
    independent of the IRF display setting.
- **`MinnesotaPrior` residual-scale calibration**: the per-series regression
  used columns `[0, i+1]` of the regressor matrix (lag-1 of the *first* and the
  *(i+1)-th* variable) instead of the intended own first lag plus intercept —
  the constant sits at the end of the `prepare_data` layout, not at column 0.
  Prior scales `std_ar` were substantially overstated for persistent series
  (interest rates, expectations), loosening cross-lag shrinkage and the
  Normal-Wishart `Scale0`. Now computed by `ar1_residual_std` (own lag +
  intercept). Posterior results change slightly; the correction matters most
  for systems mixing persistent and volatile variables.
- `conditional_forecast` no longer fails (or silently truncates) when the
  requested `fhor` exceeds the IRF horizon `hor`; IRFs are recomputed with a
  sufficient horizon on the fly.
- `BayesianVAR` no longer mutates the shared default `prior_params` dict, which
  could leak hyperparameters across instances in the same session.

## [0.1.7] - 2026-06-08

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
    parallelized over posterior draws (`compute_irfs`).
  - Informative threshold prior via `threshold_prior_mean`, `irf_1std` shock
    scaling, and the `BayesianVAR` `post_draws`/`burnin` convention.
  - **Plotting helpers** in `plots_tvarsv.py`: `plot_regimes`, `plot_volatility`,
    `plot_irfs` (uncertainty *and* recursive structural shocks, by regime).
  - New tutorial `tutorials/tutorial_tvarsv.ipynb` and dataset
    `datasets/AlessandriMumtaz_Data.csv` (monthly US data, 1973-2014).
- **State-space / unobserved-components toolkit** (`state_space.py`):
  `StateSpaceModel` with Kalman filter, RTS smoother and a Durbin-Koopman
  simulation smoother; pre-built `LocalLinearTrend` (Hodrick-Prescott as a
  special case), `ClarkModel` (1987) and `WatsonUC` (1986); `MLEEstimator` and
  `BayesianStateSpace` (Gibbs) estimators; plotting in `plots_kalman.py`.
- **Pandemic / COVID dummies** for `BayesianVAR` via exogenous regressors, with
  the new tutorial `tutorials/tutorial_bvar_pandemic.ipynb` and exogenous-
  coefficient posterior plots.
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
