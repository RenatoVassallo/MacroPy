# MacroPy — project context

Bayesian macroeconometrics toolbox (src layout, `src/MacroPy/`). Python >=3.11,<3.12.

## Layout

- `src/MacroPy/bvar.py` — `BayesianVAR`: Minnesota / Normal-Wishart / Normal-Diffuse
  priors (built in `priors.py`), block exogeneity, exogenous regressors, COVID
  treatment (Lenza-Primiceri volatility scaling or auto time dummies),
  SOC/DIO dummy priors (`soc`/`dio` in `prior_params`),
  steady-state prior (`ss_mean`/`ss_sd`, Villani 2009),
  Waggoner-Zha conditional forecasts (full shock distribution; verified
  draw-by-draw against the Canova-Ferroni `BVAR_` toolbox — reference ports in
  `tests/test_bvar_production.py`), tidy `forecast_frame()` /
  `conditional_forecast_frame()`, GLP (2015) `select_hyperparameters()`, `seed`
  for reproducibility.
- `src/MacroPy/tvarsv.py` — `ThresholdVARSV`: Alessandri & Mumtaz (2019)
  threshold VAR with stochastic volatility; GIRFs by regime; plots in
  `plots_tvarsv.py`.
- `src/MacroPy/uc.py` — `UCTrendCycle`: UC trend-cycle ladder (HP filter via
  `hp_filter`/`.hp` preset, Clark MLE via `.clark`/`fit_mle`, Bayesian UC by
  Gibbs with precision-based simulation smoother, Lenza-Primiceri COVID
  variance scales by griddy Gibbs, kappa-rule prior calibration, WAIC/DIC +
  `compare_uc`, Waggoner-Zha `conditional_forecast` with annual-growth
  targets). Port of the author's MATLAB `trend_cycle_toolbox`; plots in
  `plots_uc.py` incl. `plot_fan_chart` (BoE/BCRP-style bands).
  Course material in `classes/AppliedMacro/` (gitignored): `MA_2026/` has 5
  session folders, Spanish; `session5/uc_peru.ipynb` is the full Peru
  application (BCRP `PN02538AQ`, x13, ladder, WZ conditional fan chart,
  BCRP/Moneda peak-trough cycle dating with Harding-Pagan censoring).
  COVID convention: `covid_t0="2020Q1"` (toolbox-validated draw-level
  against `paper/output/1994/PER/cache/UC_COVID.mat`). Suppressed level noise is tied at `1e-8 * sig2_c` (never a
  hard zero: precision conditioning).
  Band convention: state bands default to the central **68%** interval
  (`decomposition_frame`, `plot_uc_decomposition`, `plot_gap_comparison`,
  `plot_covid_scales`), with `outer_cred` for a nested 68/90 look; trend and
  cycle bands coincide because `c_t = y_t - tau_t` holds draw by draw. Band
  width is prior-sensitive: `calibrate=True` (kappa rules, the default)
  tightens the trend and raises the AR(2) root cap to 0.95, which *widens*
  the gap band relative to the uncalibrated defaults (Peru full sample:
  trend sd 4.5 vs 3.0). With `calibrate=False` the sampler reproduces the
  MATLAB toolbox run within Monte Carlo error.
- `src/MacroPy/bpvar.py` — hierarchical Bayesian panel VAR. `cvar.py` — classic
  VAR. `lp.py` — smooth local projections. `state_space.py` — Kalman/UC models
  (plots in `plots_kalman.py`; for trend-cycle work prefer `uc.py`).
  `seasonal.py`/`x13.py` — X-13ARIMA-SEATS and STL seasonal adjustment.
- `priors.py` also holds the analytic conjugate-NW marginal likelihood
  (`nw_conjugate_moments`, `nw_log_marginal_likelihood`) used by GLP selection
  and the COVID scale estimation.
- `datasets/` CSVs (first column `date`); `tutorials/` executed notebooks;
  `tests/` pytest suite.

## Conventions

- Class-based API: constructor takes a `pd.DataFrame` with datetime index;
  `model_summary()` renders Markdown via `summary.py`; `sample_posterior()` uses
  tqdm ("Sampling Posterior", switch with `progress=`); plotting stays in the
  `plots*.py` modules, matplotlib only, and every compute method takes a
  `plot_*` flag that defaults to False. Plotting functions are wrapped with
  `style.styled` (a `plt.rc_context`), return `(fig, axes)`, never call
  `display` or `set_bse_style()`; figures made by methods go to
  `model.figures`. Colours come from the public `style.PALETTE`.
- Every module defines `__all__`; `__init__.py` imports names explicitly.
- `compute_hd` exists in `ClassicVAR` and `BayesianVAR` (posterior mean over
  draws) and returns `{"shock" (T, shocks, endo), "baseline", "endo", "dates"}`;
  the identity `baseline + shock.sum(axis=1) == yy` is part of the test suite.
- `post_draws` = total draws including burn-in; `burnin` = fraction discarded.
- Regressor layout from `prepare_data`: `[lag-1 vars, ..., lag-P vars,
  constant?, trend?, exog...]` — lags first, deterministic terms last.
- Keep the public API backwards compatible; new constructor arguments go at the
  end with safe defaults.
- `prior_params` always goes through `priors.normalize_prior_params` (defaults
  filled in, `lambda1..4` canonical, `lamda1..4` deprecated, unknown keys raise).
- `ClassicVAR` IRF draws: recursive-design bootstrap or IW-then-normal Monte
  Carlo, per-stage `seed`, `stable_only`. Never add module-level warning
  filters or other process-wide side effects.
- User rule: never use "--" in prose (numpydoc underlines and code syntax are
  fine).

## Testing / release

- `pytest tests/` (fast, ~2 s). Dev deps in `[dependency-groups] dev`.
- Release: bump `pyproject.toml` + `uv.lock` version, move CHANGELOG
  `[Unreleased]` to the version, `python -m build`, tag matching the README
  wheel URL (`releases/download/<ver>/macropy-<ver>-py3-none-any.whl`).

## Environment note

The committed `.venv` may point at a missing interpreter; create a fresh venv
from Homebrew `python3.11` and `pip install -e .` when running locally.
