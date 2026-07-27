# MacroPy — project context

Bayesian macroeconometrics toolbox (src layout, `src/MacroPy/`). Python >=3.11,<3.12.

## Layout

- `src/MacroPy/bvar.py` — `BayesianVAR`: Minnesota / Normal-Wishart / Normal-Diffuse
  priors (built in `priors.py`), block exogeneity, exogenous regressors, COVID
  treatment (Lenza-Primiceri volatility scaling or auto time dummies),
  Waggoner-Zha conditional forecasts (full shock distribution; verified
  draw-by-draw against the Canova-Ferroni `BVAR_` toolbox — reference ports in
  `tests/test_bvar_production.py`), tidy `forecast_frame()` /
  `conditional_forecast_frame()`, GLP (2015) `select_hyperparameters()`, `seed`
  for reproducibility.
- `src/MacroPy/tvarsv.py` — `ThresholdVARSV`: Alessandri & Mumtaz (2019)
  threshold VAR with stochastic volatility; GIRFs by regime; plots in
  `plots_tvarsv.py`.
- `src/MacroPy/bpvar.py` — hierarchical Bayesian panel VAR. `cvar.py` — classic
  VAR. `lp.py` — smooth local projections. `state_space.py` — Kalman/UC models
  (plots in `plots_kalman.py`).
- `priors.py` also holds the analytic conjugate-NW marginal likelihood
  (`nw_conjugate_moments`, `nw_log_marginal_likelihood`) used by GLP selection
  and the COVID scale estimation.
- `datasets/` CSVs (first column `date`); `tutorials/` executed notebooks;
  `tests/` pytest suite.

## Conventions

- Class-based API: constructor takes a `pd.DataFrame` with datetime index;
  `model_summary()` renders Markdown via `summary.py`; `sample_posterior()` uses
  tqdm ("Sampling Posterior"); plotting stays in the `plots*.py` modules and
  every compute method takes a `plot_*` flag.
- `post_draws` = total draws including burn-in; `burnin` = fraction discarded.
- Regressor layout from `prepare_data`: `[lag-1 vars, ..., lag-P vars,
  constant?, trend?, exog...]` — lags first, deterministic terms last.
- Keep the public API backwards compatible; new constructor arguments go at the
  end with safe defaults.
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
