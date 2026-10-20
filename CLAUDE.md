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
- `src/MacroPy/tvar.py` — `ThresholdVAR`: Alessandri & Mumtaz (2017) TVAR
  without SV (regime-specific `Sigma_i`), same API as `ThresholdVARSV`.
  Exact posterior, not Gibbs: closed-form marginal likelihood of every split
  `(d, Z*)` under the conjugate dummy prior (`split_posterior`), then iid NIW
  draws with exact rejection of unstable regimes. The authors' RW-Metropolis
  threshold step sticks to its starting mode on these models.
  `log_marginal_likelihood` (TVAR vs linear VAR). COVID treatment with the
  `BayesianVAR` arguments (`covid_mode="lenza-primiceri"` scales estimated on
  the linear VAR with the same prior, or `"dummies"` = window rows and the
  lag rows after it dropped from the likelihood); `delay=`/`threshold=` fix
  them (point-mass prior); `prior_params["sigma_scale"]` = "sd" | "variance".
  Comparison helpers
  `compare_threshold_models` and `plot_{regime,irf,volatility}_comparison`
  (in `plots_tvarsv.py`); both classes have `residual_volatility` and
  `log_predictive_score` (simulated paths, Gaussian last step);
  `recursive_log_scores` / `compare_log_scores` (Amisano-Giacomini) /
  `plot_log_scores` for out-of-sample comparisons. Threshold models start the
  sample at `max(lags + training, max_delay)`. `b_exo=` (BayesianVAR's mask
  convention) is a hard block structure, since the dummy prior is Kronecker:
  only block-recursive masks (external variables first, excluding every other
  variable) are accepted; the external block is a linear BVAR without regimes
  (`draws["Bx"]`, `["Sx"]`), the domestic block a threshold regression on
  `[domestic lags, 1, external now, external lags]` (`lamda0P` for the
  contemporaneous terms), and the split posterior uses the domestic block.
  Reference: `scripts/block_tvar_reference.py` (matched draw for draw; its test
  needs research/tvar_external_shocks or `TVAR_SHOCKS_ROOT`, else it skips).
  Tutorial: `tutorial_thresholdvar.ipynb` (about 3 min to run).
- `src/MacroPy/tvpvar.py` — `TimeVaryingVAR`: Chan & Eisenstat (2018) TVP-VARs
  with SV (`model=` C, C-SV, TVP, TVP-SV, TVP-R1/R2/R3-SV). Banded precision
  sampler (block-tridiagonal state precision in LAPACK band storage), KSC
  mixture SV; two-layer storage (hyperparameters every sweep, state paths
  thinned). IRFs/FEVD/HD date by date with recursive `A_t^{-1} Sigma_t^{1/2}`;
  optional `log_marginal_likelihood` / `dic` on the integrated likelihood
  (validated against the authors' MATLAB, Kalman and particle filters). The
  TVP posterior can be multimodal: compare seeds. Plots in `plots_tvpvar.py`
  (volatility, IRF and FEVD panels on a balanced grid, `_plotting._grid_cols`;
  FEVD plots default to 20 steps).
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
- `datasets/` CSVs (first column `date`); `tutorials/` executed notebooks
  (uniform format: `# Title`, an italic tagline, numbered `## n.` sections, a
  closing `## References`; no trailing `;`, figures are not echoed);
  `tests/` pytest suite.

## Conventions

- Class-based API: constructor takes a `pd.DataFrame` with datetime index;
  `model_summary()` renders Markdown via `summary.py`; `sample_posterior()` uses
  tqdm ("Sampling Posterior", switch with `progress=`); plotting stays in the
  `plots*.py` modules, matplotlib only, and every compute method takes a
  `plot_*` flag that defaults to False. Plotting functions are wrapped with
  `style.styled` (a temporary rcParams context; it also wraps the returned
  `(fig, axes)` so notebooks do not echo it), return `(fig, axes)`, never
  call `display` or `set_style()`; figures made by methods go to
  `model.figures`. Colours come from the public `style.PALETTE`, looked up at
  call time (no module-level colour constants, so edits to PALETTE reach every
  figure). Shared layout helpers (`_panels`, `_date_axis`, `_save`...) live in
  the private `_plotting.py`. Users opt in to the look with
  `MacroPy.set_style(context, font)` / `reset_style()` / `style_context()`;
  `macropy.mplstyle` is generated by `style._mplstyle_text()` and a test keeps
  them in sync. `set_bse_style` and `BSE_*` are deprecated aliases.
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

- `pytest tests/` (about 15 s; 40 s when the block-TVAR reference test finds
  its data). Dev deps in `[dependency-groups] dev`.
- Release: bump `pyproject.toml` + `uv.lock` version, move CHANGELOG
  `[Unreleased]` to the version, `python -m build`, tag matching the README
  wheel URL (`releases/download/<ver>/macropy-<ver>-py3-none-any.whl`).

## Environment note

The committed `.venv` may point at a missing interpreter; create a fresh venv
from Homebrew `python3.11` and `pip install -e .` when running locally.
