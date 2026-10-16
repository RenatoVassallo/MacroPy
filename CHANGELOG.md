# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-10-16

Minor-version bump: two new model classes, `TimeVaryingVAR` (Chan &
Eisenstat, 2018) and `ThresholdVAR` (Alessandri & Mumtaz, 2017, with an exact
posterior sampler); comparison tools and density forecasts for the threshold
VARs; and one MacroPy figure style with an explicit opt-in (`set_style`).
`ClassicVAR.forecast` and `ClassicVAR.run_full_analysis` are removed (they
always failed). `ThresholdVARSV` regime probabilities and GIRFs change
slightly (see Fixed). `tutorial_tvarsv.ipynb` is replaced by
`tutorial_thresholdvar.ipynb`.

### Added (time-varying parameter VARs)
- **`TimeVaryingVAR`**: time-varying parameter VARs with stochastic volatility,
  a port of the replication code of Chan & Eisenstat (2018, JAE). One class
  covers the paper's seven specifications (`model=`): `C`, `C-SV`, `TVP`,
  `TVP-SV`, `TVP-R1-SV` (drifting contemporaneous matrix `A_t`), `TVP-R2-SV`
  (drifting lag coefficients and intercepts) and `TVP-R3-SV` (drifting
  intercepts). The states are drawn jointly by the precision-based sampler,
  with the `Tk x Tk` precision assembled in LAPACK band storage, and the
  log-volatilities by the Kim-Shephard-Chib auxiliary mixture sampler.
  `presample` fixes a common estimation sample across lag lengths.
- Draws are stored in two layers: hyperparameters at every retained sweep,
  state paths thinned (`thin=None` keeps at most 1000); `storage_mb()` and a
  notice report the memory.
- Structural analysis date by date (recursive identification
  `A_t^{-1} Sigma_t^{1/2}`): `compute_irfs` (posterior median and bands at
  every date, draws only for the requested dates), `compute_fevd` (FEVD over
  time), `compute_hd` (historical decomposition with exact identity per draw),
  `forecast` / `forecast_frame` (predictive density with or without parameter
  drift), `volatility_frame`, `coefficient_frame`.
- Model comparison, optional and expensive: `log_marginal_likelihood`
  (importance sampling with the states integrated out: coefficients
  analytically, log-volatilities by an inner importance sampler at the mode
  of `p(h | y)` found by EM) and `dic`, both with a time and memory notice,
  seeded chunks and `n_jobs` parallelism; `compare_tvpvar` tabulates them.
  `logml` is the log of the mean of all importance weights; the paper's
  mean of batch log-means, biased downward when batches are small, is
  reported as `logml_batch_mean`.
- `plots_tvpvar`: `plot_tvp_volatility`, `plot_tvp_coefficients`,
  `plot_tvp_irfs`, `plot_irf_surface` (3D), `plot_irf_over_time`,
  `plot_fevd_over_time`, `plot_tvp_fan_chart`; `generate_tvpvar_summary`.
- `tutorials/tutorial_tvpvar.ipynb` (Peru: inflation, GDP growth, interbank
  rate) and `tests/test_tvpvar.py`.

### Added (threshold VARs)
- **`ThresholdVAR`**: the threshold VAR of Alessandri & Mumtaz (2017, RED),
  with regime-specific coefficients and covariance matrices: `ThresholdVARSV`
  without the stochastic-volatility block, with the same constructor, methods
  and output layouts, so both can be estimated on the same sample and
  compared. GIRFs by regime to every Cholesky shock (common random numbers);
  `irf_1std=0` gives unit shocks, comparable across models.
- **Exact posterior for `ThresholdVAR`.** Under the dummy-observation
  (Normal-inverse-Wishart) prior, the marginal likelihood of every admissible
  regime split `(d, Z*)` is computed in closed form (`split_posterior`); each
  draw takes `(d, Z*)` from that grid, then independent inverse-Wishart and
  matrix-normal draws per regime, rejecting draws with an unstable regime. Same
  target as the authors' Gibbs sampler, whose random-walk step for the
  threshold stays near its starting mode: on the Alessandri-Mumtaz data,
  chains from different starting points settled at different `(Z*, d)`.
  `log_marginal_likelihood()` reports the TVAR against the linear VAR with the
  same prior.
- Side-by-side comparison of threshold models (`{label: model}`):
  `compare_threshold_models`, `plot_regime_comparison`, `plot_irf_comparison`,
  `plot_volatility_comparison`, and `residual_volatility()` in both classes
  (posterior of the reduced-form shock standard deviations over time).
  `generate_tvar_summary`, `TVAR_PRIOR_DEFAULTS`.
- Density forecasts for both threshold classes: `log_predictive_score(future,
  horizons)` estimates `log p(y_{T+h} | y_{1:T})` at realised data, jointly
  and per variable, by simulating regimes, volatility and data to `T + h - 1`
  and averaging the Gaussian density of the last step (no kernel smoothing).
  `recursive_log_scores` repeats it over forecast origins on an expanding
  window (origins in parallel), `compare_log_scores` reports average score
  differences with Amisano-Giacomini (2007) Newey-West tests that allow for
  overlapping forecasts, and `plot_log_scores` draws cumulative differences.
- `tutorials/tutorial_thresholdvar.ipynb` replaces `tutorial_tvarsv.ipynb`:
  TVAR against TVAR-SV (regimes, volatility, financial shocks) and the
  uncertainty-shock results of Alessandri & Mumtaz (2019).
  `tests/test_tvar.py` checks the exact posterior, the marginal likelihood
  (Chib identity), the GIRFs and the predictive densities against closed
  forms.

### Added (figures and style)
- **One entry point for the MacroPy look.** `set_style(context, font)`,
  `reset_style()` and `style_context()` apply the style to the user's figures
  and to MacroPy's (one call restyles both). Contexts `"paper"` (smaller type,
  300 dpi exports), `"notebook"` and `"slides"` (larger type); `font="serif"`,
  `"sans"` or a family name, with formulas in the matching math set.
- The style now sets a categorical colour cycle (navy and orange, then
  Okabe-Ito hues), concise date ticks and tight-bbox exports, so plain
  `ax.plot` calls inherit the MacroPy identity.
- Bundled style sheet: `plt.style.use("MacroPy.macropy")`.
- `color_palette(n)`, `colormap("diverging" | "sequential")` (not registered
  globally) and the `CONTEXTS` table.
- Plotting functions return a plain `(fig, axes)` tuple (or list) that notebooks
  do not echo: a plotting call ending a cell shows only the figure, without a
  trailing `;`.

### Changed (figures and style)
- Stacked areas and bars (FEVD, historical decomposition) use the MacroPy
  categorical palette; the TVP IRF surface defaults to a diverging map centred
  at zero.
- `plots_uc`, `plots_tvarsv` and `plots_kalman` take their colours from
  `PALETTE` at call time: edits to `PALETTE` now reach them (the former
  constants were frozen at import). Hard-coded colours replaced by roles.
- Shared layout helpers moved to the private `_plotting.py`; no plotting
  module depends on `plots_kalman` any longer.
- `PanelLocalProjections.plot_irf` is drawn in the MacroPy style.
- `BayesianPanelVAR` takes a `seed` (last constructor argument): per-stage
  generators for `sample_posterior` and `forecast`, as in `BayesianVAR`;
  `failed_stability_draws` now counts the last run only.
- `SmoothLocalProjections.estimate` rejects an unknown `projection_type`
  (anything but `'reg'` used to run the smooth estimator silently).
- `plot_irfs` (threshold models) accepts `ThresholdVAR` and defaults to the
  uncertainty shock when it was computed, otherwise to the threshold
  variable's shock.

### Removed
- `ClassicVAR.forecast` and `ClassicVAR.run_full_analysis`: leftovers of an
  early version that referenced attributes and a helper that do not exist and
  always failed with `NameError`. Forecasts come from `BayesianVAR.forecast`.

### Deprecated
- `set_bse_style()` (use `set_style()`) and the `BSE_*` colour constants (use
  `PALETTE[...]`): they still work and emit a `DeprecationWarning`.

### Fixed
- Invalid escape sequence in the `plots_kalman` docstring (a
  `DeprecationWarning` on import, a `SyntaxError` under `-W error`).
- `BayesianVAR.plot_exog_posteriors` documented a plotnine return value; it
  returns matplotlib `(fig, axes)`.
- `ThresholdVARSV` failed with `max_delay > lags + training`; the estimation
  sample now starts at `max(lags + training, max_delay)`, so every delay is
  evaluated on the same observations (unchanged for the default settings).
- `ThresholdVARSV` stored with each draw the regimes implied by the threshold
  and delay before that sweep's update, but the threshold and delay after it.
  The stored regimes (`e1`) now match the stored threshold and delay, which
  `regime_probability` and the GIRF histories use; the threshold and delay
  draws themselves are unchanged. The `training` docstring now says that those
  observations are dropped (the volatility is initialised on the estimation
  sample).

### Validation (TimeVaryingVAR)
- Against the authors' MATLAB code on their US data (1955-2014, p = 2): log
  marginal likelihoods agree for C and C-SV to Monte Carlo error, and for the
  SV models once the same estimator and number of draws are used; posterior
  volatility paths correlate above 0.999. The TVP posterior is multimodal on
  those data (chains settle on different coefficients to absorb the drift);
  chains in the MATLAB run's mode reproduce its marginal likelihood.
- Integrated likelihoods checked against an exact Kalman filter, a 1M-particle
  filter (C-SV) and a Rao-Blackwellised particle filter (TVP-SV); the C-model
  marginal likelihood against exact quadrature.

## [0.2.0] - 2026-09-19

Minor-version bump: figures are now matplotlib objects that are returned
rather than displayed, nothing is plotted by default, the public API is an
explicit list, and the ClassicVAR bootstrap bands change (see below).
`statsmodels` is now a declared dependency: `seasonal_adjust` needs it and it
used to arrive only through plotnine.

### Changed (figures and API)
- **One graphics backend: matplotlib.** `plots.py` no longer uses plotnine or
  mizani (both dropped from the dependencies). `generate_series_plot`,
  `generate_coeff_plot`, `generate_exog_coeff_plot`, `generate_panel_coeff_plots`,
  `generate_irf_plots`, `generate_fevd_plot`, `generate_hd_plot` and
  `generate_forecast_plots` return matplotlib `(fig, axes)` (a list of them for
  IRFs and per-lag coefficient plots) instead of ggplot objects.
- **Figures are returned, never displayed.** No plotting path calls
  `IPython.display`: `plot_series`, `plot_unit_series` and
  `plot_exog_posteriors` return `(fig, axes)`, and the figures made by
  `compute_*`, `forecast` and `conditional_forecast` are stored in
  `model.figures`. Notebooks show them at the end of the cell; scripts call
  `plt.show()`.
- **Nothing is drawn by default.** `plot_fevd`, `plot_hd` and `plot_forecast`
  now default to False in `ClassicVAR`, `BayesianVAR` and `BayesianPanelVAR`
  (the other `plot_*` flags already did).
- **No global style changes.** Every plotting function (including
  `plots_kalman`, `plots_uc` and `plots_tvarsv`) runs inside
  `macropy_style()`, a `plt.rc_context`, and none calls `set_bse_style()`.
  `MacroPy.use_own_style = False` keeps the caller's style untouched;
  `set_bse_style()` remains as an explicit, global opt-in.
- **Explicit public API.** Every module defines `__all__` and
  `MacroPy/__init__.py` imports names explicitly instead of nine `import *`.
  Names that leaked through the star imports (`np`, `pd`, plotnine objects)
  are no longer attributes of `MacroPy`.
- `cross_validate_lambda` and `plot_irf_with_confidence` are back in
  `SmoothLocalProjections`. They had ended up inside `PanelLocalProjections`,
  which has neither `result["X"/"Y"/"P"]` nor `h_max`, so they were unusable
  from either class and `tutorial_localprojections` failed on them.
- `SmoothLocalProjections.plot_irfs` and `plot_irf_with_confidence` return
  `(fig, ax)` instead of calling `plt.show()`, and `cross_validate_lambda` only
  plots with `plot=True` (the figure goes to `cv_figure`).
- `model_summary()` works with string date indexes (it called `.date()` on
  them); `generate_series_plot` converts such indexes to dates.
- Historical-decomposition plots use the last T dates (after the lags); they
  used the first T, shifted by p periods. Forecast fans start at the last
  observation, and panel-VAR unconditional forecasts no longer use the
  conditional colour.

### Deprecated
- `lamda1`..`lamda4` in `prior_params` and in
  `select_hyperparameters(select=...)`: use `lambda1`..`lambda4`. The old keys
  still work and emit a `DeprecationWarning`; `model.prior_params` stores the
  new keys. Tutorials and course notebooks are migrated.

### Fixed (numerical results change)
- `ClassicVAR.compute_irfs(bstrap=True)` now uses the **recursive-design**
  residual bootstrap: each artificial sample iterates the estimated VAR from
  the observed initial conditions with resampled, centred residuals. Before,
  the sample was built from fitted values on the *observed* lags and then
  re-lagged with `prepare_data`, an errors-in-variables problem that attenuated
  the autoregressive coefficients and inflated `Sigma` and the Cholesky impact
  (about +28 % on the diagonal in a simulated VAR(1) with T = 2000, a bias that
  does not vanish with T). Bootstrap bands from earlier versions should be
  recomputed. On that example the 95 % impact bands are now [0.971, 1.032] and
  [0.386, 0.410], around the true 1.0 and 0.4.
- `ClassicVAR.compute_irfs(bstrap=False)` now draws `Sigma ~ IW(U'U, T - k)`
  and then `b | Sigma ~ N(b_OLS, Sigma kron (X'X)^-1)`, so the impact matrix
  carries uncertainty (it was a single point at h = 0).
- `import MacroPy` no longer runs `warnings.filterwarnings("ignore")`, which
  silenced warnings for the whole process (NumPy, pandas, user code).
- No mutable default arguments: `BayesianVAR(prior_params=None)`,
  `MinnesotaPrior`, `NormalWishartPrior` and `NormalDiffusePrior`.

### Added
- `ClassicVAR(seed=None, stable_only=True, progress=True)`: reproducible draws
  per stage (`default_rng([seed, stage])`, as in `BayesianVAR`), explicit
  filtering of explosive draws in both branches, and a switch for the progress
  bar. Axis order of `ir_draws` documented: `[draw, horizon, variable, shock]`.
- `BayesianVAR(progress=True)` to turn off the tqdm bars in rendered documents.
- `prior_params` uses `lambda1`..`lambda4` and raises a `ValueError` on
  unknown keys (`priors.normalize_prior_params`); see Deprecated.
- `coefficients_by_lag(B, n, p, constant=True)` returns `(phis, c)` and
  isolates the user from the layout (MacroPy puts the constant after the lags).
- `py.typed` marker; `get_fred_data` says what to set when the API key is
  missing.
- **Historical decomposition for `BayesianVAR`** (`compute_hd`), computed draw
  by draw with each draw's own coefficients and Cholesky impact matrix and
  reported as the posterior mean; `store_draws=True` keeps every draw. It is
  not available under the steady-state prior, which works on demeaned data.
- `data_handling.historical_decomposition(B, Sigma, yy, XX, n, p, S=None)`:
  the shared computation, with the deterministic and exogenous block read from
  the regressors so any trend convention or exogenous variable is respected.
- `HD` now has the same shape in both classes: `"shock"` (T, n_shocks, n_endo),
  `"baseline"`, `"endo"` and `"dates"`, and `baseline + shock.sum(axis=1)`
  reproduces the sample exactly (checked in `tests/test_historical_decomposition.py`).
  `ClassicVAR.HD` used to store `(T + p, variable, shock)` with `init`, `const`
  and `trend` keys, and the plot read that array as `(T, shock, variable)`, so
  panels and colours were transposed: the historical decomposition figure was
  mislabelled.
- - `MacroPy.style`: public, overridable `PALETTE` (with an Okabe-Ito,
  colour-blind-safe set for stacked FEVD and HD plots), `macropy_style()`,
  `macropy_rcparams()` and the `styled` decorator.
- `tests/test_plots.py`: figures are returned and stored, nothing is drawn by
  default, global rcParams are untouched, `use_own_style` works, and the public
  API is explicit.
- `tests/test_quick_fixes.py`: the bootstrap recovers a known impact matrix,
  Monte Carlo impact uncertainty, seed reproducibility, prior-parameter
  hygiene, `coefficients_by_lag`, and no process-wide warning filter.

### Added
- **UC trend-cycle module** (`MacroPy.uc`): one engine, `UCTrendCycle`, covers
  the whole decomposition ladder on the model
  `y = tau + c`, `tau_t = tau_{t-1} + g_{t-1} + eta_t`,
  `g_t = rho_g g_{t-1} + (1-rho_g) gbar + zeta_t`, AR(2) cycle, with
  per-shock variance-scale paths `q_{i,t} = lambda_{i,t} sigma_i^2`.
  Python port of the author's MATLAB `trend_cycle_toolbox` (building on
  Grant & Chan 2017, JEDC, and Chan 2018).
  - `hp_filter(y, lam)`: exact penalized-regression HP filter, and the
    `UCTrendCycle.hp` preset showing the filter as a state-space model
    (Harvey & Jaeger 1993); smoothed trends agree to ~1e-6.
  - `UCTrendCycle.clark(y).fit_mle()`: Clark (1987) UC by exact integrated
    maximum likelihood (precision identity), with an HP-based warm start, a
    budget-capped derivative-free polish for the classic variance pile-up
    (`polish_budget`; seconds instead of minutes at T ~ 300 with the same
    optimum) and a tqdm progress bar (`progress`).
  - `sample_posterior()`: Gibbs sampling with the Chan/McCausland
    precision-based simulation smoother, truncated-normal AR(2)/drift
    updates and inverse-gamma variances. The joint (tau, g) precision is
    assembled stencil-by-stencil in interleaved order into banded storage
    (bandwidth 4) and factored with scipy's banded Cholesky, the sparse
    strategy of the MATLAB toolbox: assembly and draw are O(T) and
    BLAS-independent (state draw ~0.1 ms at T ~ 200-300; 1000 sweeps of
    the plain model in under a second, 20000 sweeps of a COVID model in
    ~15 s). The banded system is verified against the dense reference
    assembly at 1e-10 over random configurations, and the integrated
    likelihood evaluates the minimized quadratic as a weighted residual
    sum at the mode, avoiding catastrophic cancellation (agreement with
    the independent Kalman filter ~1e-9 even on 100*log-level data). The
    griddy-Gibbs COVID conditionals are fully vectorized over the grids
    (verified against the brute-force full-path construction below 1e-9).
    A `thin` argument stores every thin-th post burn-in sweep (MATLAB
    toolbox convention); the docstring notes that thinning saves memory
    rather than adding precision.
  - **Prior-calibration kappa rules** (on by default): `sd(zeta) = 0.35 x
    sd(13-quarter MA of dy)` and `sd(eta) = 0.10 x sd(zeta)`, computed on
    the pre-COVID sample, replacing generic defaults that let the trend
    chase the data.
  - **COVID variance scales** (Lenza & Primiceri 2022): deterministic jump
    paths `(lambda1, lambda2, lambda3, rho_decay)` per shock, sampled by
    griddy Gibbs with log-normal shrinkage toward one on the trend
    channels; `covid_t0="2020Q1"` activates them (the reference toolbox
    convention: the two free scales cover 2020Q1-Q2 and the geometric tail
    starts at the rebound).
  - **Model comparison**: WAIC/DIC from the one-step-ahead predictive
    decomposition (per-period Kalman contributions at each draw) and a
    `compare_uc()` table; an independent Kalman filter cross-checks the
    precision-based likelihood at ~1e-8 (asserted in tests).
  - **Conditional forecasts** (`conditional_forecast` /
    `conditional_forecast_frame`): Waggoner-Zha style restrictions on the
    predictive distribution, per posterior draw (exact Gaussian conditioning
    of the standardized future shocks via the SVD, minimum norm plus
    null-space noise, mirroring `BayesianVAR.conditional_forecast`).
    Supports hard values for future quarters (`conditions={"2027Q1": ...}`),
    an `anchor_states` option that starts every path from the posterior
    median final state (removing the current trend/cycle split uncertainty,
    so fan charts of unobserved components start narrow and widen with the
    horizon, the central-bank convention), and calendar-year average growth
    targets
    (`annual_growth={2026: 3.2, ...}`, with already-observed quarters of a
    partially observed year entering as data). Restrictions hold to ~1e-13
    along every draw; parameter and COVID-scale uncertainty integrated
    draw by draw.
  - Tidy outputs `decomposition_frame()` / `forecast_frame()`,
    `model_summary()`, per-stage seeded reproducibility, plots in
    `plots_uc.py` (`plot_uc_decomposition`, `plot_covid_scales`,
    `plot_gap_comparison`).
  - **Fan charts** (`plots_uc.plot_fan_chart`): central-bank style nested
    prediction bands (BoE / BCRP Inflation Report look) from any draws
    matrix, continuous with the observed history, with configurable
    coverage levels, center line, reference annotations and fully
    overridable labels. `method="normal"` draws band edges from a
    per-horizon normal approximation and `smooth=` applies a light moving
    average to the band half-widths, removing Monte Carlo jaggedness
    without touching the central path. `plot_gap_comparison` gains recession shading
    (`shade`) and dashed benchmark overlays (`extra_series`, e.g. the CBO
    gap); `plot_uc_decomposition` gains `shade` and a `labels` dict for
    non-English figures.
  - **Band convention**: state bands default to the central 68% interval
    (one posterior standard deviation) in `decomposition_frame()`,
    `plot_uc_decomposition` and `plot_gap_comparison`, matching
    `plot_covid_scales`; the coverage is always written in the legend, and
    `outer_cred` adds a lighter nested band (e.g. 68% inside 90%). Trend and
    cycle bands coincide by construction, since `c_t = y_t - tau_t` holds
    draw by draw. The smoothed-state uncertainty behind them agrees with an
    independent Kalman smoother (statsmodels) to ~5e-3 away from the
    initial-condition window.
- Dataset `datasets/USGDP_Data.csv` (US real GDP, quarterly, 1947 onward)
  and executed tutorial `tutorials/tutorial_uc_gdp.ipynb`: a compact
  four-rung ladder (HP, Clark MLE, Bayesian UC, COVID scales) on US GDP
  downloaded live through the MacroPy FRED helper (`.env` key, with a
  packaged-CSV fallback), benchmarked against NBER recession shading and
  the CBO output gap (GDPC1 vs GDPPOT), with the pile-up exhibit and a
  WAIC comparison.
- Guidance notes on `ClarkModel`, `WatsonUC` and `BayesianStateSpace`
  pointing trend-cycle use cases to the new engine (all remain available).
- Validation: draw-level cross-check against the MATLAB toolbox's saved
  Peru 1994-2025 UC-COVID posterior (same input series): every prior
  hyperparameter identical to machine precision, including the calibrated
  COVID grid bounds (the r2020 rule uses the first two pandemic growth
  surprises, sample-std/var with N-1, plain rounding of the eta/zeta grid
  ratios); posterior state medians within 0.12 log points, 90 percent
  bands of equal width, lambda paths and AR dynamics at Monte Carlo error.
  Gap range on US GDP 1947-2014 also matches the golden numbers
  ([-6.7, 4.3] vs [-6.8, 4.3]). Tests cover HP equivalence, likelihood
  agreement, MLE/Gibbs parameter recovery, kappa rules, COVID scale
  absorption (pre-2020 decomposition protected) and reproducibility.

### Fixed
- `BayesianVAR` docstring for `b_exo`: the mask convention was documented backwards.
  The implementation (`priors.MinnesotaPrior`) has always used `b_exo[i, j] = 1` to
  exclude the lags of variable j from equation i (prior variance ~0), and `0` to leave
  the coefficient free; the docstring said the opposite. New regression test
  `tests/test_bvar_block_exog.py` pins the convention.

## [0.1.11] - 2026-08-06

### Added
- **Seasonal adjustment** (`seasonal_adjust`): X-13ARIMA-SEATS (official US
  Census Bureau binary bundled per platform, driven through statsmodels)
  and STL as a dependency-light cross-check, returning a tidy
  `SeasonalResult` with `observed = seasadj + seasonal` in original units.

## [0.1.10] - 2026-07-28

### Added
- **Steady-state (mean-adjusted) prior** for `BayesianVAR` (Villani, 2009,
  Journal of Applied Econometrics) via the optional `prior_params` keys
  `ss_mean` / `ss_sd`. The model is reparameterized as
  `(y_t - mu) = sum_l A_l (y_{t-l} - mu) + e_t` with `mu ~ N(m0, V0)`
  elementwise, so the VAR's long-run attractor carries an informative prior
  (e.g. judgment on medium-term growth for variables in structural slowdown)
  instead of defaulting to the estimation-sample mean, while cross-variable
  dynamics stay untouched. `ss_mean` accepts a dict `{column: value}` or a
  vector (omitted columns get a diffuse sd-1e3 prior centered on their sample
  mean); `ss_sd` accepts the same forms plus a scalar broadcast.
- **Gibbs sampler extension** for all prior types: a `(B, Sigma) | mu` block
  on data demeaned at the current draw (reusing the existing prior/posterior
  machinery, COVID weights and SOC/DIO dummies on the demeaned system) and the
  linear-Gaussian `mu | (B, Sigma)` update of Villani eqs. 6-8, GLS-weighted
  under the Lenza-Primiceri COVID mode. `mu_draws` is stored and returned.
- **Downstream integration**: forecasts iterate the companion on deviations
  and add the draw's `mu` back (long-horizon paths converge to the `mu` draws
  by construction); conditional forecasts work unchanged in levels;
  `model_summary()` reports the steady-state prior.
- **Guards**: the constant is removed when the prior is active (`mu` replaces
  it); `timetrend`, user `exog` and `covid_mode="dummies"` raise clear errors;
  `log_marginal_likelihood` / `select_hyperparameters` raise
  `NotImplementedError` (no closed form; `ss_mean`/`ss_sd` are user-set, never
  selected). Docstrings note the deliberate tension with `soc`/`dio` (unit
  roots vs reversion to `mu`).
- Tests: judgment anchoring at h=40 vs sample-mean reversion, exact backward
  compatibility, conditional forecasts under the prior, combinations with the
  COVID mode and SOC/DIO, and all guards.

## [0.1.9] - 2026-07-28

### Added
- **Sum-of-coefficients (SOC) and dummy-initial-observation (DIO) priors** for
  `BayesianVAR` via the optional `prior_params` keys `soc` and `dio`
  (Doan, Litterman & Sims, 1984; Sims, 1993; Banbura, Giannone & Reichlin,
  2010; Giannone, Lenza & Primiceri, 2015). Implemented as dummy-observation
  data augmentation anchored at the pre-sample means, so they work with every
  `prior_type` and combine with block exogeneity and both COVID modes. They
  shrink the VAR toward unit roots / no-change forecasts, anchoring
  long-horizon paths to the current level instead of the estimation-sample
  mean (the classic remedy for variables with a slowly falling mean, and what
  makes larger systems safe to estimate). Convention: pseudo-data divided by
  the hyperparameter (small = tight, 1 = GLP default, absent/None or >= 1e4 =
  off).
- `log_marginal_likelihood` applies the GLP (2015) dummy correction
  ``log m(dummies + Y) - log m(dummies)`` when SOC/DIO are active (dummies
  unweighted; the Lenza-Primiceri Jacobian stays on actual rows only), and
  `select_hyperparameters` can now search over `soc` / `dio` (bounds
  [0.1, 10], starting value 1 when off). Because the pseudo-observations
  always carry degrees-of-freedom information about Sigma, the pure ratio
  does not converge to the no-dummy value as the tightness grows; values
  >= 1e4 therefore disable the priors exactly.
- New tutorial `tutorials/tutorial_bvar_socdio.ipynb`: theory and pseudo-data
  construction of both priors, their effect on posterior persistence, marginal-
  likelihood selection of the tightness, and a pseudo out-of-sample conditional
  forecasting exercise on the Peruvian system (with Lenza-Primiceri COVID
  control) evaluated against realized data.
- `model_summary()` reports active dummy priors; new tests cover the tight-SOC
  unit-sum limit, level anchoring on a late-break mean, the ML correction and
  off-switch, exact backward compatibility (bit-identical draws without the
  new keys), and combinations with COVID modes, conditional frames and
  `b_exo`.

## [0.1.8] - 2026-07-28

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
- **`conditional_forecast` no longer depends on call order**: the no-shock
  baseline is now always rebuilt for the requested horizon. Previously a
  shape-based staleness check mistook the zero-initialized `mean_forecasts`
  array for a valid baseline whenever the requested `fhor` equaled the
  constructor's, so calling `conditional_forecast` without a prior
  `forecast()` conditioned against a *zero* baseline — the solver read the
  entire imposed path as a deviation and fired enormous shocks (explosive
  spillovers to unconstrained variables). Both `forecast()` and
  `conditional_forecast()` now also raise a clear error when called before
  `sample_posterior()`.
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
