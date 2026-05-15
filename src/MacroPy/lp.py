import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from numpy.linalg import solve
from scipy.stats import norm
from IPython.display import Markdown, display

from .data_handling import prepare_panel_data


def _b_spline_basis(horizon_grid, lower_bound, upper_bound, num_knots, degree):
    knot_step = (upper_bound - lower_bound) / num_knots
    knots = lower_bound + knot_step * np.arange(-degree, num_knots)
    T = np.tile(knots, (len(horizon_grid), 1))
    X = np.tile(np.asarray(horizon_grid).reshape(-1, 1), (1, len(knots)))
    P = (X - T) / knot_step
    B = ((T <= X) & (X < T + knot_step)).astype(float)
    for k in range(1, degree + 1):
        B = (P * B + (k + 1 - P) * np.roll(B, shift=-1, axis=1)) / k
    return B


def smooth_lp_results(
    results: pd.DataFrame,
    group_cols=None,
    shock_col: str = "shock",
    horizon_col: str = "horizon",
    estimate_col: str = "estimate",
    std_error_col: str = "std_error",
    penalty_order: int = 2,
    degree: int = 3,
    penalty_lambda: float | None = None,
    lambda_grid=None,
    conf_levels=(0.68, 0.95),
    min_variance: float = 1e-10,
) -> pd.DataFrame:
    """
    Smooth LP coefficient paths using a Barnichon-Brownlees style spline penalty.

    This helper operates on already-estimated LP coefficients. It is therefore
    closer to the intuition of smooth local projections than to a full
    one-step re-estimation, but it preserves the panel LP first stage and is
    convenient for panel settings with fixed effects and ragged edges.
    """
    if results.empty:
        return results.copy()
    if horizon_col not in results.columns or estimate_col not in results.columns:
        raise KeyError("`results` must contain horizon and estimate columns.")
    if std_error_col not in results.columns:
        raise KeyError("`results` must contain standard errors to smooth with confidence bands.")

    lambda_grid = np.asarray(
        lambda_grid if lambda_grid is not None else np.r_[0.0, np.logspace(-2, 4, 80)],
        dtype=float,
    )

    group_cols = list(group_cols or [shock_col])
    smoothed_groups = []
    for _, subset in results.groupby(group_cols, sort=False):
        ordered = subset.sort_values(horizon_col).copy().reset_index(drop=True)
        horizons = ordered[horizon_col].to_numpy(dtype=float)
        y = ordered[estimate_col].to_numpy(dtype=float)
        se = ordered[std_error_col].to_numpy(dtype=float)
        variances = np.clip(se**2, min_variance, np.inf)
        weights = np.diag(1.0 / variances)

        num_knots = max(len(horizons), degree + 1)
        basis = _b_spline_basis(
            horizons,
            lower_bound=float(horizons.min()),
            upper_bound=float(horizons.max()) + 1.0,
            num_knots=num_knots,
            degree=degree,
        )
        penalty = np.eye(basis.shape[1])
        for _ in range(penalty_order):
            penalty = np.diff(penalty, axis=0)
        penalty_matrix = penalty.T @ penalty
        precision = basis.T @ weights @ basis

        if penalty_lambda is None:
            gcv_scores = []
            for lam in lambda_grid:
                system = precision + lam * penalty_matrix
                inv_system = np.linalg.pinv(system)
                hat_matrix = basis @ inv_system @ basis.T @ weights
                fitted = hat_matrix @ y
                residual = y - fitted
                dof = np.trace(hat_matrix)
                denom = max(len(y) - dof, 1e-8) ** 2
                gcv_scores.append(float((residual @ residual) / denom))
            selected_lambda = float(lambda_grid[int(np.argmin(gcv_scores))])
        else:
            selected_lambda = float(penalty_lambda)

        system = precision + selected_lambda * penalty_matrix
        inv_system = np.linalg.pinv(system)
        beta = inv_system @ basis.T @ weights @ y
        fitted = basis @ beta
        covariance_beta = inv_system @ precision @ inv_system
        covariance_fitted = basis @ covariance_beta @ basis.T
        fitted_se = np.sqrt(np.clip(np.diag(covariance_fitted), 0.0, np.inf))

        ordered["estimate_unsmoothed"] = ordered[estimate_col]
        ordered["std_error_unsmoothed"] = ordered[std_error_col]
        ordered[estimate_col] = fitted
        ordered[std_error_col] = fitted_se
        ordered["smoothing_lambda"] = selected_lambda
        ordered["smoothing_method"] = "penalized_b_spline"

        for level in conf_levels:
            suffix = int(round(level * 100))
            z_value = norm.ppf(0.5 + level / 2.0)
            ordered[f"lower_{suffix}"] = ordered[estimate_col] - z_value * ordered[std_error_col]
            ordered[f"upper_{suffix}"] = ordered[estimate_col] + z_value * ordered[std_error_col]

        smoothed_groups.append(ordered)

    return pd.concat(smoothed_groups, ignore_index=True)


class SmoothLocalProjections:
    def __init__(self, 
                 response: pd.Series, 
                 shock: pd.Series, 
                 controls: pd.DataFrame, 
                 lags: int = 4, 
                 horizons: list = list(range(0, 21))):
        if not isinstance(response, pd.Series) or not isinstance(shock, pd.Series):
            raise ValueError("response and shock must be pandas Series.")
        if not isinstance(controls, pd.DataFrame):
            raise ValueError("controls must be a pandas DataFrame.")

        self.y = response.to_numpy().flatten()
        self.x = shock.to_numpy().flatten()
        self.w = controls.to_numpy()
        self.lags = lags
        self.horizons = horizons
        self.h_min = min(horizons)
        self.h_max = max(horizons)
        self.result = None

    def _b_spline_basis(self, horizon_grid, lower_bound, upper_bound, num_knots, degree):
        return _b_spline_basis(horizon_grid, lower_bound, upper_bound, num_knots, degree)

    def estimate(self, projection_type='smooth', penalty_order=2, penalty_lambda=100):
        T = len(self.y)
        HR = self.h_max + 1 - self.h_min

        # === Shock scaling ===
        if self.w.size == 0:
            delta = np.std(self.x)
            controls = np.ones((T, 1))  # Only intercept
        else:
            proj = self.w @ np.linalg.pinv(self.w.T @ self.w) @ self.w.T
            delta = np.std(self.x - proj @ self.x)
            controls = np.column_stack((np.ones(T), self.w))  # Add intercept

        # === Basis setup ===
        if projection_type != 'reg':
            B = self._b_spline_basis(
                np.arange(self.h_min, self.h_max + 1),
                self.h_min, self.h_max + 1,
                HR, degree=3
            )
            K = B.shape[1]
        else:
            B = None
            K = HR

        # === Preallocate structures ===
        total_rows = (self.h_max + 1) * T
        Y_all = np.full((total_rows,), np.nan)
        Xb = np.zeros((total_rows, K))
        Xc = np.zeros((total_rows, HR, controls.shape[1]))
        idx_map = []

        for t in range(T - self.h_min):
            row_start = t * HR
            row_end = row_start + HR
            idx_block = slice(row_start, row_end)
            h_range = np.arange(self.h_min, self.h_max + 1)

            # Only keep y[t+h] if within bounds, else nan
            Y_all[idx_block] = np.array([self.y[t + h] if (t + h) < T else np.nan for h in h_range])
            idx_map += [(t, h) for h in h_range]

            # Xb
            if projection_type == 'reg':
                Xb[row_start : row_start + HR, :] = np.eye(HR) * self.x[t]
            else:
                Xb[row_start : row_start + HR, :] = B * self.x[t]

            # Xc (controls)
            for i in range(controls.shape[1]):
                Xc[row_start : row_start + HR, :, i] = np.eye(HR) * controls[t, i]

        # === Stack final design matrix ===
        X = Xb
        for i in range(controls.shape[1]):
            X = np.hstack((X, Xc[:, :, i]))

        # === Apply observation mask (trim to used rows only) ===
        idx_map = np.array(idx_map)
        used_len = len(idx_map)
        valid = np.isfinite(Y_all[:used_len])
        Y_valid = Y_all[:used_len][valid]
        X_valid = sparse.csr_matrix(X[:used_len][valid, :])
        idx_map = idx_map[valid]

        # === Estimate ===
        if projection_type == 'reg':
            XtX = (X_valid.T @ X_valid).toarray()
            XtY = X_valid.T @ Y_valid
            beta, *_ = np.linalg.lstsq(XtX, XtY, rcond=None)
            IR = np.zeros(self.h_max + 1)
            IR[self.h_min:] = beta[:K] * delta
            P = None
        else:
            D = np.eye(K)
            for _ in range(penalty_order):
                D = np.diff(D, axis=0)
            P = np.zeros((X_valid.shape[1], X_valid.shape[1]))
            P[:K, :K] = D.T @ D
            beta = solve(X_valid.T @ X_valid + penalty_lambda * P, X_valid.T @ Y_valid)
            IR = np.zeros(self.h_max + 1)
            IR[self.h_min:] = B @ beta[:K] * delta

        # === Store results ===
        self.result = {
            'IR': IR,
            'theta': beta,
            'shock_scale': delta,
            'X': X_valid,
            'Y': Y_valid,
            'projection_type': projection_type,
            'lambda': penalty_lambda if projection_type != 'reg' else 0,
            'P': P,
            'B': B,
            'K': K,
            'HR': HR,
            'T': T,
            'idx_map': idx_map
        }

        return self.result

    def compute_confidence_intervals(self, lag=20, alpha=0.1):
        if self.result is None:
            raise ValueError("Run `.estimate()` before computing confidence intervals.")

        res = self.result
        X = res["X"].toarray() if sparse.issparse(res["X"]) else res["X"]
        Y = res["Y"]
        P = res["P"]
        B = res["B"]
        K = res["K"]
        HR = res["HR"]
        T = res["T"]
        delta = res["shock_scale"]
        idx = res["idx_map"]
        lamb = res["lambda"]
        theta = res["theta"]

        # === Compute point estimate "bread" ===
        XXP = X.T @ X + (lamb * P if P is not None else 0)
        bread = np.linalg.pinv(XXP)

        # === Residuals based on original estimate ===
        U = Y - X @ theta

        # === Newey-West weights ===
        weights = np.array([0.5] + [(lag + 1 - l) / (lag + 1) for l in range(1, lag + 1)])

        # === Initialize "meat" ===
        npar = theta.shape[0]
        V = np.zeros((npar, npar))

        for l in range(lag + 1):
            GplusGprime = np.zeros((npar, npar))
            for t in range(l + 1, T - HR - 1):
                it = np.where(idx[:, 0] == t)[0]
                jt = np.where(idx[:, 0] == t - l)[0]
                if len(it) == 0 or len(jt) == 0:
                    continue
                S1 = X[it, :].T @ U[it]
                S2 = X[jt, :].T @ U[jt]
                GplusGprime += np.outer(S1, S2) + np.outer(S2, S1)
            V += weights[l] * GplusGprime

        # === Sandwich estimator ===
        VC = bread @ V @ bread

        # === z-value for confidence level ===
        z = norm.ppf(1 - alpha / 2)

        # === Compute IRFs and confidence intervals ===
        conf = np.full((self.h_max + 1, 2), np.nan)

        if res["projection_type"] == 'reg':
            ster = np.sqrt(np.clip(np.diag(VC[:K, :K]), 0, np.inf))
            IR = theta[:K] * delta
            conf[self.h_min:, 0] = IR - z * ster * delta
            conf[self.h_min:, 1] = IR + z * ster * delta
        else:
            IR = B @ theta[:K] * delta
            ster = np.sqrt(np.clip(np.diag(B @ VC[:K, :K] @ B.T), 0, np.inf))
            conf[self.h_min:, 0] = IR - z * ster * delta
            conf[self.h_min:, 1] = IR + z * ster * delta

        # === Store results ===
        self.result["confidence_intervals"] = conf
        self.result["standard_errors"] = ster
        self.result["theta"] = theta
        
        if self.h_min == 1:
            # set the first element to zero if h_min is 1
            conf[0, :] = 0

        return conf

    @staticmethod
    def plot_irfs(results_list, labels=None, title="Impulse Response Comparison"):
        """
        Plot multiple IRFs from results.

        Parameters:
        - results_list: list of result dictionaries returned by `estimate()`
        - labels: optional list of labels for each IRF
        """
        plt.figure(figsize=(8, 5))
        for i, res in enumerate(results_list):
            label = labels[i] if labels else f"IRF {i+1}"
            plt.plot(res['IR'], label=label, linewidth=2)

        plt.axhline(0, color="black", linestyle="--")
        plt.xlabel("Horizon")
        plt.ylabel("Response")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def _unique_preserve_order(values):
    seen = set()
    unique = []
    for value in values:
        if value is None or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def split_shock_signs(series: pd.Series, absolute_negative: bool = True) -> pd.DataFrame:
    """
    Split one shock series into positive and negative components.

    Parameters
    ----------
    series : pd.Series
        Shock series to split.
    absolute_negative : bool, default=True
        If True, the negative component is returned in absolute value so both
        split shocks are non-negative magnitudes.

    Returns
    -------
    pd.DataFrame
        DataFrame with `positive` and `negative` components.
    """
    values = pd.Series(series, copy=False)
    positive = values.clip(lower=0.0)
    negative = (-values.clip(upper=0.0)) if absolute_negative else values.clip(upper=0.0)
    return pd.DataFrame({"positive": positive, "negative": negative}, index=values.index)


def identify_boom_periods(
    data: pd.DataFrame,
    value_col: str,
    unit_col: str = "unit",
    time_col: str = "date",
    window: int = 20,
    quantile: float = 0.90,
    min_duration: int = 6,
    state_col: str = "boom",
) -> pd.DataFrame:
    """
    Construct a commodity-boom state using a BCRP-style rolling-threshold rule.

    A period is tagged as a boom candidate when the current observation exceeds
    the rolling `quantile` computed over the previous `window` observations. A
    candidate spell becomes an effective boom only if it lasts at least
    `min_duration` consecutive periods.
    """
    if unit_col not in data.columns or time_col not in data.columns or value_col not in data.columns:
        raise ValueError("`data` must contain the unit, time, and value columns.")

    df = data[[unit_col, time_col, value_col]].copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([unit_col, time_col]).reset_index(drop=True)

    output = []
    for unit, unit_df in df.groupby(unit_col, sort=False):
        unit_df = unit_df.copy()
        rolling_q = (
            unit_df[value_col]
            .shift(1)
            .rolling(window=window, min_periods=window)
            .quantile(quantile)
        )
        candidate = (unit_df[value_col] > rolling_q) & rolling_q.notna()
        spell_id = candidate.ne(candidate.shift(fill_value=False)).cumsum()
        spell_length = candidate.groupby(spell_id).transform("sum")
        unit_df["rolling_threshold"] = rolling_q
        unit_df["boom_candidate"] = candidate.astype(int)
        unit_df[state_col] = (candidate & (spell_length >= min_duration)).astype(int)
        output.append(unit_df)

    return pd.concat(output, ignore_index=True)


def cumulative_irf_ratio(
    numerator_results: pd.DataFrame,
    denominator_results: pd.DataFrame,
    numerator_shock: str,
    denominator_shock: str,
    horizons=(4, 8, 20),
) -> pd.DataFrame:
    """
    Compute cumulative IRF ratios between two LP result tables.

    The convention is `H4 = 0..3`, `H8 = 0..7`, etc.
    """
    records = []
    for horizon in horizons:
        num_mask = (
            (numerator_results["shock"] == numerator_shock)
            & (numerator_results["horizon"] < horizon)
        )
        den_mask = (
            (denominator_results["shock"] == denominator_shock)
            & (denominator_results["horizon"] < horizon)
        )
        numerator = numerator_results.loc[num_mask, "estimate"].sum()
        denominator = denominator_results.loc[den_mask, "estimate"].sum()
        ratio = np.nan if np.isclose(denominator, 0.0) else numerator / denominator
        records.append({"horizon": horizon, "ratio": ratio})
    return pd.DataFrame(records)


class PanelLocalProjections:
    """
    Panel local projections with unit fixed effects and horizon-specific samples.

    This estimator is designed for macro panels with ragged edges. For each
    horizon `h`, it stacks all unit-time observations where both `y_{u,t+h}` and
    the contemporaneous / lagged regressors at `t` are observed.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        response: str,
        shock: str,
        controls=None,
        unit_col: str = "unit",
        time_col: str = "date",
        additional_vars=None,
        lags: int = 4,
        horizons=None,
        fixed_effects: bool = True,
        time_effects: bool = False,
        add_constant: bool = True,
        cluster: str = "unit",
        allow_unbalanced: bool = True,
    ):
        if lags < 1:
            raise ValueError("`lags` must be at least 1.")

        self.response = response
        self.shock = shock
        self.controls = [
            column
            for column in _unique_preserve_order(list(controls or []))
            if column not in {response, shock}
        ]
        self.additional_vars = [
            column
            for column in _unique_preserve_order(list(additional_vars or []))
            if column not in {response, shock, *self.controls}
        ]
        self.unit_col = unit_col
        self.time_col = time_col
        self.lags = int(lags)
        self.horizons = list(range(0, 21)) if horizons is None else list(horizons)
        self.fixed_effects = bool(fixed_effects)
        self.time_effects = bool(time_effects)
        self.add_constant = bool(add_constant)
        self.cluster = cluster
        self.allow_unbalanced = bool(allow_unbalanced)
        self.model_type = "Panel Local Projections"

        panel_vars = _unique_preserve_order([response, shock, *self.controls])
        prepared = prepare_panel_data(
            data=data,
            unit_col=unit_col,
            time_col=time_col,
            endog=panel_vars,
            allow_unbalanced=allow_unbalanced,
        )

        source_data = data.copy()
        source_data[time_col] = pd.to_datetime(source_data[time_col])
        source_data = source_data.sort_values([unit_col, time_col]).reset_index(drop=True)

        self.units = prepared["units"]
        self.unit_dates = prepared["unit_dates"]
        self.unit_sample = prepared["unit_sample"].copy()
        self.is_balanced_panel = bool(prepared["balanced"])
        self.panel_balance = "Balanced" if self.is_balanced_panel else "Unbalanced"
        self.available_variables = _unique_preserve_order([*panel_vars, *self.additional_vars])
        self.n_units = len(self.units)

        self.unit_frames = {}
        for unit, frame in zip(self.units, prepared["unit_frames"]):
            ordered = frame.copy().sort_values(time_col).reset_index(drop=True)
            ordered[time_col] = pd.to_datetime(ordered[time_col])
            ordered.attrs = {}
            if self.additional_vars:
                extra = (
                    source_data.loc[source_data[unit_col] == unit, [time_col, *self.additional_vars]]
                    .drop_duplicates(subset=[time_col])
                    .copy()
                )
                extra[time_col] = pd.to_datetime(extra[time_col])
                extra.attrs = {}
                ordered = ordered.merge(extra, on=time_col, how="left")
                ordered.attrs = {}
            self.unit_frames[unit] = ordered

        self.data = pd.concat(self.unit_frames.values(), ignore_index=True)

        self.results = None
        self.results_by_spec = {}
        self.horizon_sample_sizes = None
        self.design_columns = {}

    def model_summary(self):
        sample_start = min(dates.min() for dates in self.unit_dates).date()
        sample_end = max(dates.max() for dates in self.unit_dates).date()
        average_obs = self.unit_sample["observations"].mean()
        summary = f"""
**MacroPy Panel Local Projections**

- **Response**: {self.response}
- **Baseline Shock**: {self.shock}
- **Lag Controls**: {', '.join([self.response, self.shock, *self.controls])}
- **Horizons**: {min(self.horizons)} to {max(self.horizons)}
- **Country Fixed Effects**: {'Yes' if self.fixed_effects else 'No'}
- **Time Fixed Effects**: {'Yes' if self.time_effects else 'No'}
- **Covariance Estimator**: {'Clustered by unit' if self.cluster == 'unit' else 'HC1'}
- **Panel Structure**: {self.panel_balance}
- **Units**: {', '.join(map(str, self.units))}
- **Sample Window**: {sample_start} to {sample_end}
- **Average Observations per Unit**: {average_obs:.1f}
"""
        display(Markdown(summary))

    def sample_overview(self) -> pd.DataFrame:
        return self.unit_sample.copy()

    def horizon_overview(self) -> pd.DataFrame:
        if self.horizon_sample_sizes is None:
            raise ValueError("Run one estimation first to populate horizon-specific sample sizes.")
        return self.horizon_sample_sizes.copy()

    @staticmethod
    def _safe_se(covariance, position):
        variance = float(np.clip(covariance[position, position], 0.0, np.inf))
        return np.sqrt(variance)

    @staticmethod
    def _r_squared(y, fitted):
        centered = y - y.mean()
        denominator = float(centered @ centered)
        if np.isclose(denominator, 0.0):
            return np.nan
        residual = y - fitted
        return 1.0 - float(residual @ residual) / denominator

    def _frame_lookup_with_columns(self, extra_columns=None):
        extra_columns = extra_columns or {}
        lookup = {}
        for unit, frame in self.unit_frames.items():
            enriched = frame.copy()
            for name, values in extra_columns.items():
                if callable(values):
                    enriched[name] = values(enriched)
                else:
                    if isinstance(values, dict):
                        unit_values = values[unit]
                    else:
                        unit_values = values
                    enriched[name] = unit_values
            lookup[unit] = enriched
        return lookup

    def _build_horizon_sample(self, horizon, shock_cols, lag_source_vars, frame_lookup):
        records = []
        counts = []
        for unit in self.units:
            frame = frame_lookup[unit].copy().sort_values(self.time_col).reset_index(drop=True)
            usable = 0
            max_t = len(frame) - horizon
            for t in range(self.lags, max_t):
                row = {
                    self.unit_col: unit,
                    self.time_col: frame.at[t, self.time_col],
                    "horizon": horizon,
                    "y": frame.at[t + horizon, self.response],
                }
                if pd.isna(row["y"]):
                    continue

                missing = False
                for shock_col in shock_cols:
                    shock_value = frame.at[t, shock_col]
                    if pd.isna(shock_value):
                        missing = True
                        break
                    row[shock_col] = float(shock_value)
                if missing:
                    continue

                for variable in lag_source_vars:
                    for lag in range(1, self.lags + 1):
                        value = frame.at[t - lag, variable]
                        if pd.isna(value):
                            missing = True
                            break
                        row[f"{variable}_lag{lag}"] = float(value)
                    if missing:
                        break

                if missing:
                    continue

                records.append(row)
                usable += 1

            counts.append({self.unit_col: unit, "horizon": horizon, "nobs": usable})

        sample = pd.DataFrame.from_records(records)
        return sample, pd.DataFrame(counts)

    def _build_design_matrix(self, sample, shock_cols, lag_source_vars):
        parts = []
        column_order = []

        if self.add_constant:
            const = pd.DataFrame({"const": np.ones(len(sample), dtype=float)}, index=sample.index)
            parts.append(const)
            column_order.extend(const.columns.tolist())

        shock_block = sample[shock_cols].astype(float)
        parts.append(shock_block)
        column_order.extend(shock_cols)

        lag_columns = []
        for variable in lag_source_vars:
            for lag in range(1, self.lags + 1):
                lag_columns.append(f"{variable}_lag{lag}")
        lag_block = sample[lag_columns].astype(float)
        parts.append(lag_block)
        column_order.extend(lag_columns)

        if self.fixed_effects:
            unit_dummies = pd.get_dummies(
                sample[self.unit_col].astype(str),
                prefix="unit",
                drop_first=True,
                dtype=float,
            )
            if unit_dummies.shape[1] > 0:
                parts.append(unit_dummies)
                column_order.extend(unit_dummies.columns.tolist())

        if self.time_effects:
            time_dummies = pd.get_dummies(
                sample[self.time_col].astype(str),
                prefix="time",
                drop_first=True,
                dtype=float,
            )
            if time_dummies.shape[1] > 0:
                parts.append(time_dummies)
                column_order.extend(time_dummies.columns.tolist())

        design = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=sample.index)
        return design.astype(float), column_order

    def _covariance(self, X, residuals, clusters):
        nobs, nparams = X.shape
        rank = np.linalg.matrix_rank(X)
        xtx_inv = np.linalg.pinv(X.T @ X)

        use_cluster = clusters is not None and len(pd.unique(clusters)) > 1
        if use_cluster:
            meat = np.zeros((nparams, nparams))
            cluster_values = pd.Series(clusters).to_numpy()
            for cluster in pd.unique(cluster_values):
                mask = cluster_values == cluster
                xg = X[mask, :]
                ug = residuals[mask].reshape(-1, 1)
                score = xg.T @ ug
                meat += score @ score.T

            covariance = xtx_inv @ meat @ xtx_inv
            n_clusters = len(pd.unique(cluster_values))
            denominator = max(nobs - rank, 1)
            correction = (n_clusters / (n_clusters - 1)) * ((nobs - 1) / denominator)
            covariance *= correction
            return covariance, int(n_clusters), "cluster"

        xu = X * residuals.reshape(-1, 1)
        meat = xu.T @ xu
        covariance = xtx_inv @ meat @ xtx_inv
        denominator = max(nobs - rank, 1)
        covariance *= nobs / denominator
        return covariance, 1 if clusters is None else int(len(pd.unique(clusters))), "HC1"

    def _estimate_horizon(self, sample, shock_cols, lag_source_vars):
        X_df, column_order = self._build_design_matrix(sample, shock_cols, lag_source_vars)
        X = X_df.to_numpy(dtype=float)
        y = sample["y"].to_numpy(dtype=float)
        coefficients, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        fitted = X @ coefficients
        residuals = y - fitted

        clusters = sample[self.unit_col].to_numpy() if self.cluster == "unit" else None
        covariance, n_clusters, cov_label = self._covariance(X, residuals, clusters)
        return {
            "coefficients": coefficients,
            "covariance": covariance,
            "fitted": fitted,
            "residuals": residuals,
            "design_columns": column_order,
            "nobs": len(sample),
            "n_units": sample[self.unit_col].nunique(),
            "n_clusters": n_clusters,
            "covariance_type": cov_label,
            "r_squared": self._r_squared(y, fitted),
        }

    def _estimate_projection(
        self,
        shock_cols,
        lag_source_vars,
        shock_labels=None,
        shock_scale: float = 1.0,
        conf_levels=(0.68, 0.95),
        spec_name: str = "baseline",
        frame_lookup=None,
    ):
        if frame_lookup is None:
            frame_lookup = self.unit_frames

        shock_cols = list(shock_cols)
        lag_source_vars = _unique_preserve_order(lag_source_vars)
        shock_labels = list(shock_labels or shock_cols)
        if len(shock_labels) != len(shock_cols):
            raise ValueError("`shock_labels` must match the number of `shock_cols`.")

        all_results = []
        all_counts = []
        design_lookup = {}

        for horizon in self.horizons:
            sample, counts = self._build_horizon_sample(
                horizon=horizon,
                shock_cols=shock_cols,
                lag_source_vars=lag_source_vars,
                frame_lookup=frame_lookup,
            )
            counts["spec"] = spec_name
            all_counts.append(counts)
            if sample.empty:
                continue

            estimate = self._estimate_horizon(
                sample=sample,
                shock_cols=shock_cols,
                lag_source_vars=lag_source_vars,
            )
            design_lookup[horizon] = estimate["design_columns"]
            column_positions = {
                name: estimate["design_columns"].index(name)
                for name in shock_cols
            }

            for shock_col, shock_label in zip(shock_cols, shock_labels):
                position = column_positions[shock_col]
                beta_raw = float(estimate["coefficients"][position])
                se_raw = self._safe_se(estimate["covariance"], position)
                row = {
                    "spec": spec_name,
                    "horizon": horizon,
                    "shock": shock_label,
                    "shock_column": shock_col,
                    "estimate_raw": beta_raw,
                    "std_error_raw": se_raw,
                    "estimate": beta_raw * shock_scale,
                    "std_error": se_raw * shock_scale,
                    "shock_scale": shock_scale,
                    "nobs": estimate["nobs"],
                    "n_units": estimate["n_units"],
                    "n_clusters": estimate["n_clusters"],
                    "covariance_type": estimate["covariance_type"],
                    "r_squared": estimate["r_squared"],
                }
                for level in conf_levels:
                    suffix = int(round(level * 100))
                    z_value = norm.ppf(0.5 + level / 2.0)
                    row[f"lower_{suffix}"] = row["estimate"] - z_value * row["std_error"]
                    row[f"upper_{suffix}"] = row["estimate"] + z_value * row["std_error"]
                all_results.append(row)

        result_df = pd.DataFrame(all_results).sort_values(["shock", "horizon"]).reset_index(drop=True)
        count_df = pd.concat(all_counts, ignore_index=True) if all_counts else pd.DataFrame()
        return result_df, count_df, design_lookup

    def fit(
        self,
        shock_col: str = None,
        shock_label: str = None,
        shock_scale: float = 1.0,
        conf_levels=(0.68, 0.95),
        spec_name: str = "baseline",
    ) -> pd.DataFrame:
        shock_col = shock_col or self.shock
        if shock_col not in self.available_variables:
            raise KeyError(f"Shock `{shock_col}` is not available in this model instance.")

        lag_source_vars = _unique_preserve_order([self.response, shock_col, *self.controls])
        results, counts, design_lookup = self._estimate_projection(
            shock_cols=[shock_col],
            shock_labels=[shock_label or shock_col],
            lag_source_vars=lag_source_vars,
            shock_scale=shock_scale,
            conf_levels=conf_levels,
            spec_name=spec_name,
        )

        self.results = results
        self.results_by_spec[spec_name] = results
        self.horizon_sample_sizes = counts
        self.design_columns[spec_name] = design_lookup
        return results

    def fit_multiple(
        self,
        shock_cols,
        shock_labels=None,
        lag_source_vars=None,
        shock_scale: float = 1.0,
        conf_levels=(0.68, 0.95),
        spec_name: str = "multiple",
        frame_lookup=None,
    ) -> pd.DataFrame:
        shock_cols = list(shock_cols)
        if not shock_cols:
            raise ValueError("`shock_cols` must contain at least one shock column.")
        for shock_col in shock_cols:
            if shock_col not in self.available_variables and frame_lookup is None:
                raise KeyError(f"Shock `{shock_col}` is not available in this model instance.")

        lag_source_vars = lag_source_vars or [self.response, self.shock, *self.controls]
        lag_source_vars = _unique_preserve_order(lag_source_vars)

        results, counts, design_lookup = self._estimate_projection(
            shock_cols=shock_cols,
            shock_labels=shock_labels or shock_cols,
            lag_source_vars=lag_source_vars,
            shock_scale=shock_scale,
            conf_levels=conf_levels,
            spec_name=spec_name,
            frame_lookup=frame_lookup,
        )

        self.results_by_spec[spec_name] = results
        self.horizon_sample_sizes = counts
        self.design_columns[spec_name] = design_lookup
        return results

    def fit_asymmetric(
        self,
        shock_col: str = None,
        shock_scale: float = 1.0,
        absolute_negative: bool = True,
        conf_levels=(0.68, 0.95),
        spec_name: str = "asymmetric",
    ) -> pd.DataFrame:
        shock_col = shock_col or self.shock
        if shock_col not in self.available_variables:
            raise KeyError(f"Shock `{shock_col}` is not available in this model instance.")

        positive_col = f"{shock_col}_positive"
        negative_col = f"{shock_col}_negative"
        frame_lookup = {}
        for unit, frame in self.unit_frames.items():
            enriched = frame.copy()
            split = split_shock_signs(enriched[shock_col], absolute_negative=absolute_negative)
            enriched[positive_col] = split["positive"].to_numpy()
            enriched[negative_col] = split["negative"].to_numpy()
            frame_lookup[unit] = enriched

        lag_source_vars = _unique_preserve_order([self.response, shock_col, *self.controls])
        results, counts, design_lookup = self._estimate_projection(
            shock_cols=[positive_col, negative_col],
            shock_labels=[f"{shock_col}_positive", f"{shock_col}_negative"],
            lag_source_vars=lag_source_vars,
            shock_scale=shock_scale,
            conf_levels=conf_levels,
            spec_name=spec_name,
            frame_lookup=frame_lookup,
        )

        self.results_by_spec[spec_name] = results
        self.horizon_sample_sizes = counts
        self.design_columns[spec_name] = design_lookup
        return results

    def fit_state_dependent(
        self,
        state_col: str,
        shock_col: str = None,
        shock_scale: float = 1.0,
        conf_levels=(0.68, 0.95),
        spec_name: str = "state_dependent",
        state_labels=("boom", "nonboom"),
    ) -> pd.DataFrame:
        shock_col = shock_col or self.shock
        active_col = f"{shock_col}_{state_labels[0]}"
        inactive_col = f"{shock_col}_{state_labels[1]}"
        frame_lookup = {}

        for unit, frame in self.unit_frames.items():
            if state_col not in frame.columns:
                raise KeyError(
                    f"State column `{state_col}` is not available. "
                    "Instantiate the model with this column included in `additional_vars`."
                )
            enriched = frame.copy()
            state = enriched[state_col].astype(float)
            enriched[active_col] = enriched[shock_col] * state
            enriched[inactive_col] = enriched[shock_col] * (1.0 - state)
            frame_lookup[unit] = enriched

        lag_source_vars = _unique_preserve_order([self.response, shock_col, *self.controls])
        results, counts, design_lookup = self._estimate_projection(
            shock_cols=[active_col, inactive_col],
            shock_labels=[active_col, inactive_col],
            lag_source_vars=lag_source_vars,
            shock_scale=shock_scale,
            conf_levels=conf_levels,
            spec_name=spec_name,
            frame_lookup=frame_lookup,
        )

        self.results_by_spec[spec_name] = results
        self.horizon_sample_sizes = counts
        self.design_columns[spec_name] = design_lookup
        return results

    def fit_partitioned_shock(
        self,
        state_cols,
        shock_col: str = None,
        shock_scale: float = 1.0,
        conf_levels=(0.68, 0.95),
        spec_name: str = "partitioned_shock",
        residual_label: str | None = "other",
    ) -> pd.DataFrame:
        shock_col = shock_col or self.shock
        state_items = list(state_cols.items()) if isinstance(state_cols, dict) else list(state_cols)
        if not state_items:
            raise ValueError("`state_cols` must contain at least one state.")

        frame_lookup = {}
        interaction_cols = []
        interaction_labels = []

        for unit, frame in self.unit_frames.items():
            enriched = frame.copy()
            state_sum = np.zeros(len(enriched), dtype=float)
            for label, state_col in state_items:
                if state_col not in enriched.columns:
                    raise KeyError(
                        f"State column `{state_col}` is not available. "
                        "Instantiate the model with this column included in `additional_vars`."
                    )
                state = enriched[state_col].astype(float).fillna(0.0).to_numpy()
                state_sum += state
                interaction_col = f"{shock_col}_{label}"
                enriched[interaction_col] = enriched[shock_col].to_numpy(dtype=float) * state
                interaction_cols.append(interaction_col)
                interaction_labels.append(interaction_col)

            if np.any(state_sum > 1.0 + 1e-8):
                raise ValueError("State columns in `fit_partitioned_shock` must be mutually exclusive.")

            if residual_label is not None:
                residual_state = np.clip(1.0 - state_sum, 0.0, 1.0)
                residual_col = f"{shock_col}_{residual_label}"
                enriched[residual_col] = enriched[shock_col].to_numpy(dtype=float) * residual_state
                interaction_cols.append(residual_col)
                interaction_labels.append(residual_col)

            frame_lookup[unit] = enriched

        lag_source_vars = _unique_preserve_order([self.response, shock_col, *self.controls])
        results, counts, design_lookup = self._estimate_projection(
            shock_cols=interaction_cols,
            shock_labels=interaction_labels,
            lag_source_vars=lag_source_vars,
            shock_scale=shock_scale,
            conf_levels=conf_levels,
            spec_name=spec_name,
            frame_lookup=frame_lookup,
        )

        self.results_by_spec[spec_name] = results
        self.horizon_sample_sizes = counts
        self.design_columns[spec_name] = design_lookup
        return results

    def plot_irf(
        self,
        results: pd.DataFrame = None,
        shocks=None,
        credible_band: float = 0.68,
        ax=None,
        colors=None,
        title: str = None,
        ylabel: str = "Response",
    ):
        if results is None:
            if self.results is None:
                raise ValueError("Run `fit()` first or pass a results table explicitly.")
            results = self.results

        shocks = list(shocks or results["shock"].drop_duplicates().tolist())
        suffix = int(round(credible_band * 100))
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))

        if colors is None:
            colors = {shock: None for shock in shocks}

        for shock_name in shocks:
            subset = results.loc[results["shock"] == shock_name].sort_values("horizon")
            color = colors.get(shock_name)
            ax.plot(subset["horizon"], subset["estimate"], label=shock_name, color=color, linewidth=2.0)
            lower_col = f"lower_{suffix}"
            upper_col = f"upper_{suffix}"
            if lower_col in subset.columns and upper_col in subset.columns:
                ax.fill_between(
                    subset["horizon"],
                    subset[lower_col],
                    subset[upper_col],
                    color=color,
                    alpha=0.18,
                )

        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
        ax.set_xlabel("Horizon")
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.legend(frameon=False)
        return ax
        
    def cross_validate_lambda(self, lambdas: list):
        """
        Generalized Cross-Validation (GCV) to choose optimal lambda for smoothing.

        Parameters:
        - lambdas: List of lambda values to test

        Stores:
        - self.cv_rss: Array of RSS for each lambda
        - self.cv_lambda_opt: Lambda minimizing the RSS
        """
        if not hasattr(self, "result"):
            raise ValueError("Run `.estimate()` once with a non-zero penalty to initialize B and P.")

        Y = self.result["Y"]
        X = self.result["X"]
        P = self.result["P"]

        X = X.toarray() if sparse.issparse(X) else np.asarray(X)
        P = np.asarray(P, dtype=np.float64)

        rss_list = []
        for lam in lambdas:
            XtX = X.T @ X + lam * P
            XtY = X.T
            H = X @ solve(XtX, XtY)
            h_diag = np.diagonal(H)
            residuals = (Y - H @ Y) / (1 - h_diag)
            rss = np.sum(residuals ** 2)
            rss_list.append(rss)

        self.cv_rss = np.array(rss_list)
        self.cv_lambdas = np.array(lambdas)
        self.cv_lambda_opt = lambdas[np.argmin(rss_list)]
        
        # Step 3: Plot RSS vs lambda
        plt.plot(self.cv_lambdas, self.cv_rss, '-o')
        plt.title("Cross-Validated RSS for Lambda")
        plt.xlabel("Lambda")
        plt.ylabel("RSS")
        plt.grid(True)
        plt.show()

        return self.cv_lambda_opt
    

    def plot_irf_with_confidence(self, title="Impulse Response with Confidence Intervals", alpha=0.1):
        """
        Plot the impulse response function with confidence intervals.
        
        Parameters:
        - title: Title of the plot
        - alpha: Significance level (default: 0.05 for 95% CI)
        """
        if self.result is None or "confidence_intervals" not in self.result:
            raise ValueError("Run `.estimate()` and `.compute_confidence_intervals()` first.")

        IR = self.result["IR"]
        CI = self.result["confidence_intervals"]

        H = list(range(self.h_max + 1))

        plt.figure(figsize=(8, 5))
        plt.plot(H, IR, label="IRF", color="blue", linewidth=2)
        plt.fill_between(H, CI[:, 0], CI[:, 1], color="blue", alpha=0.2, label=f"{int((1-alpha)*100)}% CI")
        plt.axhline(0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Horizon")
        plt.ylabel("Response")
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
