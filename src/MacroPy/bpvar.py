import numpy as np
import pandas as pd
from types import SimpleNamespace
from tqdm import tqdm
from typing import Dict, List, Optional, Sequence, Union
from IPython.display import display
from numpy.linalg import eigvals
from scipy.stats import invwishart

from .data_handling import prepare_panel_data, prepare_panel_unit_data
from .plots import (
    generate_series_plot,
    generate_irf_plots,
    generate_forecast_plots,
    generate_fevd_plot,
    generate_panel_coeff_plots,
)
from .priors import HierarchicalPanelPrior, DiffusePanelExogenousPrior
from .summary import generate_summary


class BayesianPanelVAR:
    def __init__(
        self,
        data: pd.DataFrame,
        endog: Sequence[str],
        unit_col: str = "unit",
        time_col: str = "date",
        exog: Optional[Sequence[str]] = None,
        lags: int = 1,
        constant: bool = True,
        timetrend: bool = False,
        prior_params: Optional[dict] = None,
        post_draws: int = 5000,
        burnin: float = 0.5,
        hor: int = 20,
        fhor: int = 12,
        irf_1std: int = 1,
        max_tries: int = 100,
        allow_unbalanced: bool = False,
    ):
        """
        Hierarchical Bayesian Panel VAR with unit-specific dynamics and pooled lag coefficients.

        Parameters
        ----------
        data : pd.DataFrame
            Long-format panel data.
        endog : Sequence[str]
            Endogenous variable names.
        unit_col : str, default="unit"
            Column identifying the cross-sectional unit.
        time_col : str, default="date"
            Column identifying the time dimension.
        exog : Sequence[str], optional
            Common exogenous variables shared by all units at each point in time.
        lags : int, default=1
            Number of VAR lags.
        constant : bool, default=True
            Whether to include an intercept in each unit equation.
        timetrend : bool, default=False
            Whether to include a deterministic linear time trend.
        prior_params : dict, optional
            Prior hyperparameters. Defaults follow Mumtaz's MATLAB implementation.
        post_draws : int, default=5000
            Total Gibbs draws, including burn-in.
        burnin : float, default=0.5
            Fraction of draws discarded as burn-in.
        hor : int, default=20
            Impulse response and FEVD horizon.
        fhor : int, default=12
            Forecast horizon.
        irf_1std : int, default=1
            Use a one-standard-deviation shock if 1, or a unit structural shock if 0.
        max_tries : int, default=100
            Maximum stability draws for unit coefficients and pooled means.
        allow_unbalanced : bool, default=False
            If True, allow staggered country samples and trim leading/trailing
            missing observations for each unit. Internal missing observations
            inside a country sample are still rejected.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("`data` must be a pandas DataFrame.")
        if lags < 1:
            raise ValueError("`lags` must be at least 1.")
        if not 0 <= burnin < 1:
            raise ValueError("`burnin` must be in the [0, 1) interval.")
        if irf_1std not in {0, 1}:
            raise ValueError("`irf_1std` must be either 0 or 1.")

        default_prior_params = {
            "tightness": 1.0,
            "exo_precision": 1e-4,
            "lambda_scale": 0.0,
            "lambda_df": -1.0,
            "sigma_scale": 1.0,
            "sigma_df": None,
            "lambda_init": 1.0,
        }
        if prior_params is not None:
            default_prior_params.update(prior_params)

        prepared = prepare_panel_data(
            data=data,
            unit_col=unit_col,
            time_col=time_col,
            endog=list(endog),
            exog=list(exog or []),
            allow_unbalanced=allow_unbalanced,
        )

        self.data = prepared["frame"]
        self.units = prepared["units"]
        self.unit_dates = prepared["unit_dates"]
        self.dates = prepared["dates"]
        self.names = prepared["endog_names"]
        self.exo_names = prepared["exo_names"]
        self.y_panel = [np.asarray(y_unit, dtype=float) for y_unit in prepared["y"]]
        self.common_exog = prepared["exo"]
        self.common_exog_units = [
            np.asarray(exo_unit, dtype=float) for exo_unit in prepared["exo_units"]
        ]
        self.is_balanced_panel = bool(prepared["balanced"])
        self.panel_balance = "Balanced" if self.is_balanced_panel else "Unbalanced"
        self.model_type = "Bayesian Panel VAR"
        self.prior_name = "Hierarchical Minnesota Pooling"

        self.unit_col = unit_col
        self.time_col = time_col
        self.lags = lags
        self.constant = constant
        self.timetrend = timetrend
        self.prior_params = default_prior_params
        self.post_draws = int(post_draws)
        self.burnin = int(burnin * post_draws)
        self.hor = hor
        self.fhor = fhor
        self.irf_1std = irf_1std
        self.max_tries = max_tries
        self.allow_unbalanced = allow_unbalanced

        self.n_units = len(self.y_panel)
        self.n_endo = self.y_panel[0].shape[1]
        self.T_by_unit = np.asarray([y_unit.shape[0] for y_unit in self.y_panel], dtype=int)
        self.T = int(self.T_by_unit.max())
        self.min_T = int(self.T_by_unit.min())
        self.max_T = int(self.T_by_unit.max())
        self.n_common_exo = self.common_exog.shape[1]
        self.nobs_by_unit = self.T_by_unit - self.lags
        if np.any(self.nobs_by_unit <= 0):
            raise ValueError("Every panel unit must have more observations than the requested lag length.")
        self.nobs = int(self.nobs_by_unit[0]) if self.is_balanced_panel else None
        self.min_nobs = int(self.nobs_by_unit.min())
        self.max_nobs = int(self.nobs_by_unit.max())
        self.unit_weights = self.nobs_by_unit / self.nobs_by_unit.sum()
        self.yy_dates_units = [dates_u[self.lags:] for dates_u in self.unit_dates]
        self.yy_dates = self.yy_dates_units[0] if self.is_balanced_panel else None
        self.unit_sample_table = pd.DataFrame(
            {
                self.unit_col: self.units,
                "sample_start": [dates_u[0] for dates_u in self.unit_dates],
                "sample_end": [dates_u[-1] for dates_u in self.unit_dates],
                "observations": self.T_by_unit,
                "estimation_start": [dates_u[self.lags] for dates_u in self.unit_dates],
                "estimation_end": [dates_u[-1] for dates_u in self.unit_dates],
                "estimation_observations": self.nobs_by_unit,
            }
        )

        self.n_exo = int(self.constant) + int(self.timetrend) + self.n_common_exo
        self.n_lag_coeff_eq = self.n_endo * self.lags
        self.ncoeff_eq = self.n_lag_coeff_eq + self.n_exo
        self.k_beta = self.n_lag_coeff_eq * self.n_endo
        self.k_exo_total = self.n_exo * self.n_endo
        self.ncoeff = self.n_units * (self.k_beta + self.k_exo_total) + self.k_beta + 1

        sigma_df = self.prior_params["sigma_df"]
        self.sigma_prior_df = int(sigma_df) if sigma_df is not None else self.n_endo + 1
        sigma_scale = self.prior_params["sigma_scale"]
        self.sigma_prior_scale = (
            np.eye(self.n_endo) * float(sigma_scale)
            if np.isscalar(sigma_scale)
            else np.asarray(sigma_scale, dtype=float)
        )
        if self.sigma_prior_scale.shape != (self.n_endo, self.n_endo):
            raise ValueError("`sigma_scale` must be a scalar or an (n_endo x n_endo) matrix.")

        self.exo_prior = DiffusePanelExogenousPrior(
            n_exo=self.n_exo,
            n_endo=self.n_endo,
            precision=self.prior_params["exo_precision"],
        )

        self.yy_units = []
        self.X_lag_units = []
        self.Z_units = []
        self.X_units = []
        beta_ols = []
        c_ols = []
        sigma_ols = []
        prior_means = []
        prior_covs = []
        prior_precisions = []

        for unit_idx in range(self.n_units):
            yy, x_lag, z, x_full = prepare_panel_unit_data(
                y_unit=self.y_panel[unit_idx],
                lags=self.lags,
                exog=self.common_exog_units[unit_idx],
                constant=self.constant,
                timetrend=self.timetrend,
            )
            b_ols = np.linalg.lstsq(x_full, yy, rcond=None)[0]
            resid = yy - x_full @ b_ols
            denom = yy.shape[0] - x_full.shape[1]
            if denom <= 0:
                raise ValueError("Too few observations per unit to estimate the requested model.")

            self.yy_units.append(yy)
            self.X_lag_units.append(x_lag)
            self.Z_units.append(z)
            self.X_units.append(x_full)
            beta_ols.append(b_ols[: self.n_lag_coeff_eq, :].flatten(order="F"))
            c_ols.append(b_ols[self.n_lag_coeff_eq :, :].flatten(order="F"))
            sigma_ols.append((resid.T @ resid) / denom)

            prior = HierarchicalPanelPrior(
                y_unit=self.y_panel[unit_idx],
                lags=self.lags,
                tightness=self.prior_params["tightness"],
            )
            prior_means.append(prior["b0"])
            prior_covs.append(prior["H"])
            prior_precisions.append(prior["H_inv"])

        self.beta_ols = np.asarray(beta_ols)
        self.c_ols = np.asarray(c_ols)
        self.Sigma_ols = np.asarray(sigma_ols)
        self.prior_mean_units = np.asarray(prior_means)
        self.prior_cov_units = np.asarray(prior_covs)
        self.prior_precision_units = np.asarray(prior_precisions)

        lambda_df = self.prior_params["lambda_df"] + self.n_units * self.k_beta
        if lambda_df <= 0:
            raise ValueError("The implied degrees of freedom for lambda must be strictly positive.")

        self.beta_draws = []
        self.c_draws = []
        self.Sigma_draws = []
        self.bbar_draws = []
        self.lambda_draws = []
        self.ir_draws = None
        self.ir_draws_pooled = None
        self.fevd = None
        self.fevd_draws = None
        self.fevd_pooled = None
        self.fevd_draws_pooled = None
        self.forecasts = None
        self.mean_forecasts = None
        self.forecast_dates = None
        self.forecast_dates_by_unit = None
        self.failed_stability_draws = {"unit_beta": 0, "bbar": 0}

    def model_summary(self):
        """Print a summary of the Bayesian panel VAR model."""
        display(generate_summary(self))

    def sample_overview(self) -> pd.DataFrame:
        """Return unit-specific sample coverage after trimming and lag construction."""
        return self.unit_sample_table.copy()

    def plot_unit_series(
        self,
        unit: Union[int, str],
        series_titles: Optional[List[str]] = None,
        title: Optional[str] = None,
        color_scheme: int = 1,
        n_breaks: int = 10,
        zero_line: bool = False,
    ):
        """Plot the transformed endogenous variables for one panel unit."""
        unit_idx = self._resolve_unit(unit)
        plot = generate_series_plot(
            self.y_panel[unit_idx],
            yy_dates=self.unit_dates[unit_idx],
            yy_names=self.names,
            series_titles=series_titles,
            title=title or f"Panel Series: {self.units[unit_idx]}",
            color_scheme=color_scheme,
            n_breaks=n_breaks,
            zero_line=zero_line,
        )
        return display(plot)

    @staticmethod
    def _inverse(matrix):
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(matrix)

    @staticmethod
    def _safe_cholesky(matrix):
        try:
            return np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            eigenvalues = np.clip(eigenvalues, 1e-12, None)
            return eigenvectors @ np.diag(np.sqrt(eigenvalues))

    @staticmethod
    def reshape_beta(beta_vec, ncoeff_eq, n_endo):
        """Reshape a vectorized coefficient block into a coefficient matrix."""
        return beta_vec.reshape((ncoeff_eq, n_endo), order="F")

    @staticmethod
    def build_companion_matrix(B_lag, n_endo, lags):
        """Construct the VAR companion matrix from lag coefficients."""
        companion = np.zeros((n_endo * lags, n_endo * lags))
        companion[:n_endo, :] = B_lag.T
        if lags > 1:
            companion[n_endo:, :-n_endo] = np.eye(n_endo * (lags - 1))
        return companion

    @staticmethod
    def is_stable(companion):
        """Check VAR stability based on the companion matrix eigenvalues."""
        return np.all(np.abs(eigvals(companion)) < 0.9999)

    def _resolve_unit(self, unit):
        if unit is None:
            return 0
        if isinstance(unit, int):
            if not 0 <= unit < self.n_units:
                raise IndexError("Unit index out of range.")
            return unit
        for idx, label in enumerate(self.units):
            if unit == label or str(unit) == str(label):
                return idx
        raise KeyError(f"Unknown panel unit: {unit}")

    @staticmethod
    def _is_pooled_unit(unit):
        return isinstance(unit, str) and unit.lower() == "all"

    def _combine_coefficients(self, beta_vec, c_vec):
        beta_block = self.reshape_beta(beta_vec, self.n_lag_coeff_eq, self.n_endo)
        if self.n_exo == 0:
            return beta_block
        c_block = self.reshape_beta(c_vec, self.n_exo, self.n_endo)
        return np.vstack((beta_block, c_block))

    def _pooled_sigma_draws(self):
        weights = self.unit_weights.reshape(1, self.n_units, 1, 1)
        return np.sum(self.Sigma_draws * weights, axis=1)

    def _compute_single_irf(self, beta_mat, sigma):
        companion = self.build_companion_matrix(beta_mat, self.n_endo, self.lags)

        try:
            chol_sigma = np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError:
            return None

        irf = np.zeros((self.hor, self.n_endo, self.n_endo))
        for shock in range(self.n_endo):
            impulse = np.zeros((self.n_endo, 1))
            impulse[shock, 0] = 1 if self.irf_1std == 1 else 1 / chol_sigma[shock, shock]
            irf[0, :, shock] = (chol_sigma @ impulse).flatten()
            for horizon in range(1, self.hor):
                companion_power = np.linalg.matrix_power(companion, horizon)
                irf[horizon, :, shock] = (
                    companion_power[: self.n_endo, : self.n_endo] @ chol_sigma @ impulse
                ).flatten()

        return irf

    def _compute_single_fevd(self, beta_mat, sigma):
        try:
            chol_sigma = np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError:
            return None

        psi = np.zeros((self.n_endo, self.n_endo, self.hor))
        psi[:, :, 0] = np.eye(self.n_endo)

        beta_poly = np.zeros((self.n_endo, self.n_endo, self.lags))
        for lag in range(self.lags):
            beta_poly[:, :, lag] = beta_mat[lag * self.n_endo : (lag + 1) * self.n_endo, :].T

        for horizon in range(1, self.hor):
            for lag in range(1, min(self.lags, horizon) + 1):
                psi[:, :, horizon] += psi[:, :, horizon - lag] @ beta_poly[:, :, lag - 1]

        fevd = np.zeros((self.hor, self.n_endo, self.n_endo))
        for shock in range(self.n_endo):
            mse = np.zeros((self.n_endo, self.n_endo, self.hor))
            mse[:, :, 0] = sigma

            mse_shock = np.zeros((self.n_endo, self.n_endo, self.hor))
            shock_vector = chol_sigma[:, shock].reshape(-1, 1)
            mse_shock[:, :, 0] = shock_vector @ shock_vector.T

            for horizon in range(1, self.hor):
                psi_h = psi[:, :, horizon]
                mse[:, :, horizon] = mse[:, :, horizon - 1] + psi_h @ sigma @ psi_h.T
                mse_shock[:, :, horizon] = (
                    mse_shock[:, :, horizon - 1]
                    + psi_h @ (shock_vector @ shock_vector.T) @ psi_h.T
                )

            for horizon in range(self.hor):
                fevd[horizon, shock, :] = 100 * np.diag(
                    mse_shock[:, :, horizon] / mse[:, :, horizon]
                )

        return fevd

    def _draw_beta(self, unit_idx, c_vec, sigma, bbar, lambda_value, fallback_beta):
        yj = self.yy_units[unit_idx]
        xj = self.X_lag_units[unit_idx]
        zj = self.Z_units[unit_idx]
        sigma_inv = self._inverse(sigma)

        if self.n_exo > 0:
            c_mat = self.reshape_beta(c_vec, self.n_exo, self.n_endo)
            y_tilde = yj - zj @ c_mat
        else:
            y_tilde = yj

        prior_precision = self.prior_precision_units[unit_idx] / lambda_value
        post_precision = prior_precision + np.kron(sigma_inv, xj.T @ xj)
        post_cov = self._inverse(post_precision)
        rhs = prior_precision @ bbar + np.kron(sigma_inv, xj.T) @ y_tilde.reshape(-1, order="F")
        post_mean = post_cov @ rhs
        chol = self._safe_cholesky(post_cov)

        for _ in range(self.max_tries):
            beta_draw = post_mean + chol @ np.random.randn(self.k_beta)
            beta_mat = self.reshape_beta(beta_draw, self.n_lag_coeff_eq, self.n_endo)
            companion = self.build_companion_matrix(beta_mat, self.n_endo, self.lags)
            if self.is_stable(companion):
                return beta_draw, False

        return fallback_beta.copy(), True

    def _draw_exogenous(self, unit_idx, beta_vec, sigma):
        if self.n_exo == 0:
            return np.zeros((0,))

        yj = self.yy_units[unit_idx]
        xj = self.X_lag_units[unit_idx]
        zj = self.Z_units[unit_idx]
        sigma_inv = self._inverse(sigma)
        beta_mat = self.reshape_beta(beta_vec, self.n_lag_coeff_eq, self.n_endo)
        y_tilde = yj - xj @ beta_mat

        prior_precision = self.exo_prior["H_inv"]
        post_precision = prior_precision + np.kron(sigma_inv, zj.T @ zj)
        post_cov = self._inverse(post_precision)
        rhs = np.kron(sigma_inv, zj.T) @ y_tilde.reshape(-1, order="F")
        post_mean = post_cov @ rhs
        chol = self._safe_cholesky(post_cov)

        return post_mean + chol @ np.random.randn(self.k_exo_total)

    def _draw_lambda(self, beta_draws, bbar):
        scale = float(self.prior_params["lambda_scale"])
        for unit_idx in range(self.n_units):
            diff = beta_draws[unit_idx] - bbar
            scale += diff @ self.prior_precision_units[unit_idx] @ diff

        df = self.prior_params["lambda_df"] + self.n_units * self.k_beta
        scale = max(scale, 1e-12)
        return scale / np.random.chisquare(df)

    def _draw_bbar(self, beta_draws, lambda_value, fallback_bbar):
        precision = np.zeros((self.k_beta, self.k_beta))
        rhs = np.zeros(self.k_beta)
        for unit_idx in range(self.n_units):
            precision += self.prior_precision_units[unit_idx]
            rhs += self.prior_precision_units[unit_idx] @ beta_draws[unit_idx]

        precision /= lambda_value
        rhs /= lambda_value
        post_cov = self._inverse(precision)
        post_mean = post_cov @ rhs
        chol = self._safe_cholesky(post_cov)

        for _ in range(self.max_tries):
            draw = post_mean + chol @ np.random.randn(self.k_beta)
            draw_mat = self.reshape_beta(draw, self.n_lag_coeff_eq, self.n_endo)
            companion = self.build_companion_matrix(draw_mat, self.n_endo, self.lags)
            if self.is_stable(companion):
                return draw, False

        return fallback_bbar.copy(), True

    def sample_posterior(self, plot_coefficients: bool = False) -> Dict[str, np.ndarray]:
        """
        Run the hierarchical Gibbs sampler.

        Returns
        -------
        dict
            Posterior draws for unit-specific lag coefficients, exogenous coefficients,
            covariance matrices, pooled mean coefficients, and the pooling parameter.
        """
        beta_state = self.beta_ols.copy()
        c_state = self.c_ols.copy()
        sigma_state = self.Sigma_ols.copy()
        bbar_state = beta_state.mean(axis=0)
        lambda_state = float(self.prior_params["lambda_init"])

        beta_store = []
        c_store = []
        sigma_store = []
        bbar_store = []
        lambda_store = []

        for _ in tqdm(range(self.post_draws), desc="Sampling Panel Posterior"):
            for unit_idx in range(self.n_units):
                beta_draw, failed_beta = self._draw_beta(
                    unit_idx=unit_idx,
                    c_vec=c_state[unit_idx],
                    sigma=sigma_state[unit_idx],
                    bbar=bbar_state,
                    lambda_value=lambda_state,
                    fallback_beta=beta_state[unit_idx],
                )
                self.failed_stability_draws["unit_beta"] += int(failed_beta)

                c_draw = self._draw_exogenous(
                    unit_idx=unit_idx,
                    beta_vec=beta_draw,
                    sigma=sigma_state[unit_idx],
                )

                coeffs = self._combine_coefficients(beta_draw, c_draw)
                residuals = self.yy_units[unit_idx] - self.X_units[unit_idx] @ coeffs
                scale = residuals.T @ residuals + self.sigma_prior_scale
                sigma_draw = invwishart.rvs(
                    df=self.sigma_prior_df + self.yy_units[unit_idx].shape[0],
                    scale=scale,
                )

                beta_state[unit_idx] = beta_draw
                c_state[unit_idx] = c_draw
                sigma_state[unit_idx] = sigma_draw

            lambda_state = self._draw_lambda(beta_state, bbar_state)
            bbar_state, failed_bbar = self._draw_bbar(beta_state, lambda_state, bbar_state)
            self.failed_stability_draws["bbar"] += int(failed_bbar)

            beta_store.append(beta_state.copy())
            c_store.append(c_state.copy())
            sigma_store.append(sigma_state.copy())
            bbar_store.append(bbar_state.copy())
            lambda_store.append(lambda_state)

        self.beta_draws = np.asarray(beta_store[self.burnin :])
        self.c_draws = np.asarray(c_store[self.burnin :])
        self.Sigma_draws = np.asarray(sigma_store[self.burnin :])
        self.bbar_draws = np.asarray(bbar_store[self.burnin :])
        self.lambda_draws = np.asarray(lambda_store[self.burnin :])

        if plot_coefficients:
            lambda_plot, lag_plots = generate_panel_coeff_plots(self)
            display(lambda_plot)
            for plot in lag_plots[:2]:
                display(plot)
            if len(lag_plots) > 2:
                print("Note: Only showing the first 2 lag blocks of pooled mean coefficients.")

        return {
            "beta_draws": self.beta_draws,
            "c_draws": self.c_draws,
            "Sigma_draws": self.Sigma_draws,
            "bbar_draws": self.bbar_draws,
            "lambda_draws": self.lambda_draws,
        }

    def _check_posterior_draws(self):
        if len(self.beta_draws) == 0:
            raise ValueError("Run `sample_posterior()` before requesting posterior-based analysis.")

    def compute_irfs(
        self,
        plot_irfs: bool = False,
        cred_interval: Union[float, List[float]] = 0.68,
        unit: Optional[Union[int, str]] = None,
    ) -> np.ndarray:
        """
        Compute orthogonalized impulse responses for all panel units.

        Returns
        -------
        np.ndarray
            Array with shape (draws, horizon, units, variables, shocks).
        """
        self._check_posterior_draws()
        pooled_request = self._is_pooled_unit(unit)
        n_draws = self.beta_draws.shape[0]
        self.ir_draws = np.full((n_draws, self.hor, self.n_units, self.n_endo, self.n_endo), np.nan)
        self.ir_draws_pooled = np.full((n_draws, self.hor, self.n_endo, self.n_endo), np.nan)
        pooled_sigma_draws = self._pooled_sigma_draws()

        for draw_idx in tqdm(range(n_draws), desc="Computing Panel IRFs"):
            for unit_idx in range(self.n_units):
                beta_mat = self.reshape_beta(
                    self.beta_draws[draw_idx, unit_idx],
                    self.n_lag_coeff_eq,
                    self.n_endo,
                )
                sigma = self.Sigma_draws[draw_idx, unit_idx]
                irf = self._compute_single_irf(beta_mat, sigma)
                if irf is not None:
                    self.ir_draws[draw_idx, :, unit_idx, :, :] = irf

            pooled_beta = self.reshape_beta(
                self.bbar_draws[draw_idx],
                self.n_lag_coeff_eq,
                self.n_endo,
            )
            pooled_irf = self._compute_single_irf(pooled_beta, pooled_sigma_draws[draw_idx])
            if pooled_irf is not None:
                self.ir_draws_pooled[draw_idx, :, :, :] = pooled_irf

        if plot_irfs:
            if pooled_request:
                plot_settings = SimpleNamespace(
                    ir_draws=self.ir_draws_pooled,
                    hor=self.hor,
                    n_endo=self.n_endo,
                    names=self.names,
                    irf_1std=self.irf_1std,
                    plot_label="Pooled Mean Coefficients",
                )
            else:
                unit_idx = self._resolve_unit(unit)
                plot_settings = SimpleNamespace(
                    ir_draws=self.ir_draws[:, :, unit_idx, :, :],
                    hor=self.hor,
                    n_endo=self.n_endo,
                    names=self.names,
                    irf_1std=self.irf_1std,
                    plot_label=f"Unit: {self.units[unit_idx]}",
                )
            for plot in generate_irf_plots(plot_settings, cred_interval):
                display(plot)

        return self.ir_draws_pooled if pooled_request else self.ir_draws

    def compute_fevd(
        self,
        plot_fevd: bool = True,
        unit: Optional[Union[int, str]] = None,
        series_titles: Optional[List[str]] = None,
        shock_titles: Optional[List[str]] = None,
        title: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute FEVDs for all panel units.
        """
        self._check_posterior_draws()
        pooled_request = self._is_pooled_unit(unit)
        n_draws = self.beta_draws.shape[0]
        fevd_draws = np.full((n_draws, self.hor, self.n_units, self.n_endo, self.n_endo), np.nan)
        fevd_draws_pooled = np.full((n_draws, self.hor, self.n_endo, self.n_endo), np.nan)
        pooled_sigma_draws = self._pooled_sigma_draws()

        for draw_idx in tqdm(range(n_draws), desc="Computing Panel FEVD"):
            for unit_idx in range(self.n_units):
                beta_mat = self.reshape_beta(
                    self.beta_draws[draw_idx, unit_idx],
                    self.n_lag_coeff_eq,
                    self.n_endo,
                )
                sigma = self.Sigma_draws[draw_idx, unit_idx]
                fevd = self._compute_single_fevd(beta_mat, sigma)
                if fevd is not None:
                    fevd_draws[draw_idx, :, unit_idx, :, :] = fevd

            pooled_beta = self.reshape_beta(
                self.bbar_draws[draw_idx],
                self.n_lag_coeff_eq,
                self.n_endo,
            )
            pooled_fevd = self._compute_single_fevd(pooled_beta, pooled_sigma_draws[draw_idx])
            if pooled_fevd is not None:
                fevd_draws_pooled[draw_idx, :, :, :] = pooled_fevd

        self.fevd_draws = fevd_draws
        self.fevd = np.nanmean(fevd_draws, axis=0)
        self.fevd_draws_pooled = fevd_draws_pooled
        self.fevd_pooled = np.nanmean(fevd_draws_pooled, axis=0)

        if plot_fevd:
            if pooled_request:
                plot_settings = SimpleNamespace(
                    fevd=self.fevd_pooled,
                    names=self.names,
                )
                plot_title = title or "Forecast Error Variance Decomposition (Pooled Mean Coefficients)"
            else:
                unit_idx = self._resolve_unit(unit)
                plot_settings = SimpleNamespace(
                    fevd=self.fevd[:, unit_idx, :, :],
                    names=self.names,
                )
                plot_title = title or f"Forecast Error Variance Decomposition ({self.units[unit_idx]})"
            fevd_plot = generate_fevd_plot(
                plot_settings,
                series_titles=series_titles,
                shock_titles=shock_titles,
                title=plot_title,
            )
            display(fevd_plot)

        return {
            "fevd": self.fevd,
            "fevd_draws": self.fevd_draws,
            "fevd_pooled": self.fevd_pooled,
            "fevd_draws_pooled": self.fevd_draws_pooled,
        }

    def _prepare_future_exog(self, fhor, future_exog=None, unit_idx=None):
        if self.n_common_exo == 0 and not self.constant and not self.timetrend:
            return np.zeros((fhor, 0))

        blocks = []
        if self.constant:
            blocks.append(np.ones((fhor, 1)))
        if self.timetrend:
            trend_origin = (
                self.nobs_by_unit[unit_idx]
                if unit_idx is not None
                else (self.nobs if self.nobs is not None else self.max_nobs)
            )
            blocks.append(
                np.arange(trend_origin + 1, trend_origin + fhor + 1, dtype=float).reshape(-1, 1)
            )

        if self.n_common_exo > 0:
            if future_exog is None:
                source_exog = (
                    self.common_exog_units[unit_idx]
                    if unit_idx is not None
                    else self.common_exog
                )
                exo_future = np.repeat(source_exog[[-1], :], fhor, axis=0)
            elif isinstance(future_exog, pd.DataFrame):
                exo_frame = future_exog.copy()
                if self.time_col in exo_frame.columns:
                    exo_frame = exo_frame.sort_values(self.time_col)
                exo_future = exo_frame[self.exo_names].to_numpy(dtype=float)
            else:
                exo_future = np.asarray(future_exog, dtype=float)
                if exo_future.ndim == 1:
                    exo_future = exo_future.reshape(-1, self.n_common_exo)

            if exo_future.shape != (fhor, self.n_common_exo):
                raise ValueError(
                    "Future exogenous data must have shape (fhor, number_of_common_exogenous_variables)."
                )
            blocks.append(exo_future)

        return np.hstack(blocks) if blocks else np.zeros((fhor, 0))

    def forecast(
        self,
        fhor: Optional[int] = None,
        future_exog: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        plot_forecast: bool = True,
        cred_interval: List[float] = [0.68, 0.95],
        unit: Optional[Union[int, str]] = None,
        last_k: Optional[int] = None,
        n_breaks: int = 10,
        zero_line: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Generate unit-specific unconditional forecasts for the panel.
        """
        self._check_posterior_draws()
        fhor = int(fhor or self.fhor)
        n_draws = self.beta_draws.shape[0]

        self.forecasts = np.zeros((n_draws, fhor, self.n_units, self.n_endo))
        self.mean_forecasts = np.zeros((n_draws, fhor, self.n_units, self.n_endo))
        self.forecast_dates_by_unit = []
        for unit_idx in range(self.n_units):
            yy_dates_unit = self.yy_dates_units[unit_idx]
            freq = pd.infer_freq(yy_dates_unit[: min(6, len(yy_dates_unit))]) or "Q"
            self.forecast_dates_by_unit.append(
                pd.date_range(start=yy_dates_unit[-1], periods=fhor + 1, freq=freq)[1:]
            )
        self.forecast_dates = (
            self.forecast_dates_by_unit[0] if self.is_balanced_panel else None
        )

        for draw_idx in tqdm(range(n_draws), desc="Forecasting Panel VAR"):
            for unit_idx in range(self.n_units):
                z_future = self._prepare_future_exog(
                    fhor=fhor,
                    future_exog=future_exog,
                    unit_idx=unit_idx,
                )
                coeffs = self._combine_coefficients(
                    self.beta_draws[draw_idx, unit_idx],
                    self.c_draws[draw_idx, unit_idx],
                )
                sigma = self.Sigma_draws[draw_idx, unit_idx]
                history = [row.copy() for row in self.y_panel[unit_idx]]

                for horizon in range(fhor):
                    y_lags = np.hstack([history[-lag] for lag in range(1, self.lags + 1)])
                    x_t = y_lags
                    if self.n_exo > 0:
                        x_t = np.hstack((x_t, z_future[horizon]))

                    deterministic = x_t @ coeffs
                    shock = np.random.multivariate_normal(np.zeros(self.n_endo), sigma)
                    simulated = deterministic + shock

                    self.mean_forecasts[draw_idx, horizon, unit_idx, :] = deterministic
                    self.forecasts[draw_idx, horizon, unit_idx, :] = simulated
                    history.append(simulated)

        if plot_forecast:
            unit_idx = self._resolve_unit(unit)
            plot_settings = SimpleNamespace(
                yy=self.yy_units[unit_idx],
                yy_dates=self.yy_dates_units[unit_idx],
                names=self.names,
            )
            forecast_plot = generate_forecast_plots(
                plot_settings,
                self.forecasts[:, :, unit_idx, :],
                cred_interval=cred_interval,
                last_k=last_k,
                n_breaks=n_breaks,
                zero_line=zero_line,
                forecast_type=f"Unconditional ({self.units[unit_idx]})",
            )
            display(forecast_plot)

        return {
            "forecast_draws": self.forecasts,
            "mean_forecasts": self.mean_forecasts,
            "forecast_dates": self.forecast_dates,
            "forecast_dates_by_unit": self.forecast_dates_by_unit,
        }
