import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.display import display
from numpy.linalg import inv, eigvals
from typing import Optional
from scipy.stats import invwishart
from .data_handling import prepare_data, estimate_ols, historical_decomposition
from .plots import generate_series_plot, generate_irf_plots, generate_fevd_plot, generate_hd_plot
from .summary import generate_summary

__all__ = [
    "ClassicVAR",
]

class ClassicVAR:
    def __init__(self, y: pd.DataFrame, lags: int = 1, constant: bool = True, timetrend: bool = False, 
                 bstrap: bool = False, nreps: int = 1000, hor: int = 20, fhor: int = 12, irf_1std: int = 1,
                 seed: Optional[int] = None, stable_only: bool = True, progress: bool = True):
        """
        Estimate a classic (frequentist) Vector Autoregression (VAR) model and compute 
        Impulse Response Functions (IRFs) with confidence bands.

        Parameters
        ----------
        y : pd.DataFrame
            Multivariate time series dataset. Must have a datetime index.
        
        lags : int, default=1
            Number of lags to include in the VAR model.
        
        constant : bool, default=True
            If True, includes an intercept (constant) term in the model.
        
        timetrend : bool, default=False
            If True, includes a linear time trend in the model.
        
        bstrap : bool, default=False
            If True, uses bootstrap sampling to estimate IRF confidence bands.
            If False, uses Monte Carlo simulation based on estimated covariance.

        nreps : int, default=1000
            Number of repetitions for the IRF simulations (bootstrap or Monte Carlo).
        
        hor : int, default=20
            Number of periods ahead to compute impulse response functions and FEVD.
        
        fhor : int, default=12
            Forecast horizon used for out-of-sample forecasting (if applicable).
        
        irf_1std : int, default=1
            Determines the type of shock used in the IRFs:
            - 1: Use a one standard deviation structural shock.
            - 0: Use a unit (1.0) structural shock.

        seed : int, optional
            Random seed. Each simulation stage draws from its own generator
            ``default_rng([seed, stage])``, so results do not depend on the
            order of the calls. None (default) keeps non-deterministic draws.

        stable_only : bool, default=True
            Discard explosive draws (companion eigenvalue modulus >= 1) in both
            the bootstrap and the Monte Carlo branch. Discarding draws changes
            the bands, so it is an explicit option.

        progress : bool, default=True
            Show the tqdm progress bar (turn off in rendered documents).

        Notes
        -----
        This class implements a classic OLS-based VAR estimation. IRFs are derived using
        either bootstrapped resamples or draws from the asymptotic normal distribution
        of the estimated coefficients. FEVD is computed via the Wold representation.

        Use this class when you want transparent, academic VAR estimation with flexibility
        in simulation-based inference for uncertainty.
        """
        if not isinstance(y, pd.DataFrame):
            raise ValueError("Input data 'y' must be a pandas DataFrame with a datetime index.")

        self.names = y.columns
        self.dates = y.index   # Store original dates
        self.y = y.to_numpy()  # Convert DataFrame to NumPy array
        self.lags = lags
        self.n_endo = self.y.shape[1]  
        self.constant = constant
        self.timetrend = timetrend
        self.bstrap = bstrap
        self.nreps = nreps
        self.hor = hor
        self.fhor = fhor
        self.irf_1std = irf_1std
        self.seed = seed
        self.stable_only = stable_only
        self.progress = progress
        self.figures = {}   # figures from the last plot_* calls

        self.n_exo = int(constant) + int(timetrend)
        self.ncoeff_eq = self.n_endo * self.lags + self.n_exo
        self.ncoeff = self.ncoeff_eq * self.n_endo

        self.yy, self.XX = prepare_data(self.y, self.lags, self.constant, self.timetrend)
        self.yy_dates = self.dates[self.lags:]
        self.b_ols, self.Sigma_ols = estimate_ols(self.yy, self.XX)

    def model_summary(self):
        display(generate_summary(self))

    def plot_series(self, series_titles: list = None, title: str = None, color_scheme: int = 1, 
                    n_breaks: int = 10, zero_line: bool = False):
        plot = generate_series_plot(self.yy, self.yy_dates, self.names, series_titles=series_titles, title=title,
                                    color_scheme=color_scheme, n_breaks=n_breaks, zero_line=zero_line)
        return plot
    
    @staticmethod
    def is_stable(Bcomp):
        """Check VAR stability based on companion matrix eigenvalues."""
        return np.all(np.abs(eigvals(Bcomp)) < 0.9999)

    @staticmethod
    def build_companion_matrix(B, N, P, exog):
        """Construct the VAR companion matrix."""
        n_total_rows = B.shape[0]
        n_lag_coeffs = n_total_rows - exog  # rows excluding exogenous part

        Bcomp = np.zeros((N * P, N * P))
        Bcomp[:N, :] = B[:n_lag_coeffs, :].T  # only lag coefficients

        if P > 1:
            Bcomp[N:, :-N] = np.eye(N * (P - 1))

        return Bcomp

    @staticmethod
    def reshape_beta(beta_vec, ncoeff_eq, N):
        """Reshape vectorized beta into coefficient matrix."""
        return beta_vec.reshape((ncoeff_eq, N), order='F')


    def _rng(self, stage: str) -> np.random.Generator:
        """Deterministic per-stage generator; independent of the call order."""
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng([self.seed, {"bootstrap": 0, "montecarlo": 1}[stage]])

    def _irf_from(self, B, Sigma):
        """Cholesky IRFs (hor, n, n) for coefficients B and covariance Sigma; None if rejected."""
        N, P, H = self.n_endo, self.lags, self.hor
        try:
            S = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            return None
        Bcomp = self.build_companion_matrix(B, N, P, self.n_exo)
        if self.stable_only and not self.is_stable(Bcomp):
            return None
        impact = S if self.irf_1std == 1 else S / np.diag(S)   # unit shocks: scale columns
        irf = np.zeros((H, N, N))
        Bpow = np.eye(Bcomp.shape[0])
        for h in range(H):
            irf[h] = Bpow[:N, :N] @ impact
            Bpow = Bpow @ Bcomp
        return irf

    def compute_irfs(self, plot_irfs: bool = False, cred_interval=0.68):
        """
        Compute impulse response functions (IRFs) for the estimated VAR model.

        The method uses either bootstrap resampling or Monte Carlo simulation to generate
        draws of the IRFs, depending on the `bstrap` flag. Structural shocks are identified
        using a Cholesky decomposition of the residual covariance matrix, which imposes 
        contemporaneous zero restrictions (recursive identification).

        Parameters
        ----------
        plot_irfs : bool, default=False
            If True, displays IRF plots with confidence bands.

        cred_interval : float, default=0.68
            Credible interval (e.g., 0.68 or 0.95) used to compute IRF confidence bands.

        Returns
        -------
        ir_draws : np.ndarray
            Array of shape (n_draws, horizon, n_variables, n_shocks): ``ir_draws[d, h, i, k]``
            is the response of variable ``i`` at horizon ``h`` to structural shock ``k``.

        Notes
        -----
        ``bstrap=True`` uses the recursive-design residual bootstrap: each artificial
        sample is generated by iterating the estimated VAR from the observed initial
        conditions with resampled (centred) residuals, and the model is re-estimated and
        re-identified on it. Percentile bands from this bootstrap can have poor coverage
        in short samples and with roots close to one (Kilian, 1998).

        ``bstrap=False`` draws ``Sigma ~ IW(U'U, T - k)`` and then
        ``b | Sigma ~ N(b_OLS, Sigma kron (X'X)^-1)`` (Monte Carlo integration under a
        diffuse prior), so the impact matrix also carries uncertainty.
        """
        
        N, P = self.n_endo, self.lags
        B = self.b_ols.reshape((self.ncoeff_eq, N), order='F')
        U = self.yy - self.XX @ B
        T_eff, k = self.XX.shape

        self.ir_draws = []

        if self.bstrap:
            rng = self._rng("bootstrap")
            U_c = U - U.mean(axis=0)                  # centred residuals
            det = self.XX[:, N * P:]                  # constant / trend, kept as observed
            y_init = self.y[:P]                       # observed initial conditions
            with tqdm(total=self.nreps, desc="Bootstrap IRFs", disable=not self.progress) as pbar:
                while len(self.ir_draws) < self.nreps:
                    u_star = U_c[rng.integers(0, T_eff, T_eff)]
                    y_star = np.empty((T_eff + P, N))
                    y_star[:P] = y_init
                    X_star = np.empty_like(self.XX)
                    for t in range(T_eff):
                        lags = y_star[t + P - 1::-1][:P].ravel()   # y_{t-1}, ..., y_{t-p}
                        X_star[t] = np.concatenate([lags, det[t]])
                        y_star[t + P] = X_star[t] @ B + u_star[t]
                    b_star, Sigma_star = estimate_ols(y_star[P:], X_star)
                    B_star = self.reshape_beta(b_star, self.ncoeff_eq, N)
                    irf = self._irf_from(B_star, Sigma_star)
                    if irf is None:
                        continue
                    self.ir_draws.append(irf)
                    pbar.update(1)
        else:
            rng = self._rng("montecarlo")
            XtX_inv = inv(self.XX.T @ self.XX)
            scale = U.T @ U
            with tqdm(total=self.nreps, desc="Monte Carlo IRFs", disable=not self.progress) as pbar:
                while len(self.ir_draws) < self.nreps:
                    Sigma_draw = np.atleast_2d(
                        invwishart.rvs(df=T_eff - k, scale=scale, random_state=rng))
                    L = np.linalg.cholesky(np.kron(Sigma_draw, XtX_inv))
                    beta_vec = self.b_ols + L @ rng.standard_normal(self.ncoeff)
                    B_sim = self.reshape_beta(beta_vec, self.ncoeff_eq, N)
                    irf = self._irf_from(B_sim, Sigma_draw)
                    if irf is None:
                        continue
                    self.ir_draws.append(irf)
                    pbar.update(1)

        self.ir_draws = np.array(self.ir_draws)

        if plot_irfs:
            self.figures["irfs"] = generate_irf_plots(self, cred_interval)

        return self.ir_draws
    
    
    def compute_fevd(self, plot_fevd: bool = False, series_titles: list = None,
                     shock_titles: list = None, title: str = None):
        """
        Compute the Forecast Error Variance Decomposition (FEVD) for a Classic VAR model.

        Parameters:
        - steps: int, number of steps/horizons to compute the decomposition

        Returns:
        - fevd: np.ndarray of shape [steps, shocks, variables], in percentage
        """
        N = self.n_endo
        P = self.lags
        H = self.hor
        B = self.reshape_beta(self.b_ols, self.ncoeff_eq, N)
        Sigma = self.Sigma_ols

        # Compute structural impact matrix
        try:
            S = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is not positive definite")

        # Initialize Wold representation multipliers
        PSI = np.zeros((N, N, H))
        PSI[:, :, 0] = np.eye(N)

        # Reconstruct lag polynomial Bp from reshaped B (skip exogenous rows)
        Bp = np.zeros((N, N, P))
        for p in range(P):
            Bp[:, :, p] = B[p * N:(p + 1) * N, :].T

        # Compute Wold representation: PSI matrices
        for h in range(1, H):
            for j in range(1, h + 1):
                if j <= P:
                    PSI[:, :, h] += PSI[:, :, h - j] @ Bp[:, :, j - 1]

        # FEVD storage
        self.fevd = np.zeros((H, N, N))  # [horizon, shock, variable]

        for shock in range(N):
            # Initialize MSE and MSE_shock
            MSE = np.zeros((N, N, H))
            MSE[:, :, 0] = Sigma

            MSE_shock = np.zeros((N, N, H))
            S_shock = S[:, shock].reshape(-1, 1)
            MSE_shock[:, :, 0] = S_shock @ S_shock.T

            for h in range(1, H):
                PSI_h = PSI[:, :, h]
                MSE[:, :, h] = MSE[:, :, h - 1] + PSI_h @ Sigma @ PSI_h.T
                MSE_shock[:, :, h] = MSE_shock[:, :, h - 1] + PSI_h @ (S_shock @ S_shock.T) @ PSI_h.T

            # Compute FEVD
            for h in range(H):
                FECD = MSE_shock[:, :, h] / MSE[:, :, h]
                self.fevd[h, shock, :] = 100 * np.diag(FECD)
       
        if plot_fevd:
            fevd_plot = generate_fevd_plot(self, series_titles, shock_titles, title)
            self.figures["fevd"] = fevd_plot

        return self.fevd
    
    
    def compute_hd(self, plot_hd: bool = False, series_titles: list = None,
                    shock_titles: list = None, title: str = None):
        """
        Historical decomposition with a recursive (Cholesky) identification.

        Parameters
        ----------
        plot_hd : bool, default=False
            Build the stacked-bar figure (stored in ``figures["hd"]``).
        series_titles, shock_titles, title : list, list, str
            Labels for the figure.

        Returns
        -------
        dict
            ``"shock"`` (T, n_shocks, n_endo): contribution of each structural
            shock; ``"baseline"`` (T, n_endo): initial conditions plus the
            deterministic terms; ``"endo"``: their sum; ``"dates"``: the dates
            of those T observations. The identity
            ``baseline + shock.sum(axis=1) == yy`` holds exactly.
        """
        B = self.reshape_beta(self.b_ols, self.ncoeff_eq, self.n_endo)
        shock, baseline = historical_decomposition(
            B, self.Sigma_ols, self.yy, self.XX, self.n_endo, self.lags)
        self.HD = {
            "shock": shock,
            "baseline": baseline,
            "endo": baseline + shock.sum(axis=1),
            "dates": pd.DatetimeIndex(pd.to_datetime(self.dates))[self.lags:],
        }

        if plot_hd:
            self.figures["hd"] = generate_hd_plot(self, series_titles, shock_titles, title)

        return self.HD


    def forecast(self, forecast_data):
        """Compute forecasts."""
        for d in range(len(self.beta_draws)):
            frcst_no_shock, frcst_with_shocks = generate_forecasts(
                forecast_data, self.beta_draws[d], self.Sigma_draws[d], self.options
            )
            self.forecasts["no_shocks"].append(frcst_no_shock)
            self.forecasts["with_shocks"].append(frcst_with_shocks)
        
        self.forecasts["no_shocks"] = np.array(self.forecasts["no_shocks"])
        self.forecasts["with_shocks"] = np.array(self.forecasts["with_shocks"])
    
    def run_full_analysis(self, forecast_data):
        """Run the full pipeline: posterior sampling, IRFs, and forecasts."""
        self.sample_posterior()
        self.compute_irfs()
        self.forecast(forecast_data)
        
        return {
            "beta_draws": self.beta_draws,
            "Sigma_draws": self.Sigma_draws,
            "ir_draws": self.ir_draws,
            "forecasts": self.forecasts
        }
