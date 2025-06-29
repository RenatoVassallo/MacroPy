import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.display import display
from numpy.linalg import inv, eigvals
from numpy.random import multivariate_normal
from .data_handling import prepare_data, estimate_ols
from .plots import generate_series_plot, generate_irf_plots, generate_fevd_plot, generate_hd_plot
from .summary import generate_summary

class ClassicVAR:
    def __init__(self, y: pd.DataFrame, lags: int = 1, constant: bool = True, timetrend: bool = False, 
                 bstrap: bool = False, nreps: int = 1000, hor: int = 20, fhor: int = 12, irf_1std: int = 1):
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
        return display(plot)
    
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
            Array of shape (n_draws, horizon, n_variables, n_variables) containing simulated
            IRFs for each structural shock.
        """
        
        N, P, H = self.n_endo, self.lags, self.hor
        B = self.b_ols.reshape((self.ncoeff_eq, N), order='F')
        Sigma = self.Sigma_ols

        self.ir_draws = []

        if self.bstrap:
            draws = 0
            with tqdm(total=self.nreps, desc="Bootstrap IRFs") as pbar:
                while draws < self.nreps:
                    residuals = self.yy - self.XX @ B
                    idx = np.random.choice(residuals.shape[0], residuals.shape[0], replace=True)
                    resampled = residuals[idx, :]
                    Y_boot = self.XX @ B + resampled
                    yy_star, XX_star = prepare_data(Y_boot, self.lags, self.constant, self.timetrend)
                    b_star, Sigma_star = estimate_ols(yy_star, XX_star)
                    B_star = self.reshape_beta(b_star, self.ncoeff_eq, N)
                    Bcomp_star = self.build_companion_matrix(B_star, N, P, self.n_exo)

                    try:
                        S_star = np.linalg.cholesky(Sigma_star)
                    except np.linalg.LinAlgError:
                        continue

                    irf = np.zeros((H, N, N))
                    for m in range(N):
                        impulse = np.zeros((N, 1))
                        impulse[m, 0] = 1 / S_star[m, m] if self.irf_1std == 0 else 1
                        irf[0, :, m] = (S_star @ impulse).flatten()
                        for h in range(1, H):
                            Bpow = np.linalg.matrix_power(Bcomp_star, h)
                            irf[h, :, m] = (Bpow[:N, :N] @ S_star @ impulse).flatten()
                    self.ir_draws.append(irf)
                    draws += 1
                    pbar.update(1)
        else:
            omega_hat = np.kron(Sigma, inv(self.XX.T @ self.XX))
            draws = 0
            with tqdm(total=self.nreps, desc="Monte Carlo IRFs") as pbar:
                while draws < self.nreps:
                    beta_vec = self.b_ols + multivariate_normal(np.zeros(self.ncoeff), omega_hat)
                    B_sim = self.reshape_beta(beta_vec, self.ncoeff_eq, N)
                    Bcomp_sim = self.build_companion_matrix(B_sim, N, P, self.n_exo)
                    if not self.is_stable(Bcomp_sim):
                        continue

                    try:
                        S_sim = np.linalg.cholesky(Sigma)
                    except np.linalg.LinAlgError:
                        continue

                    irf = np.zeros((H, N, N))
                    for m in range(N):
                        impulse = np.zeros((N, 1))
                        impulse[m, 0] = 1 / S_sim[m, m] if self.irf_1std == 0 else 1
                        irf[0, :, m] = (S_sim @ impulse).flatten()
                        for h in range(1, H):
                            Bpow = np.linalg.matrix_power(Bcomp_sim, h)
                            irf[h, :, m] = (Bpow[:N, :N] @ S_sim @ impulse).flatten()
                    self.ir_draws.append(irf)
                    draws += 1
                    pbar.update(1)

        self.ir_draws = np.array(self.ir_draws)

        if plot_irfs:
            ir_plots = generate_irf_plots(self, cred_interval)
            for p in ir_plots:
                display(p)

        return self.ir_draws
    
    
    def compute_fevd(self, plot_fevd: bool = True, series_titles: list = None,
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
            display(fevd_plot)

        return self.fevd
    
    
    def compute_hd(self, plot_hd: bool = True, series_titles: list = None,
                    shock_titles: list = None, title: str = None):
        """
        Compute the Historical Decomposition (HD) for the VAR model.
        """
        N, P = self.n_endo, self.lags
        T = self.yy.shape[0]
        Sigma = self.Sigma_ols
        B = self.reshape_beta(self.b_ols, self.ncoeff_eq, N)
        Bcomp = self.build_companion_matrix(B, N, P, self.n_exo)
        X = self.XX
        const_cols = int(self.constant) + int(self.timetrend)

        # Get residuals and structural impact matrix
        resid = self.yy - self.XX @ B
        try:
            S = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            raise ValueError("Sigma_ols is not positive definite")
        
        eps = np.linalg.solve(S, resid.T)  # Structural shocks

        nlagX = N * P
        B_big = np.zeros((nlagX, N))
        B_big[:N, :] = S

        Icomp = np.hstack((np.eye(N), np.zeros((N, nlagX - N))))  # (N, N*P)

        # === SHOCK COMPONENTS ===
        HDshock_big = np.zeros((nlagX, T + 1, N))
        HDshock = np.zeros((N, T + 1, N))

        for j in range(N):
            eps_big = np.zeros((N, T + 1))
            eps_big[j, 1:] = eps[j, :]
            for t in range(1, T + 1):
                HDshock_big[:, t, j] = B_big @ eps_big[:, t] + Bcomp @ HDshock_big[:, t - 1, j]
                HDshock[:, t, j] = Icomp @ HDshock_big[:, t, j]

        # === INITIAL CONDITION ===
        HDinit_big = np.zeros((nlagX, T + 1))
        HDinit = np.zeros((N, T + 1))
        HDinit_big[:, 0] = X[0, :-const_cols]  # exclude constant/trend
        HDinit[:, 0] = Icomp @ HDinit_big[:, 0]
        for t in range(1, T + 1):
            HDinit_big[:, t] = Bcomp @ HDinit_big[:, t - 1]
            HDinit[:, t] = Icomp @ HDinit_big[:, t]

        # === CONSTANT TERM ===
        HDconst_big = np.zeros((nlagX, T + 1))
        HDconst = np.zeros((N, T + 1))
        if self.constant:
            C = np.zeros((nlagX,))
            C[:N] = B[-1, :]
            for t in range(1, T + 1):
                HDconst_big[:, t] = C + Bcomp @ HDconst_big[:, t - 1]
                HDconst[:, t] = Icomp @ HDconst_big[:, t]

        # === TREND TERM ===
        HDtrend_big = np.zeros((nlagX, T + 1))
        HDtrend = np.zeros((N, T + 1))
        if self.timetrend:
            T_vec = np.zeros((nlagX,))
            T_vec[:N] = B[-2 if self.constant else -1, :]
            for t in range(1, T + 1):
                HDtrend_big[:, t] = T_vec * (t - 1) + Bcomp @ HDtrend_big[:, t - 1]
                HDtrend[:, t] = Icomp @ HDtrend_big[:, t]

        # === Final HD Aggregation ===
        HDendo = HDinit + HDconst + HDtrend + np.sum(HDshock, axis=2)

        # === Save outputs ===
        HDshock_out = np.full((T + P, N, N), np.nan)
        for j in range(N):  # shock index
            for i in range(N):  # variable index
                HDshock_out[P:, i, j] = HDshock[i, 1:, j]

        self.HD = {
            'shock': HDshock_out,  # shape: [T+P, variable, shock]
            'init': np.vstack([np.full((P - 1, N), np.nan), HDinit[:, :].T]),
            'const': np.vstack([np.full((P, N), np.nan), HDconst[:, 1:].T]),
            'trend': np.vstack([np.full((P, N), np.nan), HDtrend[:, 1:].T]),
            'endo': np.vstack([np.full((P, N), np.nan), HDendo[:, 1:].T])
        }
        
        if plot_hd:
            hd_plot = generate_hd_plot(self, series_titles, shock_titles, title)
            display(hd_plot)

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
