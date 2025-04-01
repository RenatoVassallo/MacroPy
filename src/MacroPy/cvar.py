import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.display import display
from numpy.linalg import inv, eigvals
from numpy.random import multivariate_normal
from .data_handling import prepare_data, estimate_ols
from .plots import generate_series_plot, generate_irf_plots
from .summary import generate_summary

class ClassicVAR:
    def __init__(self, y: pd.DataFrame, lags: int = 1, constant: bool = True, timetrend: bool = False, 
                 bstrap: bool = False, nreps: int = 1000, hor: int = 20, fhor: int = 12, irf_1std: int = 1):
        """
        Estimate Frequentist VAR model. IRFs bands are computed using either Monte Carlo (default) or bootstrap methods.
        Parameters:
        - y: DataFrame with time series data (must have a datetime index).
        - lags: Number of lags to include in the VAR model.
        - constant: Include a constant term in the model.
        - timetrend: Include a time trend in the model.
        - bstrap: Use bootstrap method for IRF estimation.
        - nreps: Number of bootstrap repetitions.
        - hor: Forecast horizon for IRF.
        - fhor: Forecast horizon for VAR.
        - irf_1std: If 1, use 1 standard deviation shocks; if 0, use unit shocks.
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


    def compute_irfs(self, plot_irfs: bool = False, cred_interval = 0.68):
        
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
