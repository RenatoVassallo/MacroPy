import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.display import display
from numpy.linalg import inv, eigvals
from numpy.random import multivariate_normal
from scipy.stats import invwishart
from .data_handling import prepare_data, estimate_ols
from .priors import MinnesotaPrior, NormalWishartPrior, NormalDiffusePrior
from .plots import generate_series_plot, generate_coeff_plot, generate_irf_plots
from .summary import generate_summary

class BayesianVAR:
    def __init__(self, y: pd.DataFrame, lags: int = 1, constant: bool = True, timetrend: bool = False, 
                 prior_type: int = 1, prior_params: dict = {"mn_mean": 1, "lamda1": 0.2, "lamda2": 0.5, "lamda3": 1, "lamda4": 1e5},
                 post_draws: int = 5000, burnin: float = 0.5, hor: int = 20, fhor: int = 12, irf_1std: int = 1):
        """
        Bayesian VAR model.
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
        self.prior_type = prior_type
        self.prior_params = prior_params
        self.post_draws = post_draws
        self.burnin = int(burnin * post_draws)
        self.hor = hor
        self.fhor = fhor
        self.irf_1std = irf_1std
        
        # Number of exogenous variables and coefficients 
        self.n_exo = int(constant) + int(timetrend)
        self.ncoeff_eq = self.n_endo * self.lags + self.n_exo
        self.ncoeff = self.ncoeff_eq * self.n_endo
        
        # Organize data into YX format
        self.yy, self.XX = prepare_data(self.y, self.lags, self.constant, self.timetrend)
        
        # Adjust dates to match YYact (accounting for lags)
        self.yy_dates = self.dates[self.lags:]
        
        # Compute OLS estimates for initial values
        self.b_ols, self.Sigma_ols = estimate_ols(self.yy, self.XX)
        
        # Select prior distribution
        prior_dict = {1: "Minnesota", 2: "Normal-Wishart", 3: "Normal-Diffuse"}
        
        if prior_type not in prior_dict:
            raise ValueError("Invalid prior type. Choose 1 (Minnesota), 2 (Normal-Wishart), or 3 (Normal-Diffuse).")
        
        self.prior_name = prior_dict[prior_type]
        
        if prior_type == 1:
            self.prior = MinnesotaPrior(self.yy, self.XX, self.lags, self.ncoeff_eq, self.prior_params)
        elif prior_type == 2:
            self.prior = NormalWishartPrior(self.yy, self.XX, self.lags, self.ncoeff_eq, self.prior_params)
        elif prior_type == 3:
            self.prior = NormalDiffusePrior(self.yy, self.XX, self.lags, self.ncoeff_eq, self.prior_params)
        
        # Storage for draws
        self.beta_draws = []
        self.Sigma_draws = []
        
    def model_summary(self):
        """Print a summary of the Bayesian VAR model."""
        display(generate_summary(self))
                
        
    def plot_series(self, series_titles: list = None, title: str = None, color_scheme: int = 1, 
                    n_breaks: int = 10, zero_line: bool = False):
        """
        Generate and display time series plots for the model's variables.
        
        Inputs:
        - `title` (str, optional): Main title of the plot. Defaults to None.
        - `series_titles` (list, optional): Custom titles for the series, replacing default variable names. Defaults to None.
        - `color_scheme` (int, optional): Choose between 4 line colors: 1 (Dark Blue, default), 2 (Soft Red), 3 (Muted Green), and 4 (Soft Purple).
        - `n_breaks` (int, optional): Number of major x-axis ticks (date breaks). Defaults to `10`.

        Outputs:
        - Displays the generated plot using `display()`.
        """
        plot = generate_series_plot(self.yy, self.yy_dates, self.names, series_titles=series_titles, title=title,
                                    color_scheme=color_scheme, n_breaks=n_breaks, zero_line=zero_line)
        
        return display(plot)
    
    @staticmethod
    def is_stable(Bcomp):
        """Check VAR stability based on companion matrix eigenvalues."""
        return np.all(np.abs(eigvals(Bcomp)) < 1)

    @staticmethod
    def build_companion_matrix(B, N, P):
        """Construct the VAR companion matrix."""
        Bcomp = np.zeros((N * P, N * P))
        Bcomp[:N, :] = B[1:, :].T  # skip constant row
        if P > 1:
            Bcomp[N:, :-N] = np.eye(N * (P - 1))
        return Bcomp

    @staticmethod
    def reshape_beta(beta_vec, ncoeff_eq, N):
        """Reshape vectorized beta into coefficient matrix."""
        return beta_vec.reshape((ncoeff_eq, N), order='F')

    def sample_posterior(self, plot_coefficients: bool = False):
        """
        Draw posterior samples for VAR coefficients and variance-covariance matrix.
        Ensures draws are stable (companion eigenvalues < 1).
        """
        XtX = self.XX.T @ self.XX
        b_ols, Sigma_ols = self.b_ols, self.Sigma_ols
        b_prior, H_prior = self.prior["b0"], self.prior["H"]
        Scale0, alpha0 = self.prior.get("Scale0"), self.prior.get("alpha0")
        Sigma = Sigma_ols.copy()

        for _ in tqdm(range(self.post_draws), desc="Sampling Posterior"):
            Sigma_inv = inv(Sigma) if self.prior_type in [2, 3] else inv(Sigma_ols)
            invH = inv(H_prior)

            # Posterior mean and variance
            V_post = inv(invH + np.kron(Sigma_inv, XtX))
            M_post = V_post @ (invH @ b_prior + np.kron(Sigma_inv, XtX) @ b_ols)

            # Draw stable beta
            while True:
                beta_vec = multivariate_normal(mean=M_post, cov=V_post)
                B = self.reshape_beta(beta_vec, self.ncoeff_eq, self.n_endo)
                Bcomp = self.build_companion_matrix(B, self.n_endo, self.lags)
                if self.is_stable(Bcomp):
                    break

            # Draw Sigma if applicable
            if self.prior_type in [2, 3]:
                resid = self.yy - self.XX @ B
                scale_term = resid.T @ resid
                if self.prior_type == 2:
                    scale_term += Scale0
                    df = alpha0 + self.yy.shape[0]
                else:
                    df = self.yy.shape[0]
                Sigma = invwishart.rvs(df=df, scale=scale_term)

            # Store draws
            self.beta_draws.append(beta_vec)
            self.Sigma_draws.append(Sigma if self.prior_type in [2, 3] else Sigma_ols)

        # Apply burn-in
        self.beta_draws = np.array(self.beta_draws[self.burnin:])
        self.Sigma_draws = np.array(self.Sigma_draws[self.burnin:])

        # Optional: plot coefficient draws
        if plot_coefficients:
            const_plot, var_plots = generate_coeff_plot(self)
            display(const_plot)
            for i, plot in enumerate(var_plots[:2]):
                display(plot)
            if len(var_plots) > 2:
                print("Note: Only showing first 2 lags of coefficients.")


    def compute_irfs(self, plot_irfs: bool = False, cred_interval = 0.68):
        """
        Compute impulse response functions (IRFs) from posterior draws.
        Each draw produces one IRF matrix of shape [horizon, variables, shocks].
        """
        N, P, H = self.n_endo, self.lags, self.hor
        self.ir_draws = []
        n_draws = len(self.beta_draws)

        for d in tqdm(range(n_draws), desc="Computing IRFs"):
            B = self.reshape_beta(self.beta_draws[d], self.ncoeff_eq, N)
            Sigma = self.Sigma_draws[d]

            Bcomp = self.build_companion_matrix(B, N, P)

            # Cholesky of Sigma
            try:
                S = np.linalg.cholesky(Sigma)
            except np.linalg.LinAlgError:
                continue  # skip if unstable draw
            
            irf = np.zeros((H, N, N))

            for m in range(N):  # for each shock
                impulse = np.zeros((N, 1))
                if self.irf_1std == 1:
                    impulse[m, 0] = 1
                elif self.irf_1std == 0:
                    impulse[m, 0] = 1 / S[m, m]
                else:
                    raise ValueError("irf_1std must be 0 or 1")

                # Horizon 0
                irf[0, :, m] = (S @ impulse).flatten()

                # Future horizons
                for h in range(1, H):
                    Bcomp_h = np.linalg.matrix_power(Bcomp, h)
                    irf[h, :, m] = (Bcomp_h[:N, :N] @ S @ impulse).flatten()

            self.ir_draws.append(irf)

        self.ir_draws = np.array(self.ir_draws)  # [draws, horizon, variable, shock]

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
