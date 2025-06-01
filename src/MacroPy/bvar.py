import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.display import display
from numpy.linalg import inv, eigvals
from numpy.random import multivariate_normal
from scipy.stats import invwishart
from .data_handling import prepare_data, estimate_ols
from .priors import MinnesotaPrior, NormalWishartPrior, NormalDiffusePrior
from .plots import generate_coeff_plot, generate_irf_plots, generate_forecast_plots
from .summary import generate_summary

class BayesianVAR:
    def __init__(self, y: pd.DataFrame, lags: int = 1, constant: bool = True, timetrend: bool = False, 
                 prior_type: int = 1, prior_params: dict = {"mn_mean": 1, "lamda1": 0.2, "lamda2": 0.5, "lamda3": 1, "lamda4": 1e5},
                 b_exo: np.ndarray = None , post_draws: int = 5000, burnin: float = 0.5, hor: int = 20, fhor: int = 12, irf_1std: int = 1):
        """
        Bayesian Vector Autoregression (BVAR) model class.

        This class implements a Bayesian VAR for multivariate time series analysis, allowing 
        for different prior distributions and posterior simulation via Gibbs sampling. The model 
        supports inclusion of a constant and deterministic time trend, and is suitable for tasks 
        such as forecasting, impulse response analysis, and structural shock identification.

        Parameters:
            y (pd.DataFrame): Input time series data with a datetime index and variables in columns.
            lags (int): Number of autoregressive lags (default is 1).
            constant (bool): Whether to include a constant term (default is True).
            timetrend (bool): Whether to include a deterministic time trend (default is False).
            prior_type (int): Choice of prior:
                1 = Minnesota Prior
                2 = Normal-Wishart Prior
                3 = Normal-Diffuse Prior
            prior_params (dict): Dictionary of hyperparameters for the chosen prior.
                Default for Minnesota:
                    - mn_mean: Prior mean on first own lag
                    - lamda1: Own lag shrinkage
                    - lamda2: Cross lag shrinkage
                    - lamda3: Lag decay
                    - lamda4: Constant term variance
            b_exo (np.ndarray): block exogeneity mask. Variable i does not depend on lagged values of variable j.
            post_draws (int): Number of posterior draws including burn-in (default is 5000).
            burnin (float): Proportion of draws to discard as burn-in (default is 0.5).
            hor (int): Horizon for impulse response functions (default is 20).
            fhor (int): Forecast horizon (default is 12).
            irf_1std (int): Scale of structural shock for IRF (1 standard deviation, default is 1).
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
        self.b_exo = b_exo
        self.post_draws = post_draws
        self.burnin = int(burnin * post_draws)
        self.n_draws = self.post_draws - self.burnin  # Effective number of draws after burn-in
        self.hor = hor
        self.fhor = fhor
        self.irf_1std = irf_1std
        self.mean_forecasts = np.zeros((self.n_draws, self.fhor, self.n_endo))  # Forecasts without shocks
        self.forecasts = None      # With shocks
        self.cond_forecasts = np.zeros((self.n_draws, fhor, self.n_endo))
        
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
            self.prior = MinnesotaPrior(self.yy, self.XX, self.lags, self.ncoeff_eq, self.prior_params, self.b_exo)
        elif prior_type == 2:
            self.prior = NormalWishartPrior(self.yy, self.XX, self.lags, self.ncoeff_eq, self.prior_params, self.b_exo)
        elif prior_type == 3:
            self.prior = NormalDiffusePrior(self.yy, self.XX, self.lags, self.ncoeff_eq, self.prior_params, self.b_exo)
        
        # Storage for draws
        self.beta_draws = []
        self.Sigma_draws = []
        
    def model_summary(self):
        """Print a summary of the Bayesian VAR model."""
        display(generate_summary(self))
    
    @staticmethod
    def is_stable(Bcomp):
        """Check VAR stability based on companion matrix eigenvalues."""
        return np.all(np.abs(eigvals(Bcomp)) < 1)

    @staticmethod
    def build_companion_matrix(B, N, P):
        """
        Construct the VAR companion matrix from coefficient matrix B.
        Assumes B has shape [(N * P + 1), N], with the constant as the last row.
        """
        Bcomp = np.zeros((N * P, N * P))
        Bcomp[:N, :] = B[:-1, :].T  # exclude last row (constant)
        
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
                
    
    def forecast(self, fhor: int = 12, plot_forecast: bool = True, cred_interval: list = [0.68, 0.95],
                 last_k: int = None, n_breaks: int = 10, zero_line: bool = False):
        """
        Generate Bayesian forecasts using posterior draws of beta and Sigma.
        
        Parameters:
            fhor (int): Forecast horizon (e.g., 12 quarters)
            cred_interval (list): List of credible intervals to display (e.g., [0.68, 0.95])
            last_k (int): If set, show only last_k historical periods + forecast. If None, show full history.
            n_breaks (int): Number of x-axis breaks (year ticks).
            zero_line (bool): Whether to include a horizontal zero line in plots.

        Returns:
            forecasts: np.ndarray of shape (n_draws, steps, n_endo)
        """

        n_draws = len(self.beta_draws)
        n_endo = self.n_endo
        lags = self.lags
        k = self.ncoeff_eq

        # Initialize forecast array
        self.forecasts = np.zeros((n_draws, fhor, n_endo))
        Y_history = self.yy[-lags:, :]  # shape (lags, n_endo)

        # Exogenous forecast matrix (e.g., constant term)
        if self.constant:
            Xexo_future = np.ones((fhor, 1))  # constant-only
        else:
            Xexo_future = np.zeros((fhor, 0))  # no exogenous

        for i in range(n_draws):
            beta_vec = self.beta_draws[i]
            Sigma = self.Sigma_draws[i]

            # Reshape beta
            B = self.reshape_beta(beta_vec, k, n_endo)  # shape (k, n_endo)

            # Initialize history (copy for forecast path)
            Y = Y_history.copy().tolist()

            for h in range(fhor):
                Y_lags = np.hstack([Y[-lag] for lag in range(1, lags + 1)])
                X_t = Y_lags
                if self.constant:
                    X_t = np.hstack([X_t, Xexo_future[h]])

                # Compute mean forecast (no shock)
                y_deterministic = X_t @ B
                self.mean_forecasts[i, h, :] = y_deterministic

                # Add stochastic component
                eps = multivariate_normal(mean=np.zeros(n_endo), cov=Sigma)
                y_forecast = y_deterministic + eps

                self.forecasts[i, h, :] = y_forecast
                Y.append(y_forecast)
                
        if plot_forecast:
            forecast_plot = generate_forecast_plots(self, self.forecasts, cred_interval, last_k, n_breaks, zero_line, forecast_type="Unconditional")
            display(forecast_plot)

        return self.forecasts
    
    
    def _solve_shocks(self, conditions, fmat, ortirf):
        """
        Solve for structural shocks (eta) such that the conditional forecast matches the desired path.
        
        Parameters:
            conditions (steps x n_endo): matrix with np.nan for unconstrained
            fmat (steps x n_endo): baseline forecast
            ortirf (steps x n_endo x n_endo): orthogonal IRFs

        Returns:
            eta (steps x n_endo): structural shocks
        """
        steps, n = conditions.shape
        R = []
        r = []

        for t in range(steps):
            for j in range(n):
                target = conditions[t, j]
                if not np.isnan(target):
                    # Right-hand side difference
                    r.append(target - fmat[t, j])
                    
                    # Construct 1 x (n * steps) row
                    R_row = np.zeros((n * steps,))
                    for k in range(t + 1):
                        irf_block = ortirf[t - k, j, :]  # (n,)
                        R_row[k * n: (k + 1) * n] = irf_block
                    R.append(R_row)

        if not R:
            return np.zeros((steps, n))  # No conditions

        R = np.vstack(R)
        r = np.array(r)

        # Solve R * eta_vec = r
        try:
            eta_vec = np.linalg.solve(R, r)
        except np.linalg.LinAlgError:
            eta_vec, *_ = np.linalg.lstsq(R, r, rcond=None)

        # Reshape to (steps, n)
        eta = eta_vec.reshape(steps, n)

        return eta
    
    def conditional_forecast(self, conditions: np.ndarray, fhor: int = 12, plot_forecast: bool = True, 
                             cred_interval: list = [0.68, 0.95], last_k: int = None, n_breaks: int = 10, 
                             zero_line: bool = False):
        """
        Generate conditional forecasts using structural shocks.

        Parameters:
            conditions (np.ndarray): (fhor x n_endo) matrix with NaNs where no condition is imposed
            fhor (int): Forecast horizon

        Returns:
            conditional_forecast: (draws, fhor, n_endo)
            shock_record: (draws, fhor, n_endo)
        """
        n_endo = self.n_endo

        # === Assume no exogenous variables beyond constant ===
        data_exo_a = np.zeros((self.yy.shape[0], 0))
        data_exo_p = np.zeros((fhor, 0))

        if self.constant:
            data_exo_a = np.hstack([np.ones((self.yy.shape[0], 1)), data_exo_a])
            data_exo_p = np.hstack([np.ones((fhor, 1)), data_exo_p])

        # === Initialize outputs ===
        shock_record = np.zeros((self.n_draws, fhor, n_endo))

        # Compute IRFs if not already available
        if not hasattr(self, 'ir_draws') or len(self.ir_draws) == 0:
            self.compute_irfs(plot_irfs=False)

        for i in range(self.n_draws):
            # Forecasts without shocks
            fmat = self.mean_forecasts[i]  # shape (fhor, n_endo)

            # IRFs (ortho)
            ortirf = self.ir_draws[i][:fhor]  # shape (fhor, n_endo, n_endo)

            # Solve for shocks to meet conditions
            eta = self._solve_shocks(conditions, fmat, ortirf)
            eta = eta.reshape(fhor, n_endo)

            # Conditional forecast: fmat + contribution of eta via IRFs
            cdforecast = np.zeros((fhor, n_endo))
            for jj in range(fhor):
                shock_contrib = np.zeros(n_endo)
                for kk in range(jj + 1):
                    shock_contrib += ortirf[jj - kk, :, :] @ eta[kk, :]
                cdforecast[jj, :] = fmat[jj, :] + shock_contrib

            self.cond_forecasts[i, :, :] = cdforecast
            shock_record[i, :, :] = eta
            
        if plot_forecast:
            forecast_plot = generate_forecast_plots(self, self.cond_forecasts, cred_interval, last_k, n_breaks, zero_line, forecast_type="Conditional")
            display(forecast_plot)

        return self.cond_forecasts, shock_record