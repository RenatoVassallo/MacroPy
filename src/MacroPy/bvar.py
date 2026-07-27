import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Optional, Dict, Tuple, Union
from IPython.display import display
from numpy.linalg import inv, eigvals
from scipy.optimize import minimize
from scipy.stats import invwishart
from .data_handling import prepare_data, estimate_ols
from .priors import (
    MinnesotaPrior,
    NormalWishartPrior,
    NormalDiffusePrior,
    nw_conjugate_moments,
    nw_log_marginal_likelihood,
)
from .plots import (
    generate_coeff_plot,
    generate_irf_plots,
    generate_forecast_plots,
    generate_fevd_plot,
    generate_exog_coeff_plot,
)
from .summary import generate_summary

class BayesianVAR:
    def __init__(
        self,
        y: pd.DataFrame,
        lags: int = 1,
        constant: bool = True,
        timetrend: bool = False,
        prior_type: int = 1,
        prior_params: dict = {
            "mn_mean": 1,
            "lamda1": 0.2,
            "lamda2": 0.5,
            "lamda3": 1,
            "lamda4": 1e5
        },
        b_exo: np.ndarray = None,
        exog: Optional[pd.DataFrame] = None,
        post_draws: int = 5000,
        burnin: float = 0.5,
        hor: int = 20,
        fhor: int = 12,
        irf_1std: int = 1,
        covid_window: Optional[Tuple] = None,
        covid_mode: Optional[str] = None,
        covid_scales: Optional[Union[float, np.ndarray]] = None,
        covid_n_free: int = 3,
        seed: Optional[int] = None,
    ):

        """
        BayesianVAR: A flexible Bayesian Vector Autoregression model for time series analysis.

        This class implements a Bayesian VAR framework for multivariate time series, supporting
        multiple prior structures, posterior simulation via Gibbs sampling, and a range of tools 
        for structural analysis including impulse responses, variance decomposition, and forecasting.

        Parameters
        ----------
        y : pd.DataFrame
            Time series data with datetime index and one column per endogenous variable.
        
        lags : int, default=1
            Number of lags to include in the VAR specification.
        
        constant : bool, default=True
            Whether to include an intercept (constant) in the model.
        
        timetrend : bool, default=False
            Whether to include a deterministic linear time trend.
        
        prior_type : int, default=1
            Prior specification to use:
                1 = Minnesota Prior (shrinkage on own and cross lags)
                2 = Normal-Wishart Prior (conjugate prior for VAR)
                3 = Normal-Diffuse Prior (non-informative)

        prior_params (dict): Dictionary of hyperparameters for the chosen prior.
                Default for Minnesota:
                    - mn_mean: Prior mean on first own lag
                    - lamda1: Own lag shrinkage
                    - lamda2: Cross lag shrinkage
                    - lamda3: Lag decay
                    - lamda4: Constant term variance

        b_exo : np.ndarray, optional
            Block exogeneity mask (n_endo x n_endo boolean array). If variable i does not depend
            on lagged values of variable j, set b_exo[i, j] = 0. Allows partial system specification.

        exog : pd.DataFrame, optional
            Additional exogenous regressors (e.g., time dummies, oil prices). Must share its
            index with `y`. Each column is appended to every equation after the constant and
            trend, and receives a loose Minnesota prior so the likelihood drives the posterior.
            For forecasting, future values default to zero; pass `future_exog` to `forecast`
            to override.

        post_draws : int, default=5000
            Total number of posterior draws (including burn-in).

        burnin : float, default=0.5
            Proportion of posterior draws to discard as burn-in (e.g., 0.5 = discard 50%).

        hor : int, default=20
            Number of periods for impulse response function (IRF) analysis.

        fhor : int, default=12
            Number of periods to forecast in unconditional and conditional forecasts.

        irf_1std : int, default=1
            Type of IRF shock scaling:
                - 1 = 1 standard deviation shock
                - 0 = unit shock in structural space (scaled by Cholesky factor)

        covid_window : tuple, optional
            ``(start, end)`` of the extreme-volatility window, e.g.
            ``("2020-03", "2021-06")``. Endpoints are inclusive and parsed as
            periods, so ``"2021-06"`` covers the whole of June 2021 (and
            ``"2021Q2"`` the whole quarter). Required when `covid_mode` is set.

        covid_mode : {"lenza-primiceri", "dummies", None}, default=None
            How to treat the observations inside `covid_window`:
                - "lenza-primiceri": Lenza & Primiceri (2022) volatility
                  scaling. Residuals in the window have variance s_t^2 * Sigma;
                  estimation applies the exact GLS reweighting (rows divided by
                  s_t) to the data entering the OLS moments and all priors, so
                  pandemic observations stop distorting the coefficients while
                  Sigma keeps its ordinary-times scale (which is what forecasts
                  use). Scales are estimated by maximizing the analytic
                  conjugate Normal-Wishart marginal likelihood unless
                  `covid_scales` is given.
                - "dummies": one 0/1 dummy per observation in the window,
                  appended to the exogenous block (cheaper fallback; future
                  dummy values default to zero in forecasts).
                - None: no treatment.

        covid_scales : float or array-like, optional
            Fixed volatility scales s_t (>= 1) for the window observations under
            "lenza-primiceri": a scalar (same scale for the whole window) or an
            array with one value per window observation. If None, scales are
            estimated (empirical Bayes).

        covid_n_free : int, default=3
            Number of freely-estimated scales at the start of the window; the
            remaining ones decay geometrically toward 1 as in Lenza-Primiceri
            (s_t = 1 + (s_m - 1) * rho^j, with rho estimated).

        seed : int, optional
            Random seed for full reproducibility. Each stochastic method
            (`sample_posterior`, `forecast`) draws from its own deterministic
            generator derived from the seed, so two runs with the same data and
            seed produce identical draws regardless of call order, and repeated
            calls to the same method are idempotent. None (default) keeps
            non-deterministic behavior.

        Examples
        --------
        >>> from MacroPy import BayesianVAR
        >>> import pandas as pd

        >>> # Load time series data
        >>> df = pd.read_csv("Macro_Data.csv", index_col=0, parse_dates=True)

        >>> # Initialize BVAR model
        >>> bvar = BayesianVAR(df)

        >>> # Print model summary
        >>> bvar.model_summary()

        >>> # Sample posterior
        >>> post_draws = bvar.sample_posterior(plot_coefficients=True)

        >>> # Impulse response functions
        >>> irfs_results = bvar.compute_irfs(plot_irfs=True)

        >>> # Forecast error variance decomposition
        >>> fevd_results = bvar.compute_fevd(plot_fevd=True)

        >>> # Unconditional Forecasting
        >>> fore_results = bvar.forecast(plot_forecast=True)
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
        # Copy: the default dict is shared across instances and must never be
        # mutated in place (select_hyperparameters updates it per instance).
        self.prior_params = dict(prior_params)
        self.b_exo = b_exo
        self.seed = seed
        self.post_draws = post_draws
        self.burnin = int(burnin * post_draws)
        self.n_draws = self.post_draws - self.burnin  # Effective number of draws after burn-in
        self.hor = hor
        self.fhor = fhor
        self.irf_1std = irf_1std
        self.mean_forecasts = np.zeros((self.n_draws, self.fhor, self.n_endo))  # Forecasts without shocks
        self.forecasts = None      # With shocks
        self.cond_forecasts = np.zeros((self.n_draws, fhor, self.n_endo))

        # Validate and store user-supplied exogenous regressors
        self.exog_names: List[str] = []
        if exog is None:
            self.exog = None
            self.n_exog_user = 0
        else:
            if isinstance(exog, pd.Series):
                exog = exog.to_frame()
            if not isinstance(exog, pd.DataFrame):
                raise ValueError("`exog` must be a pandas DataFrame or Series aligned with `y`.")
            if not exog.index.equals(y.index):
                raise ValueError("`exog.index` must match `y.index` exactly.")
            if exog.isna().any().any():
                raise ValueError("`exog` cannot contain missing values.")
            self.exog_names = list(map(str, exog.columns))
            self.exog = exog.to_numpy(dtype=float)
            self.n_exog_user = self.exog.shape[1]

        # --- COVID treatment setup -------------------------------------------
        if covid_mode not in (None, "lenza-primiceri", "dummies"):
            raise ValueError("covid_mode must be 'lenza-primiceri', 'dummies' or None.")
        if covid_mode is not None and covid_window is None:
            raise ValueError("covid_mode requires a covid_window=(start, end).")
        self.covid_mode = covid_mode
        self.covid_window = covid_window
        self.covid_scales = None

        if covid_mode == "dummies":
            dummies = self._build_covid_dummies(covid_window)
            self.exog_names = self.exog_names + list(dummies.columns)
            dummy_arr = dummies.to_numpy(dtype=float)
            self.exog = dummy_arr if self.exog is None else np.hstack([self.exog, dummy_arr])
            self.n_exog_user = self.exog.shape[1]

        # Number of exogenous variables and coefficients
        self.n_exo = int(constant) + int(timetrend) + self.n_exog_user
        self.ncoeff_eq = self.n_endo * self.lags + self.n_exo
        self.ncoeff = self.ncoeff_eq * self.n_endo

        # Organize data into YX format
        self.yy, self.XX = prepare_data(
            self.y, self.lags, self.constant, self.timetrend, self.exog
        )

        # Adjust dates to match YYact (accounting for lags)
        self.yy_dates = self.dates[self.lags:]

        # Lenza-Primiceri volatility scaling: exact GLS reweighting of the
        # estimation sample. `yy_w`/`XX_w` feed the OLS moments, the priors and
        # the Gibbs sampler; `yy`/`XX` stay unweighted for forecasting history
        # and user-facing residuals. Without treatment the two are identical.
        self.obs_weights = np.ones(self.yy.shape[0])
        if covid_mode == "lenza-primiceri":
            widx = self._covid_window_index(covid_window)
            self.covid_scales = self._resolve_covid_scales(
                widx, covid_scales, covid_n_free
            )
            self.obs_weights[widx] = 1.0 / self.covid_scales
        self._covid_active = covid_mode == "lenza-primiceri"
        self.yy_w = self.yy * self.obs_weights[:, None] if self._covid_active else self.yy
        self.XX_w = self.XX * self.obs_weights[:, None] if self._covid_active else self.XX

        # Compute OLS estimates for initial values
        self.b_ols, self.Sigma_ols = estimate_ols(self.yy_w, self.XX_w)

        # Select prior distribution
        prior_dict = {1: "Minnesota", 2: "Normal-Wishart", 3: "Normal-Diffuse"}

        if prior_type not in prior_dict:
            raise ValueError("Invalid prior type. Choose 1 (Minnesota), 2 (Normal-Wishart), or 3 (Normal-Diffuse).")

        self.prior_name = prior_dict[prior_type]
        self._build_prior()

        # Storage for draws
        self.beta_draws = []
        self.Sigma_draws = []

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_prior(self):
        """(Re)build ``self.prior`` from the current hyperparameters."""
        args = (self.yy_w, self.XX_w, self.lags, self.ncoeff_eq,
                self.prior_params, self.b_exo)
        if self.prior_type == 1:
            self.prior = MinnesotaPrior(*args)
        elif self.prior_type == 2:
            self.prior = NormalWishartPrior(*args)
        elif self.prior_type == 3:
            self.prior = NormalDiffusePrior(*args)

    @staticmethod
    def _parse_covid_window(covid_window) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Inclusive (start, end) timestamps; bare periods cover their span."""
        start, end = covid_window
        start_ts = start if isinstance(start, pd.Timestamp) else pd.Period(str(start)).start_time
        end_ts = end if isinstance(end, pd.Timestamp) else pd.Period(str(end)).end_time
        if end_ts < start_ts:
            raise ValueError("covid_window end precedes its start.")
        return start_ts, end_ts

    def _build_covid_dummies(self, covid_window) -> pd.DataFrame:
        """One 0/1 dummy per observation of `y` inside the window."""
        start_ts, end_ts = self._parse_covid_window(covid_window)
        mask = (self.dates >= start_ts) & (self.dates <= end_ts)
        window_dates = self.dates[mask]
        if len(window_dates) == 0:
            raise ValueError("covid_window contains no observations of `y`.")
        if mask[: self.lags].any():
            raise ValueError(
                "covid_window overlaps the pre-sample (first `lags` observations)."
            )
        dummies = pd.DataFrame(0.0, index=self.dates,
                               columns=[f"covid_{d.strftime('%Y-%m')}" for d in window_dates])
        for j, d in enumerate(window_dates):
            dummies.iloc[self.dates.get_loc(d), j] = 1.0
        return dummies

    def _covid_window_index(self, covid_window) -> np.ndarray:
        """Positions of the window observations within the estimation sample."""
        start_ts, end_ts = self._parse_covid_window(covid_window)
        mask = (self.yy_dates >= start_ts) & (self.yy_dates <= end_ts)
        widx = np.flatnonzero(mask)
        if widx.size == 0:
            raise ValueError("covid_window contains no observations of the estimation sample.")
        return widx

    def _resolve_covid_scales(self, widx: np.ndarray, covid_scales,
                              covid_n_free: int) -> np.ndarray:
        """Fixed scales if provided, otherwise marginal-likelihood estimates."""
        W = widx.size
        if covid_scales is not None:
            scales = np.asarray(covid_scales, dtype=float).ravel()
            if scales.size == 1:
                scales = np.repeat(scales, W)
            if scales.size != W:
                raise ValueError(
                    f"covid_scales must be a scalar or have one value per window "
                    f"observation ({W}); got {scales.size}."
                )
            if np.any(scales < 1.0):
                raise ValueError("covid_scales must be >= 1.")
            return scales
        return self._estimate_covid_scales(widx, covid_n_free)

    def _estimate_covid_scales(self, widx: np.ndarray, n_free: int) -> np.ndarray:
        """
        Estimate Lenza-Primiceri volatility scales by maximizing the analytic
        conjugate Normal-Wishart marginal likelihood (empirical Bayes).

        The first ``n_free`` window observations get free scales; later ones
        decay geometrically toward one: ``s_{m+j} = 1 + (s_m - 1) * rho^j``.
        """
        W = widx.size
        m = int(min(max(n_free, 1), W))
        moments = nw_conjugate_moments(self.yy, self.XX, self.lags,
                                       self.n_endo, self.prior_params)

        def scales_from(theta):
            s_free = np.exp(theta[:m])          # s >= 1 via log-parametrization
            scales = np.empty(W)
            scales[:m] = s_free
            if W > m:
                rho = theta[m]
                j = np.arange(1, W - m + 1)
                scales[m:] = 1.0 + (s_free[-1] - 1.0) * rho ** j
            return scales

        def neg_logml(theta):
            w = np.ones(self.yy.shape[0])
            w[widx] = 1.0 / scales_from(theta)
            val = nw_log_marginal_likelihood(self.yy, self.XX, moments, obs_weights=w)
            return 1e12 if not np.isfinite(val) else -val

        # Start from the OLS residual-scale heuristic inside the window.
        resid = self.yy - self.XX @ np.linalg.lstsq(self.XX, self.yy, rcond=None)[0]
        base = np.median(np.abs(resid), axis=0) + 1e-12
        ratio = np.abs(resid[widx[:m]]) / base[None, :]
        s0 = np.clip(np.median(ratio, axis=1), 1.5, 50.0)
        x0 = np.concatenate([np.log(s0), [0.7]]) if W > m else np.log(s0)
        bounds = [(0.0, np.log(500.0))] * m + ([(0.01, 0.99)] if W > m else [])

        res = minimize(neg_logml, x0, method="L-BFGS-B", bounds=bounds)
        self._covid_opt = res
        return scales_from(res.x)

    _RNG_STAGES = {"posterior": 0, "forecast": 1, "conditional": 2}

    def _rng(self, stage: str) -> np.random.Generator:
        """
        Deterministic per-stage generator.

        With a seed, every call for a given stage returns a generator in the
        same initial state, so `sample_posterior` and `forecast` are idempotent
        and independent of call order. Without a seed, fresh entropy is used.
        """
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng([int(self.seed), self._RNG_STAGES[stage]])
        
    def model_summary(self):
        """Print a summary of the Bayesian VAR model."""
        display(generate_summary(self))

    def plot_exog_posteriors(
        self,
        bins: int = 30,
        ncol: int = None,
        series_titles: list = None,
        palette: list = None,
    ):
        """
        Display marginal posterior histograms of the user-supplied exogenous coefficients.

        One panel per equation in the system (auto-arranged into a balanced
        grid), with the posteriors of every exogenous regressor overlaid as
        semi-transparent histograms with a shared color legend.

        Parameters
        ----------
        bins : int, default=30
            Number of histogram bins.
        ncol : int, optional
            Override the auto-computed number of columns in the facet grid.
        series_titles : list of str, optional
            Equation titles to display instead of ``self.names``.
        palette : list of str, optional
            Hex colors for the exogenous regressors (cycled if needed).

        Returns
        -------
        plotnine.ggplot
            The plot object (also rendered inline via IPython display).
        """
        plot = generate_exog_coeff_plot(
            self, bins=bins, ncol=ncol, series_titles=series_titles, palette=palette,
        )
        display(plot)
        return plot
    
    @staticmethod
    def is_stable(Bcomp):
        """Check VAR stability based on companion matrix eigenvalues."""
        return np.all(np.abs(eigvals(Bcomp)) < 1)

    @staticmethod
    def build_companion_matrix(B, N, P):
        """
        Construct the VAR companion matrix from coefficient matrix B.

        Assumes the first ``N * P`` rows of B hold the lag coefficients and any
        remaining rows correspond to deterministic / exogenous regressors that
        do not enter the companion form.
        """
        Bcomp = np.zeros((N * P, N * P))
        Bcomp[:N, :] = B[:N * P, :].T  # drop the trailing exogenous rows

        if P > 1:
            Bcomp[N:, :-N] = np.eye(N * (P - 1))

        return Bcomp

    @staticmethod
    def reshape_beta(beta_vec, ncoeff_eq, N):
        """Reshape vectorized beta into coefficient matrix."""
        return beta_vec.reshape((ncoeff_eq, N), order='F')

    def sample_posterior(self, plot_coefficients: bool = False) -> dict:
        """
        Draw posterior samples for VAR coefficients and variance-covariance matrix.
        
        Ensures draws are stable (companion eigenvalues < 1).

        Returns
        -------
            dict: A dictionary with keys "beta_draws" and "Sigma_draws", each containing posterior samples.
        """
        rng = self._rng("posterior")
        XtX = self.XX_w.T @ self.XX_w
        b_ols, Sigma_ols = self.b_ols, self.Sigma_ols
        b_prior, H_prior = self.prior["b0"], self.prior["H"]
        Scale0, alpha0 = self.prior.get("Scale0"), self.prior.get("alpha0")
        Sigma = Sigma_ols.copy()

        self.beta_draws = []
        self.Sigma_draws = []
        self.resid_draws = []

        for _ in tqdm(range(self.post_draws), desc="Sampling Posterior"):
            Sigma_inv = inv(Sigma) if self.prior_type in [2, 3] else inv(Sigma_ols)
            invH = inv(H_prior)

            # Posterior mean and variance
            V_post = inv(invH + np.kron(Sigma_inv, XtX))
            M_post = V_post @ (invH @ b_prior + np.kron(Sigma_inv, XtX) @ b_ols)

            # Draw stable beta
            while True:
                beta_vec = rng.multivariate_normal(mean=M_post, cov=V_post)
                B = self.reshape_beta(beta_vec, self.ncoeff_eq, self.n_endo)
                Bcomp = self.build_companion_matrix(B, self.n_endo, self.lags)
                if self.is_stable(Bcomp):
                    break

            # Reduced-form residuals (unweighted, user-facing)
            resid = self.yy - self.XX @ B

            # Draw Sigma if applicable
            if self.prior_type in [2, 3]:
                # GLS-weighted residuals: homoskedastic units, so Sigma keeps
                # its ordinary-times scale under the COVID reweighting.
                resid_fit = (self.yy_w - self.XX_w @ B) if self._covid_active else resid
                scale_term = resid_fit.T @ resid_fit
                if self.prior_type == 2:
                    scale_term += Scale0
                    df = alpha0 + self.yy.shape[0]
                else:
                    df = self.yy.shape[0]
                Sigma = invwishart.rvs(df=df, scale=scale_term, random_state=rng)

            # Store draws
            self.beta_draws.append(beta_vec)
            self.Sigma_draws.append(Sigma if self.prior_type in [2, 3] else Sigma_ols)
            self.resid_draws.append(resid)

        # Apply burn-in
        self.beta_draws = np.array(self.beta_draws[self.burnin:])
        self.Sigma_draws = np.array(self.Sigma_draws[self.burnin:])
        self.resid_draws = np.array(self.resid_draws[self.burnin:])

        # Optional: plot coefficient draws
        if plot_coefficients:
            const_plot, var_plots = generate_coeff_plot(self)
            display(const_plot)
            for i, plot in enumerate(var_plots[:2]):
                display(plot)
            if len(var_plots) > 2:
                print("Note: Only showing first 2 lags of coefficients.")

        return {
            "beta_draws": self.beta_draws,
            "Sigma_draws": self.Sigma_draws,
            "resid_draws": self.resid_draws
        }


    def compute_irfs(self, plot_irfs: bool = False, cred_interval: Union[float, List[float]] = 0.68) -> np.ndarray:
        """
        Compute impulse response functions (IRFs) from posterior draws.

        Each posterior draw generates one IRF matrix of shape [horizon, variables, shocks].

        Parameters
        ----------
            plot_irfs (bool): If True, plots IRFs with given credible intervals.
            cred_interval (float or list): Credible interval(s) for IRF plots. E.g., 0.68 or [0.68, 0.95].

        Returns
        -------
            np.ndarray: Array of shape [n_draws, horizon, variables, shocks], storing IRFs.
        """
        if isinstance(cred_interval, float):
            cred_interval = [cred_interval]
    
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
    

    def compute_fevd(
        self,
        plot_fevd: bool = True,
        series_titles: Optional[List[str]] = None,
        shock_titles: Optional[List[str]] = None,
        title: Optional[str] = None
    ) -> Dict[str, Union[np.ndarray, None]]:
        """
        Compute the Forecast Error Variance Decomposition (FEVD) from posterior draws of a Bayesian VAR model.

        The FEVD quantifies the contribution of each structural shock to the forecast error variance 
        of each variable over different horizons. Computation is based on orthogonalized impulse responses 
        (via Cholesky decomposition) for each posterior draw.

        Parameters
        ----------
        plot_fevd : bool, default=True
            Whether to display the FEVD plots.

        series_titles : list of str, optional
            Custom names for the endogenous variables (for plotting purposes).

        shock_titles : list of str, optional
            Custom names for the structural shocks (for plotting purposes).

        title : str, optional
            Custom plot title.

        Returns
        -------
        dict
            A dictionary containing:
            - 'fevd' : np.ndarray of shape (horizon, shock, variable)
                Posterior mean FEVD, expressed in percentage terms.
            - 'fevd_draws' : np.ndarray of shape (n_draws, horizon, shock, variable)
                Raw FEVD draws from each posterior sample.
        """
        N, P, H = self.n_endo, self.lags, self.hor
        n_draws = len(self.beta_draws)

        # Store FEVDs across draws: [draw, horizon, shock, variable]
        fevd_draws = np.zeros((n_draws, H, N, N))

        for d in tqdm(range(n_draws), desc="Computing FEVD"):
            B = self.reshape_beta(self.beta_draws[d], self.ncoeff_eq, N)
            Sigma = self.Sigma_draws[d]

            # Structural impact matrix via Cholesky
            try:
                S = np.linalg.cholesky(Sigma)
            except np.linalg.LinAlgError:
                continue

            # Wold multipliers
            PSI = np.zeros((N, N, H))
            PSI[:, :, 0] = np.eye(N)

            # Lag polynomial Bp
            Bp = np.zeros((N, N, P))
            for p in range(P):
                Bp[:, :, p] = B[p * N:(p + 1) * N, :].T

            for h in range(1, H):
                for j in range(1, min(P, h) + 1):
                    PSI[:, :, h] += PSI[:, :, h - j] @ Bp[:, :, j - 1]

            for shock in range(N):
                MSE = np.zeros((N, N, H))
                MSE[:, :, 0] = Sigma

                MSE_shock = np.zeros((N, N, H))
                S_shock = S[:, shock].reshape(-1, 1)
                MSE_shock[:, :, 0] = S_shock @ S_shock.T

                for h in range(1, H):
                    PSI_h = PSI[:, :, h]
                    MSE[:, :, h] = MSE[:, :, h - 1] + PSI_h @ Sigma @ PSI_h.T
                    MSE_shock[:, :, h] = MSE_shock[:, :, h - 1] + PSI_h @ (S_shock @ S_shock.T) @ PSI_h.T

                for h in range(H):
                    FECD = MSE_shock[:, :, h] / MSE[:, :, h]
                    fevd_draws[d, h, shock, :] = 100 * np.diag(FECD)

        # Posterior average
        self.fevd = np.nanmean(fevd_draws, axis=0)  # [horizon, shock, variable]
        self.fevd_draws = fevd_draws  # Save raw draws for later use

        if plot_fevd:
            fevd_plot = generate_fevd_plot(self, series_titles, shock_titles, title)
            display(fevd_plot)

        return {
            "fevd": self.fevd,
            "fevd_draws": self.fevd_draws
        }
                
    
    def _build_future_exo(self, fhor: int, future_exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Build the (fhor, n_exo) regressor matrix used for forecasting.

        Layout matches ``prepare_data``: constant?, trend?, user exog columns.
        For user-supplied exog, future values default to zeros (a natural
        choice for event dummies after the event window).
        """
        blocks = []
        if self.constant:
            blocks.append(np.ones((fhor, 1)))
        if self.timetrend:
            nobs = self.yy.shape[0]
            blocks.append(np.arange(nobs + 1, nobs + fhor + 1, dtype=float).reshape(-1, 1))

        if self.n_exog_user > 0:
            if future_exog is None:
                blocks.append(np.zeros((fhor, self.n_exog_user)))
            else:
                if isinstance(future_exog, pd.DataFrame):
                    missing = [c for c in self.exog_names if c not in future_exog.columns]
                    if missing:
                        raise ValueError(f"`future_exog` is missing exog columns: {missing}")
                    fe_arr = future_exog[self.exog_names].to_numpy(dtype=float)
                else:
                    fe_arr = np.asarray(future_exog, dtype=float)
                    if fe_arr.ndim == 1:
                        fe_arr = fe_arr.reshape(-1, 1)
                if fe_arr.shape != (fhor, self.n_exog_user):
                    raise ValueError(
                        f"`future_exog` must have shape ({fhor}, {self.n_exog_user}); "
                        f"got {fe_arr.shape}."
                    )
                blocks.append(fe_arr)

        if not blocks:
            return np.zeros((fhor, 0))
        return np.hstack(blocks)

    def forecast(
        self,
        fhor: int = 12,
        plot_forecast: bool = True,
        cred_interval: list = [0.68, 0.95],
        last_k: int = None,
        n_breaks: int = 10,
        zero_line: bool = False,
        future_exog: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Generate Bayesian forecasts from posterior draws of beta and Sigma.

        Parameters
        ----------
        fhor : int
            Forecast horizon (e.g., 12 quarters).
        plot_forecast : bool
            Whether to display the forecast fan chart.
        cred_interval : list
            List of credible intervals to display (e.g., [0.68, 0.95]).
        last_k : int
            Number of recent historical periods to display. If None, show all.
        n_breaks : int
            Number of x-axis ticks (typically years).
        zero_line : bool
            Whether to add a horizontal zero line in the plot.
        future_exog : pd.DataFrame, optional
            Future values for user-supplied exogenous regressors. Required only
            if you want non-zero exog values over the forecast horizon (e.g.
            extending an oil-price path); for event dummies, leaving this as
            None correctly yields zeros after the event.

        Returns
        -------
        dict with:
            - "forecast_draws": ndarray of shape (n_draws, fhor, n_endo), predictive
              draws (future shocks added each period).
            - "mean_forecasts": ndarray of shape (n_draws, fhor, n_endo), deterministic
              no-shock paths (each iterated on its own deterministic lags, as in
              the Canova-Ferroni BVAR_ toolbox).
        """
        rng = self._rng("forecast")
        n_draws = len(self.beta_draws)
        n_endo = self.n_endo
        lags = self.lags
        k = self.ncoeff_eq

        self.forecasts = np.zeros((n_draws, fhor, n_endo))        # With shocks
        self.mean_forecasts = np.zeros((n_draws, fhor, n_endo))   # No shocks

        Y_history = self.yy[-lags:, :]  # Most recent lags
        Xexo_future = self._build_future_exo(fhor, future_exog)

        for i in range(n_draws):
            beta_vec = self.beta_draws[i]
            Sigma = self.Sigma_draws[i]
            B = self.reshape_beta(beta_vec, k, n_endo)

            # Two separate lag histories: the no-shock path must iterate on its
            # own deterministic values (Canova-Ferroni's `frcst_no_shock`),
            # while the predictive path iterates on the shocked values.
            Y_det = Y_history.copy().tolist()
            Y_sto = Y_history.copy().tolist()

            for h in range(fhor):
                exo_h = Xexo_future[h] if Xexo_future.shape[1] > 0 else None

                # No-shock (deterministic) path
                X_det = np.hstack([Y_det[-lag] for lag in range(1, lags + 1)])
                if exo_h is not None:
                    X_det = np.hstack([X_det, exo_h])
                y_det = X_det @ B
                self.mean_forecasts[i, h, :] = y_det
                Y_det.append(y_det)

                # Predictive path with Gaussian disturbances
                X_sto = np.hstack([Y_sto[-lag] for lag in range(1, lags + 1)])
                if exo_h is not None:
                    X_sto = np.hstack([X_sto, exo_h])
                eps = rng.multivariate_normal(mean=np.zeros(n_endo), cov=Sigma)
                y_sto = X_sto @ B + eps
                self.forecasts[i, h, :] = y_sto
                Y_sto.append(y_sto)

        if plot_forecast:
            forecast_plot = generate_forecast_plots(
                self, self.forecasts, cred_interval,
                last_k, n_breaks, zero_line, forecast_type="Unconditional"
            )
            display(forecast_plot)

        return {
            "forecast_draws": self.forecasts,
            "mean_forecasts": self.mean_forecasts
        }
    
    
    def _solve_shocks(self, conditions, fmat, ortirf, rng=None):
        """
        Draw the structural shocks (eta) that deliver the conditional forecast.

        Following Waggoner & Zha (1999) (as implemented in the Canova-Ferroni
        BVAR_ toolbox), the standardized structural shocks conditional on the
        restrictions ``R eta = r`` are distributed
        ``eta ~ N(R+ r, I - R+ R)``: the minimum-norm solution plus Gaussian
        noise in the null space of the restrictions.

        Parameters
        ----------
            conditions (steps x n_endo): matrix with np.nan for unconstrained
            fmat (steps x n_endo): baseline no-shock forecast
            ortirf (steps x n_endo x n_endo): 1-s.d. orthogonalized IRFs
            rng : np.random.Generator, optional
                If given, adds the null-space draw (full Waggoner-Zha
                distribution). If None, returns the conditional mean only.

        Returns
        -------
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
            # No conditions: shocks are unconstrained standard normals (the
            # conditional forecast collapses to the unconditional one).
            if rng is None:
                return np.zeros((steps, n))
            return rng.standard_normal((steps, n))

        R = np.vstack(R)
        r = np.array(r)

        # eta = V1 D^-1 U' r (+ V2 z): minimum-norm solution plus, when a
        # generator is supplied, the null-space component of Waggoner-Zha.
        U, D, Vt = np.linalg.svd(R, full_matrices=True)
        tol = max(R.shape) * np.finfo(float).eps * (D[0] if D.size else 0.0)
        rank = int(np.sum(D > tol))
        V = Vt.T
        eta_vec = V[:, :rank] @ ((U.T @ r)[:rank] / D[:rank])
        if rng is not None and V.shape[1] > rank:
            eta_vec = eta_vec + V[:, rank:] @ rng.standard_normal(V.shape[1] - rank)

        # Reshape to (steps, n)
        eta = eta_vec.reshape(steps, n)

        return eta

    def _ortho_irfs(self, hor: int) -> np.ndarray:
        """
        1-s.d. orthogonalized IRFs (n_draws, hor, N, N), independent of the
        display setting `irf_1std`. Conditioning must always operate in
        standardized structural units, otherwise the minimum-norm solution is
        computed in the wrong metric and no longer equals the Waggoner-Zha
        conditional expectation.
        """
        N, P = self.n_endo, self.lags
        n_draws = len(self.beta_draws)
        out = np.zeros((n_draws, hor, N, N))
        for d in range(n_draws):
            B = self.reshape_beta(self.beta_draws[d], self.ncoeff_eq, N)
            Bcomp = self.build_companion_matrix(B, N, P)
            try:
                S = np.linalg.cholesky(self.Sigma_draws[d])
            except np.linalg.LinAlgError:
                continue
            out[d, 0] = S
            Bpow = np.eye(N * P)
            for h in range(1, hor):
                Bpow = Bpow @ Bcomp
                out[d, h] = Bpow[:N, :N] @ S
        return out
    
    def conditional_forecast(self, conditions: Union[np.ndarray, dict], fhor: int = 12,
                             plot_forecast: bool = True,
                             cred_interval: list = [0.68, 0.95], last_k: int = None, n_breaks: int = 10,
                             zero_line: bool = False, future_exog: Optional[pd.DataFrame] = None,
                             shock_uncertainty: bool = True):
        """
        Generate conditional forecasts using structural shocks (Waggoner & Zha, 1999).

        For each posterior draw the standardized structural shocks are drawn from
        their full conditional distribution ``eta ~ N(R+ r, I - R+ R)``: the
        minimum-norm shocks that reproduce the imposed path plus Gaussian noise
        in the null space of the restrictions, as in the Canova-Ferroni BVAR_
        toolbox (`cforecasts.m`). Imposed conditions hold exactly in every draw;
        unconstrained variables carry both parameter and future-shock
        uncertainty.

        Parameters
        ----------
            conditions (np.ndarray or dict): Either an array of shape (fhor, n_endo) with
                                    NaNs for unrestricted values, or a friendly dict
                                    ``{variable_name: path}`` where each path is a list
                                    (horizons 1..len, shorter than fhor allowed, None/NaN
                                    for gaps), a ``{horizon: value}`` dict (1-based), or a
                                    scalar (horizon 1 only).
            fhor (int): Forecast horizon.
            plot_forecast (bool): Whether to display the resulting fan chart.
            cred_interval (list): Credible intervals to display (e.g., [0.68, 0.95]).
            last_k (int): If set, display only the last_k periods of history + forecast.
            n_breaks (int): Number of x-axis breaks (e.g., years).
            zero_line (bool): Whether to include a horizontal zero line in plots.
            future_exog (pd.DataFrame): Future values for exogenous regressors.
            shock_uncertainty (bool): If True (default), draw the Waggoner-Zha
                                    null-space shocks so conditional bands reflect
                                    future-shock uncertainty. If False, use only the
                                    minimum-norm (conditional-mean) shocks, so bands
                                    reflect parameter uncertainty alone.

        Returns
        -------
            cond_forecasts (np.ndarray): Shape (n_draws, fhor, n_endo), conditional forecasts.
            shock_record (np.ndarray): Shape (n_draws, fhor, n_endo), identified shocks to meet conditions.
        """
        conditions = self._conditions_to_matrix(conditions, fhor)
        rng = self._rng("conditional")
        n_draws = self.n_draws
        n_endo = self.n_endo
        self.cond_forecasts = np.zeros((n_draws, fhor, n_endo))
        shock_record = np.zeros((n_draws, fhor, n_endo))

        # Recompute mean_forecasts so they reflect the requested fhor + future_exog
        # (the base mean_forecasts may have been generated with different settings).
        if self.mean_forecasts.shape[1] != fhor or future_exog is not None:
            self.forecast(fhor=fhor, plot_forecast=False, future_exog=future_exog)

        # Conditioning operates in 1-s.d. structural units. When the display
        # setting is irf_1std=1 the cached ir_draws are exactly that and are
        # reused (computing them if absent/too short); under irf_1std=0 a
        # dedicated internal set is built so the conditional distribution does
        # not depend on the IRF display convention.
        if self.irf_1std == 1:
            if (not hasattr(self, 'ir_draws') or len(self.ir_draws) == 0
                    or self.ir_draws.shape[1] < fhor):
                prev_hor = self.hor
                self.hor = max(prev_hor, fhor)
                try:
                    self.compute_irfs(plot_irfs=False)
                finally:
                    self.hor = prev_hor
            ir_source = self.ir_draws
        else:
            ir_source = self._ortho_irfs(fhor)

        for i in range(n_draws):
            fmat = self.mean_forecasts[i]              # no-shock baseline
            ortirf = ir_source[i][:fhor]               # 1-s.d. IRFs: [h, y, shock]

            # Draw structural shocks consistent with the imposed conditions
            eta = self._solve_shocks(conditions, fmat, ortirf,
                                     rng=rng if shock_uncertainty else None)
            eta = eta.reshape(fhor, n_endo)

            # Build conditional forecast using those shocks
            cdforecast = np.zeros((fhor, n_endo))
            for h in range(fhor):
                contrib = np.zeros(n_endo)
                for j in range(h + 1):
                    contrib += ortirf[h - j] @ eta[j]
                cdforecast[h] = fmat[h] + contrib

            # Store forecast and shocks
            self.cond_forecasts[i] = cdforecast
            shock_record[i] = eta

        # Plot forecast fan chart
        if plot_forecast:
            forecast_plot = generate_forecast_plots(
                self, self.cond_forecasts, cred_interval, last_k,
                n_breaks, zero_line, forecast_type="Conditional"
            )
            display(forecast_plot)

        return self.cond_forecasts, shock_record

    # ------------------------------------------------------------------
    # Headless, tidy production outputs
    # ------------------------------------------------------------------

    def _conditions_to_matrix(self, conditions, fhor: int) -> np.ndarray:
        """
        Normalize conditions into the (fhor, n_endo) NaN matrix.

        Accepts the matrix itself (passed through after a shape check) or a
        dict ``{variable_name: path}`` where each path is a list/array over
        horizons 1..len (None/NaN for unconstrained gaps), a ``{horizon: value}``
        dict with 1-based horizons, or a scalar (horizon 1 only).
        """
        if not isinstance(conditions, dict):
            mat = np.asarray(conditions, dtype=float)
            if mat.shape != (fhor, self.n_endo):
                raise ValueError(
                    f"`conditions` must have shape ({fhor}, {self.n_endo}); got {mat.shape}."
                )
            return mat

        names = list(map(str, self.names))
        mat = np.full((fhor, self.n_endo), np.nan)
        for name, path in conditions.items():
            if str(name) not in names:
                raise ValueError(f"Unknown variable '{name}' in conditions; "
                                 f"expected one of {names}.")
            j = names.index(str(name))
            if isinstance(path, dict):
                for h, v in path.items():
                    if not 1 <= int(h) <= fhor:
                        raise ValueError(f"Condition horizon {h} for '{name}' outside 1..{fhor}.")
                    mat[int(h) - 1, j] = float(v)
            elif np.isscalar(path):
                mat[0, j] = float(path)
            else:
                arr = np.array([np.nan if v is None else float(v) for v in path])
                if arr.size > fhor:
                    raise ValueError(
                        f"Condition path for '{name}' has {arr.size} entries; fhor={fhor}."
                    )
                mat[: arr.size, j] = arr
        return mat

    def _forecast_index(self, fhor: int):
        """Future dates continuing `yy_dates` (None if frequency is not inferable)."""
        try:
            freq = pd.infer_freq(self.yy_dates)
            if freq is None:
                return None
            offset = pd.tseries.frequencies.to_offset(freq)
            last = self.yy_dates[-1]
            return pd.DatetimeIndex([last + offset * h for h in range(1, fhor + 1)])
        except (ValueError, TypeError):
            return None

    def _draws_to_frame(self, draws: np.ndarray, quantiles) -> pd.DataFrame:
        """Tidy (variable, horizon) frame with mean/median/quantiles of `draws`."""
        fhor = draws.shape[1]
        qs = sorted(float(q) for q in quantiles)
        if any(not 0.0 < q < 1.0 for q in qs):
            raise ValueError("Quantiles must lie strictly between 0 and 1.")
        future_dates = self._forecast_index(fhor)

        index = pd.MultiIndex.from_product(
            [list(map(str, self.names)), range(1, fhor + 1)],
            names=["variable", "horizon"],
        )
        data = {}
        if future_dates is not None:
            data["date"] = np.tile(future_dates, self.n_endo)
        data["mean"] = draws.mean(axis=0).T.ravel()
        data["median"] = np.median(draws, axis=0).T.ravel()
        for q in qs:
            data[f"q{round(q * 100):02d}"] = np.quantile(draws, q, axis=0).T.ravel()
        return pd.DataFrame(data, index=index)

    def forecast_frame(
        self,
        fhor: Optional[int] = None,
        quantiles=(0.05, 0.16, 0.84, 0.95),
        future_exog: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Unconditional forecast as a tidy DataFrame (no plotting).

        Runs `forecast(plot_forecast=False)` on the posterior draws and
        summarizes the predictive distribution (draws with shocks).

        Parameters
        ----------
        fhor : int, optional
            Forecast horizon (defaults to the model's `fhor`).
        quantiles : iterable of float, default=(0.05, 0.16, 0.84, 0.95)
            Predictive quantiles to report, each strictly between 0 and 1.
        future_exog : pd.DataFrame, optional
            Future exogenous values, as in `forecast`.

        Returns
        -------
        pd.DataFrame indexed by (variable, horizon 1..fhor) with columns
        ``[date?, mean, median, qXX...]`` (`date` present when the index
        frequency is inferable).
        """
        if len(self.beta_draws) == 0:
            raise RuntimeError("Call sample_posterior() before forecast_frame().")
        fhor = self.fhor if fhor is None else int(fhor)
        self.forecast(fhor=fhor, plot_forecast=False, future_exog=future_exog)
        return self._draws_to_frame(self.forecasts, quantiles)

    def conditional_forecast_frame(
        self,
        conditions: Union[np.ndarray, dict],
        fhor: Optional[int] = None,
        quantiles=(0.05, 0.16, 0.84, 0.95),
        future_exog: Optional[pd.DataFrame] = None,
        shock_uncertainty: bool = True,
    ) -> pd.DataFrame:
        """
        Conditional (Waggoner-Zha) forecast as a tidy DataFrame (no plotting).

        Parameters
        ----------
        conditions : dict or np.ndarray
            ``{variable_name: path}`` dict (see `conditional_forecast`) or the
            (fhor, n_endo) NaN matrix.
        fhor, quantiles, future_exog
            As in `forecast_frame`.
        shock_uncertainty : bool, default=True
            Include the Waggoner-Zha null-space shock draws (see
            `conditional_forecast`).

        Returns
        -------
        pd.DataFrame indexed by (variable, horizon) with columns
        ``[date?, mean, median, qXX...]``.
        """
        if len(self.beta_draws) == 0:
            raise RuntimeError("Call sample_posterior() before conditional_forecast_frame().")
        fhor = self.fhor if fhor is None else int(fhor)
        cond_draws, _ = self.conditional_forecast(
            conditions, fhor=fhor, plot_forecast=False, future_exog=future_exog,
            shock_uncertainty=shock_uncertainty,
        )
        return self._draws_to_frame(cond_draws, quantiles)

    # ------------------------------------------------------------------
    # Marginal likelihood and hyperparameter selection (GLP 2015)
    # ------------------------------------------------------------------

    def log_marginal_likelihood(self, prior_params: Optional[dict] = None) -> float:
        """
        Analytic log marginal likelihood under the conjugate Normal-Wishart
        (Minnesota-moment) prior at the given hyperparameters.

        Uses the Giannone-Lenza-Primiceri (2015) conjugate representation built
        from ``mn_mean``, ``lamda1``, ``lamda3`` and ``lamda4`` (see
        `select_hyperparameters` for why ``lamda2`` cannot enter). Under the
        Lenza-Primiceri COVID mode the estimated volatility scales are applied,
        so the value refers to the same reweighted likelihood used in
        estimation.
        """
        if self.b_exo is not None:
            raise ValueError(
                "The analytic marginal likelihood requires a Kronecker prior; "
                "per-equation block exogeneity (`b_exo`) is not representable."
            )
        params = self.prior_params if prior_params is None else prior_params
        moments = nw_conjugate_moments(self.yy, self.XX, self.lags,
                                       self.n_endo, params)
        w = self.obs_weights if self._covid_active else None
        return nw_log_marginal_likelihood(self.yy, self.XX, moments, obs_weights=w)

    def select_hyperparameters(
        self,
        select=("lamda1", "lamda3"),
        bounds: Optional[dict] = None,
        update: bool = True,
        verbose: bool = True,
    ) -> dict:
        """
        Data-driven Minnesota tightness by maximizing the marginal likelihood.

        Follows Giannone, Lenza & Primiceri (2015): the analytic marginal
        likelihood of the conjugate Normal-Wishart representation of the
        Minnesota prior is maximized over the requested hyperparameters
        (empirical Bayes). The optimum is then written into ``prior_params``
        and the sampling prior is rebuilt, so a subsequent
        `sample_posterior()` uses the selected tightness.

        Notes
        -----
        ``lamda2`` (own-versus-cross asymmetry) breaks the Kronecker structure
        that makes the marginal likelihood analytic (Kadiyala & Karlsson,
        1997), so it cannot be selected here; it keeps its current value in
        the sampling prior. Selectable: ``lamda1``, ``lamda3``, ``lamda4``.

        Parameters
        ----------
        select : iterable of str, default=("lamda1", "lamda3")
            Hyperparameters to optimize.
        bounds : dict, optional
            ``{name: (low, high)}`` overrides of the default search bounds
            ``lamda1 in [0.01, 5], lamda3 in [0.1, 5], lamda4 in [1, 1e6]``.
        update : bool, default=True
            Write the optimum into `prior_params` and rebuild the prior.
        verbose : bool, default=True
            Print the selected values.

        Returns
        -------
        dict with keys ``params`` (optimal values), ``log_ml``,
        ``initial_log_ml`` and ``success``.
        """
        allowed = {"lamda1", "lamda3", "lamda4"}
        select = list(select)
        bad = [s for s in select if s not in allowed]
        if bad:
            raise ValueError(
                f"Cannot select {bad} by marginal likelihood: only {sorted(allowed)} "
                "enter the conjugate (Kronecker) prior. In particular lamda2 "
                "requires asymmetric own/cross shrinkage, which has no analytic "
                "marginal likelihood."
            )
        default_bounds = {"lamda1": (0.01, 5.0), "lamda3": (0.1, 5.0),
                          "lamda4": (1.0, 1e6)}
        if bounds:
            default_bounds.update(bounds)

        initial_log_ml = self.log_marginal_likelihood()

        def neg_logml(x):
            trial = dict(self.prior_params)
            for name, val in zip(select, x):
                trial[name] = float(np.exp(val))
            val = self.log_marginal_likelihood(trial)
            return 1e12 if not np.isfinite(val) else -val

        x0 = np.log([float(self.prior_params.get(s, 0.2)) for s in select])
        log_bounds = [tuple(np.log(default_bounds[s])) for s in select]
        x0 = np.clip(x0, [b[0] for b in log_bounds], [b[1] for b in log_bounds])

        res = minimize(neg_logml, x0, method="L-BFGS-B", bounds=log_bounds)
        best = {name: float(np.exp(v)) for name, v in zip(select, res.x)}
        out = {
            "params": best,
            "log_ml": float(-res.fun),
            "initial_log_ml": float(initial_log_ml),
            "success": bool(res.success),
        }
        if update:
            self.prior_params.update(best)
            self._build_prior()
        if verbose:
            shown = ", ".join(f"{k} = {v:.4g}" for k, v in best.items())
            print(f"Selected hyperparameters ({shown}); "
                  f"log-ML {initial_log_ml:.2f} -> {out['log_ml']:.2f}")
        return out