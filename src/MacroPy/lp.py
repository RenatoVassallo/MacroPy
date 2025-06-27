import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from numpy.linalg import solve
from scipy.stats import norm

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
        knot_step = (upper_bound - lower_bound) / num_knots
        knots = lower_bound + knot_step * np.arange(-degree, num_knots)
        T = np.tile(knots, (len(horizon_grid), 1))
        X = np.tile(horizon_grid.reshape(-1, 1), (1, len(knots)))
        P = (X - T) / knot_step
        B = ((T <= X) & (X < T + knot_step)).astype(float)
        for k in range(1, degree + 1):
            B = (P * B + (k + 1 - P) * np.roll(B, shift=-1, axis=1)) / k
        return B

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