"""
State-space modeling, Kalman filtering and smoothing for MacroPy.

This module provides a small, didactic toolkit for linear Gaussian state-space
models used in trend/cycle decompositions (Hodrick-Prescott as a special case,
Clark 1987, generic unobserved-components models). Estimation is supported via
Maximum Likelihood (`MLEEstimator`) and Bayesian Gibbs sampling
(`BayesianStateSpace`) using a Durbin-Koopman simulation smoother.

The implementation favors clarity over speed: matrices are dense numpy arrays,
shapes are documented at every step, and intermediate quantities of the
recursion are exposed so they can be inspected from a notebook.

Canonical state-space form
--------------------------
Measurement:
    y_t   = Z_t * a_t + d_t + eps_t,    eps_t ~ N(0, H)
Transition:
    a_t   = T   * a_{t-1} + c + R * eta_t, eta_t ~ N(0, Q)

with state dimension `ns`, observable dimension `ny` and shock dimension `nq`.
Shapes:
    Z: (ny, ns),  T: (ns, ns),  R: (ns, nq),
    Q: (nq, nq),  H: (ny, ny),  c: (ns,), d: (ny,)
    a0: (ns,),    P0: (ns, ns)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import slogdet, solve
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import invgamma


# ---------------------------------------------------------------------------
# Core container
# ---------------------------------------------------------------------------


@dataclass
class StateSpaceModel:
    """
    Linear Gaussian state-space model.

    Build with the matrices `(Z, T, R, Q, H)` and the initial moments
    `(a0, P0)`, then call `filter(y)` and `smooth()`. The filter stores all
    intermediate quantities (predicted/updated states, innovations,
    log-likelihood contributions) on the instance.

    Parameters
    ----------
    Z, T, R, Q, H : np.ndarray
        State-space matrices. Q is the covariance of the shocks driving the
        states through R; H is the measurement noise covariance.
    a0 : np.ndarray, shape (ns,)
        Initial state mean.
    P0 : np.ndarray, shape (ns, ns)
        Initial state covariance.
    c : np.ndarray, optional
        Transition intercept, shape (ns,). Defaults to zero.
    d : np.ndarray, optional
        Measurement intercept, shape (ny,). Defaults to zero.
    """

    Z: np.ndarray
    T: np.ndarray
    R: np.ndarray
    Q: np.ndarray
    H: np.ndarray
    a0: np.ndarray
    P0: np.ndarray
    c: Optional[np.ndarray] = None
    d: Optional[np.ndarray] = None

    # filled by filter/smoother
    a_pred: Optional[np.ndarray] = field(default=None, repr=False)
    P_pred: Optional[np.ndarray] = field(default=None, repr=False)
    a_filt: Optional[np.ndarray] = field(default=None, repr=False)
    P_filt: Optional[np.ndarray] = field(default=None, repr=False)
    a_smooth: Optional[np.ndarray] = field(default=None, repr=False)
    P_smooth: Optional[np.ndarray] = field(default=None, repr=False)
    innov: Optional[np.ndarray] = field(default=None, repr=False)
    S_innov: Optional[np.ndarray] = field(default=None, repr=False)
    K_gain: Optional[np.ndarray] = field(default=None, repr=False)
    loglik_t: Optional[np.ndarray] = field(default=None, repr=False)
    loglik: Optional[float] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Validation / shape helpers
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self.Z = np.atleast_2d(self.Z).astype(float)
        self.T = np.atleast_2d(self.T).astype(float)
        self.R = np.atleast_2d(self.R).astype(float)
        self.Q = np.atleast_2d(self.Q).astype(float)
        self.H = np.atleast_2d(self.H).astype(float)
        self.a0 = np.asarray(self.a0, dtype=float).ravel()
        self.P0 = np.atleast_2d(self.P0).astype(float)
        ns = self.T.shape[0]
        ny = self.Z.shape[0]
        if self.c is None:
            self.c = np.zeros(ns)
        else:
            self.c = np.asarray(self.c, dtype=float).ravel()
        if self.d is None:
            self.d = np.zeros(ny)
        else:
            self.d = np.asarray(self.d, dtype=float).ravel()
        if self.Z.shape != (ny, ns):
            raise ValueError(f"Z shape {self.Z.shape} != (ny={ny}, ns={ns})")
        if self.T.shape != (ns, ns):
            raise ValueError(f"T shape {self.T.shape} != (ns, ns)")
        if self.R.shape[0] != ns:
            raise ValueError(f"R has {self.R.shape[0]} rows, expected ns={ns}")
        if self.Q.shape != (self.R.shape[1], self.R.shape[1]):
            raise ValueError("Q must be (nq, nq) with nq = R.shape[1]")
        if self.H.shape != (ny, ny):
            raise ValueError(f"H shape {self.H.shape} != (ny, ny)")
        if self.a0.shape != (ns,):
            raise ValueError(f"a0 shape {self.a0.shape} != (ns,)")
        if self.P0.shape != (ns, ns):
            raise ValueError(f"P0 shape {self.P0.shape} != (ns, ns)")

    @property
    def ns(self) -> int:
        return self.T.shape[0]

    @property
    def ny(self) -> int:
        return self.Z.shape[0]

    @property
    def nq(self) -> int:
        return self.R.shape[1]

    # ------------------------------------------------------------------
    # Kalman filter
    # ------------------------------------------------------------------

    def filter(self, y: np.ndarray, store: bool = True) -> float:
        """
        Run the Kalman filter on observations `y`.

        Parameters
        ----------
        y : np.ndarray
            Observations. Shape (T,) for univariate, (T, ny) otherwise.
            NaNs are treated as missing (the row is skipped in the update
            step, but the state prediction continues).
        store : bool, default True
            If False, only the total log-likelihood is computed and returned
            (faster path for likelihood evaluation in MLE).

        Returns
        -------
        loglik : float
            Sum of log-likelihood contributions across `t`.
        """
        Y = np.atleast_2d(y)
        if Y.shape[0] != self.ny:
            Y = Y.T  # accept (T, ny) by transposing
        if Y.shape[0] != self.ny:
            raise ValueError(f"y has ny={Y.shape[0]}, expected {self.ny}")
        nobs = Y.shape[1]
        ns, ny = self.ns, self.ny
        Z, T, R, Q, H, c, d = self.Z, self.T, self.R, self.Q, self.H, self.c, self.d
        RQR = R @ Q @ R.T

        if store:
            a_pred = np.zeros((ns, nobs))
            P_pred = np.zeros((ns, ns, nobs))
            a_filt = np.zeros((ns, nobs))
            P_filt = np.zeros((ns, ns, nobs))
            innov = np.zeros((ny, nobs))
            S_innov = np.zeros((ny, ny, nobs))
            K_gain = np.zeros((ns, ny, nobs))
            loglik_t = np.zeros(nobs)

        a, P = self.a0.copy(), self.P0.copy()
        ll_total = 0.0
        log_2pi = np.log(2 * np.pi)

        for t in range(nobs):
            # Predict
            a_p = T @ a + c
            P_p = T @ P @ T.T + RQR
            P_p = 0.5 * (P_p + P_p.T)  # symmetrize

            # Innovation
            y_t = Y[:, t]
            obs_mask = ~np.isnan(y_t)
            if obs_mask.any():
                Z_t = Z[obs_mask]
                d_t = d[obs_mask]
                H_t = H[np.ix_(obs_mask, obs_mask)]
                y_obs = y_t[obs_mask]
                v = y_obs - Z_t @ a_p - d_t
                S = Z_t @ P_p @ Z_t.T + H_t
                S = 0.5 * (S + S.T)
                try:
                    cS, low = cho_factor(S, lower=True)
                    Sinv_v = cho_solve((cS, low), v)
                    Sinv_Z = cho_solve((cS, low), Z_t @ P_p)
                    logdetS = 2.0 * np.sum(np.log(np.diag(cS)))
                except np.linalg.LinAlgError:
                    sign, logdetS = slogdet(S)
                    if sign <= 0:
                        return -np.inf if not store else self._abort_filter(t)
                    Sinv_v = solve(S, v)
                    Sinv_Z = solve(S, Z_t @ P_p)
                K = P_p @ Z_t.T @ np.linalg.inv(S)  # explicit for storage
                a = a_p + P_p @ Z_t.T @ Sinv_v
                P = P_p - (P_p @ Z_t.T) @ Sinv_Z
                P = 0.5 * (P + P.T)
                k_obs = obs_mask.sum()
                ll_t = -0.5 * (k_obs * log_2pi + logdetS + v @ Sinv_v)
            else:
                a, P = a_p, P_p
                K = np.zeros((ns, ny))
                v = np.zeros(ny)
                S = Z @ P_p @ Z.T + H
                ll_t = 0.0

            ll_total += ll_t
            if store:
                a_pred[:, t] = a_p
                P_pred[:, :, t] = P_p
                a_filt[:, t] = a
                P_filt[:, :, t] = P
                innov_full = np.full(ny, np.nan)
                innov_full[obs_mask] = v if obs_mask.any() else np.nan
                innov[:, t] = innov_full
                S_innov[:, :, t] = S if obs_mask.all() else np.nan
                if obs_mask.any():
                    K_full = np.zeros((ns, ny))
                    K_full[:, obs_mask] = K
                    K_gain[:, :, t] = K_full
                loglik_t[t] = ll_t

        if store:
            self.a_pred = a_pred
            self.P_pred = P_pred
            self.a_filt = a_filt
            self.P_filt = P_filt
            self.innov = innov
            self.S_innov = S_innov
            self.K_gain = K_gain
            self.loglik_t = loglik_t
        self.loglik = ll_total
        return ll_total

    def _abort_filter(self, t: int) -> float:
        self.loglik = -np.inf
        return -np.inf

    # ------------------------------------------------------------------
    # Rauch-Tung-Striebel smoother
    # ------------------------------------------------------------------

    def smooth(self) -> None:
        """Run the RTS smoother. Must be called after `filter(store=True)`."""
        if self.a_filt is None:
            raise RuntimeError("Run filter(store=True) before smooth().")
        T = self.T
        ns, nobs = self.a_filt.shape
        a_s = np.zeros_like(self.a_filt)
        P_s = np.zeros_like(self.P_filt)
        a_s[:, -1] = self.a_filt[:, -1]
        P_s[:, :, -1] = self.P_filt[:, :, -1]
        for t in range(nobs - 2, -1, -1):
            Pp = self.P_pred[:, :, t + 1]
            try:
                J = self.P_filt[:, :, t] @ T.T @ np.linalg.pinv(Pp)
            except np.linalg.LinAlgError:
                J = self.P_filt[:, :, t] @ T.T @ np.linalg.pinv(Pp + 1e-10 * np.eye(ns))
            a_s[:, t] = self.a_filt[:, t] + J @ (a_s[:, t + 1] - self.a_pred[:, t + 1])
            P_s[:, :, t] = self.P_filt[:, :, t] + J @ (P_s[:, :, t + 1] - Pp) @ J.T
            P_s[:, :, t] = 0.5 * (P_s[:, :, t] + P_s[:, :, t].T)
        self.a_smooth = a_s
        self.P_smooth = P_s

    # ------------------------------------------------------------------
    # Simulation and forecasting
    # ------------------------------------------------------------------

    def simulate(self, nobs: int, rng: Optional[np.random.Generator] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate `nobs` observations and the latent state path from the model.

        Returns
        -------
        y : np.ndarray, shape (ny, nobs)
        a : np.ndarray, shape (ns, nobs)
        """
        rng = np.random.default_rng() if rng is None else rng
        ns, ny, nq = self.ns, self.ny, self.nq
        a = np.zeros((ns, nobs))
        y = np.zeros((ny, nobs))
        Lq = np.linalg.cholesky(self.Q + 1e-12 * np.eye(nq))
        Lh = np.linalg.cholesky(self.H + 1e-12 * np.eye(ny)) if (self.H > 0).any() else None
        L0 = np.linalg.cholesky(self.P0 + 1e-12 * np.eye(ns))
        a_prev = self.a0 + L0 @ rng.standard_normal(ns)
        for t in range(nobs):
            eta = Lq @ rng.standard_normal(nq)
            a_curr = self.T @ a_prev + self.c + self.R @ eta
            eps = Lh @ rng.standard_normal(ny) if Lh is not None else np.zeros(ny)
            y_t = self.Z @ a_curr + self.d + eps
            a[:, t] = a_curr
            y[:, t] = y_t
            a_prev = a_curr
        return y, a

    def forecast(self, h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forecast h periods ahead from the last filtered state.

        Returns
        -------
        a_fc : np.ndarray, shape (ns, h)
        y_fc : np.ndarray, shape (ny, h)
        y_var : np.ndarray, shape (ny, ny, h)
        """
        if self.a_filt is None:
            raise RuntimeError("Run filter(store=True) before forecast().")
        a = self.a_filt[:, -1].copy()
        P = self.P_filt[:, :, -1].copy()
        RQR = self.R @ self.Q @ self.R.T
        a_fc = np.zeros((self.ns, h))
        y_fc = np.zeros((self.ny, h))
        y_var = np.zeros((self.ny, self.ny, h))
        for k in range(h):
            a = self.T @ a + self.c
            P = self.T @ P @ self.T.T + RQR
            a_fc[:, k] = a
            y_fc[:, k] = self.Z @ a + self.d
            y_var[:, :, k] = self.Z @ P @ self.Z.T + self.H
        return a_fc, y_fc, y_var

    # ------------------------------------------------------------------
    # Durbin-Koopman simulation smoother
    # ------------------------------------------------------------------

    def simulation_smoother(self, y: np.ndarray,
                            rng: Optional[np.random.Generator] = None
                            ) -> np.ndarray:
        """
        Draw one sample of the latent state path conditional on `y` using
        the Durbin-Koopman (2002) algorithm. Returns array of shape (ns, T).
        """
        rng = np.random.default_rng() if rng is None else rng
        Y = np.atleast_2d(y)
        if Y.shape[0] != self.ny:
            Y = Y.T
        nobs = Y.shape[1]
        # Durbin-Koopman (2002, Biometrika) Algorithm 2:
        # 1) Draw y+ and a+ from the unconditional model (uses self.a0, self.c).
        y_plus, a_plus = self.simulate(nobs, rng=rng)
        # 2) Smooth y* := y - y+ with the *centered* model: zero initial mean
        #    and zero intercepts (because the deviation a* = a - a+ has zero
        #    prior mean and follows the same dynamics with zero drift).
        y_star = Y - y_plus
        clone = StateSpaceModel(self.Z, self.T, self.R, self.Q, self.H,
                                np.zeros(self.ns), self.P0,
                                c=np.zeros(self.ns),
                                d=np.zeros(self.ny))
        clone.filter(y_star)
        clone.smooth()
        a_star_smooth = clone.a_smooth
        # 3) Draw = a+ + smoothed(y - y+)
        return a_plus + a_star_smooth


# ---------------------------------------------------------------------------
# Pre-built models
# ---------------------------------------------------------------------------


class LocalLinearTrend(StateSpaceModel):
    """
    Local linear trend model. Hodrick-Prescott emerges when the slope variance
    is zero, the level variance is set to 1/lambda and H = 1 (perfect signal
    rescaling).

    State a_t = (tau_t, tau_{t-1})' with
        tau_t = 2 tau_{t-1} - tau_{t-2} + eta_t
    Measurement:
        y_t = tau_t + eps_t
    """

    def __init__(self, y: np.ndarray, lam: Optional[float] = None,
                 sig2_eps: float = 1.0, sig2_eta: Optional[float] = None):
        y_arr = np.asarray(y, dtype=float).ravel()
        if lam is not None:
            sig2_eta = sig2_eps / lam
        if sig2_eta is None:
            raise ValueError("Provide either lam or sig2_eta.")
        Z = np.array([[1.0, 0.0]])
        T = np.array([[2.0, -1.0], [1.0, 0.0]])
        R = np.array([[1.0], [0.0]])
        Q = np.array([[sig2_eta]])
        H = np.array([[sig2_eps]])
        a0 = np.array([y_arr[0], y_arr[0]])
        P0 = 10.0 * np.eye(2)
        super().__init__(Z, T, R, Q, H, a0, P0)
        self.y = y_arr
        self.lam = lam

    def fit(self) -> "LocalLinearTrend":
        self.filter(self.y)
        self.smooth()
        return self

    @property
    def trend(self) -> np.ndarray:
        return self.a_smooth[0, :]

    @property
    def gap(self) -> np.ndarray:
        return self.y - self.trend


class ClarkModel(StateSpaceModel):
    """
    Clark (1987) unobserved components model. Two variants:

    - `seasonality=False` (4 states): drift g_t (RW), trend tau_t (RW with
      drift), cycle c_t (AR(2)).
    - `seasonality=True` (7 states): same plus trigonometric seasonality
      gamma_t implemented as a 3-period summation (quarterly).

    Parameters
    ----------
    y : array-like
        Observed log-level series (multiplied by 100 for percentage units).
    phi1, phi2 : float
        AR(2) coefficients for the cycle.
    sig2_g, sig2_tau, sig2_c : float
        Innovation variances of drift, trend and cycle.
    sig2_gam : float, optional
        Innovation variance of seasonal component (only if seasonality=True).
    sig2_y : float, default 0
        Measurement noise variance.
    cov_tau_c : float, default 0
        Covariance between trend and cycle innovations (Morley-Nelson-Zivot
        identification).
    seasonality : bool, default False
    """

    def __init__(self, y: np.ndarray, phi1: float, phi2: float,
                 sig2_g: float, sig2_tau: float, sig2_c: float,
                 sig2_gam: float = 0.0, sig2_y: float = 0.0,
                 cov_tau_c: float = 0.0, seasonality: bool = False):
        y_arr = np.asarray(y, dtype=float).ravel()
        self.seasonality = seasonality
        if not seasonality:
            Z = np.array([[0.0, 1.0, 1.0, 0.0]])
            T = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, phi1, phi2],
                [0.0, 0.0, 1.0, 0.0],
            ])
            R = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ])
            Q = np.array([
                [sig2_g, 0.0, 0.0],
                [0.0, sig2_tau, cov_tau_c],
                [0.0, cov_tau_c, sig2_c],
            ])
            a0 = np.array([0.0, y_arr[0], 0.0, 0.0])
            P0 = 10.0 * np.eye(4)
        else:
            Z = np.array([[0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]])
            T = np.zeros((7, 7))
            T[0, 0] = 1.0
            T[1, 0] = 1.0; T[1, 1] = 1.0
            T[2, 2] = phi1; T[2, 3] = phi2
            T[3, 2] = 1.0
            T[4, 4] = -1.0; T[4, 5] = -1.0; T[4, 6] = -1.0
            T[5, 4] = 1.0
            T[6, 5] = 1.0
            R = np.zeros((7, 4))
            R[0, 0] = 1.0; R[1, 1] = 1.0; R[2, 2] = 1.0; R[4, 3] = 1.0
            Q = np.array([
                [sig2_g, 0.0, 0.0, 0.0],
                [0.0, sig2_tau, cov_tau_c, 0.0],
                [0.0, cov_tau_c, sig2_c, 0.0],
                [0.0, 0.0, 0.0, sig2_gam],
            ])
            a0 = np.array([0.0, y_arr[0], 0.0, 0.0, 0.0, 0.0, 0.0])
            P0 = 10.0 * np.eye(7)
        H = np.array([[sig2_y]])
        super().__init__(Z, T, R, Q, H, a0, P0)
        self.y = y_arr
        self.params = dict(phi1=phi1, phi2=phi2, sig2_g=sig2_g,
                           sig2_tau=sig2_tau, sig2_c=sig2_c,
                           sig2_gam=sig2_gam, sig2_y=sig2_y,
                           cov_tau_c=cov_tau_c)

    def fit(self) -> "ClarkModel":
        self.filter(self.y)
        self.smooth()
        return self

    # Convenience accessors on the smoothed state
    @property
    def trend(self) -> np.ndarray:
        return self.a_smooth[1, :]

    @property
    def cycle(self) -> np.ndarray:
        return self.a_smooth[2, :]

    @property
    def drift(self) -> np.ndarray:
        return self.a_smooth[0, :]

    @property
    def seasonal(self) -> np.ndarray:
        if not self.seasonality:
            raise AttributeError("Model was built without seasonality.")
        return self.a_smooth[4, :]

    def component_std(self, idx: int) -> np.ndarray:
        return np.sqrt(self.P_smooth[idx, idx, :])


# ---------------------------------------------------------------------------
# MLE estimator
# ---------------------------------------------------------------------------


@dataclass
class MLEResult:
    theta_hat: np.ndarray
    loglik: float
    hessian: Optional[np.ndarray]
    std_errors: Optional[np.ndarray]
    success: bool
    message: str
    model: StateSpaceModel


class MLEEstimator:
    """
    Maximum-Likelihood estimator for a state-space model.

    The user supplies a `build` callable mapping a parameter vector theta to
    a `StateSpaceModel`, and (optionally) bounds. The negative log-likelihood
    is minimized with `scipy.optimize.minimize`.
    """

    def __init__(self, build: Callable[[np.ndarray], StateSpaceModel],
                 y: np.ndarray, bounds: Optional[list] = None,
                 method: str = "L-BFGS-B"):
        self.build = build
        self.y = np.asarray(y, dtype=float)
        self.bounds = bounds
        self.method = method

    def neg_loglik(self, theta: np.ndarray) -> float:
        try:
            mod = self.build(theta)
            ll = mod.filter(self.y, store=False)
            if not np.isfinite(ll):
                return 1e10
            return -ll
        except (np.linalg.LinAlgError, ValueError):
            return 1e10

    def fit(self, theta0: np.ndarray, compute_hessian: bool = True,
            **opts) -> MLEResult:
        res = minimize(self.neg_loglik, theta0, method=self.method,
                       bounds=self.bounds, options=opts)
        theta_hat = res.x
        hess = None
        se = None
        if compute_hessian:
            hess = _numerical_hessian(self.neg_loglik, theta_hat)
            try:
                cov = np.linalg.inv(hess)
                se = np.sqrt(np.maximum(np.diag(cov), 0))
            except np.linalg.LinAlgError:
                se = None
        mod = self.build(theta_hat)
        mod.filter(self.y)
        mod.smooth()
        return MLEResult(theta_hat=theta_hat, loglik=-res.fun, hessian=hess,
                         std_errors=se, success=res.success, message=res.message,
                         model=mod)


def _numerical_hessian(func: Callable[[np.ndarray], float], x: np.ndarray,
                       eps: float = 1e-4) -> np.ndarray:
    """Symmetric finite-difference Hessian."""
    n = len(x)
    H = np.zeros((n, n))
    fx = func(x)
    for i in range(n):
        for j in range(i, n):
            x1 = x.copy(); x1[i] += eps; x1[j] += eps
            x2 = x.copy(); x2[i] += eps; x2[j] -= eps
            x3 = x.copy(); x3[i] -= eps; x3[j] += eps
            x4 = x.copy(); x4[i] -= eps; x4[j] -= eps
            H[i, j] = (func(x1) - func(x2) - func(x3) + func(x4)) / (4 * eps * eps)
            H[j, i] = H[i, j]
    return H


# ---------------------------------------------------------------------------
# Bayesian estimator (Gibbs sampler with simulation smoother)
# ---------------------------------------------------------------------------


@dataclass
class BayesianResult:
    draws: dict
    state_draws: np.ndarray  # (ndraws, ns, T)
    model: StateSpaceModel


class BayesianStateSpace:
    """
    Generic Gibbs sampler for state-space models with Normal-Inverse-Gamma
    conjugate priors on shock variances and (optionally) a Normal prior on
    AR coefficients of a cycle block.

    The user supplies:
      - `build(params)`: maps the dict of structural parameters to a
        `StateSpaceModel`.
      - `update_variances(states, params)`: draws Inverse-Gamma variances
        from their full conditionals given the simulated states.
      - `update_ar(states, params)`: optional, draws the AR(2) cycle
        coefficients from a truncated Normal full conditional.

    For Clark and HP-like models, helpers `clark_bayesian_step` and
    `local_linear_bayesian_step` are provided.
    """

    def __init__(self, y: np.ndarray, build: Callable[[dict], StateSpaceModel],
                 update_step: Callable[[np.ndarray, dict, np.random.Generator], dict],
                 init_params: dict, ndraws: int = 5000, burnin: int = 1000,
                 seed: Optional[int] = None):
        self.y = np.asarray(y, dtype=float)
        self.build = build
        self.update_step = update_step
        self.init_params = init_params
        self.ndraws = ndraws
        self.burnin = burnin
        self.rng = np.random.default_rng(seed)

    def run(self, verbose: bool = False) -> BayesianResult:
        params = dict(self.init_params)
        keep = self.ndraws
        total = self.burnin + keep
        param_draws = {k: np.zeros(keep) for k in params if np.isscalar(params[k])}
        state_draws = None
        model = self.build(params)
        ns = model.ns
        nobs = len(self.y)
        state_draws = np.zeros((keep, ns, nobs))

        for it in range(total):
            mod = self.build(params)
            states = mod.simulation_smoother(self.y, rng=self.rng)
            params = self.update_step(states, params, self.rng)
            if it >= self.burnin:
                k = it - self.burnin
                for key in param_draws:
                    param_draws[key][k] = params[key]
                state_draws[k] = states
            if verbose and (it + 1) % max(1, total // 10) == 0:
                print(f"Gibbs iter {it + 1}/{total}")
        final_mod = self.build(params)
        final_mod.filter(self.y); final_mod.smooth()
        return BayesianResult(draws=param_draws, state_draws=state_draws,
                              model=final_mod)


# ---------------------------------------------------------------------------
# Helpers: posterior step for Clark-type models
# ---------------------------------------------------------------------------


def _ig_posterior(prior_a: float, prior_b: float, sumsq: float, n: int,
                  rng: np.random.Generator) -> float:
    """Inverse-Gamma posterior draw for a variance given iid Gaussian shocks."""
    a_post = prior_a + n / 2.0
    b_post = prior_b + 0.5 * sumsq
    return float(invgamma.rvs(a_post, scale=b_post, random_state=rng))


def clark_update_step(states: np.ndarray, params: dict,
                      rng: np.random.Generator,
                      prior_var: Tuple[float, float] = (3.0, 0.01),
                      prior_phi_mean: np.ndarray = np.array([1.3, -0.5]),
                      prior_phi_cov: np.ndarray = np.eye(2) * 0.5,
                      ) -> dict:
    """
    One Gibbs sweep for the Clark (4-state) model: draws sig2_g, sig2_tau,
    sig2_c from Inverse-Gamma full conditionals and (phi1, phi2) from a
    Normal full conditional restricted to the stationary region.

    Parameters
    ----------
    states : np.ndarray
        Simulated state path, shape (4, T).
    params : dict
        Current parameter dict.
    prior_var : tuple
        Inverse-Gamma prior shape and scale for each variance.
    prior_phi_mean, prior_phi_cov : np.ndarray
        Normal prior moments for the AR(2) cycle coefficients.
    """
    g = states[0]; tau = states[1]; c = states[2]
    eta_g = np.diff(g)
    eta_tau = tau[1:] - tau[:-1] - g[:-1]
    a, b = prior_var
    sig2_g = _ig_posterior(a, b, np.sum(eta_g ** 2), len(eta_g), rng)
    sig2_tau = _ig_posterior(a, b, np.sum(eta_tau ** 2), len(eta_tau), rng)

    # AR(2) cycle: c_t = phi1 c_{t-1} + phi2 c_{t-2} + eps
    y_c = c[2:]
    X_c = np.column_stack([c[1:-1], c[:-2]])
    sig2_c_curr = params["sig2_c"]
    V_inv = np.linalg.inv(prior_phi_cov) + X_c.T @ X_c / sig2_c_curr
    V = np.linalg.inv(V_inv)
    m = V @ (np.linalg.inv(prior_phi_cov) @ prior_phi_mean + X_c.T @ y_c / sig2_c_curr)
    # Sample phi until stationary
    for _ in range(100):
        phi_draw = rng.multivariate_normal(m, V)
        if _is_stationary_ar2(phi_draw[0], phi_draw[1]):
            break
    else:
        phi_draw = np.array([params["phi1"], params["phi2"]])
    eps_c = y_c - X_c @ phi_draw
    sig2_c = _ig_posterior(a, b, np.sum(eps_c ** 2), len(eps_c), rng)

    return dict(params, phi1=float(phi_draw[0]), phi2=float(phi_draw[1]),
                sig2_g=sig2_g, sig2_tau=sig2_tau, sig2_c=sig2_c)


def _is_stationary_ar2(phi1: float, phi2: float) -> bool:
    return (phi2 + phi1 < 1) and (phi2 - phi1 < 1) and (-1 < phi2 < 1)


def build_clark_simple(params: dict, y: np.ndarray) -> StateSpaceModel:
    """Factory used by BayesianStateSpace for the 4-state Clark model."""
    m = ClarkModel(
        y, phi1=params["phi1"], phi2=params["phi2"],
        sig2_g=params["sig2_g"], sig2_tau=params["sig2_tau"],
        sig2_c=params["sig2_c"], sig2_y=params.get("sig2_y", 0.0),
        cov_tau_c=params.get("cov_tau_c", 0.0),
        seasonality=False,
    )
    return m


# ---------------------------------------------------------------------------
# Watson (1986) UC model: RW with constant drift + AR(2) cycle
# ---------------------------------------------------------------------------


class WatsonUC(StateSpaceModel):
    """
    Watson (1986) unobserved components model: RW trend with constant drift
    plus an AR(2) cycle. Cleaner alternative to the 4-state Clark model for
    pedagogical and estimation purposes.

    State alpha_t = (tau_t, c_t, c_{t-1})':
        tau_t = mu + tau_{t-1} + eta_tau
        c_t   = phi1 c_{t-1} + phi2 c_{t-2} + eta_c
        y_t   = tau_t + c_t

    Parameters
    ----------
    y : array-like
        Observations (typically 100 * log GDP).
    mu : float
        Constant drift in the trend equation (per-period growth rate).
    phi1, phi2 : float
        AR(2) cycle coefficients (stationary region).
    sig2_tau, sig2_c : float
        Innovation variances of the trend and cycle.
    sig2_y : float, default 1e-6
        Measurement noise variance (kept tiny by default since GDP is
        essentially observed without error).
    a0 : array-like, optional
        Initial state. Defaults to (y[0], 0, 0).
    """

    def __init__(self, y: np.ndarray, mu: float, phi1: float, phi2: float,
                 sig2_tau: float, sig2_c: float, sig2_y: float = 1e-6,
                 a0: Optional[np.ndarray] = None):
        y_arr = np.asarray(y, dtype=float).ravel()
        Z = np.array([[1.0, 1.0, 0.0]])
        T = np.array([[1.0, 0.0, 0.0],
                      [0.0, phi1, phi2],
                      [0.0, 1.0, 0.0]])
        R = np.array([[1.0, 0.0],
                      [0.0, 1.0],
                      [0.0, 0.0]])
        Q = np.array([[sig2_tau, 0.0],
                      [0.0, sig2_c]])
        H = np.array([[sig2_y]])
        c = np.array([mu, 0.0, 0.0])
        a0 = np.array([y_arr[0], 0.0, 0.0]) if a0 is None else np.asarray(a0)
        P0 = np.diag([10.0, 1.0, 1.0])
        super().__init__(Z, T, R, Q, H, a0, P0, c=c)
        self.y = y_arr
        self.params = dict(mu=mu, phi1=phi1, phi2=phi2,
                           sig2_tau=sig2_tau, sig2_c=sig2_c, sig2_y=sig2_y)

    def fit(self) -> "WatsonUC":
        self.filter(self.y)
        self.smooth()
        return self

    @property
    def trend(self) -> np.ndarray:
        return self.a_smooth[0, :]

    @property
    def cycle(self) -> np.ndarray:
        return self.a_smooth[1, :]

    def component_std(self, idx: int) -> np.ndarray:
        return np.sqrt(self.P_smooth[idx, idx, :])


def build_watson_uc(params: dict, y: np.ndarray) -> WatsonUC:
    """Factory for BayesianStateSpace with the Watson UC model."""
    return WatsonUC(y, mu=params["mu"], phi1=params["phi1"],
                    phi2=params["phi2"], sig2_tau=params["sig2_tau"],
                    sig2_c=params["sig2_c"],
                    sig2_y=params.get("sig2_y", 1e-6))


def watson_gibbs_step(states: np.ndarray, params: dict,
                      rng: np.random.Generator,
                      y: Optional[np.ndarray] = None,
                      prior_tau: Tuple[float, float] = (5.0, 0.02),
                      prior_c: Tuple[float, float] = (3.0, 0.25),
                      prior_phi_mean: np.ndarray = np.array([1.5, -0.7]),
                      prior_phi_cov: np.ndarray = np.diag([0.10, 0.10]),
                      mu_fixed: bool = True,
                      ) -> dict:
    """
    One Gibbs sweep for the Watson UC model.

    Draws:
      - sig2_tau ~ IG  | trend innovations
      - sig2_c   ~ IG  | cycle innovations
      - (phi1, phi2)' ~ truncated Normal on stationary region

    Parameters
    ----------
    states : np.ndarray
        Simulated state path, shape (3, T).
    params : dict
        Current parameters (must contain mu, phi1, phi2, sig2_tau, sig2_c).
    prior_tau, prior_c : (a, b) IG hyperparameters.
        Default (5.0, 0.02) for sig2_tau gives prior mean ~0.005 (small trend
        shocks), and (3.0, 0.25) for sig2_c gives prior mean ~0.125 (loose).
    prior_phi_mean, prior_phi_cov : Normal prior for AR(2) coefficients.
    mu_fixed : bool
        If True, mu is not updated (use a calibrated value, e.g. sample
        mean growth). Recommended; estimating mu jointly is rarely worth
        the added complexity.
    """
    tau = states[0]
    c = states[1]
    eta_tau = tau[1:] - tau[:-1] - params["mu"]
    a_t, b_t = prior_tau
    sig2_tau = _ig_posterior(a_t, b_t, np.sum(eta_tau ** 2), len(eta_tau), rng)

    # AR(2) cycle
    y_c = c[2:]
    X_c = np.column_stack([c[1:-1], c[:-2]])
    sig2_c_curr = params["sig2_c"]
    P0_inv = np.linalg.inv(prior_phi_cov)
    V_inv = P0_inv + X_c.T @ X_c / sig2_c_curr
    V = np.linalg.inv(V_inv)
    m = V @ (P0_inv @ prior_phi_mean + X_c.T @ y_c / sig2_c_curr)
    phi_draw = np.array([params["phi1"], params["phi2"]])
    for _ in range(200):
        cand = rng.multivariate_normal(m, V)
        if _is_stationary_ar2(cand[0], cand[1]):
            phi_draw = cand
            break
    eps_c = y_c - X_c @ phi_draw
    a_c, b_c = prior_c
    sig2_c = _ig_posterior(a_c, b_c, np.sum(eps_c ** 2), len(eps_c), rng)

    out = dict(params, phi1=float(phi_draw[0]), phi2=float(phi_draw[1]),
               sig2_tau=sig2_tau, sig2_c=sig2_c)
    if not mu_fixed and y is not None:
        # Posterior of mu | tau, sig2_tau is Normal
        n = len(eta_tau)
        post_var = 1.0 / (1.0 / 100.0 + n / sig2_tau)  # weak prior var=100
        post_mean = post_var * (np.sum(tau[1:] - tau[:-1]) / sig2_tau)
        out["mu"] = float(rng.normal(post_mean, np.sqrt(post_var)))
    return out


# ---------------------------------------------------------------------------
# Convenience: package the smoothed decomposition as a DataFrame
# ---------------------------------------------------------------------------


def decomposition_to_frame(model: ClarkModel, index: pd.Index,
                           include_bands: bool = True, band: float = 1.96
                           ) -> pd.DataFrame:
    """Return a tidy DataFrame with trend / cycle / drift (+ bands) and
    the smoothed seasonal if present."""
    cols = {
        "y": model.y,
        "trend": model.trend,
        "cycle": model.cycle,
        "drift": model.drift,
    }
    if include_bands:
        cols["trend_lo"] = model.trend - band * model.component_std(1)
        cols["trend_hi"] = model.trend + band * model.component_std(1)
        cols["cycle_lo"] = model.cycle - band * model.component_std(2)
        cols["cycle_hi"] = model.cycle + band * model.component_std(2)
    if model.seasonality:
        cols["seasonal"] = model.seasonal
        if include_bands:
            cols["seasonal_lo"] = model.seasonal - band * model.component_std(4)
            cols["seasonal_hi"] = model.seasonal + band * model.component_std(4)
    return pd.DataFrame(cols, index=index)


__all__ = [
    "StateSpaceModel",
    "LocalLinearTrend",
    "ClarkModel",
    "WatsonUC",
    "MLEEstimator",
    "MLEResult",
    "BayesianStateSpace",
    "BayesianResult",
    "clark_update_step",
    "build_clark_simple",
    "build_watson_uc",
    "watson_gibbs_step",
    "decomposition_to_frame",
]
