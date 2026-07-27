import numpy as np
from scipy.special import multigammaln

def MinnesotaPrior(yy, XX, lags, ncoeff_eq, prior_params={"mn_mean": 1, "lamda1": 0.2, "lamda2": 0.5, "lamda3": 1, "lamda4": 1e5}, b_exo=None):
    """Compute the Minnesota prior using a parameter dictionary, supporting block exogeneity."""
    mn_mean = prior_params.get("mn_mean", 1)
    lamda1 = prior_params.get("lamda1", 0.2)
    lamda2 = prior_params.get("lamda2", 0.5)
    lamda3 = prior_params.get("lamda3", 1)
    lamda4 = prior_params.get("lamda4", 1e5)
    small_var = 1e-9  # variance for excluded coefficients

    N = yy.shape[1]

    # Residual std of an AR(1)-plus-constant regression per series. The own
    # first lag sits at column ``i`` of XX under the ``[lag-1 vars, ...,
    # constant]`` layout of ``prepare_data``.
    std = ar1_residual_std(yy, XX, N)

    # Prior mean
    B0 = np.zeros((ncoeff_eq, N))
    for i in range(N):
        B0[i, i] = mn_mean
    b0 = B0.flatten(order="F")

    # Prior variance matrix H
    H = np.zeros((ncoeff_eq * N, ncoeff_eq * N))

    # Deterministic/exogenous positions per equation (constant, trend, user
    # exog). They sit after the lag block and all receive a loose
    # Normal(0, (std_i * lamda4)^2) prior so the likelihood drives the
    # posterior.
    n_exo_positions = ncoeff_eq - N * lags

    for i in range(N):  # equation i
        for e in range(n_exo_positions):
            exoIdx = i * ncoeff_eq + N * lags + e
            H[exoIdx, exoIdx] = (std[i] * lamda4) ** 2

        for lag in range(1, lags + 1):
            for j in range(N):  # variable j
                coeffIdx = i * ncoeff_eq + (lag - 1) * N + j
                if b_exo is not None and b_exo[i, j] == 1:
                    H[coeffIdx, coeffIdx] = small_var
                    continue

                if i == j:
                    # own lags
                    if lag == 1:
                        H[coeffIdx, coeffIdx] = lamda1 ** 2
                    else:
                        H[coeffIdx, coeffIdx] = (lamda1 / (lag ** lamda3)) ** 2
                else:
                    # cross lags
                    if lag == 1:
                        H[coeffIdx, coeffIdx] = ((std[i] * lamda1 * lamda2) / std[j]) ** 2
                    else:
                        H[coeffIdx, coeffIdx] = ((std[i] * lamda1 * lamda2 / (lag ** lamda3)) / std[j]) ** 2

    return {"prior_type": 1, "b0": b0, "H": H, "std_ar": std}



def NormalWishartPrior(yy, XX, lags, ncoeff_eq, prior_params={"mn_mean": 1, "lamda1": 0.2, "lamda2": 0.5, "lamda3": 1, "lamda4": 1e5}, b_exo=None):
    """ Compute the Minnesota-Inverse Wishart prior. """
    ny = yy.shape[1]
    mn_prior = MinnesotaPrior(yy, XX, lags, ncoeff_eq, prior_params, b_exo)
    b0, H, std_ar = mn_prior["b0"], mn_prior["H"], mn_prior["std_ar"]
    std_ar = std_ar.flatten(order="F")

    alpha0 = ny + 2
    Scale0 = (alpha0 - ny - 1) * np.diag(std_ar ** 2)
    #Scale0 = np.eye(ny)
    
    return {"prior_type": 2, "b0": b0, "H": H, "Scale0": Scale0, "alpha0": alpha0}
    

def NormalDiffusePrior(yy, XX, lags, ncoeff_eq, prior_params={"mn_mean": 1, "lamda1": 0.2, "lamda2": 0.5, "lamda3": 1, "lamda4": 1e5}, b_exo=None):
    """ Compute the Minnesota-Inverse Wishart prior (diffuse version). """
    mn_prior = MinnesotaPrior(yy, XX, lags, ncoeff_eq, prior_params, b_exo)
    b0, H = mn_prior["b0"], mn_prior["H"]
    
    return {"prior_type": 3, "b0": b0, "H": H}


def ar1_residual_std(yy, XX, n_endo):
    """
    Per-series residual standard deviation from an AR(1)-plus-constant regression.

    Uses the own first-lag column of ``XX`` (column ``i`` under the
    ``[lag-1 vars, ..., constant, ...]`` layout of ``prepare_data``) together
    with an intercept. These scales calibrate the conjugate Normal-Wishart
    moments used for marginal-likelihood computations.
    """
    T = yy.shape[0]
    std = np.zeros(n_endo)
    for i in range(n_endo):
        x = np.column_stack([XX[:, i], np.ones(T)])
        b = np.linalg.lstsq(x, yy[:, i], rcond=None)[0]
        res = yy[:, i] - x @ b
        std[i] = np.sqrt(res @ res / max(T - 2, 1))
    return std


def nw_conjugate_moments(yy, XX, lags, n_endo, prior_params):
    """
    Conjugate Normal-Wishart (Minnesota-moment) prior for marginal likelihoods.

    Returns the moments of the conjugate prior ``vec(B) | Sigma ~ N(vec(B0),
    Sigma x Omega0)``, ``Sigma ~ IW(S0, nu0)`` implied by the Minnesota
    hyperparameters. This is the prior of Giannone, Lenza & Primiceri (2015),
    for which the marginal likelihood is available in closed form.

    Notes
    -----
    The Kronecker structure ties shrinkage across equations, so the
    own-versus-cross asymmetry ``lamda2`` of the sampling prior cannot be
    represented here (Kadiyala & Karlsson, 1997); only ``mn_mean``, ``lamda1``
    (overall tightness), ``lamda3`` (lag decay) and ``lamda4`` (deterministic /
    exogenous looseness) enter. Variance of the coefficient on lag ``l`` of
    variable ``j`` in any equation ``i`` is
    ``Sigma_ii * lamda1^2 / (l^(2*lamda3) * sigma_j^2)``, with
    ``E[Sigma_ii] = sigma_i^2``.

    Returns
    -------
    dict with keys ``B0`` (k x n), ``omega_diag`` (k,), ``S0`` (n x n),
    ``nu0`` (scalar) and ``sigma`` (n,).
    """
    mn_mean = prior_params.get("mn_mean", 1)
    lamda1 = prior_params.get("lamda1", 0.2)
    lamda3 = prior_params.get("lamda3", 1)
    lamda4 = prior_params.get("lamda4", 1e5)

    k = XX.shape[1]
    sigma = ar1_residual_std(yy, XX, n_endo)

    omega_diag = np.zeros(k)
    for lag in range(1, lags + 1):
        for j in range(n_endo):
            omega_diag[(lag - 1) * n_endo + j] = (
                lamda1 ** 2 / (lag ** (2 * lamda3) * sigma[j] ** 2)
            )
    omega_diag[n_endo * lags:] = lamda4 ** 2  # constant / trend / exog block

    B0 = np.zeros((k, n_endo))
    for i in range(n_endo):
        B0[i, i] = mn_mean

    nu0 = n_endo + 2
    S0 = np.diag(sigma ** 2) * (nu0 - n_endo - 1)

    return {"B0": B0, "omega_diag": omega_diag, "S0": S0, "nu0": nu0, "sigma": sigma}


def nw_log_marginal_likelihood(yy, XX, moments, obs_weights=None):
    """
    Analytic log marginal likelihood of a VAR under the conjugate NW prior.

    Standard matrix-t result: with ``Y = X B + E``, prior ``vec(B) | Sigma ~
    N(vec(B0), Sigma x Omega0)`` and ``Sigma ~ IW(S0, nu0)``,

    ``log p(Y) = -(T n / 2) log(pi) + log Gamma_n(nu_T / 2) - log Gamma_n(nu0 / 2)
    + (nu0 / 2) log|S0| - (nu_T / 2) log|S_T| - (n / 2)(log|Omega0| + log|K|)``

    where ``K = Omega0^{-1} + X'X``, ``B_T = K^{-1}(Omega0^{-1} B0 + X'Y)``,
    ``S_T = S0 + Y'Y + B0' Omega0^{-1} B0 - B_T' K B_T`` and ``nu_T = nu0 + T``.

    Parameters
    ----------
    obs_weights : np.ndarray, optional
        Per-observation GLS weights ``w_t = 1 / s_t`` (Lenza-Primiceri COVID
        volatility scaling). Rows of ``(yy, XX)`` are multiplied by ``w_t`` and
        the change-of-variables Jacobian ``+ n * sum(log w_t)`` is added, so the
        value remains the marginal likelihood of the *unweighted* data.

    Returns
    -------
    float (``-inf`` if a posterior scale matrix fails to be positive definite).
    """
    n = yy.shape[1]
    jac = 0.0
    if obs_weights is not None:
        w = np.asarray(obs_weights, dtype=float)
        yy = yy * w[:, None]
        XX = XX * w[:, None]
        jac = n * np.sum(np.log(w))

    T, k = XX.shape
    B0 = moments["B0"]
    omega = moments["omega_diag"]
    S0 = moments["S0"]
    nu0 = moments["nu0"]
    nu_T = nu0 + T

    iom = 1.0 / omega
    K = XX.T @ XX + np.diag(iom)
    XtY = XX.T @ yy
    rhs = XtY + B0 * iom[:, None]
    try:
        cK = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return -np.inf
    B_T = np.linalg.solve(K, rhs)
    S_T = S0 + yy.T @ yy + (B0 * iom[:, None]).T @ B0 - rhs.T @ B_T
    S_T = 0.5 * (S_T + S_T.T)

    try:
        cS0 = np.linalg.cholesky(S0)
        cST = np.linalg.cholesky(S_T)
    except np.linalg.LinAlgError:
        return -np.inf

    logdet_K = 2.0 * np.sum(np.log(np.diag(cK)))
    logdet_S0 = 2.0 * np.sum(np.log(np.diag(cS0)))
    logdet_ST = 2.0 * np.sum(np.log(np.diag(cST)))
    logdet_omega = np.sum(np.log(omega))

    logml = (
        -0.5 * T * n * np.log(np.pi)
        + multigammaln(nu_T / 2.0, n) - multigammaln(nu0 / 2.0, n)
        + 0.5 * nu0 * logdet_S0 - 0.5 * nu_T * logdet_ST
        - 0.5 * n * (logdet_omega + logdet_K)
    )
    return float(logml + jac)


def _panel_dummy_matrices(sigma, lags, tightness):
    """Construct the Banbura-style dummy matrices used in the Mumtaz panel VAR code."""
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    if tightness <= 0:
        raise ValueError("`tightness` must be strictly positive.")

    n_endo = sigma.size
    lag_penalty = np.diag(np.arange(1, lags + 1, dtype=float))

    yd = np.vstack(
        [
            np.diag(sigma / tightness),
            np.zeros((n_endo * (lags - 1), n_endo)),
            np.diag(sigma),
        ]
    )
    xd = np.vstack(
        [
            np.kron(lag_penalty, np.diag(sigma) / tightness),
            np.zeros((n_endo, n_endo * lags)),
        ]
    )

    return yd, xd


def HierarchicalPanelPrior(y_unit, lags, tightness=1.0):
    """
    Compute the hierarchical lag-coefficient prior used in Mumtaz's panel VAR code.

    The prior is implemented through dummy observations and returns the implied prior
    mean and covariance for the unit-specific lag coefficients.
    """
    y_unit = np.asarray(y_unit, dtype=float)
    if y_unit.ndim != 2:
        raise ValueError("`y_unit` must be a 2-dimensional array.")

    T, n_endo = y_unit.shape
    if T <= lags:
        raise ValueError("Not enough observations to build the requested panel prior.")

    sigma = np.zeros(n_endo)
    for idx in range(n_endo):
        y_series = y_unit[:, idx]
        x_lags = np.column_stack([y_series[lags - lag:T - lag] for lag in range(1, lags + 1)])
        x_lags = np.hstack((x_lags, np.ones((T - lags, 1))))
        y_target = y_series[lags:]
        coeffs = np.linalg.lstsq(x_lags, y_target, rcond=None)[0]
        residuals = y_target - x_lags @ coeffs
        dof = y_target.shape[0] - x_lags.shape[1]
        if dof <= 0:
            raise ValueError("Too few observations to calibrate the panel prior.")
        sigma[idx] = np.sqrt((residuals.T @ residuals) / dof)

    yd, xd = _panel_dummy_matrices(sigma, lags, tightness)
    b0 = np.linalg.lstsq(xd, yd, rcond=None)[0].flatten(order="F")
    H = np.kron(np.diag(sigma ** 2), np.linalg.pinv(xd.T @ xd))
    H_inv = np.linalg.pinv(H)

    return {"b0": b0, "H": H, "H_inv": H_inv, "sigma_scale": sigma}


def DiffusePanelExogenousPrior(n_exo, n_endo, precision=1e-4):
    """Diffuse Gaussian prior for unit-specific intercept and exogenous coefficients."""
    size = int(n_exo) * int(n_endo)
    return {"b0": np.zeros(size), "H_inv": np.eye(size) * precision}
