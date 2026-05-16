import numpy as np

def MinnesotaPrior(yy, XX, lags, ncoeff_eq, prior_params={"mn_mean": 1, "lamda1": 0.2, "lamda2": 0.5, "lamda3": 1, "lamda4": 1e5}, b_exo=None):
    """Compute the Minnesota prior using a parameter dictionary, supporting block exogeneity."""
    mn_mean = prior_params.get("mn_mean", 1)
    lamda1 = prior_params.get("lamda1", 0.2)
    lamda2 = prior_params.get("lamda2", 0.5)
    lamda3 = prior_params.get("lamda3", 1)
    lamda4 = prior_params.get("lamda4", 1e5)
    small_var = 1e-9  # variance for excluded coefficients

    N = yy.shape[1]
    
    # Estimate std deviation of residuals for each series
    std = np.zeros(N)
    for i in range(N):
        x = XX[:, [0, i + 1]]  # constant + own first lag
        y = yy[:, i]
        b0 = np.linalg.lstsq(x, y, rcond=None)[0]
        res = y - x @ b0
        std[i] = np.sqrt((res.T @ res) / (len(y) - 2))

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
