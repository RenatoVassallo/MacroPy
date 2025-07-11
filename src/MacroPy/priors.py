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

    for i in range(N):  # equation i
        constIdx = (i + 1) * ncoeff_eq - 1
        H[constIdx, constIdx] = (std[i] * lamda4) ** 2

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
