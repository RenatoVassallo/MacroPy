"""Block exogeneity convention: b_exo[i, j] = 1 excludes the lags of j from equation i."""
import numpy as np
import pandas as pd

from MacroPy import BayesianVAR


def _simulated_open_economy(T=300, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros((T, 2))
    for t in range(1, T):
        y[t, 0] = 0.5 * y[t - 1, 0] + rng.normal()                       # external
        y[t, 1] = 0.3 * y[t - 1, 1] + 0.4 * y[t - 1, 0] + rng.normal()   # domestic
    index = pd.date_range("2000-01-01", periods=T, freq="MS")
    return pd.DataFrame(y, index=index, columns=["external", "domestic"])


def test_block_exogeneity_one_excludes_the_lag():
    df = _simulated_open_economy()
    b_exo = np.array([[0, 1],
                      [0, 0]])
    model = BayesianVAR(df, lags=1, prior_type=1, b_exo=b_exo,
                        post_draws=400, burnin=0.5, seed=1)
    model.sample_posterior()
    draws = np.asarray(model.beta_draws)
    k = model.ncoeff_eq
    excluded = draws[:, 0 * k + 1]   # equation 0, first lag of variable 1
    free = draws[:, 1 * k + 0]       # equation 1, first lag of variable 0
    assert np.abs(excluded).max() < 1e-3
    assert free.std() > 1e-3
    assert abs(free.mean() - 0.4) < 0.2
