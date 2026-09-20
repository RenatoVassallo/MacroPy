import numpy as np
import pandas as pd

__all__ = [
    "prepare_data",
    "estimate_ols",
    "prepare_panel_unit_data",
    "prepare_panel_data",
    "coefficients_by_lag",
    "historical_decomposition",
]


def prepare_data(YY, lags, constant=True, timetrend=False, exog=None):
    """
    Organize the data in the form of Y = XB + E.

    Parameters
    ----------
    YY : numpy.ndarray
        The input time series data.
    lags : int
        Number of lags.
    constant : bool
        Whether to include a constant column.
    timetrend : bool
        Whether to include a time trend column.
    exog : numpy.ndarray, optional
        Additional exogenous regressors aligned with `YY` (shape (T, K)). They
        are appended after the constant/trend block. Pre-sample observations
        are dropped to match the lagged Y matrix.

    Returns
    -------
    YYact : numpy.ndarray
        The dependent variable matrix.
    XXact : numpy.ndarray
        The independent variable matrix. Column layout per equation:
        ``[lag-1 vars, ..., lag-P vars, constant?, trend?, exog cols...]``.
    """

    T0 = lags  # Pre-sample size
    nv = YY.shape[1]  # Number of variables
    nobs = YY.shape[0] - T0  # Number of observations

    # Actual observations
    YYact = YY[T0:T0 + nobs, :]
    XXact = np.zeros((nobs, nv * lags))

    for i in range(1, lags + 1):
        XXact[:, (i - 1) * nv:i * nv] = YY[T0 - i:T0 + nobs - i, :]

    if constant:
        XXact = np.hstack((XXact, np.ones((nobs, 1))))

    if timetrend:
        XXact = np.hstack((XXact, np.arange(1, nobs + 1).reshape(-1, 1)))

    if exog is not None:
        exog_arr = np.asarray(exog, dtype=float)
        if exog_arr.ndim == 1:
            exog_arr = exog_arr.reshape(-1, 1)
        if exog_arr.shape[0] != YY.shape[0]:
            raise ValueError(
                "`exog` must have the same number of rows as `YY` "
                f"(got {exog_arr.shape[0]} vs {YY.shape[0]})."
            )
        XXact = np.hstack((XXact, exog_arr[T0:T0 + nobs, :]))

    return YYact, XXact


def estimate_ols(yy, XX):
    """Estimate OLS coefficients."""
    
    # Parameters
    T, _ = yy.shape
    K = XX.shape[1]
    
    # OLS Estimation 
    XtX = XX.T @ XX
    XtY = XX.T @ yy
    B_OLS = np.linalg.inv(XtX) @ XtY  
    b_ols = B_OLS.flatten(order='F')
    residuals = yy - XX @ B_OLS 
    Sigma_ols = (residuals.T @ residuals) / (T - K)
    
    return b_ols, Sigma_ols


def prepare_panel_unit_data(y_unit, lags, exog=None, constant=True, timetrend=False):
    """
    Organize one panel unit in the form Y = X B + E.

    Parameters
    ----------
    y_unit : np.ndarray
        Array of shape (T, N) with endogenous variables for one unit.
    lags : int
        Number of lags in the VAR.
    exog : np.ndarray, optional
        Array of shape (T, K) with common exogenous regressors.
    constant : bool, default=True
        Whether to include a constant.
    timetrend : bool, default=False
        Whether to include a linear trend.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple with:
        - dependent variables `yy`
        - lagged endogenous regressors `x_lag`
        - deterministic/common exogenous regressors `z`
        - full regressor matrix `x_full`
    """
    if y_unit.ndim != 2:
        raise ValueError("`y_unit` must be a 2-dimensional array.")

    if lags < 1:
        raise ValueError("`lags` must be at least 1.")

    if exog is None:
        exog = np.zeros((y_unit.shape[0], 0))
    else:
        exog = np.asarray(exog, dtype=float)
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)
        if exog.shape[0] != y_unit.shape[0]:
            raise ValueError("`exog` must have the same number of rows as `y_unit`.")

    T, n_endo = y_unit.shape
    nobs = T - lags
    if nobs <= 0:
        raise ValueError("Not enough observations to build the requested lag structure.")

    yy = y_unit[lags:, :]
    x_lag = np.zeros((nobs, n_endo * lags))

    for lag in range(1, lags + 1):
        start = (lag - 1) * n_endo
        stop = lag * n_endo
        x_lag[:, start:stop] = y_unit[lags - lag:T - lag, :]

    z_blocks = []
    if constant:
        z_blocks.append(np.ones((nobs, 1)))
    if timetrend:
        z_blocks.append(np.arange(1, nobs + 1, dtype=float).reshape(-1, 1))
    if exog.shape[1] > 0:
        z_blocks.append(exog[lags:, :])

    z = np.hstack(z_blocks) if z_blocks else np.zeros((nobs, 0))
    x_full = np.hstack((x_lag, z)) if z.shape[1] > 0 else x_lag.copy()

    return yy, x_lag, z, x_full


def _trim_panel_unit_frame(unit_df, value_cols, allow_unbalanced):
    """
    Trim leading and trailing missing observations for one panel unit.

    When `allow_unbalanced=True`, MacroPy accepts ragged-edge panels where
    some countries start later or end earlier. Internal missing observations
    inside the usable sample are still rejected because the current sampler
    assumes contiguous lags.
    """
    complete_mask = ~unit_df[value_cols].isna().any(axis=1)
    if complete_mask.all():
        return unit_df.reset_index(drop=True)

    if not allow_unbalanced:
        raise ValueError(
            "Missing values are not supported in panel variables. "
            "Trim the panel first or set `allow_unbalanced=True` to drop "
            "leading/trailing missing observations."
        )

    complete_positions = np.flatnonzero(complete_mask.to_numpy())
    if complete_positions.size == 0:
        raise ValueError("Each panel unit must contain at least one complete observation.")

    first_complete = int(complete_positions[0])
    last_complete = int(complete_positions[-1])
    usable_mask = complete_mask.iloc[first_complete : last_complete + 1]
    if not usable_mask.all():
        raise ValueError(
            "Internal missing observations are not yet supported in `BayesianPanelVAR`. "
            "Please interpolate/impute externally, or pass a panel where missingness only "
            "appears at the beginning or end of each unit sample."
        )

    return unit_df.iloc[first_complete : last_complete + 1].reset_index(drop=True)


def _extract_common_exogenous(frame, time_col, exog):
    """Validate and align common exogenous variables on the union time index."""
    if not exog:
        unique_dates = pd.DatetimeIndex(frame[time_col].drop_duplicates().sort_values())
        return pd.DataFrame(index=unique_dates)

    grouped = frame.groupby(time_col, sort=True)
    for exo_name in exog:
        disagreements = grouped[exo_name].nunique(dropna=False)
        if (disagreements > 1).any():
            raise ValueError(
                "Common exogenous variables must take the same value across units "
                "for each time period."
            )

    exo_by_time = grouped[exog].first().sort_index()
    if exo_by_time.isna().any().any():
        raise ValueError("Missing values are not supported in common exogenous variables.")

    return exo_by_time


def prepare_panel_data(
    data,
    unit_col="unit",
    time_col="date",
    endog=None,
    exog=None,
    allow_unbalanced=False,
):
    """
    Validate and reshape a long-format panel into unit-level NumPy arrays.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    unit_col : str, default="unit"
        Column identifying cross-sectional units.
    time_col : str, default="date"
        Column identifying the time dimension.
    endog : list[str]
        Endogenous variable names.
    exog : list[str], optional
        Common exogenous variable names. These must match across units for
        each time period.
    allow_unbalanced : bool, default=False
        If True, allow unit-specific sample starts/ends and trim leading or
        trailing missing observations. Internal missing observations inside a
        unit sample are still rejected.

    Returns
    -------
    dict
        Dictionary with the validated dataframe, panel metadata, unit-specific
        endogenous arrays, and aligned exogenous arrays.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a pandas DataFrame.")

    endog = list(endog or [])
    exog = list(exog or [])
    if not endog:
        raise ValueError("`endog` must contain at least one endogenous variable name.")

    df = data.copy()
    if unit_col not in df.columns or time_col not in df.columns:
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        elif df.index.name in {unit_col, time_col}:
            df = df.reset_index()

    required_columns = [unit_col, time_col, *endog, *exog]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required panel columns: {missing}")

    df = df[required_columns].copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([unit_col, time_col]).reset_index(drop=True)

    if df.duplicated(subset=[unit_col, time_col]).any():
        raise ValueError("Each (unit, time) pair must appear only once.")

    units = df[unit_col].drop_duplicates().tolist()
    if not units:
        raise ValueError("No panel units were found in the provided data.")

    unit_frames = []
    unit_dates = []
    y_units = []
    for unit in units:
        unit_df = df.loc[df[unit_col] == unit, [time_col, *endog, *exog]].reset_index(drop=True)
        unit_df = _trim_panel_unit_frame(
            unit_df=unit_df,
            value_cols=[*endog, *exog],
            allow_unbalanced=allow_unbalanced,
        )
        unit_frames.append(unit_df)
        unit_dates.append(pd.DatetimeIndex(unit_df[time_col]))
        y_units.append(unit_df[endog].to_numpy(dtype=float))

    first_dates = unit_dates[0]
    balanced = all(dates.equals(first_dates) for dates in unit_dates[1:])
    if not balanced and not allow_unbalanced:
        raise ValueError(
            "The panel must be balanced and share the same ordered time index across all units. "
            "Set `allow_unbalanced=True` to estimate a ragged-edge hierarchical panel VAR."
        )

    dates = first_dates if balanced else pd.DatetimeIndex(
        sorted({timestamp for unit_index in unit_dates for timestamp in unit_index})
    )

    exo_by_time = _extract_common_exogenous(df, time_col=time_col, exog=exog)
    if exog:
        exo_panel = exo_by_time.reindex(dates).to_numpy(dtype=float)
        exo_units = []
        for dates_u in unit_dates:
            exo_unit = exo_by_time.reindex(dates_u).to_numpy(dtype=float)
            if np.isnan(exo_unit).any():
                raise ValueError(
                    "Common exogenous variables must be observed for every retained "
                    "unit-time observation."
                )
            exo_units.append(exo_unit)
    else:
        exo_panel = np.zeros((len(dates), 0))
        exo_units = [np.zeros((len(dates_u), 0)) for dates_u in unit_dates]

    unit_sample = pd.DataFrame(
        {
            unit_col: units,
            "sample_start": [dates_u[0] for dates_u in unit_dates],
            "sample_end": [dates_u[-1] for dates_u in unit_dates],
            "observations": [len(dates_u) for dates_u in unit_dates],
        }
    )

    return {
        "frame": df,
        "units": units,
        "balanced": balanced,
        "dates": dates,
        "unit_dates": unit_dates,
        "unit_frames": unit_frames,
        "unit_sample": unit_sample,
        "y": y_units,
        "exo": exo_panel,
        "exo_units": exo_units,
        "endog_names": endog,
        "exo_names": exog,
    }


def coefficients_by_lag(B, n, p, constant=True):
    """
    Split a MacroPy coefficient matrix into lag matrices and the constant.

    MacroPy stacks the regressors as ``[y_{t-1}, ..., y_{t-p}, constant, trend,
    exog]`` (``prepare_data``), so the constant sits **after** the lags in
    ``B`` (k x n, one column per equation). Many textbooks put it in the first
    row instead; this helper isolates the user from the layout.

    Parameters
    ----------
    B : array (k x n)
        Coefficients by equation in columns, e.g. ``b_ols.reshape((k, n),
        order="F")`` in ``ClassicVAR`` or ``beta_draws[d].reshape(n, k).T`` in
        ``BayesianVAR``.
    n, p : int
        Number of variables and lags.
    constant : bool, default=True
        Whether the model has a constant (the row right after the lags).

    Returns
    -------
    phis : list of p arrays (n x n)
        ``phis[l - 1][i, j]`` is the coefficient on variable ``j`` at lag ``l``
        in equation ``i``.
    c : array (n,) or None
        The constant, or None when ``constant=False``.
    """
    B = np.asarray(B)
    phis = [B[lag * n:(lag + 1) * n, :].T for lag in range(p)]
    c = B[n * p, :].copy() if constant else None
    return phis, c


def historical_decomposition(B, Sigma, yy, XX, n, p, S=None):
    """
    Historical decomposition of a reduced-form VAR, identified recursively.

    Each observation is written as the part explained by the initial conditions
    and the deterministic terms plus the accumulated effect of every structural
    shock, using the companion recursion
    ``state_t = F state_{t-1} + [S eps_t; 0]``.

    Parameters
    ----------
    B : array (k x n)
        Coefficients in MacroPy's layout: ``[lag-1 vars, ..., lag-p vars,
        constant?, trend?, exog...]`` down the rows, one column per equation.
    Sigma : array (n x n)
        Covariance of the reduced-form innovations.
    yy, XX : arrays (T x n) and (T x k)
        Estimation sample as returned by `prepare_data`.
    n, p : int
        Number of variables and lags.
    S : array (n x n), optional
        Impact matrix. Defaults to the Cholesky factor of `Sigma`, that is, a
        recursive identification in the order of the columns of `yy`.

    Returns
    -------
    shock : array (T, n, n)
        ``shock[t, j, i]``: contribution of structural shock ``j`` to variable
        ``i`` at time ``t``.
    baseline : array (T, n)
        Initial conditions plus deterministic and exogenous terms.

    Notes
    -----
    The decomposition is exact: ``baseline + shock.sum(axis=1)`` reproduces
    ``yy`` up to floating point error.
    """
    B = np.asarray(B, dtype=float)
    yy = np.asarray(yy, dtype=float)
    XX = np.asarray(XX, dtype=float)
    S = np.linalg.cholesky(Sigma) if S is None else np.asarray(S, dtype=float)

    T, lag_block = yy.shape[0], n * p
    eps = np.linalg.solve(S, (yy - XX @ B).T)          # structural shocks (n x T)

    F = np.zeros((lag_block, lag_block))               # companion matrix
    F[:n, :] = B[:lag_block, :].T
    if p > 1:
        F[n:, :n * (p - 1)] = np.eye(n * (p - 1))
    impact = np.zeros((lag_block, n))                  # column j: shock j on impact
    impact[:n, :] = S

    # Deterministic and exogenous part, read from the regressors themselves so
    # that the trend convention and any exogenous block are respected.
    deterministic = (XX[:, lag_block:] @ B[lag_block:, :] if XX.shape[1] > lag_block
                     else np.zeros((T, n)))

    shock = np.zeros((T, n, n))
    baseline = np.zeros((T, n))
    state = np.zeros((lag_block, n))                   # one column per shock
    base_state = XX[0, :lag_block].copy()              # initial conditions
    lifted = np.zeros(lag_block)
    for t in range(T):
        state = F @ state + impact * eps[:, t]
        shock[t] = state[:n, :].T
        lifted[:n] = deterministic[t]
        base_state = F @ base_state + lifted
        baseline[t] = base_state[:n]
    return shock, baseline
