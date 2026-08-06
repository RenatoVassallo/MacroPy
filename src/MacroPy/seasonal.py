"""Seasonal adjustment in one call.

``seasonal_adjust`` fronts two engines behind the same tidy result:

- ``method="x13"`` (default): the US Census Bureau's X-13ARIMA-SEATS, the
  reference program used by statistical agencies (BCRP, INEI, BEA,
  Eurostat members). MacroPy bundles the official binary per platform
  (see ``MacroPy.x13``) and drives it through ``statsmodels``: a regARIMA
  model extends the series with forecasts and removes deterministic
  effects (outliers, optionally trading day), then the X-11 filter
  cascade extracts trend, seasonal and irregular components.
- ``method="stl"``: STL (Cleveland et al. 1990), a transparent
  LOESS-based decomposition computed natively in Python. Useful as a
  fast, dependency-light cross-check; it is NOT what agencies publish.

The returned components always satisfy ``observed = seasadj + seasonal``
in the original units (for X13 the seasonal effect is defined as
``observed - seasadj``, which is well defined under both additive and
log-multiplicative regARIMA transforms). The ``irregular`` component is
passed through in the engine's own convention: under X13's automatic
log/multiplicative transform it is a ratio fluctuating around 1 (table
D13), while STL returns an additive residual.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = ["SeasonalResult", "seasonal_adjust"]

_PERIODS = {"M": 12, "Q": 4}


@dataclass
class SeasonalResult:
    """Decomposition in original units; ``observed = seasadj + seasonal``."""

    observed: pd.Series
    seasadj: pd.Series
    trend: pd.Series
    irregular: pd.Series
    seasonal: pd.Series
    method: str

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "observed": self.observed, "seasadj": self.seasadj,
            "trend": self.trend, "seasonal": self.seasonal,
            "irregular": self.irregular,
        })


def _prepare(series: pd.Series | pd.DataFrame) -> tuple[pd.Series, str]:
    """Coerce to a clean Series on a monthly or quarterly date index."""
    s = series.squeeze()
    if not isinstance(s, pd.Series):
        raise TypeError("seasonal_adjust expects a single series")
    if isinstance(s.index, pd.PeriodIndex):
        freq = s.index.freqstr[0].upper()
        s = pd.Series(s.to_numpy(), index=s.index.to_timestamp(how="start"),
                      name=s.name)
    else:
        idx = pd.DatetimeIndex(pd.to_datetime(s.index))
        s = pd.Series(s.to_numpy(), index=idx, name=s.name).sort_index()
        inferred = pd.infer_freq(s.index)
        if inferred is None:
            raise ValueError("could not infer a monthly or quarterly frequency "
                             "from the index")
        freq = inferred[0].upper()
        if freq == "Q" or inferred.upper().startswith("QS"):
            freq = "Q"
    if freq not in _PERIODS:
        raise ValueError(f"only monthly and quarterly series are supported, "
                         f"got frequency {freq!r}")
    s = s.loc[s.first_valid_index():s.last_valid_index()].astype(float)
    if s.isna().any():
        raise ValueError("interior missing values: fill or trim them first")
    if len(s) < 3 * _PERIODS[freq]:
        raise ValueError(f"need at least three full years of data, "
                         f"got {len(s)} observations")
    # a freq-carrying index keeps statsmodels from guessing
    s.index = pd.PeriodIndex(s.index, freq=freq).to_timestamp(how="start")
    return s, freq


def seasonal_adjust(series, method: str = "x13", *, outlier: bool = True,
                    trading_day: bool = False,
                    x13_binary: str | None = None) -> SeasonalResult:
    """Seasonally adjust a monthly or quarterly series.

    Parameters
    ----------
    series : pd.Series (or one-column DataFrame) on a date or period index.
    method : "x13" (Census X-13ARIMA-SEATS, the bundled reference engine)
        or "stl" (native LOESS-based decomposition).
    outlier : let regARIMA detect additive outliers and level shifts (x13).
    trading_day : include trading-day regressors (x13).
    x13_binary : explicit path to an x13as executable; defaults to the
        binary bundled with MacroPy, then to what statsmodels finds on
        PATH / X13PATH.
    """
    s, freq = _prepare(series)

    if method == "stl":
        from statsmodels.tsa.seasonal import STL

        fit = STL(s, period=_PERIODS[freq], robust=True).fit()
        seasadj = s - fit.seasonal
        return SeasonalResult(observed=s, seasadj=seasadj, trend=fit.trend,
                              irregular=fit.resid, seasonal=fit.seasonal,
                              method="stl")

    if method != "x13":
        raise ValueError(f"unknown method {method!r}; use 'x13' or 'stl'")

    from statsmodels.tsa.x13 import x13_arima_analysis
    from statsmodels.tools.sm_exceptions import X13NotFoundError

    path = x13_binary
    bundle_error = None
    if path is None:
        try:
            from .x13 import x13_path
            path = str(x13_path())
        except FileNotFoundError as exc:
            bundle_error = exc     # let statsmodels search PATH / X13PATH
    try:
        res = x13_arima_analysis(s, x12path=path, prefer_x13=True,
                                 outlier=outlier, trading=trading_day, log=None)
    except X13NotFoundError as exc:
        raise RuntimeError(
            "no x13as binary available for this platform. "
            + (f"{bundle_error} " if bundle_error else "")
            + "Alternatively, use method='stl' for a pure-Python decomposition."
        ) from exc
    seasadj = res.seasadj.astype(float)
    seasadj.index = s.index
    trend = res.trend.astype(float)
    trend.index = s.index
    irregular = res.irregular.astype(float)
    irregular.index = s.index
    return SeasonalResult(observed=s, seasadj=seasadj, trend=trend,
                          irregular=irregular, seasonal=s - seasadj,
                          method="x13")
