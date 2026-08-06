"""seasonal_adjust: both engines remove seasonality and keep the identity
observed = seasadj + seasonal in original units."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from MacroPy import seasonal_adjust


def _seasonal_series(n_years: int = 10) -> pd.Series:
    idx = pd.period_range("2014-01", periods=12 * n_years, freq="M")
    t = np.arange(len(idx))
    rng = np.random.default_rng(7)
    y = 100 + 0.4 * t + 12 * np.sin(2 * np.pi * (t % 12) / 12) + rng.normal(0, 1.5, len(idx))
    return pd.Series(y, index=idx, name="y")


def _month_amplitude(s: pd.Series) -> float:
    """Spread of month-of-year means after detrending: seasonality gauge."""
    detr = s - s.rolling(13, center=True, min_periods=7).mean()
    return float(detr.groupby(detr.index.month).mean().std())


@pytest.mark.parametrize("method", ["stl", "x13"])
def test_removes_seasonality_and_recomposes(method):
    if method == "x13":
        pytest.importorskip("statsmodels.tsa.x13")
        from MacroPy import x13_path
        try:
            x13_path()
        except FileNotFoundError:
            pytest.skip("no bundled x13as for this platform")
    y = _seasonal_series()
    res = seasonal_adjust(y, method=method)
    assert res.method == method
    assert len(res.seasadj) == len(y)
    pd.testing.assert_index_equal(res.seasadj.index, res.observed.index)
    # identity in original units
    np.testing.assert_allclose(res.observed, res.seasadj + res.seasonal, rtol=1e-8)
    # the seasonal pattern must be essentially gone
    assert _month_amplitude(res.seasadj) < 0.15 * _month_amplitude(res.observed)


def test_input_validation():
    y = _seasonal_series()
    with pytest.raises(ValueError, match="three full years"):
        seasonal_adjust(y.iloc[:20], method="stl")
    bad = pd.Series(np.arange(60.0), index=range(60))
    with pytest.raises((ValueError, TypeError)):
        seasonal_adjust(bad, method="stl")
    gappy = _seasonal_series()
    gappy.iloc[50] = np.nan
    with pytest.raises(ValueError, match="missing"):
        seasonal_adjust(gappy, method="stl")
