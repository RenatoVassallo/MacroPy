# 🧠 MacroPy

**A Toolbox for Macroeconometric Analysis in Python**  
Built for researchers. Designed for clarity. Ready for publication.

---

## 🚀 What is MacroPy?

MacroPy is a flexible and intuitive Python package designed for macroeconometric modeling and forecasting. While it originated with a focus on **Bayesian VARs**, it has evolved into a broader toolbox supporting structural analysis, **Smooth Local Projections** and seamless **API integrations** for macroeconomic data.

Whether you’re forecasting GDP or tracing the effects of policy shocks, `MacroPy` delivers publication-ready results with minimal code and academic-grade rigor.

---

## 📦 Key Features

`MacroPy` includes robust functionality for:

✅ APIs for data: FRED (St. Louis Fed), BIS and Peruvian Central Bank
✅ Seasonal adjustment using X-13ARIMA-SEATS (U.S. Census Bureau) and STL decomposition   
✅ Frequentist and Bayesian VAR estimation  
✅ Structural impulse response functions and Forecast Error Variance Decomposition  
✅ Unconditional and conditional forecasts  
✅ Smooth Local Projections (Barnichon & Brownlees, 2019)  
✅ Threshold VARs with regime-specific covariances (exact posterior) or with stochastic volatility and uncertainty shocks, compared side by side and by out-of-sample log scores (Alessandri & Mumtaz, 2017, 2019)  
✅ Time-varying parameter VARs with stochastic volatility, compared by marginal likelihood and DIC (Chan & Eisenstat, 2018)  
✅ Trend-cycle decomposition: HP filter, Clark UC by MLE, and Bayesian UC with COVID variance scales (Grant & Chan, 2017; Lenza & Primiceri, 2022)  
✅ Publication-ready matplotlib figures in one consistent MacroPy style, for notebooks, papers and slides  

**Specifically for the `BayesianVAR` class:**

- Minnesota, Normal-Wishart, and Normal-Diffuse priors  
- Structural IRFs using Cholesky decomposition  
- Support for block exogeneity (zero restrictions on lag structures)
- Conditional forecasting à la Waggoner & Zha (1999) with the full conditional shock distribution, and friendly `{variable: path}` scenario dicts  
- Fan chart-style forecast plots with flexible credibility intervals  
- COVID-window treatment: Lenza & Primiceri (2022) volatility scaling or automatic time dummies  
- Full reproducibility via a `seed` argument, and tidy `forecast_frame()` / `conditional_forecast_frame()` outputs for production pipelines  
- Data-driven prior tightness via marginal-likelihood maximization (Giannone, Lenza & Primiceri, 2015)  
- Sum-of-coefficients and dummy-initial-observation priors (Doan-Litterman-Sims, 1984; Sims, 1993) to anchor long-horizon forecasts to current levels  
- Steady-state (mean-adjusted) prior (Villani, 2009) to anchor the long-run mean to judgment, e.g. medium-term growth forecasts  

---

## 🔧 Installation

You can install the latest pre-release directly from GitHub:

```bash
pip install https://github.com/RenatoVassallo/MacroPy/releases/download/0.3.0/macropy-0.3.0-py3-none-any.whl
```

> 🆕 **New to Python?** Follow this practical guide to set up Python and VS Code before you start:
> [Python + VS Code: a practical setup](https://renatovassallo.github.io/posts/python-vscode-practical-setup/)

---

## 🧪 Quick Start

```python
from MacroPy import BayesianVAR

# Estimate a BVAR(4) with standard settings
bvar = BayesianVAR(df, lags=4, hor=24, post_draws=50000)
bvar.model_summary()
post_draws = bvar.sample_posterior(plot_coefficients=True)

# IRFs with 68% and 95% credible intervals
irfs_results = bvar.compute_irfs(plot_irfs=True, cred_interval=[0.68, 0.95])

# Forecast with fan chart
forecasts_results = bvar.forecast(fhor=12, plot_forecast=True, cred_interval=[0.90, 0.60, 0.30])
```

---

## 📁 Tutorials

Explore full examples in the [tutorials/](https://github.com/RenatoVassallo/MacroPy/tree/main/tutorials) folder:

- 📡 [`tutorial_api.ipynb`](https://github.com/RenatoVassallo/MacroPy/blob/main/tutorials/tutorial_api.ipynb): Accessing macroeconomic data via APIs  
- 🧮 [`tutorial_cvar.ipynb`](https://github.com/RenatoVassallo/MacroPy/blob/main/tutorials/tutorial_cvar.ipynb): Classic VAR estimation  
- 🧠 [`tutorial_bvar.ipynb`](https://github.com/RenatoVassallo/MacroPy/blob/main/tutorials/tutorial_bvar.ipynb): Bayesian VAR estimation and forecasting  
- 🦠 [`tutorial_bvar_pandemic.ipynb`](https://github.com/RenatoVassallo/MacroPy/blob/main/tutorials/tutorial_bvar_pandemic.ipynb): COVID treatment — pandemic dummies vs Lenza-Primiceri (2022) volatility scaling  
- ⚓ [`tutorial_bvar_socdio.ipynb`](https://github.com/RenatoVassallo/MacroPy/blob/main/tutorials/tutorial_bvar_socdio.ipynb): Sum-of-coefficients and dummy-initial-observation priors, with a pseudo out-of-sample conditional forecasting exercise  
- 🏗️ [`tutorial_bpvar.ipynb`](https://github.com/RenatoVassallo/MacroPy/blob/main/tutorials/tutorial_bpvar.ipynb): Hierarchical Bayesian Panel VAR  
- 📉 [`tutorial_localprojections.ipynb`](https://github.com/RenatoVassallo/MacroPy/blob/main/tutorials/tutorial_localprojections.ipynb): Smooth Local Projections (Barnichon & Brownlees, 2019)  
- ⛓️ [`tutorial_thresholdvar.ipynb`](https://github.com/RenatoVassallo/MacroPy/blob/main/tutorials/tutorial_thresholdvar.ipynb): Threshold VARs with and without stochastic volatility (Alessandri & Mumtaz, 2017, 2019): regimes, volatility, and financial and uncertainty shocks by regime
- 🕰️ [`tutorial_tvpvar.ipynb`](https://github.com/RenatoVassallo/MacroPy/blob/main/tutorials/tutorial_tvpvar.ipynb): Time-varying parameter VARs with stochastic volatility on Peruvian data: IRFs by date and in 3D, FEVD over time, historical decomposition, fan charts, and seven specifications compared by marginal likelihood and DIC (Chan & Eisenstat, 2018)
- 📐 [`tutorial_uc_gdp.ipynb`](https://github.com/RenatoVassallo/MacroPy/blob/main/tutorials/tutorial_uc_gdp.ipynb): Trend-cycle decomposition ladder on US GDP, from the HP filter to a Bayesian UC model with COVID variance scales

---

## 📚 Citation

If you use **MacroPy** in academic work, please cite:

> Vassallo, R. (2026). *MacroPy: A Toolbox for Macroeconometric Analysis in Python*, Version 0.3.1.  
> [GitHub Repository](https://github.com/RenatoVassallo/MacroPy)
