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

✅ APIs for data: FRED (St. Louis Fed) and Peruvian Central Bank  
✅ Frequentist and Bayesian VAR estimation  
✅ Structural impulse response functions and Forecast Error Variance Decomposition  
✅ Unconditional and conditional forecasts  
✅ Smooth Local Projections (Barnichon & Brownlees, 2019)  
✅ High-quality plots using `ggplot2`-style, ready for LaTeX or reports  

**Specifically for the `BayesianVAR` class:**

- Minnesota, Normal-Wishart, and Normal-Diffuse priors  
- Structural IRFs using Cholesky decomposition  
- Support for block exogeneity (zero restrictions on lag structures)
- Conditional forecasting à la Waggoner & Zha (1999)  
- Fan chart-style forecast plots with flexible credibility intervals  

---

## 🔧 Installation

You can install the latest pre-release directly from GitHub:

```bash
pip install https://github.com/RenatoVassallo/MacroPy/releases/download/0.1.5/macropy-0.1.5-py3-none-any.whl
```

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
- 📉 [`tutorial_localprojections.ipynb`](https://github.com/RenatoVassallo/MacroPy/blob/main/tutorials/tutorial_localprojections.ipynb): Smooth Local Projections (Barnichon & Brownlees, 2019)

---

## 🔮 Coming Soon

The roadmap for future versions includes:

- 🦠 Pandemic-specific priors for crisis episodes  
- ⛓️ Threshold and regime-switching BVARs  
- 🔍 Unobserved components models  
- 🖼️ Enhanced plotting themes and publication export options  
- 🧭 DSGE-style simulation support  

---

## 📚 Citation

If you use **MacroPy** in academic work, please cite:

> Vassallo, R. (2025). *MacroPy: A Toolbox for Macroeconometric Analysis in Python*, Version 0.1.5.  
> [GitHub Repository](https://github.com/RenatoVassallo/MacroPy)
