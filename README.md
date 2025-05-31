# 🧠 MacroPy

**Bayesian Toolbox for Macroeconometric Analysis in Python**  
Built for researchers. Designed for clarity. Ready for journals.

---

## 🚀 What is MacroPy?

**MacroPy** is a flexible Python package for estimating Frequentist and Bayesian Vector Autoregressions (BVARs), computing Impulse Response Functions (IRFs), Variance Decompositions and producing publication-ready macroeconomic insights — all with minimal syntax and robust methods.

---

## 📦 Features

- Bayesian VAR estimation with customizable priors
- Impulse Response Functions with credible intervals
- Posterior diagnostics and coefficient visualizations
- Ready-to-publish IRF and variance decomposition plots

---

## 📂 Project Structure

MacroPy/
├── src/
│   └── MacroPy/
│       ├── init.py
│       ├── cvar.py
│       ├── bvar.py
│       ├── data_handling.py
│       ├── get_macrodata.py
│       ├── plots.py
│       ├── priors.py
│       ├── summary.py
├── tests/
├── notebooks/
├── pyproject.toml
└── README.md

---

## 🧪 Quick Example

```python
# Estimate a Bayesian VAR
from MacroPy import BayesianVAR
bvar = BayesianVAR(df_sw, lags=4, hor=24, irf_1std=0)

# Model summary & posterior draws
bvar.model_summary()
bvar.sample_posterior(plot_coefficients=True)

# Compute and plot impulse responses
irfs = bvar.compute_irfs(plot_irfs=True)
```

## 📚 Citation

If you use MacroPy in academic work, please consider citing it as:

Vassallo, R. (2025), “MacroPy: A Bayesian Toolbox for Macroeconometrics in Python,” Version 0.1.2.