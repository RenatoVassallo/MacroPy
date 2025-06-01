# 🧠 MacroPy

**A Bayesian Toolbox for Macroeconometric Analysis in Python**
Built for researchers, designed for clarity, and ready for publication.

---

## 🚀 What is MacroPy?

**MacroPy** is a flexible and intuitive Python library for estimating **Bayesian** and **Frequentist** Vector Autoregressions (VARs). It enables robust structural analysis with a focus on:

* Impulse Response Functions (IRFs)
* Variance Decompositions
* Forecasting (unconditional and conditional)
* Visual and publication-ready diagnostics

All with minimal syntax and academic-grade rigor.

---

## 📦 Key Features

* Bayesian and Frequentist VAR estimation with flexible prior settings
* Structural IRFs with credible intervals
* Support for block exogeneity restrictions
* Conditional forecasting à la Waggoner & Zha (1999)
* Posterior diagnostics and coefficient visualization
* Unconditional forecasts with customizable fan charts

---

## 🧪 Quick Start

```python
# Bayesian VAR estimation
from MacroPy import BayesianVAR

bvar = BayesianVAR(df, lags=4, hor=24, irf_1std=0)
bvar.model_summary()
bvar.sample_posterior(plot_coefficients=True)

# Compute and plot impulse responses
irfs = bvar.compute_irfs(plot_irfs=True)
```

---

## 🔬 Coming Soon

Next releases will introduce:

* Pandemic Priors for crisis-specific modeling
* Threshold Bayesian VARs for regime-switching dynamics
* Unobserved Components Models
* Even tighter plotting and inference tools

---

## 📚 Citation

If you use **MacroPy** in academic work, please cite:

> Vassallo, R. (2025). *MacroPy: A Bayesian Toolbox for Macroeconometrics in Python*, Version 0.1.3.
