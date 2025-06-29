from IPython.display import Markdown

def generate_summary(settings):
    """
    Generate a structured Markdown summary of the VAR model.
    Adapts to both ClassicVAR and BayesianVAR settings objects.
    """
    linkedin_text = "Renato Vassallo"
    linkedin_url = "https://www.linkedin.com/in/renatovassallo"

    # === VAR EQUATIONS IN LATEX ===
    var_equations = r"\begin{align*}" + "\n"
    for i, var in enumerate(settings.names):
        equation = f"{var}_{{t}} &="
        for lag in range(1, settings.lags + 1):
            for j, other_var in enumerate(settings.names):
                equation += f" b_{{{i+1},{j+1}}}^{({lag})} {other_var}_{{t-{lag}}} +"
        equation += f" c_{{{i+1}}} + e_{{t}}^{{{var}}} \\\\\n"
        var_equations += equation
    var_equations += r"\end{align*}"

    # === HEADER ===
    summary = f"""
**MacroPy Toolbox for Macroeconometric Analysis in Python**  
Developed by [{linkedin_text}]({linkedin_url}), Institute for Economic Analysis (IAE-CSIC)  
Version 0.1.4, June 2025  

---

**Model Specifications**  
- **Model Type**: {'Bayesian VAR' if hasattr(settings, 'prior_name') else 'Classic VAR'}  
- **Endogenous Variables**: {', '.join(settings.names)}  
- **Exogenous Variables**: {'Constant' if settings.constant else ''} {'Trend' if settings.timetrend else ''}  
- **Number of Lags**: {settings.lags}  
- **Total Number of Coefficients to Estimate**: {settings.ncoeff}  
"""

    # === BAYESIAN SECTION (IF APPLICABLE) ===
    if hasattr(settings, "prior_name"):
        summary += f"""
---

**MCMC Algorithm**: Gibbs Sampling  
- **Prior Type**: {settings.prior_name}  
- **Iterations**: {settings.post_draws}  
- **Burn-in**: {settings.burnin} ({settings.burnin/settings.post_draws:.0%})  
"""

    # === FORECAST / IRF SECTION (Shared) ===
    summary += f"""
---

**Forecast & IRF Details**  
- **Impulse Response Horizon**: {settings.hor}  
- **Forecast Horizon**: {settings.fhor}  
- **IRF Computation**: {'1 Standard Deviation' if getattr(settings, 'irf_1std', 1) == 1 else 'Unit Shock'}  

---

**VAR Model Equations**

$$
{var_equations}
$$
"""

    return Markdown(summary)
