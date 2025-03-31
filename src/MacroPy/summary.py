from IPython.display import Markdown

def generate_summary(settings):
    """
    Print a summary of the Bayesian VAR model in a structured format with a clickable LinkedIn link.
    Also includes the mathematical representation of the VAR equations in LaTeX format.
    """
    linkedin_text = "Renato Vassallo"
    linkedin_url = "https://www.linkedin.com/in/renatovassallo"
    
    # Generate VAR equations dynamically
    var_equations = r"\begin{align*}" + "\n"
    
    for i, var in enumerate(settings.names):
        equation = f"    {var}_{{t}} &= c_{{{i+1}}} "  # Constant term
        
        for lag in range(1, settings.lags + 1):
            for j, other_var in enumerate(settings.names):
                equation += f"+ b_{{{i+1},{j+1}}} {other_var}_{{t-{lag}}} "
        
        equation += f"+ e_{{t}}^{{{var}}} \\\\\n"  # Error term
        var_equations += equation
    
    var_equations += r"\end{align*}"
    
    # Full summary including LaTeX equations
    summary = f"""
**MacroPy Toolbox for Bayesian Macroeconometric Analysis in Python**  
Developed by [{linkedin_text}]({linkedin_url}), Institute for Economic Analysis (IAE-CSIC)  
Version 0.1, March 2025  

---
**1. Model Specifications**  
- **Endogenous Variables**: {', '.join(settings.names)}  
- **Exogenous Variables**: {'Constant' if settings.constant else ''} {'Trend' if settings.timetrend else ''}  
- **Number of Lags**: {settings.lags}  
- **Total Number of Coefficients to Estimate**: {settings.ncoeff}  

---
**2. MCMC Algorithm**: Gibbs Sampling  
- **Prior Type**: {settings.prior_name}  
- **Iterations**: {settings.post_draws}  
- **Burn-in**: {settings.burnin} ({settings.burnin/settings.post_draws:.0%})  

---
**3. Forecast & IRF Details**  
- **Impulse Response Horizon**: {settings.hor}  
- **Forecast Horizon**: {settings.fhor}  
- **IRF Computation**: {'1 Standard Deviation' if settings.irf_1std else 'Unit Shock'}  

---
**4. VAR Model Equations**
  
$$
{var_equations}
$$
"""

    return Markdown(summary)  
