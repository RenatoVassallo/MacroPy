from importlib.metadata import PackageNotFoundError, version

from IPython.display import Markdown

try:
    __version__ = version("MacroPy")
except PackageNotFoundError:  # package is not installed (e.g. running from a source checkout)
    __version__ = "0.0.0+unknown"


def _format_exogenous(settings):
    pieces = []
    if getattr(settings, "constant", False):
        pieces.append("Constant")
    if getattr(settings, "timetrend", False):
        pieces.append("Trend")
    # BayesianVAR uses `exog_names`; BayesianPanelVAR uses `exo_names`. Cover both.
    pieces.extend(getattr(settings, "exo_names", []) or getattr(settings, "exog_names", []))
    return ", ".join(pieces) if pieces else "None"


def _panel_equations():
    return r"""
\begin{align*}
y_{u,t} &= \sum_{\ell=1}^{p} A_{u,\ell} y_{u,t-\ell} + C_{u} x_t + e_{u,t},
\qquad e_{u,t} \sim \mathcal{N}(0, \Sigma_u) \\
\beta_u &= \mathrm{vec}\left(A_{u,1}, \ldots, A_{u,p}\right) \\
\beta_u &\sim \mathcal{N}\!\left(\bar{\beta}, \lambda \Omega_u\right)
\end{align*}
"""


def _var_equations(settings):
    equations = r"\begin{align*}" + "\n"
    for i, var in enumerate(settings.names):
        equation = f"{var}_{{t}} &="
        for lag in range(1, settings.lags + 1):
            for j, other_var in enumerate(settings.names):
                equation += f" b_{{{i+1},{j+1}}}^{({lag})} {other_var}_{{t-{lag}}} +"
        equation += f" c_{{{i+1}}} + e_{{t}}^{{{var}}} \\\\\n"
        equations += equation
    equations += r"\end{align*}"
    return equations


def generate_summary(settings):
    """
    Generate a structured Markdown summary for MacroPy models.
    """
    linkedin_text = "Renato Vassallo"
    linkedin_url = "https://www.linkedin.com/in/renatovassallo"

    model_type = getattr(
        settings,
        "model_type",
        "Bayesian VAR" if hasattr(settings, "prior_name") else "Classic VAR",
    )
    exog_text = _format_exogenous(settings)

    if hasattr(settings, "n_units"):
        if getattr(settings, "is_balanced_panel", True):
            sample_start = settings.yy_dates[0].date()
            sample_end = settings.yy_dates[-1].date()
            obs_text = f"{settings.nobs} observations per unit"
        else:
            sample_start = min(dates[0] for dates in settings.yy_dates_units).date()
            sample_end = max(dates[-1] for dates in settings.yy_dates_units).date()
            obs_text = (
                f"{settings.min_nobs} to {settings.max_nobs} observations per unit "
                "(unit-specific windows)"
            )
        panel_block = f"""
- **Panel Units**: {', '.join(map(str, settings.units))}
- **Number of Units**: {settings.n_units}
- **Panel Structure**: {getattr(settings, "panel_balance", "Balanced")}
"""
        equations = _panel_equations()
    else:
        sample_start = settings.dates[settings.lags].date()
        sample_end = settings.dates[-1].date()
        obs_text = f"{settings.yy.shape[0]} observations"
        panel_block = ""
        equations = _var_equations(settings)

    summary = f"""
**MacroPy: A Toolbox for Bayesian Macroeconometric Analysis in Python**
Developed by [{linkedin_text}]({linkedin_url}) - Institute for Economic Analysis (IAE-CSIC)
Version {__version__}

---

**Model Specifications**  
- **Model Type**: {model_type}
- **Endogenous Variables**: {', '.join(map(str, settings.names))}
- **Exogenous Variables**: {exog_text}
- **Number of Lags**: {settings.lags}
- **Sample Period**: {sample_start} to {sample_end} ({obs_text})
{panel_block}- **Total Parameters Estimated**: {settings.ncoeff}
"""

    if hasattr(settings, "prior_name"):
        summary += f"""
---

**Bayesian Estimation Settings**
- **Posterior Simulation**: Gibbs Sampling
- **Prior Type**: {settings.prior_name}
- **Total Draws**: {settings.post_draws}
- **Burn-in**: {settings.burnin} ({settings.burnin / settings.post_draws:.0%})
"""
        if hasattr(settings, "n_units"):
            summary += f"- **Hierarchical Pooling Parameter**: lambda\n"

    summary += f"""
---

**Forecast & IRF Details**
- **Impulse Response Horizon**: {settings.hor}
- **Forecast Horizon**: {settings.fhor}
- **IRF Computation**: {'1 Standard Deviation' if getattr(settings, 'irf_1std', 1) == 1 else 'Unit Shock'}

---

**Model Equations**

$$
{equations}
$$
"""

    return Markdown(summary)


def _tvarsv_equations():
    return r"""
\begin{align*}
Z_t &= \Big(c_1 + \sum_{j=1}^{P} B_{1,j} Z_{t-j} + \sum_{j=0}^{J}\gamma_{1,j}\ln\lambda_{t-j} + \Omega_{1,t}^{1/2} e_t\Big)\,\tilde S_t \\
&\;+ \Big(c_2 + \sum_{j=1}^{P} B_{2,j} Z_{t-j} + \sum_{j=0}^{J}\gamma_{2,j}\ln\lambda_{t-j} + \Omega_{2,t}^{1/2} e_t\Big)(1-\tilde S_t) \\
\tilde S_t &= 1 \iff F_{t-d} \le Z^{*} \\
\Omega_{i,t} &= A_i^{-1} H_t A_i^{-1\prime}, \qquad H_t = \lambda_t \, \mathrm{diag}(s_1,\dots,s_N) \\
\ln \lambda_t &= \alpha + F \ln \lambda_{t-1} + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, Q)
\end{align*}
"""


def generate_tvarsv_summary(settings):
    """
    Generate a structured Markdown summary for the Threshold VAR-SV model,
    matching the layout of :func:`generate_summary`.
    """
    linkedin_text = "Renato Vassallo"
    linkedin_url = "https://www.linkedin.com/in/renatovassallo"

    sample_start = settings.dates[settings.lags + settings.training].date()
    sample_end = settings.dates[-1].date()
    var_params = 2 * settings.k * settings.N
    total_draws = settings.post_draws  # post_draws is the total (incl. burn-in)

    summary = f"""
**MacroPy: A Toolbox for Bayesian Macroeconometric Analysis in Python**
Developed by [{linkedin_text}]({linkedin_url}) - Institute for Economic Analysis (IAE-CSIC)
Version {__version__}

---

**Model Specifications**
- **Model Type**: Threshold VAR with Stochastic Volatility (Alessandri & Mumtaz, 2019)
- **Endogenous Variables**: {', '.join(map(str, settings.names))}
- **Threshold Variable**: {settings.threshold_name}
- **Number of Lags (P)**: {settings.lags}
- **Volatility-in-Mean Lags (J)**: {settings.vol_lags}
- **Maximum Threshold Delay**: {settings.max_delay}
- **Sample Period**: {sample_start} to {sample_end} ({settings.T} observations)
- **VAR Parameters (both regimes)**: {var_params}

---

**Bayesian Estimation Settings**
- **Posterior Simulation**: Gibbs Sampling
- **Regimes**: 2 (calm / crisis), endogenous threshold and delay
- **Stochastic Volatility**: single common factor, AR(1) in logs
- **Total Draws**: {total_draws}
- **Burn-in**: {settings.burnin} ({settings.burnin / total_draws:.0%})

---

**Impulse Response Details**
- **GIRF Horizon**: {settings.hor}
- **Identification**: Generalized IRF (Koop, Pesaran & Potter, 1996)

---

**Model Equations**

$$
{_tvarsv_equations()}
$$
"""

    return Markdown(summary)
