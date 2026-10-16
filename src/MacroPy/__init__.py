"""MacroPy: Bayesian macroeconometric analysis in Python."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("MacroPy")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

#: Set to False so that MacroPy figures keep the caller's matplotlib style.
use_own_style = True

from .bvar import BayesianVAR
from .bpvar import BayesianPanelVAR
from .cvar import ClassicVAR
from .lp import (
    SmoothLocalProjections, PanelLocalProjections, smooth_lp_results,
    split_shock_signs, identify_boom_periods, cumulative_irf_ratio,
)
from .data_handling import (
    prepare_data, estimate_ols, prepare_panel_unit_data, prepare_panel_data,
    coefficients_by_lag,
)
from .priors import (
    MinnesotaPrior, NormalWishartPrior, NormalDiffusePrior, HierarchicalPanelPrior,
    DiffusePanelExogenousPrior, normalize_prior_params, ar1_residual_std,
    nw_conjugate_moments, nw_log_marginal_likelihood, PRIOR_DEFAULTS, PRIOR_KEYS,
)
from .summary import (
    generate_summary, generate_uc_summary, generate_tvarsv_summary, generate_tvar_summary,
    generate_tvpvar_summary,
)
from .state_space import (
    StateSpaceModel, LocalLinearTrend, ClarkModel, WatsonUC, MLEEstimator, MLEResult,
    BayesianStateSpace, BayesianResult, clark_update_step, build_clark_simple,
    build_watson_uc, watson_gibbs_step, decomposition_to_frame,
)
from .style import (
    PALETTE, CONTEXTS, set_style, reset_style, style_context, styled, macropy_rcparams,
    macropy_style, color_palette, colormap, set_bse_style,
)
from .plots import (
    generate_series_plot, generate_coeff_plot, generate_exog_coeff_plot,
    generate_panel_coeff_plots, generate_irf_plots, generate_fevd_plot,
    generate_hd_plot, generate_forecast_plots,
)
from .plots_kalman import (
    plot_series_with_trend, plot_decomposition, plot_with_bands, plot_kalman_step,
    plot_posterior_traces,
)
from .tvarsv import ThresholdVARSV
from .tvar import (
    ThresholdVAR, compare_threshold_models, recursive_log_scores, compare_log_scores,
    TVAR_PRIOR_DEFAULTS,
)
from .plots_tvarsv import (
    plot_regimes, plot_volatility, plot_irfs, plot_regime_comparison, plot_irf_comparison,
    plot_volatility_comparison, plot_log_scores,
)
from .tvpvar import TimeVaryingVAR, compare_tvpvar, TVP_PRIOR_DEFAULTS
from .plots_tvpvar import (
    plot_tvp_volatility, plot_tvp_coefficients, plot_tvp_irfs, plot_irf_surface, plot_irf_over_time,
    plot_fevd_over_time, plot_tvp_fan_chart,
)
from .uc import (
    hp_filter, UCTrendCycle, compare_uc, make_priors, covid_lambda_path,
    integrated_loglik, kalman_loglik,
)
from .plots_uc import plot_uc_decomposition, plot_covid_scales, plot_gap_comparison, plot_fan_chart
from .x13 import x13_path
from .seasonal import seasonal_adjust, SeasonalResult

__all__ = [
    "BayesianVAR", "BayesianPanelVAR", "ClassicVAR",
    "SmoothLocalProjections", "PanelLocalProjections", "smooth_lp_results",
    "split_shock_signs", "identify_boom_periods", "cumulative_irf_ratio",
    "prepare_data", "estimate_ols", "prepare_panel_unit_data", "prepare_panel_data",
    "coefficients_by_lag",
    "MinnesotaPrior", "NormalWishartPrior", "NormalDiffusePrior", "HierarchicalPanelPrior",
    "DiffusePanelExogenousPrior", "normalize_prior_params", "ar1_residual_std",
    "nw_conjugate_moments", "nw_log_marginal_likelihood", "PRIOR_DEFAULTS", "PRIOR_KEYS",
    "generate_summary", "generate_uc_summary", "generate_tvarsv_summary",
    "generate_tvar_summary", "generate_tvpvar_summary",
    "StateSpaceModel", "LocalLinearTrend", "ClarkModel", "WatsonUC", "MLEEstimator",
    "MLEResult", "BayesianStateSpace", "BayesianResult", "clark_update_step",
    "build_clark_simple", "build_watson_uc", "watson_gibbs_step", "decomposition_to_frame",
    "PALETTE", "CONTEXTS", "set_style", "reset_style", "style_context", "styled",
    "macropy_rcparams", "macropy_style", "color_palette", "colormap", "use_own_style",
    "generate_series_plot", "generate_coeff_plot", "generate_exog_coeff_plot",
    "generate_panel_coeff_plots", "generate_irf_plots", "generate_fevd_plot",
    "generate_hd_plot", "generate_forecast_plots",
    "plot_series_with_trend", "plot_decomposition", "plot_with_bands", "plot_kalman_step",
    "plot_posterior_traces",
    "ThresholdVARSV", "plot_regimes", "plot_volatility", "plot_irfs",
    "ThresholdVAR", "compare_threshold_models", "recursive_log_scores", "compare_log_scores",
    "TVAR_PRIOR_DEFAULTS", "plot_regime_comparison", "plot_irf_comparison",
    "plot_volatility_comparison", "plot_log_scores",
    "TimeVaryingVAR", "compare_tvpvar", "TVP_PRIOR_DEFAULTS", "plot_tvp_volatility",
    "plot_tvp_coefficients",
    "plot_tvp_irfs", "plot_irf_surface", "plot_irf_over_time", "plot_fevd_over_time",
    "plot_tvp_fan_chart",
    "hp_filter", "UCTrendCycle", "compare_uc", "make_priors", "covid_lambda_path",
    "integrated_loglik", "kalman_loglik",
    "plot_uc_decomposition", "plot_covid_scales", "plot_gap_comparison", "plot_fan_chart",
    "x13_path", "seasonal_adjust", "SeasonalResult",
]


def __getattr__(name):
    """Deprecated names (PEP 562): the former BSE_* colour constants."""
    from .plots_kalman import _DEPRECATED_COLORS
    if name in _DEPRECATED_COLORS:
        import warnings
        key = _DEPRECATED_COLORS[name]
        warnings.warn(f"{name} is deprecated; use MacroPy.PALETTE[{key!r}].",
                      DeprecationWarning, stacklevel=2)
        return PALETTE[key]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


try:
    # Data-download helpers depend on third-party APIs that the estimators do not need.
    from .get_macrodata import get_bcrp_data, get_fred_data, get_bis_data, get_bis_data_single
except ModuleNotFoundError:
    pass
else:
    __all__ += ["get_bcrp_data", "get_fred_data", "get_bis_data", "get_bis_data_single"]
