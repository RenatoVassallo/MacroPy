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
from .summary import generate_summary, generate_uc_summary, generate_tvarsv_summary
from .state_space import (
    StateSpaceModel, LocalLinearTrend, ClarkModel, WatsonUC, MLEEstimator, MLEResult,
    BayesianStateSpace, BayesianResult, clark_update_step, build_clark_simple,
    build_watson_uc, watson_gibbs_step, decomposition_to_frame,
)
from .style import PALETTE, macropy_rcparams, macropy_style, set_bse_style, styled
from .plots import (
    generate_series_plot, generate_coeff_plot, generate_exog_coeff_plot,
    generate_panel_coeff_plots, generate_irf_plots, generate_fevd_plot,
    generate_hd_plot, generate_forecast_plots,
)
from .plots_kalman import (
    plot_series_with_trend, plot_decomposition, plot_with_bands, plot_kalman_step,
    plot_posterior_traces, BSE_TEAL, BSE_TEAL_SOFT, BSE_ORANGE, BSE_NAVY, BSE_GRAY,
    BSE_LIGHT,
)
from .tvarsv import ThresholdVARSV
from .plots_tvarsv import plot_regimes, plot_volatility, plot_irfs
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
    "StateSpaceModel", "LocalLinearTrend", "ClarkModel", "WatsonUC", "MLEEstimator",
    "MLEResult", "BayesianStateSpace", "BayesianResult", "clark_update_step",
    "build_clark_simple", "build_watson_uc", "watson_gibbs_step", "decomposition_to_frame",
    "PALETTE", "macropy_rcparams", "macropy_style", "set_bse_style", "styled", "use_own_style",
    "generate_series_plot", "generate_coeff_plot", "generate_exog_coeff_plot",
    "generate_panel_coeff_plots", "generate_irf_plots", "generate_fevd_plot",
    "generate_hd_plot", "generate_forecast_plots",
    "plot_series_with_trend", "plot_decomposition", "plot_with_bands", "plot_kalman_step",
    "plot_posterior_traces", "BSE_TEAL", "BSE_TEAL_SOFT", "BSE_ORANGE", "BSE_NAVY",
    "BSE_GRAY", "BSE_LIGHT",
    "ThresholdVARSV", "plot_regimes", "plot_volatility", "plot_irfs",
    "hp_filter", "UCTrendCycle", "compare_uc", "make_priors", "covid_lambda_path",
    "integrated_loglik", "kalman_loglik",
    "plot_uc_decomposition", "plot_covid_scales", "plot_gap_comparison", "plot_fan_chart",
    "x13_path", "seasonal_adjust", "SeasonalResult",
]

try:
    # Data-download helpers depend on third-party APIs that the estimators do not need.
    from .get_macrodata import get_bcrp_data, get_fred_data, get_bis_data, get_bis_data_single
except ModuleNotFoundError:
    pass
else:
    __all__ += ["get_bcrp_data", "get_fred_data", "get_bis_data", "get_bis_data_single"]
