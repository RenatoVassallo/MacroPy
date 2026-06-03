from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("MacroPy")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from .bvar import *
from .bpvar import *
from .cvar import *
from .lp import *
from .data_handling import *
from .plots import *
from .priors import *
from .summary import *
from .state_space import *
from .plots_kalman import *

try:
    from .get_macrodata import *
except ModuleNotFoundError:
    # Optional data-download helpers depend on third-party APIs that are not
    # required for the core VAR estimators.
    pass
