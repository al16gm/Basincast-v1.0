"""BasinCast Engine 1.0: generic, endogeneous, monthly forecast core."""
from .schema import SeriesData, load_series
from .engine import EngineConfig, adaptive_config, run_series, run_short_series
from .input import normalize_user_file

__version__ = "1.0.0"
__all__ = ["SeriesData", "load_series", "EngineConfig", "adaptive_config", "run_series", "run_short_series", "normalize_user_file"]
