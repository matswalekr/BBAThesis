from pathlib import Path
from .schema import PATH_CONFIG, PROJECT_ROOT, PATH_ANALYSIS
from enum import StrEnum

# Paths that are hidden from the PATHCONFIG
DATA_DIR: Path = PROJECT_ROOT / "data"
SQL_DIR: Path = PROJECT_ROOT / "sql"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# Paths for the analysis
ANALYSIS_PATHS = PATH_ANALYSIS(
    PORTFOLIO_DATA_DIR=DATA_DIR / "portfolios",
    # Result directories in the RESULTS_DIR
    RESULT_DATA_DIR=RESULTS_DIR / "data",
    RESULT_IMAGES_DIR=RESULTS_DIR / "images",
)

# Paths for the main
PATHCONFIG = PATH_CONFIG(
    # SQL directory
    SQL_DIR=PROJECT_ROOT / "sql",
    # Data directories in the DATA_DIR
    RAW_DATA_DIR=DATA_DIR / "raw",
    PROCESSED_DATA_DIR=DATA_DIR / "processed",
    PORTFOLIO_DATA_DIR=DATA_DIR / "portfolios",
    # Result directories in the RESULTS_DIR
    RESULT_DATA_DIR=RESULTS_DIR / "data",
    RESULT_IMAGES_DIR=RESULTS_DIR / "images",
)


# Enum for the analysis pathnames
class FILENAMES_ANALYSIS(StrEnum):

    # Portfolio Info
    Portfolio_returns: str = "portfolio_returns_monthly"
    Portfolio_construction_details: str = "portfolio_construction_details"

    # Model outcomes
    Comp_pred_actual_portfolio: str = "factor_model_predictions_monthly_obs_universe"
    Factor_loadings_monthly: str = "factor_loadings_monthly_portfolios"
    Factor_loadings_differentperiods: str = "factor_loadings_overtime_portfolios"


# Enum for all pathnames
class FILENAMES(StrEnum):
    # Fama French Files
    FF5_factors_monthly: str = "fama_french_monthly_factors"
    FF5_factors_yearly: str = "fama_french_yearly_factors"

    # Stock info files
    Stock_prices: str = "stock_prices_wrds_monthly_obs_universe"

    # Firm info
    Firm_info: str = "firm_info_wrds"

    # Sic Description
    Sic_description: str = "sic_description"

    # Portfolio Info
    Portfolio_returns: str = "portfolio_returns_monthly"
    Portfolio_construction_details: str = "portfolio_construction_details"

    # Model outcomes
    Comp_pred_actual_portfolio: str = "factor_model_predictions_monthly_obs_universe"
    Factor_loadings_monthly: str = "factor_loadings_monthly_portfolios"
    Factor_loadings_differentperiods: str = "factor_loadings_overtime_portfolios"
