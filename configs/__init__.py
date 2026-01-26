# src/configs/__init__.py
from .schema import CONFIGURATION
from .paths import ANALYSIS_PATHS, FILENAMES_ANALYSIS, FILENAMES
from .configurations import CONFIG, PLOTTING_CONFIG

__all__ = [
    "CONFIGURATION",  # For type hinting
    "CONFIG",  # Configuration of the project
    "ANALYSIS_PATHS",  # Paths for the analysis
    "FILENAMES_ANALYSIS",  # Enum for the filenames for the analysis
    "FILENAMES",  # Enum for all filenames
    "PLOTTING_CONFIG",  # Configurations for plotting
]
