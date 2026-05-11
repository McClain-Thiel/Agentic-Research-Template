"""[[ package_name ]] — Modern ML research project template.

This package provides a structured foundation for machine learning research
projects, including configuration management, experiment tracking, data loading,
model training, and result analysis.
"""

__version__ = "0.1.0"

from [[ package_name ]].config.experiment import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)
from [[ package_name ]].config.settings import Settings, settings

__all__ = [
    "__version__",
    "settings",
    "Settings",
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "EvalConfig",
    "ExperimentConfig",
]
