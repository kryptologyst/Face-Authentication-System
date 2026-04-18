"""Main package."""

from .models import FaceAuthenticator, AuthenticationResult
from .data import FaceDataset, FaceDataProcessor
from .eval import BiometricEvaluator
from .utils import setup_logging, get_device, ConfigManager

__all__ = [
    "FaceAuthenticator",
    "AuthenticationResult",
    "FaceDataset", 
    "FaceDataProcessor",
    "BiometricEvaluator",
    "setup_logging",
    "get_device",
    "ConfigManager"
]
