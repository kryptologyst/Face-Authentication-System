"""Core utilities for the face authentication system."""

import os
import random
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OmegaConf configuration object
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, output_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_path)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def hash_sensitive_data(data: str) -> str:
    """Hash sensitive data for privacy protection.
    
    Args:
        data: Sensitive string data to hash
        
    Returns:
        Hashed string
    """
    import hashlib
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def redact_pii(text: str) -> str:
    """Redact personally identifiable information from text.
    
    Args:
        text: Text that may contain PII
        
    Returns:
        Text with PII redacted
    """
    import re
    
    # Redact email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                  '[EMAIL_REDACTED]', text)
    
    # Redact IP addresses
    text = re.sub(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', 
                  '[IP_REDACTED]', text)
    
    # Redact phone numbers
    text = re.sub(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
                  '[PHONE_REDACTED]', text)
    
    return text


class ConfigManager:
    """Configuration manager for the face authentication system."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.config: Optional[DictConfig] = None
        
        if self.config_path and self.config_path.exists():
            self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file."""
        if self.config_path:
            self.config = load_config(self.config_path)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if self.config is None:
            return default
        
        try:
            return OmegaConf.select(self.config, key, default=default)
        except Exception:
            return default
    
    def update(self, key: str, value: Any) -> None:
        """Update configuration value.
        
        Args:
            key: Configuration key
            value: New value
        """
        if self.config is None:
            self.config = OmegaConf.create({})
        
        OmegaConf.set(self.config, key, value)
    
    def save(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to file.
        
        Args:
            output_path: Output file path (uses original path if None)
        """
        if self.config is None:
            raise ValueError("No configuration to save")
        
        save_path = output_path or self.config_path
        if save_path:
            save_config(self.config, save_path)


def validate_image_path(image_path: Union[str, Path]) -> bool:
    """Validate that image path exists and is a supported format.
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if valid, False otherwise
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        return False
    
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    return image_path.suffix.lower() in supported_formats


def get_model_info(model: torch.nn.Module) -> Dict[str, Any]:
    """Get information about a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
    }
