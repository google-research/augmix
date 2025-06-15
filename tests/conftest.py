"""Shared pytest fixtures and configuration."""
import os
import sys
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing.
    
    Yields:
        Path: Path to the temporary directory
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_config() -> dict:
    """Provide a mock configuration dictionary for testing.
    
    Returns:
        dict: Mock configuration with common settings
    """
    return {
        "batch_size": 32,
        "num_workers": 2,
        "learning_rate": 0.001,
        "epochs": 10,
        "augmentation": {
            "severity": 3,
            "width": 3,
            "depth": -1,
            "alpha": 1.0
        }
    }


@pytest.fixture
def sample_image_path(temp_dir: Path) -> Path:
    """Create a dummy image file for testing.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Path: Path to the created dummy image file
    """
    image_path = temp_dir / "test_image.jpg"
    # Create a dummy file (actual image content not needed for most tests)
    image_path.write_bytes(b"dummy image content")
    return image_path


@pytest.fixture
def mock_dataset_config() -> dict:
    """Provide mock dataset configuration.
    
    Returns:
        dict: Mock dataset configuration
    """
    return {
        "name": "cifar10",
        "data_dir": "./data",
        "num_classes": 10,
        "image_size": 32,
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2023, 0.1994, 0.2010]
    }


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    import random
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    
    # Only set torch seed if torch is available
    try:
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    except ImportError:
        pass


@pytest.fixture
def capture_output():
    """Capture stdout and stderr for testing print statements.
    
    Yields:
        tuple: (stdout, stderr) StringIO objects
    """
    from io import StringIO
    import sys
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    yield sys.stdout, sys.stderr
    
    sys.stdout = old_stdout
    sys.stderr = old_stderr


# Markers for different test types
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that test individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that test component interactions"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run"
    )