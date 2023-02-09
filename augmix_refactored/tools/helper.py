
from typing import Any
import numpy as np

__all__ = [
    "int_parameter",
    "float_parameter",
    "sample_level"
]

def sample_level(n: float) -> np.ndarray:
    """Sampling out of a uniform distribution
    in between [0.1, n)

    Parameters
    ----------
    n : float
        High for the uniform distribution.

    Returns
    -------
    np.ndarray
        The drawn distribution.
    """
    return np.random.uniform(low=0.1, high=n)

def int_parameter(level: Any, maxval: float):
    """Helper function to scale `val` between 0 and maxval .

    Args:
        level: Level of the operation that will be between [0, `PARAMETER_MAX`].
        maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.

    Returns:
        An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)

def float_parameter(level: Any, maxval: float):
    """Helper function to scale `val` between 0 and maxval.

    Args:
        level: Level of the operation that will be between [0, `PARAMETER_MAX`].
        maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.

    Returns:
        A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.