from enum import Enum

import numpy as np

from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


class SmoothingMethod(str, Enum):
    MOVING_AVERAGE = 'moving_average'
    GAUSSIAN = 'gaussian'
    SAVITZKY_GOLAY = 'savitzky_golay'
    EXPONENTIAL = 'exponential'


def smooth(
        data: NDArray[np.float64],
        method: str | SmoothingMethod,
        **kwargs
        ) -> NDArray[np.float64]:
    if isinstance(method, str):
        method = SmoothingMethod(method.lower())

    if method == SmoothingMethod.MOVING_AVERAGE:
        kernel_size = kwargs.get('kernel_size', None)
        weights = kwargs.get('weights', None)
        if kernel_size is None:
            raise ValueError('"moving_average()" requires a "window_size".')
        if weights is not None:
            if isinstance(weights, list):
                weights = np.array(weights, dtype=np.float64)
            if weights.ndim != 1 and weights.shape[0] != kernel_size:
                raise ValueError('"moving_average()" requires "weights" '
                                 'to have the size as "window_size".')
        smoothed_data = moving_average(data, kernel_size, weights)
    elif method == SmoothingMethod.GAUSSIAN:
        sigma = kwargs.get('sigma', None)
        if sigma is None:
            raise ValueError('"gaussian()" requires "sigma".')
        kernel_size = kwargs.get('kernel_size', None)
        smoothed_data = gaussian(data, sigma, kernel_size)
    elif method == SmoothingMethod.SAVITZKY_GOLAY:
        kernel_size = kwargs.get('kernel_size', None)
        polyorder = kwargs.get('polyorder', None)
        smoothed_data = savitzky_golay(data, kernel_size, polyorder)
    elif method == SmoothingMethod.EXPONENTIAL:
        alpha = kwargs.get('alpha', None)
        if alpha is None:
            raise ValueError('"exponential()" requires a "alpha".')
        if not (0 < alpha <= 1):
            raise ValueError('"alpha" must be between 0 and 1.')
        smoothed_data = exponential(data, alpha)
    else:
        raise ValueError('Unknown smoothing method: {}'.format(method))

    return smoothed_data


def moving_average(
        data: NDArray[np.float64],
        window_size: int,
        weights: NDArray[np.float64] | None = None
        ) -> NDArray[np.float64]:
    """
    Moving average smoothing.

    Args:
        data (NDArray[np.float64]): The input data
        window_size (int): window size
        weights (NDArray[np.float64] | None, optional): weights filter
                                                        (Default: None)

    Returns:
        NDArray[np.float64]: Data smoothed with moving average
    """
    if weights is None:
        weights = np.ones(window_size) / window_size
    else:
        weights = weights / np.sum(weights)
    smoothed_data = np.convolve(data, weights, 'valid')
    return smoothed_data


def gaussian(
        data: NDArray[np.float64],
        sigma: float | None = None,
        kernel_size: int | None = None
        ) -> NDArray[np.float64]:
    """
    Gaussian kernel smoothing.

    Args:
        data (NDArray[np.float64]): The input data
        sigma (float): The standard deviation of the Gaussian distribution.
        kernel_size (int, optional): The desired size of the kernel. The size
                                     should be odd number. If None, "truncate"
                                     parameter in Scipy will be recalculate.
                                     (Default: None)

    Returns:
        NDArray[np.float64]: Data smoothed with a Gaussian kernel
    """
    if kernel_size is None:
        smoothed_data = gaussian_filter1d(data, sigma, axis=0)
    else:
        if kernel_size % 2 == 0:
            raise ValueError('"kernel_size" should be an odd number.')
        radius = (kernel_size - 1) // 2
        smoothed_data = gaussian_filter1d(data, sigma, axis=0, radius=radius)

    return smoothed_data


def savitzky_golay(
            data: NDArray[np.float64],
            kernel_size: int,
            polyorder: int
        ) -> NDArray[np.float64]:
    """
    Savitzky-Golay smoothing.

    Args:
        data (NDArray[np.float64]): The input data
        kernel_size (int): Filter window size
        polyorder (int): Polynominal order

    Returns:
        NDArray[np.float64]: Data smoothed with Savztky-Golay filter
    """
    smoothed_data = savgol_filter(data, kernel_size, polyorder=polyorder)
    return smoothed_data

def exponential(
        data: NDArray[np.float64],
        alpha: float
        ) -> NDArray[np.float64]:
    """
    Simple exponential smoothing.

    Args:
        data (NDArray[np.float64]): The input data
        alpha (float): The smoothing parameter (0 < alpha <= 1)

    Returns:
        NDArray[np.float64]: Data smoothed with exponential smoothing
    """
    smoothed_data = np.zeros(data.shape[0], dtype=np.float64)
    
    # Initialize the first smoothed value with the first data point
    smoothed_data[0] = data[0]

    for i in range(1, data.shape[0]):
        smoothed_data[i] = alpha * data[i] + (1 - alpha) * smoothed_data[i - 1]
        
    return smoothed_data


if __name__ == '__main__':
    # Example usage:
    import matplotlib.pyplot as plt

    x = np.linspace(0, 10, 200)
    y = np.sin(x) + 0.2 * np.random.randn(200)

    methods = [
        (SmoothingMethod.MOVING_AVERAGE, dict(kernel_size=5)),
        (SmoothingMethod.GAUSSIAN, dict(sigma=2)),
        (SmoothingMethod.SAVITZKY_GOLAY, dict(kernel_size=7, polyorder=3)),
        (SmoothingMethod.EXPONENTIAL, dict(alpha=0.3)),
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Original", color="gray", alpha=0.6)

    for method, params in methods:
        y_smooth = smooth(y, method, **params)
        # Moving Average, Savitzky-Golay might have different lengths
        x_adj = x
        if len(y_smooth) != len(x):
            offset = (len(x) - len(y_smooth)) // 2
            x_adj = x[offset:offset + len(y_smooth)]
        plt.plot(x_adj, y_smooth, label=method.value)

    plt.title("Smoothing Methods Demo")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()