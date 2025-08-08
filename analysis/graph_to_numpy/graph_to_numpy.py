import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def graph_to_numpy(fig: plt.Figure | None = None,
                   remove_axis: bool = False,
                   transparent: bool = False) -> NDArray[np.uint8]:
    """
    Convert matplotlib canvas into a RGB(A) Numpy array.

    Args:
        fig (plt.Figure | None): Target figure. Defaults to current figure.
        remove_axis (bool): Remove axes and margins if True.
        transparent (bool): Output RGBA image with transparent background
                            if True.

    Returns:
        np.ndarray: (H, W, 3) RGB or (H, W, 4) RGBA image.
    """
    # Select the current figure if None
    if fig is None:
        fig = plt.gcf()

    # Remove axes
    if remove_axis:
        for ax in fig.axes:
            ax.set_axis_off()

    with io.BytesIO() as buf:
        # Set transparency
        fig.savefig(buf,
                    format='png',
                    bbox_inches='tight' if remove_axis else None,
                    pad_inches=0 if remove_axis else 0.1,
                    transparent=transparent)
        buf.seek(0)
        img = np.frombuffer(buf.getvalue(), dtype=np.uint8)

    if transparent:
        img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    return img


if __name__ == '__main__':
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    x = np.linspace(5, 10, 100)
    y = np.random.rand(100)

    ax.plot(x, y)

    img = graph_to_numpy(fig, remove_axis=True, transparent=False)

    cv2.imwrite('example.png', img)
