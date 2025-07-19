from igakit.nurbs import NURBS
import numpy as np


def circle(radius):
    """
    Create circular NURBS surface with given radius.

    Parameters
    ----------
    radius: float
        Radius of the circle.

    Returns
    -------
    circle: igakit.nurbs.NURBS
        NURBS surface.
    """
    # Control points and weights
    ctrlpts = radius * np.array(
        [
            [-1 / np.sqrt(2), -1 / np.sqrt(2), 0],
            [0, -np.sqrt(2), 0],
            [1 / np.sqrt(2), -1 / np.sqrt(2), 0],
            [-np.sqrt(2), 0, 0],
            [0, 0, 0],
            [np.sqrt(2), 0, 0],
            [-1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            [0, np.sqrt(2), 0],
            [1 / np.sqrt(2), 1 / np.sqrt(2), 0],
        ]
    ).reshape((3, 3, 3))
    weights = np.array(
        [
            [1, 1 / np.sqrt(2), 1],
            [1 / np.sqrt(2), 1, 1 / np.sqrt(2)],
            [1, 1 / np.sqrt(2), 1],
        ]
    )

    return NURBS(
        knots=[[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]], control=ctrlpts, weights=weights
    )
