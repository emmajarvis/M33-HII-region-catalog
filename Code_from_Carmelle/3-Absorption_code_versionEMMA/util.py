import numpy as np
import numpy.typing as npt
import orb
from matplotlib.ticker import AutoMinorLocator


def filter1_to_filter2_map(
    map: npt.NDArray[np.float64], transform: dict
) -> npt.NDArray[np.float64]:
    """
    TODO: description
    
    Args:
        map (numpy.ndarray): Map to align with another filter.
        transform (dict): Translation and rotation parameters.

    Returns:
        map_aligned (numpy.ndarray):
    """
    map_aligned = orb.utils.image.transform_frame(
        map,
        0,
        map.shape[0],
        0,
        map.shape[1],
        [-transform["dy"], -transform["dx"], transform["dtheta"], 0, 0],
        [0, 0],
        1,
        0,
    )

    return map_aligned


def set_tick_params(axes: list, label_outer: bool = True) -> None:
    """
    Sets ticks' parameters (inside ticks' direction, shows minor ticks, etc.).

    params:
        axes (list): List of axes objects.
        label_outer (bool): Only show "outer" labels and tick labels.
    """
    for ax in axes:
        # Ticks parameter
        ax.tick_params(
            "both", which="both", direction="in", length=10, top=True, right=True
        )  # width=2
        ax.tick_params(
            "both", which="both", direction="in", length=10, top=True, right=True
        )  # width=2
        ax.tick_params("both", which="minor", length=4)
        ax.tick_params("both", which="minor", length=4)

        # Shows minor thicks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        # Hide x labels and tick labels for all but bottom plot.
        if label_outer:
            ax.label_outer()


def transform_position(
    x: int | npt.NDArray[np.int32],
    y: int | npt.NDArray[np.int32],
    transform: dict,
) -> tuple:
    """
    Transforms pixels' position from SN3 filter to their respective position in SN1 or SN2 filters.

    params:
        x (float or ArrayLike): Pixels' position in x-axis in SN3 filter.
        y (float or ArrayLike): Pixels' position in y-axis in SN3 filter.
        transrot (dict): Translation and rotation parameters.

    retrun: Pixels's position in SN1 or SN2 filters.
    """
    r = np.array([x, y])  # Position vector
    u = np.array([transform["dx"], transform["dy"]])  # Translation vector
    theta = -np.radians(transform["dtheta"])  # Convert rotation angle to radians

    c, s = np.cos(theta), np.sin(theta)  # cos(theta) and sin(theta)
    R = np.array(((c, -s), (s, c)))  # Rotation matrix

    x_prime, y_prime = (np.dot(R, r).T - u).T.astype(int)  # New coordinates

    return x_prime, y_prime


def intersect2D(a: list | npt.NDArray, b: list | npt.NDArray) -> npt.NDArray:
    """
    Find row intersection between 2D numpy arrays, a and b.
    Returns another numpy array with shared rows.

    Source: https://gist.github.com/Robaina/b742f44f489a07cd26b49222f6063ef7
    """
    return np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])


if __name__ == "__main__":
    pass
