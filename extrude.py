import matplotlib.pyplot as plt
import numpy as np


def extrude(t_sec, x_m, y_m, h_deg, footprint_x, footprint_y):
    """
    Extrude footprint along the x y t coordinates

    t_sec : t (z) coordinates of refernce point
    x_m : x coordinates of refernce point
    y_m : y coordinates of refernce point
    heading_deg : heading angles at each time step

    footprint_x : footprint x coordinates w. r. t. (0, 0)
    footprint_y : footprint y coordinates w. r. t. (0, 0)

    Example
    =======
    >>> import numpy as np
    >>> l_m = 4.8
    >>> w_m = 1.83
    >>> l_half_m = l_m * 0.5
    >>> w_half_m = w_m * 0.5
    >>> footprint_x, footprint_y = zip((l_m, w_half_m), (-l_m, w_half_m), (-l_m, -w_half_m), (l_m, -w_half_m), (l_m, w_half_m))
    >>> t = np.arange(0, 10.01, 0.1)
    >>> x = np.arange(0, len(t)+0.1)
    >>> y = np.zeros_like(t)
    >>> heading_deg = np.linspace(0, 60, len(t))

    """

    foot_x, foot_y = np.array(footprint_x), np.array(footprint_y)
    h_rad = np.deg2rad(h_deg)
