import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def extrude(t_sec, x_m, y_m, h_deg, footprint_x, footprint_y):
    """
    Extrude footprint along the x y t coordinates

    t_sec : t (z) coordinates of refernce point
    x_m : x coordinates of refernce point
    y_m : y coordinates of refernce point

    h_deg : heading angles at each time step
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

    # Time steps
    t_vec = np.matrix([t_sec]).T

    # Reference coordinates at each time step
    rx_vec = np.matrix([x_m]).T
    ry_vec = np.matrix([y_m]).T

    # Heading angle at each time step
    heading_rad = np.deg2rad(h_deg)

    # Foot print relative coordinate
    fx_vec = np.matrix([footprint_x])
    fy_vec = np.matrix([footprint_y])

    c_vec = np.matrix(np.cos(heading_rad)).T
    s_vec = np.matrix(np.sin(heading_rad)).T
    one_vec = np.ones_like(fx_vec)

    # Foot print with rotation at each time step
    exij = c_vec * fx_vec - s_vec * fy_vec + rx_vec * one_vec
    eyij = s_vec * fx_vec + c_vec * fy_vec + ry_vec * one_vec

    # Repeat time step
    tij = t_vec * one_vec

    # plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # https://stackoverflow.com/questions/9170838/surface-plots-in-matplotlib
    ax.plot_surface(exij, eyij, tij)
    return exij, eyij, ax


def main():

    t = np.arange(0, 10.01, 0.1)

    theta_deg = 135
    theta_deg_array = np.linspace(0, theta_deg, len(t))
    theta_rad_array = np.deg2rad(theta_deg_array)
    R_m = 10

    x = R_m * np.cos(theta_rad_array)
    y = R_m * np.sin(theta_rad_array)
    heading_deg = np.linspace(0, theta_deg, len(t))

    l_m = 4.8
    w_m = 1.83
    l_half_m = l_m * 0.5
    w_half_m = w_m * 0.5
    footprint_x, footprint_y = zip((l_half_m, w_half_m), (-l_half_m, w_half_m), (-l_half_m, -w_half_m), (l_half_m, -w_half_m), (l_half_m, w_half_m))

    exij, eyij, ax = extrude(t, x, y, heading_deg, footprint_x, footprint_y)

    x_min, x_max = min(exij.flatten().tolist()[0]), max(exij.flatten().tolist()[0])
    y_min, y_max = min(eyij.flatten().tolist()[0]), max(eyij.flatten().tolist()[0])

    delta_x = x_max - x_min
    delta_y = y_max - y_min

    center_x = (x_max + x_min) * 0.5
    center_y = (y_max + y_min) * 0.5

    half_delta = max([delta_x, delta_y]) * 0.5

    plt.xlim((center_x - half_delta, center_x + half_delta))
    plt.ylim((center_y - half_delta, center_y + half_delta))

    # https://stackoverflow.com/questions/12904912/how-to-set-camera-position-for-3d-plots-using-python-matplotlib
    ax.view_init(elev=50., azim=-60)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('t')

    filename = 'surface.svg'
    print(filename)
    plt.savefig(filename)


if "__main__" == __name__:
    main()
