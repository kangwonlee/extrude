import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def extrude(t_sec, x_m, y_m, h_deg, footprint_x, footprint_y, alpha=0.7, ax=None):
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

    if isinstance(footprint_x, (int, float)) and isinstance(footprint_y, (int, float)):
        footprint_x, footprint_y = get_rect_footprint(footprint_x, footprint_y)

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
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    # https://stackoverflow.com/questions/9170838/surface-plots-in-matplotlib
    ax.plot_surface(exij, eyij, tij, alpha=alpha)
    return exij, eyij, ax


def get_rect_footprint(l_m, w_m):
    l_half_m = l_m * 0.5
    w_half_m = w_m * 0.5
    footprint_x, footprint_y = zip((l_half_m, w_half_m), (-l_half_m, w_half_m), (-l_half_m, -w_half_m), (l_half_m, -w_half_m), (l_half_m, w_half_m))
    return footprint_x, footprint_y


def constant_radius_turning(t_max_sec, R_m, t_sample_sec=0.1, initial_angle_deg=0, final_angle_deg=90,):
    """
    Generate reference trajectory of constant radius turning centered at (0, 0), starting at (R, 0)

    Arguments
    =========
    t_max_sec : Max time
    R_m : Turning radius
    t_sample_sec : Sampling time
    initial_angle_deg : Start point angle
    final_angle_deg : End point angle

    Return Values
    =============
    t_sec_array : Time steps
    x_m_array : X coordinates
    y_m_array : Y coordinates
    Heading_deg_array : Heading angle
    """

    t = np.arange(0, t_max_sec + t_sample_sec * 0.5, t_sample_sec)

    # Angle arrays
    theta_deg_array = np.linspace(initial_angle_deg, initial_angle_deg+final_angle_deg, len(t))
    theta_rad_array = np.deg2rad(theta_deg_array)

    # Heading angle in degree
    heading_deg_array = 90 + theta_deg_array

    # Reference coordinates
    x = R_m * np.cos(theta_rad_array)
    y = R_m * np.sin(theta_rad_array)

    return t, x, y, heading_deg_array


def axis_equal_xy(exij, eyij):
    # Extremums in x, y coordinates
    x_min, x_max = min(exij.flatten().tolist()[0]), max(exij.flatten().tolist()[0])
    y_min, y_max = min(eyij.flatten().tolist()[0]), max(eyij.flatten().tolist()[0])

    # x, y span
    delta_x = x_max - x_min
    delta_y = y_max - y_min

    # center of x, y coordinate
    center_x = (x_max + x_min) * 0.5
    center_y = (y_max + y_min) * 0.5

    # choose the larger of the half spans
    half_delta = max([delta_x, delta_y]) * 0.5

    # set axis ranges
    plt.xlim((center_x - half_delta, center_x + half_delta))
    plt.ylim((center_y - half_delta, center_y + half_delta))


def main():

    l_m = 4.8
    w_m = 1.83

    t_max = 10
    R_m = 20

    exij0, eyij0, ax = helix(l_m, w_m, t_max, R_m, start_deg=0, end_deg=90,)
    exij2, eyij2, _ = helix(l_m, w_m, t_max, R_m, start_deg=90, end_deg=90, ax=ax)
    exij3, eyij3, _ = helix(l_m, w_m, t_max, R_m, start_deg=180, end_deg=90, ax=ax)
    exij4, eyij4, _ = helix(l_m, w_m, t_max, R_m, start_deg=270, end_deg=90, ax=ax)

    exij = np.hstack((exij0, exij2, exij3, exij4,))
    eyij = np.hstack((eyij0, eyij2, eyij3, eyij4,))

    axis_equal_xy(exij, eyij)

    # https://stackoverflow.com/questions/12904912/how-to-set-camera-position-for-3d-plots-using-python-matplotlib
    ax.view_init(elev=30., azim=-60)

    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_zlabel('t(sec)')

    filename = 'surface.svg'
    print(filename)
    plt.savefig(filename)


def helix(l_m, w_m, t_max, R_m, start_deg, end_deg, ax=None):
    t, x, y, heading_deg = constant_radius_turning(t_max, R_m, initial_angle_deg=start_deg, final_angle_deg=end_deg)

    exij, eyij, ax = extrude(t, x, y, heading_deg, l_m, w_m, ax=ax)
    return exij, eyij, ax


if "__main__" == __name__:
    main()
