import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os


def extrude(t_sec, x_m, y_m, h_deg, footprint_x, footprint_y, alpha=0.7, ax=None, color='orange'):
    """
    Extrude footprint along the x y t coordinates

    t_sec : t (z) coordinates of refernce point
    x_m : x coordinates of refernce point
    y_m : y coordinates of refernce point

    h_deg : Heading angles at each time step
    footprint_x : Footprint x coordinates w. r. t. (0, 0)
    footprint_y : Footprint y coordinates w. r. t. (0, 0)

    alpha : Transparency level (default 0.7)
    ax : Axis capable of 3D plotting. Would make one of None. (default None)
    color : Surface fill color (default 'orange')

    Return Values
    =============
    exij : Surface x coordinates
    eyij : Surface y coordinates
    tij : Surface z coordinates
    ax : matplotlib 3D axis

    Example
    =======
    >>> import numpy as np
    >>> l_m = 4.8
    >>> w_m = 1.83
    >>> l_half_m = l_m * 0.5
    >>> w_half_m = w_m * 0.5
    >>> footprint_x, footprint_y = zip((l_half_m, w_half_m), 
                                       (-l_half_m, w_half_m), 
                                       (-l_half_m, -w_half_m), 
                                       (l_half_m, -w_half_m), 
                                       (l_half_m, w_half_m))
    >>> t = np.arange(0, 10.01, 0.1)
    >>> x = np.arange(0, len(t)+0.1)
    >>> y = np.zeros_like(t)
    >>> heading_deg = np.linspace(0, 60, len(t))
    >>> exij, eyij, ax = extrude(t, x, y, heading_deg, footprint_x, footprint_y)

    """

    # Time steps in column vector
    t_vec = np.matrix([t_sec]).T

    # Reference coordinates at each time step
    # in column vector
    rx_vec = np.matrix([x_m]).T
    ry_vec = np.matrix([y_m]).T

    # Heading angle at each time step
    heading_rad = np.deg2rad(h_deg)

    # Footprint relative coordinates
    if isinstance(footprint_x, (int, float)) and isinstance(footprint_y, (int, float)):
        footprint_x, footprint_y = get_rect_footprint(footprint_x, footprint_y)

    # In row vectors
    fx_vec = np.matrix([footprint_x])
    fy_vec = np.matrix([footprint_y])

    # Direction cosines to rotate the footprint at each time step
    # In column vectors
    c_vec = np.matrix(np.cos(heading_rad)).T
    s_vec = np.matrix(np.sin(heading_rad)).T

    # To repeat in column direction
    # May consider numpy.tile() as an alternative
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html#numpy.tile
    one_vec = np.ones_like(fx_vec)

    # Footprint with rotation at each time step
    # [xs] = [cos, -sin][fx] + [x]
    # [ys]   [sin,  cos][fy]   [y]
    exij = c_vec * fx_vec - s_vec * fy_vec + rx_vec * one_vec
    eyij = s_vec * fx_vec + c_vec * fy_vec + ry_vec * one_vec

    # Repeat time step
    tij = t_vec * one_vec

    # Prepare 3D axis if necessary
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    # https://stackoverflow.com/questions/9170838/surface-plots-in-matplotlib
    # To make it easy to calculate which surface is closer to the point of view,
    # slice the duct at each time step and plot as a separate surface.
    # Would be possible to parallelize.
    for step in range(exij.shape[0] - 1):
        ax.plot_surface(exij[step:(step+2), :], 
                        eyij[step:(step+2), :], 
                        tij[step:(step+2), :], 
                        alpha=alpha, color=color)

    return exij, eyij, ax


def get_rect_footprint(l_m, w_m):
    """
    Get rectangular footprint of length (x, longitudinal direction) and width (y, lateral direction)

    l_m : length
    w_m : width

    Return Values
    =============
    footprint_x : tuple of x coordinates of the footprint
    footprint_y : tuple of y coordinates of the footprint

    """
    l_half_m = l_m * 0.5
    w_half_m = w_m * 0.5
    footprint_x, footprint_y = zip((l_half_m, w_half_m), (-l_half_m, w_half_m), (-l_half_m, -w_half_m), (l_half_m, -w_half_m), (l_half_m, w_half_m))
    return footprint_x, footprint_y


def constant_radius_turning(t_max_sec, R_m, t_sample_sec=0.1, initial_angle_deg=0, final_angle_deg=90,):
    """
    Generate reference trajectory of constant radius turning centered at (0, 0), starting at (R, 0) if initial angle is 0 degree.

    Arguments
    =========
    t_max_sec : Max time
    R_m : Turning radius
    t_sample_sec : Sampling time
    initial_angle_deg : Start point angle
    final_angle_deg : Angular length

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
    """
    axis('equal') for a 3D axis

    For matplotlib 3D axis, axis('equal') seems working differently then expected
    Hence created one similar to it.

    If plot contains multiple plots, please use numpy.hstack() (or numpy.vstack()) 
    to collect all x y coordinates of the surfaces

    Example: multiple surface case
    ==============================
    >>> import numpy.random as nr
    >>> exij0, exij2, exij3, exij4 = nr.random((10, 10)), nr.random((10, 10)), nr.random((10, 10)), nr.random((10, 10))
    >>> eyij0, eyij2, eyij3, eyij4 = nr.random((10, 10)), nr.random((10, 10)), nr.random((10, 10)), nr.random((10, 10))
    >>> exij = np.hstack((exij0, exij2, exij3, exij4,)) # assume these are from multiple surfaces
    >>> eyij = np.hstack((eyij0, eyij2, eyij3, eyij4,))
    >>> axis_equal_xy(exij, eyij)
    """

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

    # Size of a possible vehicle
    l_m = 4.8
    w_m = 1.83

    # Max simulation time
    t_max = 10

    # Trajectory radius
    R_m = 20

    # First duct
    exij0, eyij0, ax = helix(l_m, w_m, t_max, R_m, start_deg=0, end_deg=90,)

    # Second duct
    t2, x2, y2, heading_deg2 = constant_radius_turning(t_max, R_m, initial_angle_deg=90, final_angle_deg=90)
    x2 += R_m
    exij2, eyij2, _ = extrude(t2, x2, y2, heading_deg2, l_m, w_m, ax=ax, color='red')

    # Third duct
    t3, x3, y3, heading_deg3 = constant_radius_turning(t_max, R_m, initial_angle_deg=180, final_angle_deg=90)
    x3 += R_m
    y3 += R_m
    exij3, eyij3, _ = extrude(t3, x3, y3, heading_deg3, l_m, w_m, ax=ax, color='green')

    # Fourth duct
    t4, x4, y4, heading_deg4 = constant_radius_turning(t_max, R_m, initial_angle_deg=270, final_angle_deg=90)
    y4 += R_m
    exij4, eyij4, _ = extrude(t4, x4, y4, heading_deg4, l_m, w_m, ax=ax, color='blue')

    # Adjust x y limits to include all surfaces
    exij = np.hstack((exij0, exij2, exij3, exij4,))
    eyij = np.hstack((eyij0, eyij2, eyij3, eyij4,))

    axis_equal_xy(exij, eyij)

    # Adjust view points
    azim = (0.46875+15.234375)*0.5
    print('azim =', azim)

    # https://stackoverflow.com/questions/12904912/how-to-set-camera-position-for-3d-plots-using-python-matplotlib
    ax.view_init(elev=30., azim=azim)

    # Mark axes labels
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_zlabel('t(sec)')

    filename = 'surface.svg'
    if os.path.exists(filename):
        os.remove(filename)
    plt.savefig(filename)
    if os.path.exists(filename):
        print('saved to %s' % filename)
    else:
        raise Warning('Unable to save to %s' % filename)


def helix(l_m, w_m, t_max, R_m, start_deg, end_deg, ax=None):
    """
    Plots a helix centered at (0, 0)
    """
    t, x, y, heading_deg = constant_radius_turning(t_max, R_m, initial_angle_deg=start_deg, final_angle_deg=end_deg)

    exij, eyij, ax = extrude(t, x, y, heading_deg, l_m, w_m, ax=ax)
    return exij, eyij, ax


if "__main__" == __name__:
    main()
