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
    """
