import unittest

import numpy as np
import numpy.linalg as nl

import extrude


class BaseTestExtrude(unittest.TestCase):
    epsilon = 1e-9

    def assertArrayEqual(self, x, y):
        # check norm small enough
        return self.assertLess(float(nl.norm(x - y)), BaseTestExtrude.epsilon, '\nx = %r\ny = %r' % (x, y))


class TestExtrude(BaseTestExtrude):
    def test_get_extrusion_surface(self):
        t_sec = (0, 1)
        x_m = (0, 1)
        y_m = (1, 0)
        
        h_deg = (0, 90)

        footprint_x = (0.1, 0, 0, 0.1)
        footprint_y = (0.0, 1.0, -1.0, 0.0)

        result = extrude.get_extrusion_coordinates(t_sec, x_m, y_m, h_deg, footprint_x, footprint_y)

        expected_x = np.matrix([[0.1, 0. , 0. , 0.1],
                                [1. , 0. , 2. , 1. ]])
        expected_y = np.matrix([[ 1.000000e+00,  2.000000e+00,  0.000000e+00,  1.000000e+00],
                                [ 1.000000e-01,  6.123234e-17, -6.123234e-17,  1.000000e-01]])
        expected_t = np.matrix([[0., 0., 0., 0.],
                                [1., 1., 1., 1.]])

        self.assertArrayEqual(expected_x, result[0])
        self.assertArrayEqual(expected_y, result[1])
        self.assertArrayEqual(expected_t, result[2])

    def test_get_rect_footprint(self):
        x = 2
        y = 1
        result_x, result_y = extrude.get_rect_footprint(2*x, 2*y)
        self.assertAlmostEqual(2 * x, max(result_x) - min(result_x))
        self.assertAlmostEqual(2 * y, max(result_y) - min(result_y))

        dx = result_x[2] - result_x[0]
        dy = result_y[2] - result_y[0]

        self.assertAlmostEqual(2*(x**2 + y**2)**0.5, (dx**2 + dy**2)**0.5)

    def test_get_rect_footprint_default(self):
        result_x, result_y = extrude.get_rect_footprint()

        points = np.array(tuple(zip(result_x, result_y)))
        for i in range(points.shape[0] - 1):
            vec_i = points[i + 1, :] - points[i, :]
            vec_j = points[i - 1, :] - points[i, :]
            self.assertLess(np.dot(vec_i, vec_j), BaseTestExtrude.epsilon)

    def test_constant_radius_turning(self):
        t_max = 1.0
        t_sample = 0.1
        R = 2.0
        t, x, y, heading_deg_array = extrude.constant_radius_turning(t_max, R, t_sample_sec=t_sample, initial_angle_deg=0, central_angle_deg=90,)
    
        self.assertAlmostEqual(t[-1], t_max)
        self.assertAlmostEqual(t[1] - t[0], t_sample)

        for xi, yi, theta_i_deg in zip(x, y, heading_deg_array):
            self.assertAlmostEqual(np.deg2rad(theta_i_deg - 90), np.arctan2(yi, xi))

    def test_get_extrusion_coordinates_default_footprint(self):
        t_sec = (0, 1)
        x_m = (0, 1)
        y_m = (1, 0)
        
        h_deg = (0, 90)

        result = extrude.get_extrusion_coordinates(t_sec, x_m, y_m, h_deg)

        expected_x = np.array([[ 2.4  , -2.4  , -2.4  ,  2.4  ,  2.4  ], 
                               [ 0.085,  0.085,  1.915,  1.915,  0.085]])
        expected_y = np.array([[ 1.915,  1.915,  0.085,  0.085,  1.915], 
                               [ 2.4  , -2.4  , -2.4  ,  2.4  ,  2.4  ]])
        expected_t = np.array([[0., 0., 0., 0., 0.0], 
                               [1., 1., 1., 1., 1.0]])

        self.assertArrayEqual(expected_x, result[0])
        self.assertArrayEqual(expected_y, result[1])
        self.assertArrayEqual(expected_t, result[2])


if "__main__" == __name__:
    unittest.main()
