import unittest

import numpy as np
import numpy.linalg as nl

import extrude


class BaseTestExtrude(unittest.TestCase):
    epsilon = 1e-9

    def assertArrayEqual(self, x, y):
        # check norm small enough
        return self.assertLess(float(nl.norm(x - y)), BaseTestExtrude.epsilon)


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
        

if "__main__" == __name__:
    unittest.main()
