from Pixel import Coordinate
from scipy.optimize import leastsq


def nonlinear_triangulation(camera_pose_1, camera_pose_2, triangulated_points, associated_pixels):
    
    return


def error(u, v, x_homogeneous, projection_matrix_1, projection_matrix_2):
    sum = 0
    for Pj in [projection_matrix_1, projection_matrix_2]:
        P1 = Pj[0, 0:3].reshape((1,3))  # We reshape these matrices otherwise the result
        P2 = Pj[1, 0:3].reshape((1,3))  # is shape (3,) which is not the same as (1,3)
        P3 = Pj[2, 0:3].reshape((1,3))
        result = 0
        continue
    return 0
