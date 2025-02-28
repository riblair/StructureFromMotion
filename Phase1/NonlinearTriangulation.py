from Pixel import Coordinate
from scipy.optimize import leastsq
import Utilities as util
import numpy as np
def nonlinear_triangulation(camera_pose_1: np.ndarray, camera_pose_2: np.ndarray, triangulated_points: list[Coordinate], inliers_dict: dict):
    # NOTE: We are assuming that triangulated_points has the same order as inliers_dict. Basically,
    # the first element of list(inliers_dict) corresponds to the first element of triangulated_points
    # Returns a list of non-homogenous point arrays [u, v]^T
    points1_list, points2_list = util.pointlist_from_dict(inliers_dict)
    for i in range(triangulated_points):  # Pixel objects
        point1 = points1_list[i]
        point2 = points2_list[i]
        X0 = triangulated_points[i].to_arr(homogenous=True)  # Initial guess from linear trigulation
        u1 = point1[0]
        v1 = point1[1]
        u2 = point2[0]
        v2 = point2[1]
        out = leastsq(error, X0, args=((u1, u2), (v1, v2), camera_pose_1, camera_pose_2), )
    return

#         Optimized for
def error(x_homogeneous, u_set, v_set, projection_matrix_1, projection_matrix_2):
    sum = 0
    p_list = [projection_matrix_1, projection_matrix_2]
    for i in range(2):
        P1 = p_list[i][0, 0:3].reshape((1,3))  # We reshape these matrices otherwise the result
        P2 = p_list[i][1, 0:3].reshape((1,3))  # is shape (3,) which is not the same as (1,3)
        P3 = p_list[i][2, 0:3].reshape((1,3))
        reproj_x = (P1 @ x_homogeneous) / (P3 @ x_homogeneous)
        reproj_y = (P2 @ x_homogeneous) / (P3 @ x_homogeneous)
        u = u_set[i]
        v = v_set[i]
        sum += (u - reproj_x)**2 + (v - reproj_y)**2
    return sum
