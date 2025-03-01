from Pixel import Coordinate
from scipy.optimize import least_squares
import Utilities as util
import numpy as np
import cv2
import copy

def nonlinear_triangulation(camera_pose_1: np.ndarray, camera_pose_2: np.ndarray, triangulated_points: list[Coordinate], inliers_dict: dict):
    # NOTE: We are assuming that triangulated_points has the same order as inliers_dict. Basically,
    # the first element of list(inliers_dict) corresponds to the first element of triangulated_points
    # Returns a list of non-homogenous point arrays [u, v]^T
    points1_list, points2_list = util.pointlist_from_dict(inliers_dict)
    pixel_list = list(inliers_dict)
    new_points = []
    correspondances = {}  # Used in the future for PnP
    for i in range(len(triangulated_points)):  # Pixel objects
        point1 = points1_list[i]
        point2 = points2_list[i]
        X0 = triangulated_points[i].to_arr(homogenous=True).flatten()  # Initial guess from linear trigulation
        u1 = point1[0]
        v1 = point1[1]
        u2 = point2[0]
        v2 = point2[1]
        out = least_squares(error, X0, args=((u1, u2), (v1, v2), camera_pose_1, camera_pose_2), ftol=None)
        coord = Coordinate(out.x)
        correspondances[pixel_list[i]] = coord
        new_points.append(coord)
    return new_points, correspondances

#         Optimized for
def error(x_homogeneous, u_set, v_set, projection_matrix_1, projection_matrix_2):
    out = []
    p_list = [projection_matrix_1, projection_matrix_2]
    for i in range(2):
        P1 = p_list[i][0, :].reshape((1,4))  # We reshape these matrices otherwise the result
        P2 = p_list[i][1, :].reshape((1,4))  # is shape (3,) which is not the same as (1,3)
        P3 = p_list[i][2, :].reshape((1,4))
        reproj_x = (P1 @ x_homogeneous) / (P3 @ x_homogeneous)
        reproj_y = (P2 @ x_homogeneous) / (P3 @ x_homogeneous)
        u = u_set[i]
        v = v_set[i]
        out.append(float(u - reproj_x)**2 + (v - reproj_y)**2)
    return sum(out)


def compare_triangulations(im_pair, K, P_identity, P_best_pose, non_linear_points: list[Coordinate], linear_points: list[Coordinate], inliers_dict: dict):

    im_1 = copy.deepcopy(im_pair[0])
    im_2 = copy.deepcopy(im_pair[1])
    p1_list = list(inliers_dict)
    for point in p1_list: # ground truth
        cv2.circle(im_1, (int(point.u), int(point.v)), radius=2, color=(0, 255, 0), thickness=-1)
        cv2.circle(im_2, (int(point.u), int(point.v)), radius=2, color=(0, 255, 0), thickness=-1)
    for point in linear_points:  # reprojection of linear
        point_homogenous = point.to_arr(homogenous=True)
        reproj_x = (P_identity[0,:] @ point_homogenous) / (P_identity[2, :] @ point_homogenous)
        reproj_y = (P_identity[1,:] @ point_homogenous) / (P_identity[2, :] @ point_homogenous)
        cv2.circle(im_1, (int(reproj_x), int(reproj_y)), radius=2, color=(0, 0, 255), thickness=-1)
    for point in non_linear_points:  # reprojection of non-linear
        point_homogenous = point.to_arr(homogenous=True)
        reproj_x = (P_identity[0,:] @ point_homogenous) / (P_identity[2, :] @ point_homogenous)
        reproj_y = (P_identity[1,:] @ point_homogenous) / (P_identity[2, :] @ point_homogenous)
        cv2.circle(im_2, (int(reproj_x), int(reproj_y)), radius=2, color=(0, 0, 255), thickness=-1)
    
    new_im = np.hstack((im_1, im_2))
    cv2.imshow("LT (left) NLT (Right)", new_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calc_error(camera_pose_1: np.ndarray, camera_pose_2: np.ndarray, triangulated_points: list[Coordinate], inliers_dict: dict):
    points1_list, points2_list = util.pointlist_from_dict(inliers_dict)
    err_total = 0
    for i in range(len(triangulated_points)):  # Pixel objects
        point1 = points1_list[i]
        point2 = points2_list[i]
        X0 = triangulated_points[i].to_arr(homogenous=True).flatten()  # Initial guess from linear trigulation
        u1 = point1[0]
        v1 = point1[1]
        u2 = point2[0]
        v2 = point2[1]
        err = error(X0, (u1, u2), (v1, v2), camera_pose_1, camera_pose_2)
        err_total += err
    return err_total
    
def compare_triangulations_top_down(points_3d, best_x_set, best_t):
    xl_list = []
    xn_list = []

    zl_list = []
    zn_list = []
    for pointL, pointN in zip(best_x_set, points_3d):
        xl_list.append(pointL.x)
        xn_list.append(pointN.x)

        zl_list.append(pointL.z)
        zn_list.append(pointN.z)

    # First subplot (top-left)
    plt.scatter(xl_list, zl_list, c='red', linewidths=0.5, s=10)
    plt.scatter(xn_list, zn_list, c='blue', linewidths=0.5, s=10)
    plt.legend(["Linear", "Non-Linear"])
    plt.scatter(0, 0, c='red', marker="^")
    plt.scatter(best_t[0, 3], best_t[2, 3], c='Blue', marker="^")
    plt.xlabel("X")
    plt.xlabel("Z")
    plt.title("Linear vs Non-Linear Triangulation")
    plt.show()
