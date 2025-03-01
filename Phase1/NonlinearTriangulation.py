from Pixel import Coordinate
from scipy.optimize import least_squares
import Utilities as util
import numpy as np
import cv2
def nonlinear_triangulation(camera_pose_1: np.ndarray, camera_pose_2: np.ndarray, triangulated_points: list[Coordinate], inliers_dict: dict):
    # NOTE: We are assuming that triangulated_points has the same order as inliers_dict. Basically,
    # the first element of list(inliers_dict) corresponds to the first element of triangulated_points
    # Returns a list of non-homogenous point arrays [u, v]^T
    points1_list, points2_list = util.pointlist_from_dict(inliers_dict)
    new_points = []
    for i in range(len(triangulated_points)):  # Pixel objects
        point1 = points1_list[i]
        point2 = points2_list[i]
        X0 = triangulated_points[i].to_arr(homogenous=True).flatten()  # Initial guess from linear trigulation
        u1 = point1[0]
        v1 = point1[1]
        u2 = point2[0]
        v2 = point2[1]
        out = least_squares(error, X0, args=((u1, u2), (v1, v2), camera_pose_1, camera_pose_2), ftol=1e-10)
        new_points.append(Coordinate(out.x))
    return new_points

#         Optimized for
def error(x_homogeneous, u_set, v_set, projection_matrix_1, projection_matrix_2):
    out = np.zeros((2,1))
    p_list = [projection_matrix_1, projection_matrix_2]
    for i in range(2):
        P1 = p_list[i][0, :].reshape((1,4))  # We reshape these matrices otherwise the result
        P2 = p_list[i][1, :].reshape((1,4))  # is shape (3,) which is not the same as (1,3)
        P3 = p_list[i][2, :].reshape((1,4))
        reproj_x = (P1 @ x_homogeneous) / (P3 @ x_homogeneous)  # 3x1 times 4,  TODO fix me
        reproj_y = (P2 @ x_homogeneous) / (P3 @ x_homogeneous)
        u = u_set[i]
        v = v_set[i]
        out[i] = (u - reproj_x)**2 + (v - reproj_y)**2
    return np.sum(out)


def compare_triangulations(im_pair, K, P_identity, P_best_pose, non_linear_points: list[Coordinate], linear_points: list[Coordinate], inliers_dict: dict):
                        #  (image, same image)
    # Needed: Pi, P1, image[0], inliers dict,
    # 1. project 3d poses into image 0 using P_eye
    # 2. use pixels from image[1] and P1 to make new 3d point from perspective of image[1]
    # 3. use P_eye to project new 3d point from image[1] onto image[0]


    im1_shape = im_pair[0].shape
    # p1_list, p2_list = util.pointlist_from_dict(inliers_dict)
    p1_list = list(inliers_dict)
    for point in p1_list: # ground truth
        cv2.circle(im_pair[0], (int(point.u), int(point.v)), radius=1, color=(0, 255, 0), thickness=-1)
        cv2.circle(im_pair[1], (int(point.u), int(point.v)), radius=1, color=(0, 255, 0), thickness=-1)
    for point in linear_points:  # may not be what we are looking for?
        pixel_coords = P_identity @ point.to_arr(homogenous=True)  #3x4 @ 4x1 = 3x1         Cyan
        pixel_coords /= pixel_coords[2]
        cv2.circle(im_pair[0], (int(pixel_coords[0]), int(pixel_coords[1])), radius=1, color=(0, 0, 0), thickness=-1)
    for pixel in list(inliers_dict.values()):
        pixel = pixel.to_arr(homogenous=True)
        point_3D = np.linalg.pinv(P_best_pose) @ pixel
        # x_in_image_0 = (P_identity[0,:] @ point_3D) / (P_identity[2, :] @ point_3D) + P_best_pose[0, 3]
        # y_in_image_0 = (P_identity[1,:] @ point_3D) / (P_identity[2, :] @ point_3D) + P_best_pose[1, 3]
        final = np.array([
            [point_3D[0, 0] * K[0,0]],
            [point_3D[1,0] * K[1,1]],
            [point_3D[2, 0]],
            [1]
        ])
        result = P_identity @ final
        cv2.circle(im_pair[0], (int(final[0]), int(final[1])), radius=1, color=(0, 0, 255), thickness=-1)
        
    cv2.imshow("LT (left)", im_pair[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # new_im = np.hstack((im_pair[0], im_pair[1]))
    
    # for point in linear_points:  # reprojection of linear
    #     point_homogenous = point.to_arr(homogenous=True)
    #     reproj_x = (P[0,:] @ point_homogenous) / (P[2, :] @ point_homogenous)
    #     reproj_y = (P[1,:] @ point_homogenous) / (P[2, :] @ point_homogenous)
    #     cv2.circle(new_im, (int(reproj_x), int(reproj_y)), radius=1, color=(0, 0, 255), thickness=-1)
    # for point in non_linear_points:  # reprojection of non-linear
    #     point_homogenous = point.to_arr(homogenous=True)
    #     reproj_x = (P[0,:] @ point_homogenous) / (P[2, :] @ point_homogenous)
    #     reproj_y = (P[1,:] @ point_homogenous) / (P[2, :] @ point_homogenous)
    #     cv2.circle(new_im, (int(reproj_x+im1_shape[1]), int(reproj_y)), radius=1, color=(0, 0, 255), thickness=-1)
    # cv2.imshow("LT (left) NLT (Right)", new_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
