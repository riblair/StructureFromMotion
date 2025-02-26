import numpy as np
from Pixel import Pixel, Coordinate
import Utilities as util
import cv2
import matplotlib.pyplot as plt
import copy

def linear_triangulation(camera_pose_1, camera_pose_2, correspondances: dict):
    #TODO debug...
    keys = list(correspondances)
    x_set = [] 
    for key in keys:
        point2 = correspondances[key].to_arr(homogenous=True)
        point1 = key.to_arr(homogenous=True)
        mat = util.skew_sym(point1) @ camera_pose_1
        mat_2 = util.skew_sym(point2) @ camera_pose_2
        
        rank_1 = np.linalg.matrix_rank(mat)
        rank_2 = np.linalg.matrix_rank(mat_2)
        if rank_1 != 2 or rank_2 != 2:
            raise ValueError(f"Rank is not 2! Instead is (1) {rank_1} or (2) {rank_2}")
        
        big_mat = np.vstack((mat, mat_2))
        
        __, S, Vt = np.linalg.svd(big_mat)
        solution_idx = np.argmin(S)
        solution = Vt[solution_idx, :]  # Estimated Pose
        solution_coord = Coordinate(solution)
        x_set.append(solution_coord)
    return x_set

def visualize_triangulation(image, original_features, triangulated_features, P):

    im_copy = copy.deepcopy(image)
    
    for point in original_features:
        cv2.circle(im_copy, (int(point.u), int(point.v)), radius=1, color=(0, 255, 0), thickness=-1)
    for point in triangulated_features:
        point_homogenous = point.to_arr(homogenous=True)
        reproj_x = (P[0,:] @ point_homogenous) / (P[2, :] @ point_homogenous)
        reproj_y = (P[1,:] @ point_homogenous) / (P[2, :] @ point_homogenous)
        print(f"Calced_Val: ({reproj_x}, {reproj_y})")
        cv2.circle(im_copy, (int(reproj_x), int(reproj_y)), radius=1, color=(0, 0, 255), thickness=-1)
    cv2.imshow("Linear Triangulation", im_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_ambiguity(triangulated_features_list):
    xs_list = []
    ys_list = []
    zs_list = []

    for triangulated_features in triangulated_features_list:
        xi_list = []
        yi_list = []
        zi_list = []
        for point in triangulated_features:
            xi_list.append(point.x)
            yi_list.append(point.y)
            zi_list.append(point.z)
        xs_list.append(xi_list)
        ys_list.append(yi_list)
        zs_list.append(zi_list)
    # plt.scatter(xs_list[0], ys_list[0], zs_list[0])
    # plt.scatter(xs_list[1], ys_list[1], zs_list[1])
    # plt.scatter(xs_list[2], ys_list[2], zs_list[2])
    # plt.scatter(xs_list[3], ys_list[3], zs_list[3])
    plt.scatter(xs_list[0], zs_list[0], c='red', linewidths=0.5)
    plt.scatter(xs_list[1], zs_list[1], c="blue", linewidths=0.5)
    plt.scatter(xs_list[2], zs_list[2], c="green", linewidths=0.5)
    plt.scatter(xs_list[3], zs_list[3], c="black", linewidths=0.5)
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.show()
    pass
