import numpy as np
from Pixel import Pixel, Coordinate
import Utilities as util
import cv2
import matplotlib.pyplot as plt

def linear_triangulation(camera_pose_1, camera_pose_2, correspondances: dict):
    #TODO debug...
    keys = list(correspondances)
    x_set = [] 
    for key in keys:
        point2 = correspondances[key].to_hom_arr()
        point1 = key.to_hom_arr()
        mat = util.skew_sym(point1) @ camera_pose_1
        mat_2 = util.skew_sym(point2) @ camera_pose_2
        
        big_mat = np.vstack((mat, mat_2))
        
        __, S, Vt = np.linalg.svd(big_mat)
        solution_idx = np.argmin(S)
        solution = Vt[solution_idx, :]  # Estimated Pose
        solution_coord = Coordinate(solution)
        x_set.append(solution_coord)
    return x_set

def visualize_triangulation(image, original_features, triangulated_features):
    for point in triangulated_features:
        cv2.circle(image, (int(point.x), int(point.x)), radius=1, color=(0, 0, 255), thickness=-1)
    for point in original_features:
        cv2.circle(image, (int(point.u), int(point.v)), radius=1, color=(0, 255, 0), thickness=-1)
    cv2.imshow("Linear Triangulation", image)
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

    print(len(xs_list))
    print(len(ys_list))
    print(len(zs_list))
    plt.scatter(xs_list[0], ys_list[0], zs_list[0])
    plt.scatter(xs_list[1], ys_list[1], zs_list[1])
    plt.scatter(xs_list[2], ys_list[2], zs_list[2])
    plt.scatter(xs_list[3], ys_list[3], zs_list[3])
    plt.show()
    pass