import numpy as np
from Pixel import Pixel, Coordinate
import Utilities as util
import cv2
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt

def linear_triangulation(camera_pose_1, camera_pose_2, correspondances: dict):
    keys = list(correspondances)
    x_set = [] 
    # Usually a sign error.....
    # independant of the scale factor sign 
    for key in keys:
        point2 = correspondances[key].to_arr(homogenous=True)
        point1 = key.to_arr(homogenous=True)
        mat = util.skew_sym(point1) @ camera_pose_1
        mat_2 = util.skew_sym(point2) @ camera_pose_2
        big_mat = np.vstack((mat, mat_2))
        
        __, S, Vt = np.linalg.svd(big_mat)
        solution_idx = np.argmin(S)
        solution = Vt[solution_idx, :]  # Estimated Pose
        solution_coord = Coordinate(solution, norm=True)
        x_set.append(solution_coord)
    return x_set

def cv2triangulate(p1, p2, inliers_dict):
    # x1_list, x2_list = util.pointlist_from_dict(inliers_dict, homogenous=True)
    x_mat1 = np.ndarray((2,len(inliers_dict.keys())), dtype=np.float32)
    x_mat2 = np.ndarray((2,len(inliers_dict.keys())), dtype=np.float32)
    iterator = 0
    for key,value in inliers_dict.items():
        x_mat1[:, iterator] = key.to_arr().flatten()
        x_mat2[:, iterator] = value.to_arr().flatten()
        iterator+=1

    coords = cv2.triangulatePoints(p1,p2, x_mat1, x_mat2)
    return coords

def reproject_points() -> list[Pixel]: 

    pass

def visualize_triangulation(image, original_features, triangulated_features, proj_mat):
    for coord in triangulated_features: # reprojection
        coord_arr = coord.to_arr(homogenous=True)
        x_to_u = (proj_mat[0,:] @ coord_arr) / (proj_mat[2,:] @ coord_arr) 
        y_to_v = (proj_mat[1,:] @ coord_arr) / (proj_mat[2,:] @ coord_arr) 
        cv2.circle(image, (int(x_to_u), int(y_to_v)), radius=1, color=(0, 0, 255), thickness=-1)
    for point in original_features: # ground truth
        cv2.circle(image, (int(point.u), int(point.v)), radius=1, color=(0, 255, 0), thickness=-1)
    cv2.imshow("Linear Triangulation", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_ambiguity(triangulated_features_list):
    xs_list = []
    ys_list = []
    zs_list = []

    for triangulated_features in triangulated_features_list:
        xi_mat = np.zeros((len(triangulated_features)))
        yi_mat = np.zeros((len(triangulated_features)))
        zi_mat = np.zeros((len(triangulated_features)))
        # xi_list = []
        # yi_list = []
        # zi_list = []
        for i in range(len(triangulated_features)):
            xi_mat[i] = triangulated_features[i].x
            yi_mat[i] = triangulated_features[i].y
            zi_mat[i] = triangulated_features[i].z
            # xi_list.append(point.x)
            # yi_list.append(point.y)
            # zi_list.append(point.z)
        xs_list.append(xi_mat)
        ys_list.append(yi_mat)
        zs_list.append(zi_mat)
        # xs_list.append(xi_list)
        # ys_list.append(yi_list)
        # zs_list.append(zi_list)

    # plt.scatter(xs_list[0], ys_list[0], zs_list[0])
    # plt.scatter(xs_list[1], ys_list[1], zs_list[1])
    # plt.scatter(xs_list[2], ys_list[2], zs_list[2])
    # plt.scatter(xs_list[3], ys_list[3], zs_list[3])
    plt.scatter(xs_list[0], zs_list[0], c='red')
    plt.scatter(xs_list[1], zs_list[1], c='green')
    plt.scatter(xs_list[2], zs_list[2], c='blue')
    plt.scatter(xs_list[3], zs_list[3], c='black')
    plt.show()
    pass