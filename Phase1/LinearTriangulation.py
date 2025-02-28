import numpy as np
from Pixel import Pixel, Coordinate
import Utilities as util
import cv2
# import matplotlib
# matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import copy

def linear_triangulation_lstsq(camera_pose_1, camera_pose_2, correspondances: dict):
    keys = list(correspondances)
    x_set = [] 
    # Usually a sign error.....
    # independant of the scale factor sign 
    for key in keys:
        point2 = correspondances[key].to_arr(homogenous=True)
        point1 = key.to_arr(homogenous=True)
        mat = util.skew_sym(point1) @ camera_pose_1
        mat_2 = util.skew_sym(point2) @ camera_pose_2
        
        rank_1 = np.linalg.matrix_rank(mat)
        rank_2 = np.linalg.matrix_rank(mat_2)
        if rank_1 != 2 or rank_2 != 2:
            raise ValueError(f"Rank is not 2! Instead is: (1) {rank_1} (2) {rank_2}")
        
        big_mat = np.vstack((mat, mat_2))
        zero_mat = np.zeros((6, 1))
        x, residuals, rank, singular_values = np.linalg.lstsq(big_mat, zero_mat)
    return x_set
    

def linear_triangulation(camera_pose_1, camera_pose_2, correspondances: dict):
    #TODO debug...
    keys = list(correspondances)
    x_set = [] 
    for key in keys:  # THESE ARE PIXEL OBJECTS
        point2 = correspondances[key].to_arr(homogenous=True)
        point1 = key.to_arr(homogenous=True)
        mat = util.skew_sym(point1) @ camera_pose_1
        mat_2 = util.skew_sym(point2) @ camera_pose_2
        
        rank_1 = np.linalg.matrix_rank(mat)
        rank_2 = np.linalg.matrix_rank(mat_2)
        if rank_1 != 2 or rank_2 != 2:
            raise ValueError(f"Rank is not 2! Instead is: (1) {rank_1} (2) {rank_2}")
        
        big_mat = np.vstack((mat, mat_2))
        
        __, S, Vt = np.linalg.svd(big_mat)
        X = Vt[-1]
        X = X / X[-1]
        solution_coord = Coordinate(X)
        x_set.append(solution_coord)
        # Theoretically, the output should be a normalized coordinate. Or at least the X and Y values are
        # normalized since the input was normalized (pixel coords). This can be rescaled by using the 
        # focal lengths from K (fx and fy). The Z coordinate remains unchanged since we are calculating
        # for this via the perspective matrices.
        # final = np.array([
        #     [X[0] * K[0,0]],
        #     [X[1] * K[1,1]],
        #     [X[2]],
        #     [1]
        # ])
        
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

def visualize_triangulation(image, original_features, triangulated_features, P):

    im_copy = copy.deepcopy(image)
    
    for point in original_features: # ground truth
        cv2.circle(im_copy, (int(point.u), int(point.v)), radius=1, color=(0, 255, 0), thickness=-1)
    for point in triangulated_features:  # reprojection
        point_homogenous = point.to_arr(homogenous=True)
        reproj_x = (P[0,:] @ point_homogenous) / (P[2, :] @ point_homogenous)
        reproj_y = (P[1,:] @ point_homogenous) / (P[2, :] @ point_homogenous)
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

    # Create a 2x2 grid for subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # Adjust the size as needed

    # First subplot (top-left)
    axs[0, 0].scatter(xs_list[0], zs_list[0], c='red', linewidths=0.5, s=10)
    axs[0, 0].set_xlabel("X")
    axs[0, 0].set_ylabel("Z")
    axs[0, 0].set_title("Pose 1: R1, t")

    # Second subplot (top-right)
    axs[0, 1].scatter(xs_list[1], zs_list[1], c='blue', linewidths=0.5, s=10)
    axs[0, 1].set_xlabel("X")
    axs[0, 1].set_ylabel("Z")
    axs[0, 1].set_title("Pose 2: R2, t")

    # Third subplot (bottom-left)
    axs[1, 0].scatter(xs_list[2], zs_list[2], c='green', linewidths=0.5, s=10)
    axs[1, 0].set_xlabel("X")
    axs[1, 0].set_ylabel("Z")
    axs[1, 0].set_title("Pose 3: R1, t")

    # Fourth subplot (bottom-right)
    axs[1, 1].scatter(xs_list[3], zs_list[3], c='black', linewidths=0.5, s=10)
    axs[1, 1].set_xlabel("X")
    axs[1, 1].set_ylabel("Z")
    axs[1, 1].set_title("Pose 4: R2, -t")

    # Adjust the layout for better spacing
    plt.tight_layout()

    # Show the plots
    plt.show()

    pass
