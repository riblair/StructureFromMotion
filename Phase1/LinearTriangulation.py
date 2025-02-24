import numpy as np
from Pixel import Pixel, Coordinate
import Utilities as util
import cv2

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