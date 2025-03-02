import copy
import logging
import cv2
import numpy as np
import Utilities as util
import matplotlib
matplotlib.use("tkagg")

import EstimateFundamentalMatrix as EFM
import GetInlierRANSANC as GIR
import EssentialMatrixFromFundamentalMatrix as EMFFM
import ExtractCameraPose as ECP
import LinearTriangulation as LT
import DisambiguateCameraPose as DCP
import NonlinearTriangulation as NLT
from Pixel import Pixel, Coordinate
import PnPRANSAC as PnP
import NonlinearPnP as NLPnP

logger = logging.getLogger(__name__)

def main():
    Args = util.setup_environment()
    DataPath = Args.DataPath

    """Parsing the data"""
    images, image_names = util.load_images(DataPath, -1, cv2.IMREAD_ANYCOLOR)
    match_dictionaries = util.parse_matching_txt(DataPath)
    k_Mat = util.parse_Camera_Instrinsics(DataPath)

    """Estimating F matrix between two images"""
    F, inliers_dict = GIR.getInlierRANSAC(match_dictionaries[(1,2)])
    inlier_pair_dict = dict()
    for i in range(1,len(images)):
        for j in range(i+1, len(images)+1):
            # print(f"({i},{j})")
            __, inlier_coorespondances = GIR.getInlierRANSAC(match_dictionaries[(i,j)])
            inlier_pair_dict[(i,j)] = inlier_coorespondances

    pair_lines = []
    for key,value in inliers_dict.items():
        pair_lines.append((key, value))
    
    print(f"Percentage of inliers found: {round(100*len(inliers_dict)/len(match_dictionaries[(1,2)]))}%")
   
    """Estimate Essential Matrix"""
    e_Mat = EMFFM.getEssentialFromF(F,k_Mat)

    """Extract Pose from Essential Matrix"""
    R1, R2, t = ECP.extract_camera_pose(e_Mat)

    T1 = R1 @ np.hstack((np.eye(3), t))
    T2 = R2 @ np.hstack((np.eye(3), t))
    T3 = R1 @ np.hstack((np.eye(3), -t))
    T4 = R2 @ np.hstack((np.eye(3), -t))

    P1 = k_Mat @ T1
    P2 = k_Mat @ T2
    P3 = k_Mat @ T3
    P4 = k_Mat @ T4

    t_list = [T1, T2, T3, T4]
    p_list = [P1, P2, P3, P4]

    """Linear Triangulation"""
    x_set_list = []
    # x_set2_list = []
    P_identity = k_Mat @ np.hstack((np.eye(3), np.zeros((3,1))))
    for i in range(4):
        x_set = LT.linear_triangulation(P_identity, p_list[i], inliers_dict)
        # x_set2 = LT.cv2triangulate(P_identity, p_list[i], inliers_dict)
        x_set_list.append(x_set)
    best_t, best_x_set = DCP.disambiguate_camera_pose(t_list, x_set_list)
    best_pose = k_Mat @ best_t
    # When Projecting points onto image 1, use keys and Identity. 
    # When projecting points onto image 2, use values and p_list[i]

    """Non-Linear Optimization of correspondances"""
    X, p1_to_X, p2_to_X = NLT.nonlinear_triangulation(P_identity, best_pose, best_x_set, inliers_dict)
    LT.visualize_triangulation(images[0], list(inliers_dict), X, P_identity)
    LT.visualize_triangulation(images[1], list(inliers_dict.values()), X, best_pose)
    # print(f"LINEAR ERROR: {NLT.calc_error(P_identity, best_pose, best_x_set, inliers_dict)}")
    # print(f"NON-LINEAR ERROR: {NLT.calc_error(P_identity, best_pose, X, inliers_dict)}")

    # exit(1)
    print(f"R: \n{best_t[0:3,0:3]}")
    print(f"t: \n{-best_t[:, 3]}")
    ########################
    # Rest of Images
    ########################
    R_set = [np.eye(3), best_t[:, 0:3]]
    C_set = [np.zeros((3,1)), -best_t[:,3]]
    P_prev = P_identity
    for i in range(2,len(images)):
        R_new, C_new = PnP.linear_pnp_RANSAC(p1_to_X, inlier_pair_dict[(1,i+1)], k_Mat, images[0])
        print(f"R: \n{R_new}")
        print(f"C: \n{C_new}")
        R_set.append(R_new)
        C_set.append(C_new)
        
        # Add new 3D points
        P_new = k_Mat @ np.hstack((R_new, -C_new))
        pixel_correspondances = inlier_pair_dict[(1, i+1)]
        X_new = LT.linear_triangulation(P_prev, P_new, pixel_correspondances)  # Returns list[Coordinate]
        LT.visualize_triangulation(images[i], list(pixel_correspondances.values()), X_new, P_new)
        X_new, __, p2_to_X = NLT.nonlinear_triangulation(P_prev, P_new, X_new, pixel_correspondances)
        
        # P_prev = P_new
        X = set(X) | set(X_new)
        
        # Build Visability Matrix
        # Bundle Adjustment
        util.draw_pointcloud2D(X_new, R_set, C_set)

if __name__ == '__main__':
    main()
