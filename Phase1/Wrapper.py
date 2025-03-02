import argparse
import copy
import csv
import cv2
from datetime import datetime
import logging
import numpy as np
import random
import os
import Utilities as util
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('qtagg')

import EstimateFundamentalMatrix as EFM
import GetInlierRANSANC as GIR
import EssentialMatrixFromFundamentalMatrix as EMFFM
import ExtractCameraPose as ECP
import LinearTriangulation as LT
import DisambiguateCameraPose as DCP
import NonlinearTriangulation as NLT
from Pixel import Pixel, Coordinate
# import LinearPnP as LPnP
import PnPRANSAC as PnP
import NonlinearPnP as NLPnP

def main():
    Parser = argparse.ArgumentParser()

    Parser.add_argument(
        "--LoggingPath",
        default="Phase1/Logs/",
        type=str,
        help="Path for the Logging files to be created in. Default: Phase1/Logs/"
    )
    Parser.add_argument(
        "--DataPath",
        default="Phase1/P2Data/",
        type=str,
        help="Path for the image / matches and calibration data. default: Phase1/P2Data/",
    )
    Parser.add_argument(
        "--OutputPath",
        default="Output/",
        type=str,
        help="Path for the outputs default: Output/"
    )
    Parser.add_argument(
        "--DebugLevel",
        default="INFO",
        type=str,
        help="Path for the outputs default: Output/"
    )

    Args = Parser.parse_args()
    LoggingFilePath = Args.LoggingPath
    DataPath = Args.DataPath
    OutputPath = Args.OutputPath
    os.makedirs(LoggingFilePath, exist_ok=True)
    os.makedirs(OutputPath, exist_ok=True)

    if Args.DebugLevel == "INFO":
        DebugLevel = logging.INFO
    elif Args.DebugLevel == "WARNING":
        DebugLevel = logging.WARNING
    elif Args.DebugLevel == "DEBUG":
        DebugLevel = logging.DEBUG
    elif Args.DebugLevel == "CRITICAL":
        DebugLevel = logging.CRITICAL
    elif Args.DebugLevel == "ERROR":
        DebugLevel = logging.ERROR
    else:
        print(f"Unknown debug level {Args.DebugLevel}.\n Defaulting to INFO\n")
        DebugLevel = logging.INFO


    # This initializes the python logger.
    logging.basicConfig(filename=LoggingFilePath+f"{datetime.now().strftime('%b_%d_%H:%M:%S')}.logging", level=DebugLevel)
    log = logging.getLogger()
    log.info(f"Beginning SfM")

    """Parsing the data"""
    images, image_names = util.load_images(DataPath, -1, cv2.IMREAD_ANYCOLOR)
    match_dictionaries = util.parse_matching_txt(DataPath)
    k_Mat = util.parse_Camera_Instrinsics(DataPath)
    # util.show_im_match_pair((images[0], images[1]), match_dictionaries[(1,2)], True)

    """Estimating F matrix between two images"""
    F, inliers_dict = GIR.getInlierRANSAC(match_dictionaries[(1,2)])
    inlier_pair_dict = dict()
    for i in range(1,len(images)):
        for j in range(i+1, len(images)+1):
            __, inlier_coorespondances = GIR.getInlierRANSAC(match_dictionaries[(i,j)])
            inlier_pair_dict[(i,j)] = inlier_coorespondances

    pair_lines = []
    for key,value in inliers_dict.items():
        pair_lines.append((key, value))
    # EFM.visualizeEpipolarLines(F, pair_lines, copy.deepcopy(images[0]))
    
    print(f"Percentage of inliers found: {round(100*len(inliers_dict)/len(match_dictionaries[(1,2)]))}%")
    # GIR.visualize_RANSAC((images[0], images[1]), match_dictionaries[(1,2)], inliers_dict)

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
    P_Iden_list = [P_identity,P_identity,P_identity,P_identity]
    for i in range(4):
        x_set = LT.linear_triangulation(P_identity, p_list[i], inliers_dict)
        # x_set2 = LT.cv2triangulate(P_identity, p_list[i], inliers_dict)
        x_set_list.append(x_set)
        # x_set2_list.append(x_set2)
        """ 
            When Projecting points onto image 1, use keys and Identity. 
            When projecting points onto image 2, use values and p_list[i]
        """
        # LT.visualize_triangulation(images[0], list(inliers_dict.keys()), x_set, P_identity)
        # LT.visualize_triangulation(images[1], list(inliers_dict.values()), x_set, p_list[i])
    # LT.visualize_ambiguity(x_set_list)
    # LT.visualize_ambiguity(x_set2_list)
    best_t, best_x_set = DCP.disambiguate_camera_pose(t_list, x_set_list)
    best_pose = k_Mat @ best_t
    # LT.visualize_triangulation(images[0], list(inliers_dict.keys()), best_x_set, P_identity)
    """Non-Linear Optimization of correspondances"""
    X, p1_to_X, p2_to_X = NLT.nonlinear_triangulation(P_identity, best_pose, best_x_set, inliers_dict)
    print(f"LINEAR ERROR: {NLT.calc_error(P_identity, best_pose, best_x_set, inliers_dict)}")
    print(f"NON-LINEAR ERROR: {NLT.calc_error(P_identity, best_pose, X, inliers_dict)}")
    # NLT.compare_triangulations_top_down(X, best_x_set, best_t)
    # NLT.compare_triangulations((images[0], images[0]), k_Mat, P_identity, best_pose, points_3d, best_x_set, inliers_dict)
    # print(f"R: \n{best_t[0:3,0:3]}")
    # print(f"t: \n{best_t[:, 3]}")
    
    ########################
    # Rest of Images
    ########################
    
    # inlier_pair_dict -> inliers match dict with key as (from, to) and value as aas
    
    R_set = [np.eye(3), best_t[:, 0:3]]
    C_set = [np.zeros((3,1)), best_t[:,3]]
    P_prev = best_pose
    for i in range(2,len(images)):
        # util.draw_features_on_image(images[i-1], list(p2_to_X), list(inlier_pair_dict[(i, i+1)] ))
        # util.draw_features_on_image(images[i-1], list(inlier_pair_dict[(i, i+1)]))
        # Register ith image using PnP       # THIS SHOULD BE THE last_p_to_X
        R_new, C_new = PnP.linear_pnp_RANSAC(p1_to_X, inlier_pair_dict[(1,i+1)], k_Mat, images[0])
        print(f"R: \n{R_new}")
        print(f"C: \n{C_new}")
        
        # R_new, C_new = NLPnP.nonlinear_pnp(R_new, C_new, p2_to_X, k_Mat)
        # print(f"R: \n{R_new}")
        # print(f"C: \n{C_new}")
        # exit(1)
        R_set.append(R_new)
        C_set.append(C_new)
        
        # Add new 3D points
        P_new = k_Mat @ np.hstack((R_new, -C_new))
        pixel_correspondances = inlier_pair_dict[(i-1, i)]
        X_new = LT.linear_triangulation(P_prev, P_new, pixel_correspondances)  # Returns list[Coordinate]
        # X_new, __, p2_to_X = NLT.nonlinear_triangulation(P_prev, P_new, X_new, pixel_correspondances)
        
        P_prev = P_new
        X = set(X) | set(X_new)
        
        # Build Visability Matrix
        
        # Bundle Adjustment
    
        util.draw_pointcloud2D(X, R_set, C_set)

if __name__ == '__main__':
    main()
