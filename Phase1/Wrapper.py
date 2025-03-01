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
import matplotlib
matplotlib.use('qtagg')

import EstimateFundamentalMatrix as EFM
import GetInlierRANSANC as GIR
import EssentialMatrixFromFundamentalMatrix as EMFFM
import ExtractCameraPose as ECP
import LinearTriangulation as LT
import DisambiguateCameraPose as DCP
import NonlinearTriangulation as NLT

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
    pair_lines = []
    for key,value in inliers_dict.items():
        pair_lines.append((key, value))
    # EFM.visualizeEpipolarLines(F, pair_lines, copy.deepcopy(images[0]))
    # F = EFM.estimate_F2(match_dictionaries[(1,2)])
    # EFM.visualizeEpipolarLines(F, pair_lines, copy.deepcopy(images[0]))
    
    print(f"Percentage of inliers found: {round(100*len(inliers_dict)/len(match_dictionaries[(1,2)]))}%")
    # GIR.visualize_RANSAC((images[0], images[1]), match_dictionaries[(1,2)], inliers_dict)
    # log.info(f"Fundamental Matricies:\n {F},\n {F2}")

    """Estimate Essential Matrix"""
    e_Mat = EMFFM.getEssentialFromF(F,k_Mat)
    log.info(f" E FROM OUR VALUES\n {e_Mat}")
    e_Mat2 = EMFFM.getEssentialFromcv2(match_dictionaries[(1,2)], k_Mat)
    log.info(f" E FROM CV2 ALL features\n {e_Mat2}")
    e_Mat3 = EMFFM.getEssentialFromcv2(inliers_dict, k_Mat)
    log.info(f" E FROM CV2 Inliers\n {e_Mat3}")
    log.info(f"DIFFERENCES\n{e_Mat-e_Mat2}\n{e_Mat-e_Mat3}\n{e_Mat2-e_Mat3}")
    # exit(1)
    # print(e_Mat2)
    # log.info(f"Essential Matricies:\n {e_Mat},\n {e_Mat2}")
    R1, R2, t = cv2.decomposeEssentialMat(e_Mat)

    # p1 = k_Mat @ R1 @ np.hstack((np.eye(3), -t))
    # p2 = k_Mat @ R1 @ np.hstack((np.eye(3), -t))
    # p3 = k_Mat @ R2 @ np.hstack((np.eye(3), -t))
    # p4 = k_Mat @ R2 @ np.hstack((np.eye(3), -t))


    p_list = ECP.extract_camera_pose(e_Mat, k_Mat)

    """Linear Triangulation"""
    # P_Ident = k_Mat @ np.eye(3) @ np.hstack((np.eye(3), np.zeros((3,1))))
    # log.info(P_Ident)
    x_set_list = []
    P_identity = k_Mat @ np.hstack((np.eye(3), np.zeros((3,1))))
    for i in range(4):
        x_set = LT.linear_triangulation(P_identity, p_list[i], inliers_dict)
        # x_set2 = LT.cv2triangulate(p_list[i], P_identity, inliers_dict)
        # pixel_points = cv2.convertPointsFromHomogeneous(x_set2)
        # log.info(F"\nX_SET {i}\n")
        # log.info(pixel_points)
        #     log.info(f"ours: {x_set[i].to_arr(homogenous=True).flatten()}\n cv2: {x_set2[:,i]}")
        x_set_list.append(x_set)
        #                            I1       points in I1        triag  P back to I1
    # LT.visualize_ambiguity(x_set_list)
    best_pose, best_x_set = DCP.disambiguate_camera_pose(p_list, x_set_list)
    """Non-Linear Optimization of correspandances"""
    points_3d = NLT.nonlinear_triangulation(best_pose, best_pose, best_x_set, inliers_dict)

    # LT.visualize_triangulation(images[0], list(inliers_dict.values()), best_x_set, best_pose)
    # LT.visualize_triangulation(images[0], list(inliers_dict.values()), points_3d, best_pose)
    NLT.compare_triangulations((copy.deepcopy(images[0]),copy.deepcopy(images[0])), k_Mat, P_identity, best_pose, points_3d, best_x_set, inliers_dict)

if __name__ == '__main__':
    main()
