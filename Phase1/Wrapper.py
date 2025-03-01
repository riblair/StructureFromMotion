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
from Pixel import Pixel, Coordinate

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
    
    print(f"Percentage of inliers found: {round(100*len(inliers_dict)/len(match_dictionaries[(1,2)]))}%")
    # GIR.visualize_RANSAC((images[0], images[1]), match_dictionaries[(1,2)], inliers_dict)

    """Estimate Essential Matrix"""
    e_Mat = EMFFM.getEssentialFromF(F,k_Mat)
    p_list = ECP.extract_camera_pose(e_Mat, k_Mat)
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
    LT.visualize_ambiguity(x_set_list)
    # LT.visualize_ambiguity(x_set2_list)
    best_pose, best_x_set = DCP.disambiguate_camera_pose(p_list, x_set_list)
    LT.visualize_triangulation(images[1], list(inliers_dict.values()), best_x_set, best_pose)
    """Non-Linear Optimization of correspondances"""
    # TODO non-linear triangulation does improve the points, but not by any significant margin. 
    # example improvement 3656.84-> 3647.73, or 2800.15 -> 2791.05
    # Perhaps we need to be doing the projection of point 

    points_3d = NLT.nonlinear_triangulation(P_identity, best_pose, best_x_set, inliers_dict)
    print(f"LINEAR ERROR: {NLT.calc_error(P_identity, best_pose, best_x_set, inliers_dict)}")
    print(f"NON-LINEAR ERROR: {NLT.calc_error(P_identity, best_pose, points_3d, inliers_dict)}")
    NLT.compare_triangulations((images[0], images[0]), k_Mat, P_identity, best_pose, points_3d, best_x_set, inliers_dict)

if __name__ == '__main__':
    main()
