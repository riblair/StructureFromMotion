import argparse
import csv
import cv2
from datetime import datetime
import logging
import numpy as np
import random
import os
import Utilities as util

from EstimateFundamentalMatrix import estimate_F, visualizeEpipolarLines
from GetInlierRANSANC import getInlierRANSAC, visualize_RANSAC
from EssentialMatrixFromFundamentalMatrix import getEssentialFromF, getEssentialFromF2
from ExtractCameraPose import extract_camera_pose
from LinearTriangulation import linear_triangulation, visualize_triangulation, visualize_ambiguity


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
    inliers_dict = getInlierRANSAC(match_dictionaries[(1,2)])
    
    print(f"Percentage of inliers found: {round(100*len(inliers_dict)/len(match_dictionaries[(1,2)]))}%")
    # visualize_RANSAC((images[0], images[1]), match_dictionaries[(1,2)], matches_dict)
    key_list = random.sample(inliers_dict.keys(), 8)
    eight_pair = []
    for i in range(8):
        eight_pair.append((key_list[i], inliers_dict[key_list[i]]))
    F = estimate_F(eight_pair)

    """Estimate Essential Matrix"""
    e_Mat = getEssentialFromF2(F,k_Mat)
    # log.info(getEssentialFromF2(round(e_Mat, 4)))
    
    # c_list, r_list = extract_camera_pose(e_Mat, k_Mat)
    p_list = extract_camera_pose(e_Mat, k_Mat)
    

    """Linear Triangulation"""
    x_set_list = []
    for i in range(4):
        if i == 3:
            x_set = linear_triangulation(p_list[i], p_list[0], inliers_dict)
        else:
            x_set = linear_triangulation(p_list[i], p_list[i+1], inliers_dict)
        x_set_list.append(x_set)
        # visualize_triangulation(images[0], list(inliers_dict), x_set)

    visualize_ambiguity(x_set_list)
if __name__ == '__main__':
    main()