import argparse
import csv
import cv2
from datetime import datetime
import logging
import numpy as np
import random
import os
import Utilities as util

from EstimateFundamentalMatrix import estimate_F, visualizeEpipolarLines, estimate_F2
from GetInlierRANSANC import getInlierRANSAC, visualize_RANSAC
from EssentialMatrixFromFundamentalMatrix import getEssentialFromF, getEssentialFromF2
from ExtractCameraPose import extract_camera_pose
from LinearTriangulation import linear_triangulation, visualize_triangulation, visualize_ambiguity, linear_triangulation_lstsq


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
    key_list = random.sample(list(inliers_dict), 8)
    eight_pair = []
    eight_pair_mat_1 = np.zeros((8,2))
    eight_pair_mat_2 = np.zeros((8,2))
    for i in range(8):
        eight_pair.append((key_list[i], inliers_dict[key_list[i]]))
        eight_pair_mat_1[i, 0] = key_list[i].u
        eight_pair_mat_1[i, 1] = key_list[i].v
        eight_pair_mat_2[i, 0] = inliers_dict[key_list[i]].u
        eight_pair_mat_2[i, 1] = inliers_dict[key_list[i]].v
        
    # Estimates fundamental matrix using point correspondances between image 1 and image 2.
    # F = estimate_F(eight_pair)
    F = estimate_F2(match_dictionaries[(1,2)]) # Output maps points 1 onto image 2
    visualizeEpipolarLines(F, eight_pair, images[0])  # Give a few points and the second image to draw on.

    eight_pair_arr = np.array(eight_pair)
    """Estimate Essential Matrix"""
    # e_Mat = getEssentialFromF2(F,k_Mat)
    e_Mat, _ = cv2.findEssentialMat(eight_pair_mat_1, eight_pair_mat_2, cameraMatrix=k_Mat)
    # log.info(getEssentialFromF2(round(e_Mat, 4)))
    
    # c_list, r_list = extract_camera_pose(e_Mat, k_Mat)
    # p_list = extract_camera_pose(e_Mat, k_Mat)
    S = cv2.decomposeEssentialMat(e_Mat)
    R1 = S[0]
    R2 = S[1]
    t = S[2]
    p1 = k_Mat @ np.hstack((R1, t))
    p2 = k_Mat @ np.hstack((R1, -t))
    p3 = k_Mat @ np.hstack((R2, t))
    p4 = k_Mat @ np.hstack((R2, -t))
    p_list = [p1, p2, p3, p4]
    
    """Linear Triangulation"""
    x_set_list = []
    P_identity = k_Mat @ np.hstack((np.eye(3), np.zeros((3,1))))
    for i in range(4):
        x_set = linear_triangulation(P_identity, p_list[i], inliers_dict, k_Mat)
        x_set_list.append(x_set)
        # keys = list(inliers_dict)
        # points1_2d = []
        # points2_2d = []
        # for point in keys:
        #     point1_arr = point.to_arr()
        #     point2_arr = inliers_dict[point].to_arr()
        #     points1_2d.append(point1_arr)
        #     points2_2d.append(point2_arr)
        # points_np_1 = np.array(points1_2d)
        # points_np_2 = np.array(points2_2d)
        # result = cv2.triangulatePoints(P_identity, p_list[i], points_np_1[0:2], points_np_2[0:2])
        # points_3d = result[:3] / result[3]
        # points_3d = np.reshape(points_3d, (2, 1, 3))
        # reprojected = cv2.projectPoints(points_3d, p_list[i][:, 0:3], p_list[i][:, 3], k_Mat, np.zeros((5, 1), np.float32) )
        
        visualize_triangulation(images[0], list(inliers_dict), x_set, p_list[i])

    visualize_ambiguity(x_set_list)

if __name__ == '__main__':
    main()
