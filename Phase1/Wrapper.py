# Create Your Own Starter Code :)
import cv2
import numpy as np
import logging
import argparse
import Utilities as util
from datetime import datetime
import os
import csv

from EstimateFundamentalMatrix import estimate_F, visualizeEpipolarLines
from GetInlierRANSANC import getInlierRANSAC, visualize_RANSAC
import random


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


    # This initializes the python loggingger.
    logging.basicConfig(filename=LoggingFilePath+f"{datetime.now().strftime('%b_%d_%H:%M:%S')}.logging", level=DebugLevel)
    log = logging.getLogger()
    log.info(f"Beginning SfM")

    """Parsing the data"""
    images, image_names = util.load_images(DataPath, -1, cv2.IMREAD_ANYCOLOR)
    match_dictionaries = util.parse_matching_txt(DataPath)

    # log.info(a)
    # util.show_im_match_pair((images[0], images[1]), match_dictionaries[(1,2)], True)

    """Estimating F matrix between two images"""
    # match_dict = match_dictionaries[(1,2)]
    
    # # randomly select eight pairs from dict
    # key_list = random.sample(match_dict.keys(), 8)
    # eight_pair = []
    # for i in range(8):
    #     eight_pair.append((key_list[i], match_dict[key_list[i]]))
    # F = estimate_F(eight_pair)
    # log.info(F)
    # log.info(np.linalg.matrix_rank(F))

    # visualizeEpipolarLines(F, eight_pair, images[1], 1)

    matches_dict = getInlierRANSAC(match_dictionaries[(1,2)])
    print(f"Percentage of inliers found: {round(100*len(matches_dict)/len(match_dictionaries[(1,2)]))}%")


    visualize_RANSAC((images[0], images[1]), match_dictionaries[(1,2)], matches_dict)

    # VISUALIZATION OF INLIERS,
    # Figure out a good threshold imperical


    return

if __name__ == '__main__':
    main()