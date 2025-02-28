import numpy as np
import cv2
from Pixel import Pixel

import Utilities as util

"""
    `eight_point_pair` is a list of length 8 with tuples of pixels.
"""
def estimate_F(eight_point_pair: list):
    if len(eight_point_pair) != 8:
        raise("[estimate_F]: Wrong sized list given for Estimating Fundamental Matrix")

    A_mat = np.zeros((8,9), dtype=np.float32)

    for i in range(0, 8):
        point_tup = eight_point_pair[i]
        A_mat[i, :] = np.array([
            point_tup[0].u * point_tup[1].u,
            point_tup[0].u * point_tup[1].v,
            point_tup[0].u,
            point_tup[0].v * point_tup[1].u,
            point_tup[0].v * point_tup[1].v,
            point_tup[0].v,
            point_tup[1].u,
            point_tup[1].v,
            1])
        
    U, S, Vt = np.linalg.svd(A_mat)

    S_prime = np.zeros((8,9), dtype=np.float32)
    S_prime[0,0] = S[0]
    S_prime[1,1] = S[1]
    S_prime[2,2] = S[2]
    S_prime[3,3] = S[3]
    S_prime[4,4] = S[4]
    S_prime[5,5] = S[5]
    S_prime[6,6] = S[6]
    S_prime[7,7] = S[7]

    F_prime = U @ S_prime @ Vt

    U, S, Vt = np.linalg.svd(F_prime)
    F = np.reshape(Vt[-1, :], (3,3))
    # print(F)
    return F

def estimate_F2(match_dict: dict):
    points1, points2 = util.pointlist_from_dict(match_dict)

    points1 = np.array(points1)
    points2 = np.array(points2)
    F, __ = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, 0.1, 0.99)
    return F

def visualizeEpipolarLines(F: np.ndarray, points: list[tuple[Pixel, Pixel]], image, from_image=1):
    lines = []
    pixel_list = []
    ## Generates a list of pixels in first tmage 
    for point_pair in points:
        pixel_list.append(point_pair[0])

    if from_image == 2:
        F = np.transpose(F)
    for point in pixel_list:
        lines.append(np.matmul(F, point.to_arr(homogenous=True)))
    row, col, depth = image.shape
    for line in lines:
        x0,y0 = map(int, [0, -line[2]/line[1] ])
        x1,y1 = map(int, [col, -(line[2]+line[0]*col)/line[1] ])
        cv2.line(image, (x0, y0), (x1, y1), (255, 0, 255), 1)
    cv2.imshow("Epipolar Lines", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return