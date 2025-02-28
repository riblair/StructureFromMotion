import numpy as np
import cv2
from Pixel import Pixel
import random
from EstimateFundamentalMatrix import estimate_F, estimate_F2, visualizeEpipolarLines
import matplotlib.pyplot as plt


MAX_ITER = 100
THRESHOLD = 0.15

def getInlierRANSAC(matches_dict):
    inliers = {}
    best_F = None
    for i in range(MAX_ITER):
        key_list = random.sample(matches_dict.keys(), 8)
        current_inliers = dict()
        errors = []
        
        eight_pair = []
        for i in range(8):
            eight_pair.append((key_list[i], matches_dict[key_list[i]]))
        F = estimate_F(eight_pair)
        # F = estimate_F2(matches_dict)
        for pt1, pt2 in matches_dict.items():
            err = loss((pt1,pt2), F)
            errors.append(float(err))
            if abs(err) < THRESHOLD:
                current_inliers[pt1] = pt2

        # visualize_err_graph(errors)
        if len(current_inliers.keys()) > len(inliers.keys()):
            inliers = current_inliers
            best_F = F
    return best_F, inliers

def loss(point_pair: tuple[Pixel, Pixel], F_mat: np.ndarray):
    return point_pair[1].to_arr(homogenous=True).T @ F_mat @ point_pair[0].to_arr(homogenous=True)

def visualize_err_graph(errors):
    
    plt.hist(errors, bins=1000)
    plt.axvline(x=THRESHOLD, color='red', linestyle='--',
                linewidth=2, label=f'x = {THRESHOLD}')
    # plt.ylim([0, 500])
    plt.show()

def visualize_RANSAC(image_pair, matches_dict_old, matches_dict_new):
    im1_shape = image_pair[0].shape
    new_im = np.hstack((image_pair[0], image_pair[1]))

    outliers = matches_dict_old.keys() - matches_dict_new.keys()
    for key in outliers:
        value = matches_dict_old[key]
        center1 = key.to_arr(typecast=np.int32)
        center1 = list(center1.flatten())
        center2 = value.to_arr(typecast=np.int32) + np.array([[im1_shape[1]], [0]], dtype=np.int32)
        center2 = list(center2.flatten())
        cv2.circle(new_im, center1, 2, (0,0,255), -1)
        cv2.circle(new_im, center2, 2, (0,0,255), -1)
        cv2.line(new_im, center1, center2, (0, 0, 255), 1)

    for key,value in matches_dict_new.items():
        center1 = key.to_arr(typecast=np.int32)
        center1 = list(center1.flatten())
        center2 = value.to_arr(typecast=np.int32) + np.array([[im1_shape[1]], [0]], dtype=np.int32)
        center2 = list(center2.flatten())
        cv2.circle(new_im, center1, 2, (0,0,255), -1)
        cv2.circle(new_im, center2, 2, (0,0,255), -1)
        cv2.line(new_im, center1, center2, (0, 255, 0), 1)

    cv2.imshow("image", new_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




