import numpy as np
import LinearPnP as LPnP
from Pixel import Pixel, Coordinate
import random
import matplotlib.pyplot as plt
import Utilities as util

MAX_ITER = 500
THRESHOLD = 20000.0

def linear_pnp_RANSAC(correspondances: dict[Pixel, Coordinate], K: np.ndarray):
    best_inliers = dict()
    # Pixels in 1 to X 
    # pixel_list_old = list(correspondances)    
    # # Pixels in 1 to 3
    # pixel_list_in_new = list(pix_correspondances)
    # pixel_1_2, pixel_2_1 = util.extract_intersection(pixel_list_old, pixel_list_in_new)
    # new_correspondances = dict()
    # for pix_1_2, pix_2_1 in zip(pixel_1_2, pixel_2_1):
    #     new_correspondances[pix_correspondances[pix_2_1]] = correspondances[pix_1_2]
    pixel_list_corr = list(correspondances)
    best_pose = None
    errors = []
    for __ in range(MAX_ITER):
        six_pixels = random.sample(pixel_list_corr, 6)
        sub_correspondances = {k: correspondances[k] for k in six_pixels if k in correspondances}
        R, t = LPnP.linear_pnp(sub_correspondances, K)
        C = -np.linalg.inv(np.transpose(R)) @ t
        P = K @ np.hstack((R, -C))
        inliers = dict()
        for j in range(len(pixel_list_corr)):
            pixel = pixel_list_corr[j]
            reproj_pix = util.reproject_point(P, correspondances[pixel])
            pixel_diff = pixel-reproj_pix
            error = float(pixel_diff.u**2 + pixel_diff.v**2)
            errors.append(error)
            if error < THRESHOLD:
                inliers[pixel] = correspondances[pixel]
        if len(best_inliers) < len(inliers):
            best_inliers = inliers
            best_pose = (R, C)
        
    # visualize_err_graph(errors)
    print(f"PnP RANSAC Results:\n-------------------\n{len(best_inliers)} inliers found: {len(best_inliers)/len(pixel_list_corr)*100}%")
    print(f"Removed {len(pixel_list_corr) - len(best_inliers)} outliers.")

    return best_pose

# def refine_PnP(best_inliers):
#     pass


def visualize_err_graph(errors):
    print(len(errors))
    plt.hist(errors, bins=1000)
    plt.axvline(x=THRESHOLD, color='red', linestyle='--',
                linewidth=2, label=f'x = {THRESHOLD}')
    plt.ylim([0, 200])
    # plt.xlim([0, 10])
    plt.show()
