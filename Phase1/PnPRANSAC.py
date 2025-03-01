import numpy as np
import LinearPnP as LPnP
from Pixel import *
import random
import matplotlib.pyplot as plt

MAX_ITER = 300
THRESHOLD = 3e6

def linear_pnp_RANSAC(correspondances: dict[Pixel, Coordinate], K: np.ndarray):
    n=0
    pixel_list = list(correspondances)
    best_inliers = []
    best_pose = None
    errors = []
    for i in range(MAX_ITER):
        eight_pixels = random.sample(pixel_list, 6)
        sub_correspondances = {k: correspondances[k] for k in eight_pixels if k in correspondances}
        R, t = LPnP.linear_pnp(sub_correspondances, K)
        P = K @ np.hstack((R,t))
        inliers = []
        for j in range(len(pixel_list)):
            pixel = pixel_list[j]
            coord = correspondances[pixel].to_arr(homogenous=True)
            error = (pixel.u - (P[0,:] @ coord / P[2, :] @ coord))**2 + (pixel.v - (P[1,:] @ coord / P[2, :] @ coord))**2
            error = abs(error)
            errors.append(float(error))
            if error < THRESHOLD:
                inliers.append(j)  # TODO: Actually append something useful? Unsure where this is used later since we are estimating P
        if n < len(inliers):
            n = len(inliers)
            best_inliers = inliers
            best_pose = P
    visualize_err_graph(errors)
    print(f"{n} inliers found: {n/len(pixel_list)*100}%")
    return best_pose

def visualize_err_graph(errors):
    
    plt.hist(errors, bins=10000)
    plt.axvline(x=THRESHOLD, color='red', linestyle='--',
                linewidth=2, label=f'x = {THRESHOLD}')
    # plt.ylim([0, 500])
    # plt.xlim([0, 10])
    plt.show()
