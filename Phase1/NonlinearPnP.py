import numpy as np
from scipy.optimize import least_squares
import Utilities as util
from Pixel import *

def nonlinear_pnp(R0: np.ndarray, C0: np.ndarray, correspondances:dict[Pixel, Coordinate], K: np.ndarray):
    q0 = util.r_to_quaternion(R0)
    X0 = np.vstack((q0, C0.reshape(3,1))).flatten()
    out = least_squares(loss, X0, args=(correspondances, K))
    result = out.x
    print(f"Result: {out.status}, Message: {out.message}")
    q = result[0:4]
    C = result[4:]
    R = util.quaternion_to_r(q)
    return R, C

def nonlinear_pnp_2(P0: np.ndarray, correspondances: dict[Pixel,Coordinate], K=None):
    if P0.shape == (3,4):
        P0 = P0.flatten()
    out = least_squares(loss_from_student_paper, P0, args=(correspondances, K), method='lm')
    return out.x

def loss(X0: np.ndarray, correspondances: dict[Pixel, Coordinate], K: np.ndarray):
    q = X0[0:4]  # Quaternion
    C = X0[4:].reshape((3,1))  # Camera center position
    R = util.quaternion_to_r(q)
    P = K @ np.hstack((R, C))
    list_to_sum = []  # TODO: Turn into np.array after for loop to use np.sum
    pixel_list = list(correspondances)
    for j in range(len(pixel_list)):
        pixel = pixel_list[j]
        coord = correspondances[pixel]
        pix_reproj = util.reproject_point(P, coord)
        pixel_diff = pixel - pix_reproj
        error = float(pixel_diff.u**2 + pixel_diff.v**2)
        list_to_sum.append(error)
    sum = np.sum(np.array(list_to_sum))
    return sum

def loss_from_student_paper(X0, correspondances: dict[Pixel,Coordinate], K):
    pixel_list = list(correspondances)
    list_to_sum = []
    P = X0.reshape((3,4))
    for pixel in pixel_list:
        coord = correspondances[pixel].to_arr(homogenous=True)
        pixel = pixel.to_arr(homogenous=True)
        loss = (pixel - (P @ coord))**2
        list_to_sum.append(loss)
    return np.sum(np.array(list_to_sum))
