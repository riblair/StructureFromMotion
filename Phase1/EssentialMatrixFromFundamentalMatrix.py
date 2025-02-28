import numpy as np
import cv2
import Utilities as util

def getEssentialFromF(f_Mat: np.ndarray, k_Mat: np.ndarray):
    E_Estim = k_Mat.T @ f_Mat @ k_Mat

    U, S, Vt = np.linalg.svd(E_Estim)
    S_prime = np.diag(np.array([1, 1, 0]))
    E_actual = U @ S_prime @ Vt
    return E_actual

def getEssentialFromcv2(match_dict, k_mat):
    points1, points2 = util.pointlist_from_dict(match_dict)

    points1 = np.array(points1)
    points2 = np.array(points2)
    E, __ = cv2.findEssentialMat(points1,points2,k_mat)
    return E