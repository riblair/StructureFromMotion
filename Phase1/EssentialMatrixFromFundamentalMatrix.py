import numpy as np

def getEssentialFromF(f_Mat: np.ndarray, k_Mat: np.ndarray):
    return k_Mat.T @ f_Mat @ k_Mat

def getEssentialFromF2(f_Mat: np.ndarray, k_Mat: np.ndarray):
    E_Estim = k_Mat.T @ f_Mat @ k_Mat

    U, S, Vt = np.linalg.svd(E_Estim)
    S_prime = np.diag(np.array([1, 1, 0]))
    E_actual = U @ S_prime @ Vt
    return E_actual

