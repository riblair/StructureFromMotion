import numpy as np
from Pixel import *

def linear_pnp(correspondances: dict[Pixel, Coordinate], K: np.ndarray):
    points_2d = list(correspondances)
    A = np.zeros((1,12))  # This just starts the matrix. Before SVD, we remove this part.
    for point_2d in points_2d:
        point_3d = correspondances[point_2d]
        Ai = np.array([
            [point_3d.x, point_3d.y, point_3d.z, 1,          0,          0,          0, 0, -point_2d.u * point_3d.x, -point_2d.u * point_3d.y, -point_2d.u * point_3d.z, -point_2d.u],
            [         0,          0,          0, 0, point_3d.x, point_3d.y, point_3d.z, 1, -point_2d.v * point_3d.x, -point_2d.v * point_3d.y, -point_2d.v * point_3d.z, -point_2d.v],
        ])
        A = np.vstack((A, Ai))
    A = A[1:, :]
    U, S, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape((3,4)) #???
    K_inv = np.linalg.inv(K)
    R = K_inv @ P[:, 0:3]
    t = -(K_inv @ P[:, 3])
    
    # SVD Cleanup
    U, D, Vt = np.linalg.svd(R)
    R = U @ Vt
    t = t / D[0]
    if(abs(np.linalg.det(R)-(-1)) < 0.001):  # Accounting for float math, basically checking for det(R)=0
        R = -R
        t = -t
    return R, t.reshape((3,1))


