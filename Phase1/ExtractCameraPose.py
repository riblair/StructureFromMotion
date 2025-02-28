import numpy as np

""" 
THIS FUNCTION CONFIRMED WORKING!!! TESTED USING CV2. DO NOT DEBUG, YOU ARE WASTING YOUR TIME.
"""

# E == Essential Matrix
def extract_camera_pose(E: np.ndarray, K: np.ndarray):
    # Given from assignment
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    # E = UDV^T
    U, __, Vt = np.linalg.svd(E)

    t = np.reshape(U[:,2], (3,1))

    R1 = np.matmul(np.matmul(U, W), Vt)
    R2 = np.matmul(np.matmul(U, W.transpose()), Vt)

    # print(np.linalg.det(R1))
    if(abs(np.linalg.det(R1)-(-1)) < 0.001):
        R1 = -R1
        t = -t
    # print(np.linalg.det(R2))
    if(abs(np.linalg.det(R2)-(-1)) < 0.001):
        R2 = -R2
        t = -t

    P1 = K @ R1 @ np.hstack((np.eye(3), t))
    P2 = K @ R2 @ np.hstack((np.eye(3), t))
    P3 = K @ R1 @ np.hstack((np.eye(3), -t))
    P4 = K @ R2 @ np.hstack((np.eye(3), -t))
    return [P1,P2,P3,P4]
    # c_list = [c1, c2, c3 ,c4]
    # R_list = [R1, R2, R3, R4]
    # return R1, R3, c1
