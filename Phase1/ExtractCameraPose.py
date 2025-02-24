import numpy as np

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

    c1 = np.reshape(U[:,2], (3,1))
    c2 = -np.reshape(U[:,2], (3,1))
    c3 = np.reshape(U[:,2], (3,1))
    c4 = -np.reshape(U[:,2], (3,1))

    R1 = np.matmul(np.matmul(U, W), Vt)
    R2 = np.matmul(np.matmul(U, W), Vt)
    R3 = np.matmul(np.matmul(U, W.transpose()), Vt)
    R4 = np.matmul(np.matmul(U, W.transpose()), Vt)

    print(np.linalg.det(R1))
    if(abs(np.linalg.det(R1)-(-1)) < 0.001):
        R1 = -R1
        c1 = -c1
    print(np.linalg.det(R2))
    if(abs(np.linalg.det(R2)-(-1)) < 0.001):
        R2 = -R2
        c2 = -c2
    print(np.linalg.det(R3))
    if(abs(np.linalg.det(R3)-(-1)) < 0.001):
        R3 = -R3
        c3 = -c3
    print(np.linalg.det(R4))
    if(abs(np.linalg.det(R4)-(-1)) < 0.001):
        R4 = -R4
        c4 = -c4
    P1 = np.matmul(np.matmul(K, R1), np.hstack((np.eye(3), -c1)))
    P2 = np.matmul(np.matmul(K, R2), np.hstack((np.eye(3), -c2)))
    P3 = np.matmul(np.matmul(K, R3), np.hstack((np.eye(3), -c3)))
    P4 = np.matmul(np.matmul(K, R4), np.hstack((np.eye(3), -c4)))
    return [P1,P2,P3,P4]
    # c_list = [c1, c2, c3 ,c4]
    # R_list = [R1, R2, R3, R4]
    # return c_list, R_list
