import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import Utilities as util
from Pixel import *
from scipy.spatial.transform import Rotation as R
from scipy.sparse import csr_matrix

""" 
NEW IDEA:
Loop through list of X poses coming from params/X0.
Find all cameras that see that point
Run error function only on those cameras with that point
Append sum of error to residuals
"""


def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 4]
    k1 = camera_params[:, 5]
    k2 = camera_params[:, 6]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 7].reshape((n_cameras, 7))
    points_3d = params[n_cameras * 7:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()

def error(X0, n_cameras, K, V, V_points: list[Coordinate]):
    residuals = []
    for idx in range(len(V_points)):  # Gives idx for point, can find in V
        submatrix = V[V[:, 1] == idx]  # gets all rows with same point_idx
        point_3d = V_points[idx]
        for jdk in range(submatrix.shape[0]):
            real_camera_idx = submatrix[jdk, 0] - 1  # Image 1 has camera idx of 1, this accounts for 0 index.
            pixel_u = submatrix[jdk, 2]
            pixel_v = submatrix[jdk, 3]
            start = int(real_camera_idx)*7  # i-1 accounts for 0-index
            end = start + 7
            camera_params = X0[start:end]
            q = camera_params[0:4]  # first 4 elements
            R = util.quaternion_to_r(q)
            C = camera_params[4:].reshape((3,1))
            P = K @ np.hstack((R, -C))
            proj_point = util.reproject_point(P, point_3d)
            residuals.append(pixel_u - proj_point.u)
            residuals.append(pixel_v - proj_point.v)
    residuals = np.array(residuals).flatten()
    return residuals
    

# def error(X0, n_cameras, K,  V, V_points):
#     # 7 params for each camera
#     errors = []
#     for i in range(1, int(n_cameras)+1):
#         start = (i-1)*7  # i-1 accounts for 0-index
#         end = start + 7
#         camera_params = X0[start:end]
#         q = camera_params[0:4]  # first 4 elements
#         R = util.quaternion_to_r(q)
#         C = camera_params[4:].reshape((3,1))
#         P = K @ np.hstack((R, -C))
#         submatrix = V[V[:, 0] == i]
#         # Column 1 is camera_indicies (same as i)
#         # Column 2 is point_indicies for V_points
#         # Columns 3 and 4 are the 2D pixel correspondance (u,v) with V_points[Column 2]
#         for j in range(submatrix.shape[0]):  # Loop through every row
#             point_3d_idx = int(submatrix[j, 1])
#             point_3d = V_points[point_3d_idx]
#             pixel_u = submatrix[j, 2]
#             pixel_v = submatrix[j, 3]
#             reproj_point = util.reproject_point(P, point_3d)
#             reproj_error = (pixel_u - reproj_point.u)**2 + (pixel_v - reproj_point.v)**2
#             errors.append(reproj_error)
#     errors = np.array(errors)
#     print(f"Shape of residuals: {errors.shape}")
#     return errors.flatten()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = int(camera_indices.size * 2)
    n = int(n_cameras * 7 + n_points * 3)
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(7):
        A[2 * i, camera_indices * 7 + s] = 1
        A[2 * i + 1, camera_indices * 7 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 7 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 7 + point_indices * 3 + s] = 1

    return A

def bundle_adjustment(C_set, R_set, X: list[Coordinate], K: np.ndarray, V: np.ndarray, V_points: list[Coordinate]):
    """Performs bundle adjustment on the pointcloud

    Args:
        C_set (list[np.ndarray]): 3x1 vector describing camera's 3D world pose
        R_set (list[np.ndarray]): 3x3 rotation matrix for each camera
        X (list): Contains initial estimates of point coordinates in the world frame.
        K (np.ndarray): Intrinsic parameters of the camera
        V (np.ndarray): Visibility Matrix

    Returns:
        _type_: _description_
    """
    # First, convert each rotation matrix into a quaternion so it is easier to represent during optimization.
    q_set = []
    for R in R_set:
        q_set.append(util.r_to_quaternion(R))
    
    x_set = []
    for x in X:
        x_set.append(x.to_arr().ravel())  # 1x3 vector. Purposefully not homogenous.

    # Convert list[Pixel] into nx2 matrix.
    points_2d = np.hstack((V[:, 2], V[:, 3]))

    params = np.hstack((np.array(q_set).ravel(), np.array(C_set).ravel(), np.array(x_set).ravel()))
    
    n_cameras = V[-1, 0]
    n_points = len(V_points)  # Columns = number of points
    n = 9 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]

    
    # Weird function but basically removes a ton of 0s from the visibility matrix and makes it faster to
    # compute. Basically, it becomes a matrix with only 2 columns. The first represents the camera that sees
    # the point, and the second row represents the index of the roow itself. In Wrapper, you can see a
    # list named 'V_coordinates'. If you were to take an index from point_indices to use on V_coordinates,
    # you would get the actual 3D point. 
    # https://numpy.org/doc/2.1/reference/generated/numpy.nonzero.html
    # https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html 
    camera_indices = V[:, 0]
    point_indices = V[:, 1]
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    
    print(f"Shape of A (Jacobian Matrix): {A.shape}")
    print(f"Shape of observed_2d_points: {V[:, 2:].shape}")
    print(f"Shape of camera_indices: {camera_indices.shape}")
    print(f"Shape of point_indices: {point_indices.shape}")
    expected_params = 4 * n_cameras + 3 * n_cameras + 3 * n_points
    print(f"Expected total number of parameters: {expected_params}")
    print(f"Number of Camera Parameters: {4 * n_cameras + 3 * n_cameras}")
    print(f"Number of Points: {n_points}")
    # def error(X0, n_cameras, K,  V, V_points):
    A_csr = csr_matrix(A)
    print(f"Shape of A_csr (Jacobian Matrix): {A_csr.shape}")
    print(f"Shape of input params: {params.shape}")
    result = least_squares(error, params, method='trf', jac_sparsity=A_csr,
                           args=(n_cameras, K, V, V_points))
    return C_set, R_set, X


