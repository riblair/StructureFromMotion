import numpy as np
import cv2
from Pixel import Pixel
import Utilities as util
import copy
import matplotlib
matplotlib.use("tkagg")

"""
    `eight_point_pair` is a list of length 8 with tuples of pixels.
"""
def estimate_F(eight_point_pair: list):
    if len(eight_point_pair) != 8:
        raise("[estimate_F]: Wrong sized list given for Estimating Fundamental Matrix")

    A_mat = np.zeros((8,9), dtype=np.float32)

    for i in range(0, 8):
        point_tup = eight_point_pair[i]
        A_mat[i, :] = np.array([
            point_tup[0].u * point_tup[1].u,
            point_tup[0].v * point_tup[1].u,
            point_tup[1].u,
            point_tup[0].u * point_tup[1].v,
            point_tup[0].v * point_tup[1].v,
            point_tup[1].v,
            point_tup[0].u,
            point_tup[0].v,
            1])
        
    U, S, Vt = np.linalg.svd(A_mat)

    S_prime = np.zeros((8,9), dtype=np.float32)
    S_prime[0,0] = S[0]
    S_prime[1,1] = S[1]
    S_prime[2,2] = S[2]
    S_prime[3,3] = S[3]
    S_prime[4,4] = S[4]
    S_prime[5,5] = S[5]
    S_prime[6,6] = S[6]
    S_prime[7,7] = S[7]

    F_prime = U @ S_prime @ Vt

    U, S, Vt = np.linalg.svd(F_prime)
    F = np.reshape(Vt[-1, :], (3,3))
    f_rank = np.linalg.matrix_rank(F)
    return F

def estimate_F2(match_dict: dict):
    points1, points2 = util.pointlist_from_dict(match_dict)

    points1 = np.array(points1)
    points2 = np.array(points2)
    F, __ = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, 0.1, 0.99)
    return F

def visualizeEpipolarLines(img: np.ndarray, F: np.ndarray, pixel_correspondances: list):
    """Visualizes the epipolar lines from stereo camera using the fundamental matrix F.

    Args:
        img (_type_): The image to display the epipolar lines on. If F goes from image 1 to image 2, img should be image 2.
        F (np.ndarray): Fundamental Matrix
        points (list): List of Pixel correspondances between image 1 and image 2
        
    Details:
        We draw the point correspondance from image 2 onto image 1. Since they are correspondances.
        We use the point correspondances from image 1 to calculate the epipolar lines for image 2.
        Since the Fundamental matrix relates the geometries from image 1 into image 2, we are multiplying
        the correspondances from image 1 with F, which results in the epipolar lines created by image 2 from
        the perspective of image 1, which can be drawn onto image 1.
        
        Alternatively, if we wanted to visualize the actual point correspondances from image one with the
        epipolar lines by image 2, we would have to draw all points from image 1, then multiply the points
        from image 2 by the transpose of the Fundamental matrix (F^T). Both options are valid, but this
        function visualizes the latter.
    """
    image = copy.deepcopy(img)
    row, col, depth = image.shape
    # F = np.transpose(F)
    
    for point in pixel_correspondances:
        point_from_image_1 = point[0].to_arr(homogenous=True, dtype=np.uint16)
        point_from_image_2 = point[1].to_arr(homogenous=True, dtype=np.uint16)
        cv2.circle(image, (int(point_from_image_1[0]), int(point_from_image_1[1])), radius=3, color=(0, 255, 255), thickness=-1)
        line=np.dot(np.transpose(F), point_from_image_2)  # Returns coeffs for ax + by + c = 0
        
        # Drawing the epipolar line. We need to essentially draw from the left edge to the right edge.
        # So the first point is on the y intercept (x=0). The second point is when x is the width of the image.
        a = line[0]
        b = line[1]
        c = line[2]
        
        x1 = 0
        y1 = int(-c/b)  # Make this an int since we are drawing on an image (discrete).
        point1 = (x1, y1)
        
        x2 = col  # Image width
        y2 = int(-a*x2/b - c/b)
        point2 = (x2, y2)

        cv2.line(image, point1, point2, (255, 100, 255), 1)
    cv2.imshow("Epipolar Lines", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return
