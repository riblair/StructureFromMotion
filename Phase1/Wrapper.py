import copy
import logging
import cv2
import numpy as np
import Utilities as util
import matplotlib
matplotlib.use("tkagg")

import EstimateFundamentalMatrix as EFM
import GetInlierRANSANC as GIR
import EssentialMatrixFromFundamentalMatrix as EMFFM
import ExtractCameraPose as ECP
import LinearTriangulation as LT
import DisambiguateCameraPose as DCP
import NonlinearTriangulation as NLT
from Pixel import Pixel, Coordinate
import PnPRANSAC as PnP
import NonlinearPnP as NLPnP

logger = logging.getLogger(__name__)

def main():
    Args = util.setup_environment()
    DataPath = Args.DataPath

    """Parsing the data"""
    images, image_names = util.load_images(DataPath, -1, cv2.IMREAD_ANYCOLOR)
    match_dictionaries = util.parse_matching_txt(DataPath)
    k_Mat = util.parse_Camera_Instrinsics(DataPath)

    """Estimating F matrix between two images"""

    # BAD MATCHES:  1-3, 1-4, 1-5, 2-5, 3-4 (PAIN),
    # Good matches: 1-2, 2-3, 2-4, 3-5, 4-5
    inlier_pair_dict = dict()
    F_dict = dict()
    for i in range(1,len(images)):
        for j in range(i+1, len(images)+1):
            F, inlier_coorespondances = GIR.getInlierRANSAC(match_dictionaries[(i,j)])
            inlier_pair_dict[(i,j)] = inlier_coorespondances
            F_dict[(i,j)] = F

    first_pair = (1,2)
    # F, first_inlier_set = GIR.getInlierRANSAC(match_dictionaries[first_pair])
    F, first_inlier_set = F_dict[first_pair], inlier_pair_dict[first_pair]

    # pair_lines = []
    # for key,value in first_inlier_set.items():
    #     pair_lines.append((key, value))
    # EFM.visualizeEpipolarLines(images[first_pair[0]-1], F, pair_lines)
    
    print(f"Percentage of inliers found: {round(100*len(first_inlier_set)/len(match_dictionaries[first_pair]))}%")
    # F2 = EFM.estimate_F2(match_dictionaries[first_pair])
    # EFM.visualizeEpipolarLines(images[first_pair[0]-1], F2, pair_lines)
    # exit(1)
    """Estimate Essential Matrix"""
    e_Mat = EMFFM.getEssentialFromF(F,k_Mat)

    """Extract Pose from Essential Matrix"""
    R1, R2, t = ECP.extract_camera_pose(e_Mat)

    T1 = R1 @ np.hstack((np.eye(3), t))
    T2 = R2 @ np.hstack((np.eye(3), t))
    T3 = R1 @ np.hstack((np.eye(3), -t))
    T4 = R2 @ np.hstack((np.eye(3), -t))

    P1 = k_Mat @ T1
    P2 = k_Mat @ T2
    P3 = k_Mat @ T3
    P4 = k_Mat @ T4

    t_list = [T1, T2, T3, T4]
    p_list = [P1, P2, P3, P4]

    """Linear Triangulation"""
    x_set_list = []
    # x_set2_list = []
    P_identity = k_Mat @ np.hstack((np.eye(3), np.zeros((3,1))))
    for i in range(4):
        x_set = LT.linear_triangulation(P_identity, p_list[i], first_inlier_set)
        # x_set2 = LT.cv2triangulate(P_identity, p_list[i], first_inlier_set)
        x_set_list.append(x_set)
    # LT.visualize_ambiguity(x_set_list)
    best_t, best_x_set = DCP.disambiguate_camera_pose(t_list, x_set_list)
    best_pose = k_Mat @ best_t
    # When Projecting points onto image 1, use keys and Identity. 
    # When projecting points onto image 2, use values and p_list[i]
    # LT.visualize_triangulation(images[first_pair[0]-1], list(first_inlier_set), best_x_set, P_identity)
    # util.draw_pointcloud2D(best_x_set, [np.eye(3), best_t[:,0:3]], [np.zeros((3,1)), -best_t[:, 3]])
    """Non-Linear Optimization of correspondances"""
    X, p1_to_X, p2_to_X = NLT.nonlinear_triangulation(P_identity, best_pose, best_x_set, first_inlier_set)
    # LT.visualize_triangulation(images[first_pair[0]-1], list(first_inlier_set), X, P_identity)
    # util.draw_pointcloud2D(X, [np.eye(3), best_t[:,0:3]], [np.zeros((3,1)), -best_t[:, 3]])
    print(f"LINEAR ERROR: {NLT.calc_error(P_identity, best_pose, best_x_set, first_inlier_set)}")
    print(f"NON-LINEAR ERROR: {NLT.calc_error(P_identity, best_pose, X, first_inlier_set)}")
    print(f"R: \n{best_t[0:3,0:3]}")
    print(f"t: \n{-best_t[:, 3]}")
    # usually around [0.85823854 0.15040683 0.49071824] for 1-2
    # usually around [0.82190836 0.16795243 0.54429645] for 2-3 ...
    # usually around [0.9311055  0.03814694 0.36274971] for 2-4... [0.96525889 0.04845211 0.25676384]
    # usually around [ 0.68206627 -0.1470562   0.71635193] for 3-5 [0.72168504 0.11093733 0.68327418]
    # usually around [0.59251233 0.17518985 0.7862809 ] for 4-5
    ########################
    # Rest of Images
    ########################

    """Building relevent data structures for assembly"""
    R_set = [np.eye(3), best_t[:, 0:3]]
    C_set = [np.zeros((3,1)), -best_t[:,3]]
    
    poses = []
    poses.append(P_identity)
    poses.append(best_pose)

    pixel_world_mappings = []
    pixel_world_mappings.append(p1_to_X)
    pixel_world_mappings.append(p2_to_X) # the same thing 


    ## THIS SHOULD FAIL BECAUSE WE CANNOT DO THE INTERSECTION OF PIXELS FOR SOME REASON... INVESTIGATE
    for i in range(3,len(images)+1):
        #Dictionary containing pixels in new image that HAVE a corresponding world point already calculated
        pix_new_world_mapped = util.build_all_correspondences(i, pixel_world_mappings, inlier_pair_dict)

        R_new, C_new = PnP.linear_pnp_RANSAC(pix_new_world_mapped, k_Mat)
        print(f"PnP - R: \n{R_new}")
        print(f"PnP - C: \n{C_new}")
        P_new = k_Mat @ np.hstack((R_new, -C_new))

        # R_new, C_new = NLPnP.nonlinear_pnp(R_new, C_new, pix_new_world_mapped, k_Mat)
        # P_best = NLPnP.nonlinear_pnp_2(P_new,pix_new_world_mapped,k_Mat)
        print(f"NLPnP - R: \n{R_new}")
        print(f"NLPnP - C: \n{C_new}")
        R_set.append(R_new)
        C_set.append(C_new)
        # cv2.solvePnP() # should investigate...
        # Add new 3D points
        # Dict of pixels that DO NOT have a currently calculated world point
        pix_old_pix_new_unmapped = util.build_pix_pix_correspondences_unmapped(pix_new_world_mapped, inlier_pair_dict[(i-1,i)])

        # WE NEED TO GET ALL OF THE CORRESPONDENCES BETWEEN pixels_new_unmapped and pixels_prev to pixels new_unmapped.
        # We should be very careful when making X_New NOT to recalculate points that have already been calculated. 
        # We dont want to run this on the inliers dictionary, but rather the subset of pixels that dont already have a correspondances


        X_new = LT.linear_triangulation(poses[-1], P_new, pix_old_pix_new_unmapped)  # Returns list[Coordinate]
        # LT.visualize_triangulation(images[i], list(pixel_correspondances.values()), X_new, P_new)
        X_new, __, pix_new_world_unmapped = NLT.nonlinear_triangulation(poses[-1], P_new, X_new, pix_old_pix_new_unmapped)
        util.draw_pointcloud2D(X_new, R_set, C_set)
        
        X.extend(X_new) # X and X_New are garunteed to be discrete sets

        poses.append(P_new)
        pixel_world_mappings.append(pix_new_world_unmapped)
        util.draw_colored_pc2d(pixel_world_mappings, R_set, C_set)

    # Build Visability Matrix

    # Bundle Adjustment

if __name__ == '__main__':
    main()
