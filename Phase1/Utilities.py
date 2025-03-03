import argparse
import copy
import csv
import cv2
from datetime import datetime
import numpy as np
import os
import logging
from Pixel import Pixel, Coordinate
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def setup_environment():
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--LoggingPath",
        default="Phase1/Logs/",
        type=str,
        help="Path for the Logging files to be created in. Default: Phase1/Logs/"
    )
    Parser.add_argument(
        "--DataPath",
        default="Phase1/P2Data/",
        type=str,
        help="Path for the image / matches and calibration data. default: Phase1/P2Data/",
    )
    Parser.add_argument(
        "--OutputPath",
        default="Output/",
        type=str,
        help="Path for the outputs default: Output/"
    )
    Parser.add_argument(
        "--DebugLevel",
        default="INFO",
        type=str,
        help="Path for the outputs default: Output/"
    )
    Args = Parser.parse_args()

    os.makedirs(Args.LoggingPath, exist_ok=True)
    os.makedirs(Args.OutputPath, exist_ok=True)

    if Args.DebugLevel == "INFO":
        DebugLevel = logging.INFO
    elif Args.DebugLevel == "WARNING":
        DebugLevel = logging.WARNING
    elif Args.DebugLevel == "DEBUG":
        DebugLevel = logging.DEBUG
    elif Args.DebugLevel == "CRITICAL":
        DebugLevel = logging.CRITICAL
    elif Args.DebugLevel == "ERROR":
        DebugLevel = logging.ERROR
    else:
        print(f"Unknown debug level {Args.DebugLevel}.\n Defaulting to INFO\n")
        DebugLevel = logging.INFO
    # This initializes the python logger.
    logging.basicConfig(filename=Args.LoggingPath+f"{datetime.now().strftime('%b_%d_%H:%M:%S')}.logging", level=DebugLevel)
    log = logging.getLogger()
    log.info(f"Beginning SfM")
    return Args

def load_images(im_path: str, num_images: int, flags: int = cv2.IMREAD_GRAYSCALE) -> tuple[list[cv2.Mat], list[str]]:
    images = []
    image_names = []
    filenames = os.listdir(im_path)
    i = 0
    while( i < len(filenames)):
        if ".png" not in filenames[i]:
            filenames.remove(filenames[i]) 
        else:
            i+=1

    filenames.sort(key=lambda x: int(x.split('.')[0]))  # Sort based on the number before the '.'
    count = 0
    for filename in filenames:
        full_image_path = im_path+filename
        image = cv2.imread(full_image_path, flags=flags)
        image_names.append(filename)
        images.append(image)
        count +=1
        if count == num_images:
            break
    return images, image_names

def parse_Camera_Instrinsics(file_dir: str) -> np.ndarray:
    for root, dirs, files in os.walk(file_dir, topdown=True, onerror=None, followlinks=False):
        for file in files:
            if "calibration" in file:
                with open(os.path.join(root,file), newline='') as csvfile:
                    reader = csv.reader(csvfile, delimiter=' ')
                    Camera_Calib = np.ndarray((3,3), dtype=np.float32)
                    row_iter = 0
                    for row in reader:
                        Camera_Calib[row_iter,:] = np.array([row[0], row[1], row[2]], dtype=np.float32)
                        row_iter+=1
                    return Camera_Calib

def parse_matching_txt(file_dir: str):

    # Dict((Tuple),(Dict))
    # Tuple is im IDX, im IDX - Assume that im_IDX1 < im_IDX 2
    # Dict is D_IDX_IDX of matches with (Pixel, Pixel) 
    master_dictionary = dict() # (Tuple, Dictionary)
    im_count = 1
    for root, dirs, files in os.walk(file_dir, topdown=True, onerror=None, followlinks=False):
        for file in files:
            if "matching" not in file:
                continue
            im_count +=1

    for i in range(1,im_count+1):
        for j in range(i+1,im_count+1):
            key = (i, j)
            master_dictionary[key] = dict()

    for root, dirs, files in os.walk(file_dir, topdown=True, onerror=None, followlinks=False):
        for file in files: 
            if "matching" not in file:
                continue
            with open(os.path.join(root,file), newline='') as csvfile:
                src_IDX = int(file[-5])
                reader = csv.reader(csvfile, delimiter=' ')
                num_features = 0
                for row in reader:
                    if len(row) == 2:  # This is the header
                        num_features = int(row[1])
                        continue

                    num_matches = int(row[0]) 
                    pixel_RGB = (int(row[1]),int(row[2]), int(row[3]))
                    u_src = float(row[4])
                    v_src = float(row[5])
                    
                    current_pixel = Pixel(u_src, v_src, RGB=pixel_RGB)
                    for i in range(num_matches-1):
                        dst_IDX = int(row[6+i*3])
                        u_dst = float(row[(7+i*3)])
                        v_dst = float(row[(8+i*3)])
                        dict_key = (src_IDX, dst_IDX)
                        master_dictionary[dict_key][current_pixel] = Pixel(u_dst, v_dst, RGB=pixel_RGB)
    return master_dictionary


def show_im_match_pair(image_pair: tuple[np.ndarray, np.ndarray], match_dict: dict, line:bool=False):
    # the dictionary should be the specific match_pair dictionary
    im1_shape = image_pair[0].shape
    print(im1_shape)
    new_im = np.hstack((image_pair[0], image_pair[1]))

    for key,value in match_dict.items():
        center1 = key.to_arr(typecast=np.int32)
        center1 = list(center1)
        center2 = value.to_arr(typecast=np.int32) + np.array([im1_shape[1], 0], dtype=np.int32)
        center2 = list(center2)
        cv2.circle(new_im, center1, 2, (0,0,255), -1)
        cv2.circle(new_im, center2, 2, (0,0,255), -1)
        if line: cv2.line(new_im, center1, center2, (0, 255, 255), 1)

    cv2.imshow("image", new_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def skew_sym(w: np.ndarray):
    if not(w.shape == (1,3) or w.shape == (3,1)):
        raise(f"Size Error, Expected (1,3) or (3,1). given: {w.shape}")
    out = np.array([
        [         0,   -w[2,0],   w[1,0]],
        [    w[2,0],        0,  -w[0,0]],
        [   -w[1,0],  w[0,0],        0]
    ], dtype=np.float32)
    return out

def reproject_point(p_mat: np.ndarray, X_coord: Coordinate) -> Pixel:
    if isinstance(X_coord, Coordinate):
        point_homogenous = X_coord.to_arr(homogenous=True)
    reproj_u = (p_mat[0,:] @ point_homogenous) / (p_mat[2, :] @ point_homogenous)
    reproj_v = (p_mat[1,:] @ point_homogenous) / (p_mat[2, :] @ point_homogenous)
    return Pixel(reproj_u, reproj_v)

def pointlist_from_dict(match_dict:dict, homogenous=False) -> tuple[list[np.ndarray], list[np.ndarray]]:
    points1 = []
    points2 = []
    for key,value in match_dict.items():
        points1.append(key.to_arr(homogenous=homogenous))
        points2.append(value.to_arr(homogenous=homogenous))
    return points1, points2

def r_to_quaternion(r_mat: np.ndarray):
    quat = R.from_matrix(r_mat).as_quat()
    return np.array(quat).reshape((4,1))

def quaternion_to_r(q: np.ndarray):
    r_mat = R.from_quat(q).as_matrix()
    return np.array(r_mat)

def draw_features_on_image(image: np.ndarray, feature_list: list[Pixel], second_list: list[Pixel]=None):
    im_copy = copy.deepcopy(image)
    for pix in feature_list:
        cv2.circle(im_copy, (int(pix.u), int(pix.v)), radius=3, color=(0, 0, 255), thickness=-1)

    if second_list:
        for pix in second_list:
            cv2.circle(im_copy, (int(pix.u), int(pix.v)), radius=2, color=(0, 255, 0), thickness=-1)
    cv2.imshow("sift features", im_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def build_pix_pix_correspondences_unmapped(pixel_j_world_j: dict[Pixel, Coordinate], pixel_i_pixel_j:dict[Pixel, Pixel]):

    # need to get values in pix_i_pix_j that are not in pixel_j_world_j 
    unmapped_dict = dict()
    for pix_i, pix_j in pixel_i_pixel_j.items():
        if pix_j not in pixel_j_world_j.keys():
            unmapped_dict[pix_i] = pix_j
    return unmapped_dict




    # pixels_new_unmapped = build_all_differences(idx_pair[1], pixel_j_world_j, pixel_pixel_mappings)

    # unmapped_dict = dict() # pix_i to pix_j unmapped
    # inlier_dict = pixel_pixel_mappings[idx_pair]

    # for pix_i, pix_j in inlier_dict.items():
    #     if pix_j in pixels_new_unmapped:
    #         unmapped_dict[pix_i] = pix_j


""" Function Builds a new pixel to world correspondnces pixel_j_to_world_j based on intersection of sets"""
def correspondences_intersection(pixel_i_world_i: dict[Pixel, Coordinate], pixel_i_pixel_j: dict[Pixel, Pixel]):
    # This grabs all of the pixels that exist in both the mapping of pixel to world points, and pixels in image i to pixels in im j 
    mapped_pixels_i_correpondences = set(pixel_i_world_i.keys()) & set(pixel_i_pixel_j.keys())
    pixel_j_world_j = dict()
    for pix_i in mapped_pixels_i_correpondences:
        world_j_mapping = pixel_i_world_i[pix_i]
        pixel_j_mapping = pixel_i_pixel_j[pix_i]
        pixel_j_world_j[pixel_j_mapping] = world_j_mapping
    return pixel_j_world_j

def correspondences_union(pixel_world: dict[Pixel, Coordinate], new_pixel_world: dict[Pixel, Coordinate]):
    combined_pixel_world = copy.deepcopy(pixel_world) # Copying ensure old pixel_world dict is unaffected THIS MIGHT BE A BAD IDEA, AS DEEP COPYS MAY CAUSE FUTURE SET STUFF TO FAIL!
    for pixel, world_p in new_pixel_world.items():
        if pixel not in pixel_world:
            combined_pixel_world[pixel] = world_p
        else:
            print(f"Difference of Xs: {pixel_world[pixel]-world_p}")
    return combined_pixel_world

# """Used for generating all unmapped pixels in new image"""
# def build_all_differences(new_im_index: int, pixel_j_world_j: dict[Pixel, Coordinate], pixel_pixel_mappings:dict[tuple[int,int], dict[Pixel, Pixel]]):
#     pixel_pixel_j_mappings = set()
#     for iter in range(1, new_im_index): # extend the list 
#         pixel_pixel_j_mappings |= set(pixel_pixel_mappings[(iter,new_im_index)].keys()) # Values???
#     unmapped_pixels = pixel_pixel_j_mappings-set(pixel_j_world_j.keys())
#     return unmapped_pixels

"""Function builds ALL pixel_j_world_j mappings from the list of pixel_word and pixel_pixel dictionaries"""
def build_all_correspondences(new_im_index: int, pixel_world_mappings: list[dict[Pixel, Coordinate]], pixel_pixel_mappings:dict[tuple[int,int], dict[Pixel, Pixel]]):
    pixel_j_world_j = dict()
    pixel_pixel_j_mappings = []

    for iter in range(1, new_im_index):
        pixel_pixel_j_mappings.append(pixel_pixel_mappings[(iter,new_im_index)])

    if len(pixel_world_mappings) != len(pixel_pixel_j_mappings): # one less as p1_w1 and p2_w2 are the same, so we exclude the latter
        raise RuntimeError(f"Length of pixel_world+1 and pixel_pixel mappings should agree. p_w:{len(pixel_world_mappings)}, p_p:{len(pixel_pixel_j_mappings)}")

    for pixel_i_world_i, pixel_i_pixel_j in zip(pixel_world_mappings, pixel_pixel_j_mappings):
        new_correspondences = correspondences_intersection(pixel_i_world_i, pixel_i_pixel_j)
        pixel_j_world_j = correspondences_union(pixel_j_world_j, new_correspondences)

    return pixel_j_world_j

def draw_pointcloud3D(X, R_set, C_set):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    color_maps = ['black', 'red', 'orange', 'green', 'blue']
    iterator = 0
    for R,C in zip(R_set, C_set):
        ax.scatter(C[0],C[1], C[2], c=color_maps[iterator], marker="^", label=f"Camera {iterator+1} Location")
        iterator+=1
    
    
    x_list = []
    y_list = []
    z_list = []

    for point in X:
        x_list.append(point.x)
        y_list.append(point.y)
        z_list.append(point.z)
    ax.scatter(x_list, y_list, z_list, s=0.5)

    
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])
    ax.set_zlim([-20, 20])

    ax.set_title("Point cloud of Unity Hall")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.legend()
    plt.show()


def draw_pointcloud2D(X, R_set, C_set):

    fig = plt.figure()
    ax = fig.add_subplot()
    
    x_list = []
    z_list = []

    for point in X:
        x_list.append(point.x)
        z_list.append(point.z)
    ax.scatter(x_list, z_list, s=0.5)

    for R,C in zip(R_set, C_set):
        ax.scatter(C[0], C[2], c='Blue', marker="^")
    
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])

    plt.show()


def draw_colored_pc2d(pix_world_mappings:list[dict[Pixel, Coordinate]], R_set, C_set):

    fig = plt.figure()
    ax = fig.add_subplot()
    
    ax.scatter(0, 0, c='black', marker="^", label=f"Camera {1} Origin")
    color_maps = ['red', 'orange', 'green', 'blue']

    for i in range(len(pix_world_mappings)-1):
        x_list = []
        z_list = []
        value_list = pix_world_mappings[i+1].values()
        for point in value_list:
            x_list.append(point.x)
            z_list.append(point.z)
        ax.scatter(x_list, z_list, c=color_maps[i], s=1, label=f"Camera {i+2} points")
        
        ax.scatter(C_set[i+1][0], C_set[i+1][2], c=color_maps[i], marker="^", label=f"Camera {i+2} location")
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])
    ax.set_title("Point cloud of Unity Hall")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.legend()
    plt.show()

