import numpy as np
import cv2
import os
import csv
import logging
from Pixel import Pixel
logger = logging.getLogger(__name__)


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
                    
                    current_pixel = Pixel(pixel_RGB, u_src, v_src)
                    for i in range(num_matches-1):
                        dst_IDX = int(row[6+i*3])
                        u_dst = float(row[(7+i*3)])
                        v_dst = float(row[(8+i*3)])
                        dict_key = (src_IDX, dst_IDX)
                        master_dictionary[dict_key][current_pixel] = Pixel(pixel_RGB, u_dst, v_dst)
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
    out = np.array([
        [0, w[0,0], -w[1,0]],
        [-w[0,0], 0, w[2,0]],
        [w[1,0], -w[2,0], 0]
    ], dtype=np.float32)
    return out
