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

