import argparse
import glob
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
import imageio
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import json
from scipy.spacial.transform import Rotation as R

from NeRFModel import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

def loadDataset(data_path, mode):
    """
    Input:
        data_path: dataset path
        mode: train or test
    Outputs:
        camera_info: image width, height, camera matrix 
        images: images
        pose: corresponding camera pose in world frame
    """
    image_width = 100
    image_height = 100
    K = np.array([[1, 0, image_width/2], [0, 1, image_height/2], [0, 0, 1]])
    im_path = data_path+mode+'/'
    transforms_file_name = "transforms_"+ mode + ".json"
    # list of dictionaries...
    # Turn Transformation matrix into Camera Pose 
        # "pose": Camera Pose can be a (6,1) => [x, y, z, r, p, y]^T
        # "idx": Image index (number appended to file name)
    
    with open(transforms_file_name, 'r') as fp:
        data = json.load(fp)
        
    camera_angle_x = data["camera_angle_x"]
    

    poses = dict()
    poses["camera_angle_x"] = camera_angle_x
    pose_list = []

    for frame in data["frames"]:
        pose_dictionary = dict()
        yaw = frame["rotation"] # NOTE: we are assuming the "rotation" is the yaw value...
        t_mat = np.array([frame["transform_matrix"]])
        r_mat = t_mat[0:3, 0:3]

        rot = R.from_matrix(r_mat)

        roll_r,pitch_r,yaw_r = rot.as_euler('xyz')

        # COMPARE yaw and yaw_r
        print(f"Given Yaw? {yaw}, Euler yaw {yaw_r}")
        exit(1)
        pose_dictionary["camera_pose"] = np.array([t_mat[0,3], t_mat[1,3], t_mat[2, 3], roll_r, pitch_r, yaw]).T
        pose_dictionary["idx"] = int(frame["file_path"].split("_")[1])
        pose_list.append(pose_dictionary)
        # use rot matrix to get euler angles...
        
    poses["pose_list"] = pose_list
    # Two images lists:
    # 1. RGB images (used by all modes)
    # 2. Depth images (used by only test mode, None otherwise)

    images = []
    depth_images = [] if mode == 'test' else None
    filenames = os.listdir(im_path)

    i = 0
    while( i < len(filenames)):
        # NOTE: If there actually was a file that wasn't png format, then directly removing it
        # will skip the next index. If two .jpg files were next to each other, this would
        # only remove 1 .jpg file.
        if ".png" not in filenames[i]:
            filenames.remove(filenames[i]) 
        else:
            i+=1

    for filename in filenames:
        full_image_path = im_path+filename
        image = cv2.imread(full_image_path)
        if "depth" in filename:
            if depth_images is None:
                raise ValueError(f"""Depth image was found in dataset, but input mode is 
                                 not test. Mode is currently {mode} instead""")
            depth_images.append(image)
        else:
            images.append(image)

    return poses, images, depth_images

def PixelToRay(camera_info, pose, pixelPosition, args):
    """
    Input:
        camera_info: image width, height, camera matrix 
        pose: camera pose in world frame
        pixelPoition: pixel position in the image
        args: get near and far range, sample rate ...
    Outputs:
        ray origin and direction
    """

def generateBatch(images, poses, camera_info, args):
    """
    Input:
        images: all images in dataset
        poses: corresponding camera pose in world frame
        camera_info: image width, height, camera matrix
        args: get batch size related information
    Outputs:
        A set of rays
    """

def render(model, rays_origin, rays_direction, args):
    """
    Input:
        model: NeRF model
        rays_origin: origins of input rays
        rays_direction: direction of input rays
    Outputs:
        rgb values of input rays
    """

def loss(groundtruth, prediction):
    pass

def train(images, poses, camera_info, args):
    pass

def test(images, poses, camera_info, args):
    pass

def main(args):
    # load data
    print("Loading data...")
    images, poses, camera_info = loadDataset(args.data_path, args.mode)

    if args.mode == 'train':
        print("Start training")
        train(images, poses, camera_info, args)
    elif args.mode == 'test':
        print("Start testing")
        args.load_checkpoint = True
        test(images, poses, camera_info, args)

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default="./Phase2/Data/lego/",help="dataset path")
    parser.add_argument('--mode',default='train',help="train/test/val")
    parser.add_argument('--lrate',default=5e-4,help="training learning rate")
    parser.add_argument('--n_pos_freq',default=10,help="number of positional encoding frequencies for position")
    parser.add_argument('--n_dirc_freq',default=4,help="number of positional encoding frequencies for viewing direction")
    parser.add_argument('--n_rays_batch',default=32*32*4,help="number of rays per batch")
    parser.add_argument('--n_sample',default=400,help="number of sample per ray")
    parser.add_argument('--max_iters',default=10000,help="number of max iterations for training")
    parser.add_argument('--logs_path',default="./logs/",help="logs path")
    parser.add_argument('--checkpoint_path',default="./Phase2/example_checkpoint/",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=True,help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter',default=1000,help="num of iteration to save checkpoint")
    parser.add_argument('--images_path', default="./image/",help="folder to store images")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)