import argparse
import glob
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
# import imageio
import torch
import matplotlib.pyplot as plt

import numpy as np
import cv2

from scipy.spatial.transform import Rotation as R
import Utilities as util

from NeRFModel import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

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
    t_mat = util.transform_from_pose(pose)
    R = t_mat[0:3, 0:3]
    x = np.array([[pixelPosition[0]], [pixelPosition[1]], [1]])
    direction = np.linalg.inv(camera_info["K"] @ R) @ x
    util.show_ray(t_mat, pose, direction)
    exit(1)


def generateBatch(sample_space, images, poses, camera_info, args):
    """
    Input:
        sample_space: the indices of all pixels that can still be sampled
        images: all images in dataset
        poses: corresponding camera pose in world frame
        camera_info: image width, height, camera matrix
        args: get batch size related information
    Outputs:
        A set of rays
    """

    # given a set of camera images, choose a random set of pixels from dataset 
    sample_indices = np.random.choice(sample_space.shape[0], size=args.n_rays_batch, replace=False)
    samples = sample_space[sample_indices]
    new_sample_space = np.delete(sample_space, sample_indices)
    cam_index_helper = camera_info["W"] * camera_info["H"]
    print(len(new_sample_space))

    # Returned obj creations
    ground_truths = [] # list of pixel RBG values
    ray_origins = []
    ray_directions = []


    for index in samples:
        # each index is a pixel on one of the images, we just need to extract the image, row, col indices and get its ray, 
        camera_index = index // cam_index_helper # index of camera from 0-99
        remainder = index % cam_index_helper # index of pixel within image
        v = remainder // camera_info["W"]
        u = remainder % camera_info["W"]
        if v > 799 or u > 799 or v < 0 or u < 0 or camera_index > 99 or camera_index < 0:
            raise RuntimeError(f"Bad camera index or pixel encountered: Image: {camera_index}, Pixel: ({u},{v})")

        ground_truths.append(images[camera_index][v,u])
        ray_o, ray_d = PixelToRay(camera_info, poses[camera_index]["camera_pose"], (u,v), args)
        # turn pixels to 


    return ray_origins, ray_directions, ground_truths, new_sample_space


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

    NUM_EPOCHS = 30
    MAX_ITER = args.max_iters
    BATCH_SIZE = args.n_rays_batch
    # total amount of data
    data_total  = camera_info["W"] * camera_info["H"] * len(images)
    batch_iterations = min(data_total / BATCH_SIZE, MAX_ITER)

    # Init NeRF Model
    # NOTE: with hierarchical sampling, we actually optimze two models at the same time...
    model = NeRFmodel(60, 24, False)
    # Init Optimizer
    #NOTE: Paper includes a decaying lr.  
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=[0.9, 0.999], eps=1e-7) 

    for i in tqdm(range(NUM_EPOCHS)):
        epoch_loss = 0
        # all pixels in image set arranges as indices.
        samples = np.linspace(0, data_total-1, num=data_total, dtype=np.int64)
        for j in tqdm(range(batch_iterations)):
            # generate batch
            ray_origins, ray_directions, ground_truths, samples = generateBatch(samples, images, poses["pose_list"], camera_info, args)
            exit(1)
            rgb = render()
            mse_loss = loss()
            
            optimizer.zero_grad()
            mse_loss.backwards()
            optimizer.step()

            epoch_loss += mse_loss.item()
            # tabulate trainning error
        
        """ Validation step"""
        with torch.no_grad():
            pose_batch, direction_batch = generateBatch(...)
            rgb = render()
            val_mse_loss = loss().item()
        
        print(f"Validation Loss: {val_mse_loss}, Training Loss: {epoch_loss}")
        # save batch
        SaveName = args.checkpoint_path + str(i) + "model.ckpt"

        torch.save(
            {
                "epoch" : i,
                "coarse_model_state_dict" : model.state_dict(),
                # "fine_model_state_dict" : model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
                "loss" : mse_loss,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")
    pass

def test(images, poses, camera_info, args):
    pass

def main(args):
    # load data
    print("Loading data...")
    camera_info, poses, images, depth_images = util.loadDataset(args.data_path, args.mode)
    # util.show_camera_frames(poses)
    # exit(1)
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
    parser.add_argument('--n_rays_batch',default=4096,help="number of rays per batch")
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