import argparse
# import glob
import math
from tqdm import tqdm
# import random
# from torch.utils.tensorboard import SummaryWriter
# import imageio
import torch
import matplotlib.pyplot as plt

import numpy as np
import cv2

from scipy.spatial.transform import Rotation as R
import Utilities as util

from NeRFModel import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(3)

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
    x_dir = (pixelPosition[0]- camera_info["W"]*0.5) / camera_info["focal_length"]
    y_dir = (pixelPosition[1] - camera_info["H"]*0.5) / camera_info["focal_length"]
    x = np.array([[x_dir], [y_dir], [1]]) # NOTE: unsure if we need to normalize or not...
    # x_norm = x / np.linalg.norm(x)
    direction = t_mat[0:3,0:3] @ x
    # print(direction)
    d_norm = direction / np.linalg.norm(direction)
    # util.show_ray(t_mat, pose, d_norm)
    # exit(1)
    return t_mat[0:3, -1], d_norm.flatten()

def generateBatch(sample_space, images, poses, camera_info, args, epoch_count):
    """
    Input:
        sample_space: the indices of all pixels that can still be sampled
        images: all images in dataset
        poses: corresponding camera pose in world frame
        camera_info: image width, height, camera matrix
        args: get batch size related information
        epoch_count: the current epoch count.
    Outputs:
        A set of rays
    """

    # given a set of camera images, choose a random set of pixels from dataset 
    max = 799 if epoch_count>5 else 599
    min = 0 if epoch_count>5 else 200
    image_idxs = np.random.random_integers(0, 99, (args.n_rays_batch,1))
    us = np.random.random_integers(min, max, (args.n_rays_batch,1))
    vs = np.random.random_integers(min, max, (args.n_rays_batch,1))
    # Returned obj creations
    ground_truths = [] # list of pixel RGB values
    ray_origins = []
    ray_directions = []
    for i in range(0,args.n_rays_batch):
        u = us[i]
        v = vs[i]
        camera_index = image_idxs[i]
        
        gt_pixel = np.float32(images[camera_index][v,u]) # each element is (0-255)
        gt_pixel /= 255.0  # Normalizing the pixel before being inputted into the model
        ground_truths.append(gt_pixel)
        ray_o, ray_d = PixelToRay(camera_info, poses[camera_index]["camera_pose"], (u,v), args)
        ray_origins.append(ray_o)
        ray_directions.append(ray_d)

    ray_origins = np.array(ray_origins)
    ray_directions = np.array(ray_directions)
    ground_truths = torch.tensor(np.array(ground_truths)) # gets sent straight to the loss fcn, so needs to be a tensor
    return ray_origins, ray_directions, ground_truths

def volume_rendering(rgbs, sigmas, args):
    ##NOTE: as I needed to detach(), this methodology wont work...
    # will probably need to do the operations in place...
    """
    Input:
        ray_ts: linspace of ray distances 
        rgbs: flattened tensor of model RGB outputs [B*S 3] 
        sigmas: flattened tensor of model sigma outputs [B*S 1]
        args: additional params
    """
    ray_ts = np.linspace(args.near, args.far, args.n_sample)
    RGB_out = np.zeros((args.n_rays_batch, 3))

    for i in range(0,rgbs.shape[0], args.n_sample):
        T_sum = 0
        color_sum = np.zeros((1,3))
        ray_RGBS = rgbs[i:(i+args.n_sample), :] # [S 3]
        ray_Sigmas = sigmas[i:(i+args.n_sample), :] # [S 1]

        for j in range(len(ray_ts)-1):
            delta_j = ray_ts[j+1] - ray_ts[j]
            T_j = math.exp(-T_sum)
            a_j = 1 - math.exp(-ray_Sigmas[j]*delta_j)
            color_sum += T_j * a_j * ray_RGBS[j]
            T_sum += ray_Sigmas[j] * delta_j

        print(color_sum)
        RGB_out[int(i/args.n_sample), :] = color_sum
    return RGB_out

def volume_rendering2(rgbs, sigmas, args):
    # will probably need to do the operations in place...
    """
    Input:
        ray_ts: linspace of ray distances 
        rgbs: flattened tensor of model RGB outputs [B*S 3] 
        sigmas: flattened tensor of model sigma outputs [B*S 1]
        args: additional params
    """
    ray_ts = torch.linspace(args.near, args.far, args.n_sample)
    deltas = ray_ts[1:] - ray_ts[:-1]
    deltas = torch.cat([deltas, torch.tensor([1e10])]) # as mentioned in the NeRF repo, distance to the last element in 'infinity'   
    deltas.to(device)
    # deltas = deltas.broadcast_to((args.n_sample, 1))
    """ Dangerous operations incoming..."""
    # each "ray" is comprised of sample points all ran through the forward pass. We unflatten the tensor to mirror this relation [B*S, 3] -> [B, S, 3]
    
    rgbs_reshaped = rgbs.reshape((args.n_rays_batch, args.n_sample, 3)) 
    sigmas_reshaped = sigmas.reshape((args.n_rays_batch, args.n_sample)) # [B, S] 
    sigmas_reshaped = nn.functional.relu(sigmas_reshaped)

    # alphas = 1 - torch.exp(-sigmas_reshaped*deltas)

    # weights = alphas * torch.cumprod(1.0 - alphas+1e-10, dim=-1)
    # rgb_out = torch.sum(rgbs_reshaped*weights.unsqueeze(-1), 1)
    # return rgb_out
    a = sigmas_reshaped*deltas
    alphas = 1 - torch.exp(-a)
    transmittance = torch.exp(-torch.cumsum(a, -1))
    # transmittance = torch.cumprod(1-alphas+1e-10,-1)
    weights = alphas * transmittance
        # [4096, 100, 3] # [4096,100] -> [4096,100, 1]
    out = rgbs_reshaped * weights.unsqueeze(-1)
    rgb_out = torch.sum(out, 1)
    print(f"random sigmas: {sigmas[0:2,0:2]}")
    return rgb_out

def render(model, rays_origin, rays_direction, args):
    """
    Input:
        model: NeRF model
        rays_origin: origins of input rays [B 3]
        rays_direction: direction of input rays [B 3]
    Outputs:
        rgb values of input rays
    """

    """ Generate a set of points along the ray to query"""
    ray_ts = np.linspace(args.near, args.far, args.n_sample).reshape((args.n_sample,1)) * np.ones((1,3))

    batch_ray_points = np.zeros((args.n_sample * args.n_rays_batch, 3)) # [B*S, 3]
    iterator = 0

    for ray_o, ray_d in zip(rays_origin, rays_direction):           # element wise mult
        ray_points = np.broadcast_to(ray_o, (args.n_sample, 3)) + np.broadcast_to(ray_d, (args.n_sample,3)) *  ray_ts
        # util.show_ray_points(ray_o, ray_points)
        # exit(1)
        batch_ray_points[iterator:(iterator+args.n_sample), :] = ray_points
        iterator += args.n_sample
    
    """ Feed points into model to get RGB and sigma"""
    batch_ray_points = torch.tensor(batch_ray_points, dtype=torch.float32)
    batch_ray_points.to(device)
    model.to(device)
    print(f"Device (in render function): {device}")
    rgbs, sigmas = model.forward(batch_ray_points, None)
    # print(rgbs.shape)
    # print(sigmas.shape)
    # exit(1)
    """ Use volumetric rendering equation to generate actual RGB output from summated points"""
    # need a way to do this output running detach...
    batch_rgb = volume_rendering2(rgbs, sigmas, args)
    return batch_rgb

def loss(mse_obj: nn.MSELoss, groundtruth, prediction):
    # square diff of G.T RGB and pred RGB
    # diff = groundtruth - prediction
    # loss = torch.norm(diff)
    # if loss > 1e6:
    #     print(f"Big Loss {loss}")
    # return torch.norm(diff)
    return mse_obj(prediction, groundtruth)

def train(images, images_val, poses, poses_val, camera_info, args):

    NUM_EPOCHS = 30
    MAX_ITER = args.max_iters
    BATCH_SIZE = args.n_rays_batch
    # total amount of data
    data_total  = camera_info["W"] * camera_info["H"] * len(images)
    data_total_val  = camera_info["W"] * camera_info["H"] * len(images_val)
    batch_iterations = min(data_total / BATCH_SIZE, MAX_ITER)
    # batch_iterations = 2 # for testing val  code...
    mse_obj = nn.MSELoss()
    # Init NeRF Model
    # NOTE: with hierarchical sampling, we actually optimze two models at the same time...
    model = NeRFmodel(60, 24, False, False)
    model.to(device)
    # Init Optimizer
    #NOTE: Paper includes a decaying lr.  
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=[0.9, 0.999], eps=1e-7) 
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    for i in tqdm(range(NUM_EPOCHS)):
        epoch_loss = 0
        # all pixels in image set arranges as indices.
        samples = np.linspace(0, data_total-1, num=data_total, dtype=np.int64)
        for j in tqdm(range(batch_iterations)):
            # generate batch
            ray_origins, ray_directions, ground_truths, samples = generateBatch(samples, images, poses["pose_list"], camera_info, args, i)
            ground_truths.to(device)
            rgb = render(model, ray_origins, ray_directions, args)
            square_loss = loss(mse_obj, ground_truths, rgb)
            print(f"Loss: {square_loss}, rand_rgb: {rgb[0, :]}")
            optimizer.zero_grad()
            square_loss.backward()
            optimizer.step()

            epoch_loss += square_loss.item()
            # tabulate trainning error
        
        """ Validation step"""
        with torch.no_grad():
            samples2 = np.linspace(0, data_total_val-1, num=data_total_val, dtype=np.int64)
            ray_origins_val, ray_directions_val, ground_truths_val, __ = generateBatch(samples2, images_val, poses_val["pose_list"], camera_info, args)
            ground_truths.to()
            rgb_val = render(model, ray_origins_val, ray_directions_val, args)
            val_mse_loss = loss(mse_obj, ground_truths_val, rgb_val).item()
        
        print(f"Validation Loss: {val_mse_loss}, Training Loss: {epoch_loss}")
        # save batch
        SaveName = args.checkpoint_path + str(i) + "model.ckpt"

        torch.save(
            {
                "epoch" : i,
                "coarse_model_state_dict" : model.state_dict(),
                # "fine_model_state_dict" : model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")
    pass

def generateImageBatch(image, pose, camera_info, args):
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
    indices = int((camera_info["H"] / args.scale_factor) * (camera_info["W"] / args.scale_factor))

    samples = np.linspace(0,indices-1, indices, dtype=np.int32) * args.scale_factor # [0 4 8 12 ...]
    # Returned obj creations
    ground_truths = [] # list of pixel RGB values
    ray_origins = []
    ray_directions = []

    for index in samples:
        # each index is a pixel on one of the images, we just need to extract the image, row, col indices and get its ray, 
        v = index // camera_info["H"]
        u = index % camera_info["W"]
        if v > 799 or u > 799 or v < 0 or u < 0:
            raise RuntimeError(f"Bad pixel encountered: Pixel: ({u},{v})")

        gt_pixel = np.float32(image[v,u]) # each element is (0-255)
        gt_pixel /= 255.0
        ground_truths.append(gt_pixel)
        ray_o, ray_d = PixelToRay(camera_info, pose, (u,v), args)
        ray_origins.append(ray_o)
        ray_directions.append(ray_d)

    ray_origins = np.array(ray_origins)
    ray_directions = np.array(ray_directions)
    ground_truths = torch.tensor(np.array(ground_truths)) # gets sent straight to the loss fcn, so needs to be a tensor
    return ray_origins, ray_directions, ground_truths

def test(images, poses, camera_info, args):

    """Generate an image from camera poses"""
    model = NeRFmodel(60,24,False,False)
    args.n_rays_batch = int(camera_info["H"] / args.scale_factor * camera_info["W"] / args.scale_factor)
    checkpoint = torch.load(args.checkpoint_path+"0model.ckpt", weights_only=True)

    model.load_state_dict(checkpoint["coarse_model_state_dict"])
    model.eval()
    rays_origin, rays_direction, ground_truths = generateImageBatch(images[0], poses["pose_list"][0]["camera_pose"], camera_info, args)    
    with torch.no_grad():
        rgb = render(model, rays_origin, rays_direction, args)
    rgb = rgb.detach().numpy()
    rgb = np.uint8(rgb.reshape((int(camera_info["H"] / args.scale_factor), int(camera_info["W"] / args.scale_factor), 3)))
    cv2.imshow("recreation", rgb)
    cv2.imshow("original", images[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(args):
    # load data
    print("Loading data...")
    camera_info, poses, images, depth_images = util.loadDataset(args.data_path, args.mode)
    __, poses_val, images_val, depth_images_val = util.loadDataset(args.data_path, 'val') 
    # util.show_camera_frames(poses)
    # exit(1)
    if args.mode == 'train':
        print("Start training")
        print(f"Device: {device}")
        train(images, images_val, poses, poses_val, camera_info, args)
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
    parser.add_argument('--n_sample',default=100,help="number of sample per ray")
    parser.add_argument('--near',default=0,help="starting distance for sampling points on rays")
    parser.add_argument('--far',default=3,help="ending distance for sampling points on rays")
    parser.add_argument('--max_iters',default=10000,help="number of max iterations for training")
    parser.add_argument('--logs_path',default="./logs/",help="logs path")
    parser.add_argument('--checkpoint_path',default="./Phase2/checkpoint/",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=True,help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter',default=1000,help="num of iteration to save checkpoint")
    parser.add_argument('--images_path', default="./image/",help="folder to store images")
    parser.add_argument('--scale_factor', default=8, type=int, help='reduction ratio for final image')
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)
