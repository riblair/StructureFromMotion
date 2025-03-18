import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("tkagg")
from scipy.spatial.transform import Rotation as R
import json
import os

Q = 1
X_TRANS = np.array([[1,0,0,Q], 
                    [0,1,0,0], 
                    [0,0,1,0], 
                    [0,0,0,1]])

Y_TRANS = np.array([[1,0,0,0], 
                    [0,1,0,Q], 
                    [0,0,1,0], 
                    [0,0,0,1]])

Z_TRANS = np.array([[1,0,0,0], 
                    [0,1,0,0], 
                    [0,0,1,Q], 
                    [0,0,0,1]])

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
    image_width = 800
    image_height = 800
    K = np.array([[1, 0, image_width/2], [0, 1, image_height/2], [0, 0, 1]])
    im_path = data_path+mode+'/'
    transforms_file_name = data_path+"transforms_"+ mode + ".json"
    # list of dictionaries...
    # Turn Transformation matrix into Camera Pose 
        # "pose": Camera Pose can be a (6,1) => [x, y, z, r, p, y]^T
        # "idx": Image index (number appended to file name)
    
    with open(transforms_file_name, 'r') as fp:
        data = json.load(fp)
    
    # translation from camera_angle_x to focal length given on line 77 of NeRF repo load_blender.py
    focal = .5 * image_width / np.tan(.5 * float(data["camera_angle_x"]))
    camera_info = {"W": image_width, "H": image_height, "K": K, "focal_length": focal}
    poses = dict()
    pose_list = []

    for frame in data["frames"]:
        pose_dictionary = dict()
        yaw = frame["rotation"] # NOTE: we currently think this represents the Objects rotation in 3D space, and is seperate from the cameras rotation.
        t_mat = np.squeeze(np.array([frame["transform_matrix"]]))
        x_rot = np.array([[1, 0, 0], [0,-1,0], [0,0,-1]])
        r_mat = t_mat[0:3, 0:3] @ x_rot

        rot = R.from_matrix(r_mat)

        roll_r,pitch_r,yaw_r = rot.as_euler('XYZ')
        pose_dictionary["camera_pose"] = np.array([t_mat[0,3], t_mat[1,3], t_mat[2, 3], roll_r, pitch_r, yaw_r]).T
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

    return camera_info, poses, images, depth_images

def transform_from_pose(pose):
    t_list = [pose[0], pose[1], pose[2]]
    t_vec = np.array(t_list).reshape((3,1))
    r_list = [float(pose[3]), float(pose[4]), float(pose[5])]
    rot = R.from_euler('XYZ', np.array(r_list))
    r_mat = rot.as_matrix()
    t_mat = np.vstack((np.hstack((r_mat, t_vec)), np.array([[0,0,0,1]])))
    return t_mat

def show_camera_frames(pose_dictionary):

    origin = np.zeros(shape=(4,1))
    xp_list = []
    yp_list = []
    zp_list = []

    x0_ax_list = []
    x1_ax_list = []
    x2_ax_list = []

    y0_ax_list = []
    y1_ax_list = []
    y2_ax_list = []

    z0_ax_list = []
    z1_ax_list = []
    z2_ax_list = []

    for frame in pose_dictionary["pose_list"]:
        t_mat = transform_from_pose(frame["camera_pose"])
        x2 = t_mat @ X_TRANS
        y2 = t_mat @ Y_TRANS 
        z2 = t_mat @ Z_TRANS  

        xp_list.append(frame["camera_pose"][0])
        yp_list.append(frame["camera_pose"][1])
        zp_list.append(frame["camera_pose"][2])

        x0_ax_list.append([float(frame["camera_pose"][0]), x2[0, 3]])
        x1_ax_list.append([float(frame["camera_pose"][1]), x2[1, 3]])
        x2_ax_list.append([float(frame["camera_pose"][2]), x2[2, 3]])

        y0_ax_list.append([float(frame["camera_pose"][0]), y2[0, 3]])
        y1_ax_list.append([float(frame["camera_pose"][1]), y2[1, 3]])
        y2_ax_list.append([float(frame["camera_pose"][2]), y2[2, 3]])

        z0_ax_list.append([float(frame["camera_pose"][0]), z2[0, 3]])
        z1_ax_list.append([float(frame["camera_pose"][1]), z2[1, 3]])
        z2_ax_list.append([float(frame["camera_pose"][2]), z2[2, 3]])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_zlabel('Z-Axis')
    # add camera points
    ax.scatter(origin[0],origin[0],origin[0], c='black', marker='o', label='Origin')
    ax.scatter(xp_list,yp_list,zp_list, marker='*')

    ax.plot([0, 1], [0,0], [0,0], color='red')
    ax.plot([0, 0], [0,1], [0,0], color='green')
    ax.plot([0, 0], [0,0], [0,1], color='blue')

    for i in range(len(x0_ax_list)):
        ax.plot(x0_ax_list[i], x1_ax_list[i], x2_ax_list[i], color='red')
        ax.plot(y0_ax_list[i], y1_ax_list[i], y2_ax_list[i], color='green')
        ax.plot(z0_ax_list[i], z1_ax_list[i], z2_ax_list[i], color='blue')

    plt.show()

def show_ray(t_mat, pose, direction):
    ## Show camera origin and axes
    x2 = t_mat @ X_TRANS
    y2 = t_mat @ Y_TRANS 
    z2 = t_mat @ Z_TRANS 

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_zlabel('Z-Axis')
    # add camera points
    ax.scatter([pose[0]],[pose[1]],[pose[2]], marker='*')

    # add camera axes

    ax.plot([float(pose[0]), x2[0, 3]], [float(pose[1]), x2[1, 3]], [float(pose[2]), x2[2, 3]], color='red')
    ax.plot([float(pose[0]), y2[0, 3]], [float(pose[1]), y2[1, 3]], [float(pose[2]), y2[2, 3]], color='green')
    ax.plot([float(pose[0]), z2[0, 3]], [float(pose[1]), z2[1, 3]], [float(pose[2]), z2[2, 3]], color='blue')

    # add origin
    ax.scatter(0,0,0, c='black', marker='o', label='Origin')
    ax.plot([0, 1], [0,0], [0,0], color='red')
    ax.plot([0, 0], [0,1], [0,0], color='green')
    ax.plot([0, 0], [0,0], [0,1], color='blue')

    # add ray
    ray = np.reshape(pose[0:3], (3,1)) + direction
    ray2 = np.reshape(pose[0:3], (3,1)) + 2*direction
    # ray3 = np.reshape(pose[0:3], (3,1)) + 3*direction

    ax.plot([float(pose[0]), float(ray[0])], [float(pose[1]), float(ray[1])], [float(pose[2]), float(ray[2])], color='black')
    ax.plot([float(pose[0]), float(ray2[0])], [float(pose[1]), float(ray2[1])], [float(pose[2]), float(ray2[2])], color='yellow')
    # ax.plot([float(pose[0]), float(ray3[0])], [float(pose[1]), float(ray3[1])], [float(pose[2]), float(ray3[2])], color='cyan')

    plt.show()

def show_ray_points(ray_o, ray_points):
    ## Show camera origin and axes
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_zlabel('Z-Axis')
    # add camera points
    ax.scatter([ray_o[0]],[ray_o[1]],[ray_o[2]], marker='*')
    # add origin
    ax.scatter(0,0,0, c='black', marker='o', label='Origin')
    ax.plot([0, 1], [0,0], [0,0], color='red')
    ax.plot([0, 0], [0,1], [0,0], color='green')
    ax.plot([0, 0], [0,0], [0,1], color='blue')
    # add ray

    ax.scatter(ray_points[:, 0].tolist(), ray_points[:, 1].tolist(), ray_points[:, 2].tolist(), marker='x')
    # ax.plot([float(pose[0]), float(ray[0])], [float(pose[1]), float(ray[1])], [float(pose[2]), float(ray[2])], color='black')
    # ax.plot([float(pose[0]), float(ray2[0])], [float(pose[1]), float(ray2[1])], [float(pose[2]), float(ray2[2])], color='yellow')
    # ax.plot([float(pose[0]), float(ray3[0])], [float(pose[1]), float(ray3[1])], [float(pose[2]), float(ray3[2])], color='cyan')

    plt.show()
