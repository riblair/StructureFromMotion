import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("tkagg")
from scipy.spatial.transform import Rotation as R

def show_camera_frames(pose_dictionary):

    Q = 1
    x_trans = np.array([[1,0,0,Q], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
    y_trans = np.array([[1,0,0,0], [0,1,0,Q], [0,0,1,0], [0,0,0,1]])
    z_trans = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,Q], [0,0,0,1]])

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
        t_list = [frame["camera_pose"][0], frame["camera_pose"][1], frame["camera_pose"][2]]
        t_vec = np.array(t_list).reshape((3,1))
        r_list = [float(frame["camera_pose"][3]), float(frame["camera_pose"][4]), float(frame["camera_pose"][5])]
        rot = R.from_euler('xyz', np.array(r_list))

        r_mat = rot.as_matrix()
        r_mat = np.vstack((np.hstack((r_mat, t_vec)), np.array([[0,0,0,1]])))
        x2 = r_mat @ x_trans
        y2 = r_mat @ y_trans 
        z2 = r_mat @ z_trans  

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
