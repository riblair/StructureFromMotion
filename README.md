# README

_**ONE LATE DAY USED**_

This repository contains the contents of Phase One for Project Two for RBE/CS 549 Computer Vision. The project was completed by Riley Blair and Scott Pena, AKA Team 6.

Project Two Phase One requires the implementation of the Structure From Motion algorithm. The inputs of this algorithm is a set of images of a single scene from multiple poses. From these images, we can create a point cloud to represent a 3D reconstruction of the scene in the provided images. To quickly review the process, it begins by estimating the fundamental and essential matrix between the first two images. Using the essential matrix, we can estimate the camera's pose that best satisfies the cheirality condition (least points with negative depth). Using camera poses and pixel correspondances between images, we can triangulate the 3D point that corresponds with each pixel match. To add more images and points to the point cloud, we utilize the Perspective-n-Point algorithm to refine the new camera's pose and add new 3D points into the point cloud. After adding a new camera, we perform bundle adjustment to refine all the 3D points and camera poses to minimize reprojection error. PnP and bundle adjustment is repeated for all new images and points added to the pointcloud.

To run our algorithm, please run Wrapper.py.

When running our algorithm, the following (optional) arguments can be provided:
1. --DataPath: The filepath to the data to be inputted into the algorithm. This includes both the images and the feature matching files.
