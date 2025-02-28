from Pixel import Coordinate

def disambiguate_camera_pose(camera_poses, triangulated_point_list: list[list[Coordinate]]):
    best_pose = None
    best_points = None
    highest_positive_count = 0
    for camera_pose, triangulated_points in zip(camera_poses, triangulated_point_list):
        # Currently, this assumes that the cheirality condition uses the R from the
        # perspective matrix AFTER we multiply it by K. If we run into errors, the 
        # first thing I would check is to input the raw R and t pairings.
        R3 = camera_pose[2, 0:3].reshape((1,3))  # 3rd row of R representing z-axis
        t = camera_pose[:, 3].reshape((3,1))  # 3x1 translation vector
        num_positive_depth_points = 0
        for point in triangulated_points:  # Point is Coordinate class
            point = point.to_arr()  # Do not make homogenous for 3x1 - 3x1
            point_3d = R3 @ (point - t)  # 1x3 @ 3x1 makes singular matrix
            if point_3d[0] > 0:
                num_positive_depth_points += 1
        if num_positive_depth_points > highest_positive_count:
            best_pose = camera_pose
            best_points = triangulated_points
            highest_positive_count = num_positive_depth_points
    return best_pose, best_points
