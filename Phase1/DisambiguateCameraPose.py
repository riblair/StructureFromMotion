from Pixel import Coordinate

def disambiguate_camera_pose(t_list, triangulated_point_list: list[list[Coordinate]]):
    best_translation = None
    best_points = None
    highest_positive_count = 0
    iter = 1

    for transformation, triangulated_points in zip(t_list, triangulated_point_list):
        R3 = transformation[2, 0:3].reshape((1,3))  # 3rd row of R representing z-axis
        t = transformation[:, 3].reshape((3,1))  # 3x1 translation vector
        num_positive_depth_points = 0
        for point in triangulated_points:  # Point is Coordinate class
            point = point.to_arr()  # Do not make homogenous for 3x1 - 3x1
            point_3d = R3 @ (point - t)  # 1x3 @ 3x1 makes singular matrix
            if point_3d[0] > 0:
                num_positive_depth_points += 1
        if num_positive_depth_points > highest_positive_count:
            best_translation = transformation
            best_points = triangulated_points
            highest_positive_count = num_positive_depth_points
        print(f"[{iter} num_pos_depth = {num_positive_depth_points}]")
        iter+=1
    return best_translation, best_points
