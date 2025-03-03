import numpy as np

from Pixel import *

# def build_V_from_stratch(pixels_from_1_to_X: dict):
#     pixels = list(pixels_from_1_to_X)
#     V = np.ones((1, len(pixels)))
#     V_coordinates = list(pixels_from_1_to_X.values())
#     V_pixels = list(pixels_from_1_to_X)
#     return V, V_coordinates, V_pixels

def build_V_from_stratch(pixels_from_1_to_X: dict[Pixel, Coordinate]):
    pixels = list(pixels_from_1_to_X)
    V = np.ones((len(pixels), 4))
    V_points = []
    for row_idx in range(len(pixels)):
        V[row_idx, 1] = row_idx  # We just care about the point index (some cameras see the same point)
        V[row_idx, 2] = pixels[row_idx].u
        V[row_idx, 3] = pixels[row_idx].v
        V_points.append(pixels_from_1_to_X[pixels[row_idx]])
    return V, V_points

def add_to_V(V, V_points:list, new_pixels_to_X: dict[Pixel, Coordinate]):
    new_pixels = list(new_pixels_to_X)
    new_camera_idx = V[-1, 0] + 1
    for pixel in new_pixels:
        new_coord = new_pixels_to_X[pixel]
        if new_coord in V_points:
            point_idx = V_points.index(new_coord)
        else:
            # If we are here, then we are going to add a point to V_points.
            point_idx = len(V_points)
            V_points.append(new_coord)
        V = np.vstack((V, np.array([new_camera_idx, point_idx, pixel.u, pixel.v])))
    return V, V_points

# def add_to_V(V: np.ndarray, V_coordinates: list, V_pixels: list, new_pixels_to_X):
#     new_pixels = list(new_pixels_to_X)
#     found_pixels = []
#     new_row = np.zeros((1, V.shape[1]))
#     for new_pixel in new_pixels:
#         new_coord = new_pixels_to_X[new_pixel]
#         if new_coord in V_coordinates:
#             i = V_coordinates.index(new_coord)
#             new_row[0][i] = 1
#             found_pixels.append(new_pixel)
#     remaining_pixels = [x for x in new_pixels if x not in found_pixels]
#     remaining_points = [new_pixels_to_X[x] for x in remaining_pixels]
#     V_coordinates.extend(remaining_points)
#     V_pixels.extend(remaining_pixels)
#     V = np.vstack((V, new_row))
#     for i in range(len(remaining_points)):
#         new_column = np.zeros((V.shape[0], 1))
#         new_column[-1] = 1
#         V = np.hstack((V, new_column))
#     return V, V_coordinates, V_pixels
        
