#!/usr/bin/env python

# Copyright (c) 2024 Tananaev Denis
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
import sys

class EquidistantProjection:
    """This is implementation of the fish eye equidistant camera projection.

    
    Args:
        fx: focal length in x direction
        fy: focal lenght in y direction
        cx: principle point x coordinate
        cy: principle point y coordinate
    """

    def __init__(self, fx: float, fy: float, cx: float, cy: float)-> None:
        """Init."""
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.intrinsic_matrix = np.asarray([[fx, 0.0, cx],
                                            [0.0, fy, cy],
                                            [0.0, 0.0, 1.0]])
        
        self.inv_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)
        
    def from_3d_to_2d(self, points3d: np.ndarray)-> np.ndarray:
        """The camera projection from 3D to 2D image.
        
        Args:
            points3d: the 3d points array of the shape [..., 3]
        
        Returns:
            points2d: the 2d pixels positions on the image [..., 2]
        """
        shape = points3d.shape
    
        if shape[-1] != 3:
            raise ValueError(f"Incorrect channels shape should be 3 but it is {shape[-1]}!")

        points3d = points3d.reshape(-1, 3)

        # First find sqrt(X**2 + Y**2)
        norm_xy = np.linalg.norm(points3d[:, :2], axis=-1)

        # Find theta angle between optical axis and ray
        theta = np.arctan2(norm_xy, points3d[:, 2])

        # Compute normalized coordinates here we add small epsilon to avoid division by zero
        x_normalized_coordinate = points3d[:, 0] / (norm_xy + sys.float_info.epsilon) * theta
        y_normalized_coordinate = points3d[:, 1] / (norm_xy + sys.float_info.epsilon) * theta
        
        ones = np.ones_like(x_normalized_coordinate)

        # Here homogeneous coordinates of the shape [3, ...]
        homogeneous_coords = np.concatenate( (x_normalized_coordinate[None, :], y_normalized_coordinate[None, :], ones[None, :]), axis=0)

        # Here pixels coordinates of the shape [..., 3]
        pixels_coords = (self.intrinsic_matrix @ homogeneous_coords).T
        pixels_coords = np.reshape(pixels_coords, shape)

        return pixels_coords[..., :2]


    def from_2d_to_3d(self, pixels_coords: np.ndarray)-> np.ndarray:
        """The inverse projection from 2d image to 3d space.
        
        Note: to get original 3d point cloud you need to multiply final rays by depth.
        e.g. rays3d * Depth where Depth = sqrt(X**2 + Y**2 + Z**2) and X, Y, Z original position
        of the 3d point.

        Args:
            pixels_coords: the pixels coordinates of the shape [..., 2]
        
        Returns:
            rays3d: the unit rays of the shape [..., 3]
        """
        shape = pixels_coords.shape
    
        if shape[-1] != 2:
            raise ValueError(f"Incorrect channels shape should be 2 but it is {shape[-1]}!")
        
        pixels_coords = pixels_coords.reshape(-1, 2)

        # Homogeneous coordinates of the shape [3, ...]
        homogeneous_coords = np.concatenate((pixels_coords, np.ones_like(pixels_coords[..., 0][..., None])), axis=-1).T
        #  Normalized coordinates of the shape [..., 3]
        normalized_coords = (self.inv_intrinsic_matrix @ homogeneous_coords).T

        # Get sqrt(x_norm**2 + y_norm**2)
        theta = np.linalg.norm(normalized_coords[:, :2], axis=-1)

        X = normalized_coords[:, 0] / theta
        Y = normalized_coords[:, 1] / theta
        Z = 1.0 / np.tan(theta)

        rays3d = np.concatenate((X[:, None], Y[:, None], Z[:, None]), axis=-1)
        rays3d = rays3d / np.linalg.norm(rays3d, axis=-1, keepdims=True)

        rays3d = np.reshape(rays3d, (*shape[:-1], 3))
        return rays3d

