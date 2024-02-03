#!/usr/bin/env python

# Copyright (c) 2024 Tananaev Denis
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
from carla_fsd.camera_fisheye.camera_models.base_projection import BaseProjection
from carla_fsd.camera_fisheye.camera_models.lense_distortion import LenseDistortion
import sys

class EquidistantProjection(BaseProjection):
    """This is implementation of the fish eye equidistant camera projection.

    Note: The derived model equasions could be found in paper:
        Steffen Abraham, Wolfgang Förstner. Fish-Eye-Stereo Calibration and Epipolar Rectification, (2005)
        http://www.ipb.uni-bonn.de/pdfs/Steffen2005Fish.pdf
    
    Args:
        fx: focal length in x direction
        fy: focal lenght in y direction
        cx: principle point x coordinate
        cy: principle point y coordinate
    """

    def __init__(self, fx: float, fy: float, cx: float, cy: float, k0: float, k1: float, k2: float, k3: float, k4: float)-> None:
        """Init."""
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.intrinsic_matrix = np.asarray([[fx, 0.0, cx],
                                            [0.0, fy, cy],
                                            [0.0, 0.0, 1.0]])
        
        self.inv_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)


        self.lense_distortion = LenseDistortion(fx=fx, fy=fy, k0=k0, k1=k1, k2=k2, k3=k3, k4=k4)

    @classmethod
    def from_fov(cls, width: int, height: int, fov: float,  k0: float, k1: float, k2: float, k3: float, k4: float)-> None:
        """Creates calibration from fov.
        
          Note: estimate calibration for the equidistant projection
        The relation is r = f * theta
        Here r = width / 2
        theta = np.deg2rad(FOV / 2.0 ) or FOV * pi /360
        for more info see: 
        https://www.researchgate.net/publication/6899685_A_Generic_Camera_Model_and_Calibration_Method_for_Conventional_Wide-Angle_and_Fish-Eye_Lenses

        Args:
            width: width of the image
            height: height of the image
            fov: horizontal field of view
        """

        calibration = np.identity(3)
        calibration[0, 2] = float(width) / 2.0 
        calibration[1, 2] = float(height) / 2.0 
        calibration[0, 0] =calibration[1, 1] =  float(width) / (2.0 * float(fov) * np.pi / 360.0)
        return cls(fx=calibration[0,0], fy=calibration[1,1], cx=calibration[0,-1], cy=calibration[1,-1], k0=k0, k1=k1, k2=k2, k3=k3, k4=k4)


    def projection(self, points3d: np.ndarray)-> np.ndarray:
        """The camera projection model from 3d points to normalized coordinates.
        
        Args:
            points3d: the 3d points array of the shape [..., 3]

        Returns:
            normalized_coordinates: with z=1 of the shape [..., 3]
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
        x_normalized_coordinate = points3d[:, 0] * theta / (norm_xy + sys.float_info.epsilon) 
        y_normalized_coordinate = points3d[:, 1] * theta/ (norm_xy + sys.float_info.epsilon)

        ones = np.ones_like(x_normalized_coordinate)
        # Here homogeneous coordinates of the shape  [..., 3]
        normalized_coordinates = np.concatenate((x_normalized_coordinate[:, None], y_normalized_coordinate[:, None], ones[:, None]), axis=-1)
        return normalized_coordinates.reshape(shape)


    def inverse_projection(self, normalized_coords: np.ndarray)-> np.ndarray:
        """Iverse projection from normalized camera coordinates to 3d rays on the unit sphere.
        
        
        Args:
            normalized_coords: normmalized camera coordinates (z=1) of the shape [..., 3]
        
        Returns:
            rays3d: 3d rays on the unit sphere of the shape [..., 3]
        """
        shape = normalized_coords.shape
    
        if shape[-1] != 3:
            raise ValueError(f"Incorrect channels shape should be 3 but it is {shape[-1]}!")
        # Get sqrt(x_norm**2 + y_norm**2)
        theta = np.linalg.norm(normalized_coords[:, :2], axis=-1)
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        X = normalized_coords[:, 0] * sin_t/ (theta + sys.float_info.epsilon)
        Y = normalized_coords[:, 1] * sin_t  / (theta + sys.float_info.epsilon)
        Z = cos_t

        rays3d = np.concatenate((X[:, None], Y[:, None], Z[:, None]), axis=-1)
        return rays3d.reshape(shape)


    def from_3d_to_2d(self, points3d: np.ndarray)-> np.ndarray:
        """The camera projection from 3D to 2D image.
        
        Args:
            points3d: the 3d points array of the shape [..., 3]
        
        Returns:
            points2d: the 2d pixels positions on the image [..., 2]
        """
       
        # 1. Apply fisheye projection
        # Aplly equidistant projection to get normalized camera coordinates (z=1)
        homogeneous_coords = self.projection(points3d)
        shape = homogeneous_coords.shape
        homogeneous_coords = homogeneous_coords.reshape(-1, 3)

        # 2. Apply distortion
        undist_x, undist_y, ones = homogeneous_coords[:, 0], homogeneous_coords[:, 1], homogeneous_coords[:, 2]
        dist_x, dist_y =  self.lense_distortion.distortion(undist_x, undist_y)
        # Here distorted homoheneous coords z=1 of the shape [3, ...]
        dist_homogeneous_coords = np.concatenate(( dist_x[None, :], dist_y[None, :], ones[None, :]), axis=0)

        # 3. Apply pinhole projection
        # Here pixels coordinates of the shape [..., 3]
        pixels_coords = (self.intrinsic_matrix @ dist_homogeneous_coords).T
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
        
        # 1. Here inverse pinhole projection
        pixels_coords = pixels_coords.reshape(-1, 2)
        # Homogeneous coordinates of the shape [3, ...]
        homogeneous_coords = np.concatenate((pixels_coords, np.ones_like(pixels_coords[..., 0][..., None])), axis=-1).T
        #  Normalized coordinates of the shape [..., 3]
        normalized_coords = (self.inv_intrinsic_matrix @ homogeneous_coords).T
        
        # 2. Apply undistortion 
        dist_x, dist_y, ones = normalized_coords[:, 0], normalized_coords[:, 1], normalized_coords[:, 2]
        undist_x, undist_y = self.lense_distortion.undistortion(dist_x, dist_y)
        undist_homogeneous_coords = np.concatenate(( undist_x[:, None], undist_y[:, None], ones[:, None]), axis=-1)
        undist_homogeneous_coords = undist_homogeneous_coords.reshape((*shape[:-1], 3))

        # 3. Apply inverse fisheye projection
        rays3d = self.inverse_projection(undist_homogeneous_coords)
        return rays3d

