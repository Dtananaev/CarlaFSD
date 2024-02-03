#!/usr/bin/env python

# Copyright (c) 2024 Tananaev Denis
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import abc
import numpy as np


class BaseProjection(abc.ABC):
    """This is base projection class which should be inherit from."""

    @classmethod
    @abc.abstractmethod
    def from_fov(cls, width: int, height: int, fov: float,  k0: float, k1: float, k2: float, k3: float, k4: float)-> None:
        """Constructor from fov."""

    @abc.abstractmethod
    def projection(self, points3d: np.ndarray)-> np.ndarray:
        """The camera projection model from 3d points to normalized coordinates.
        
        Args:
            points3d: the 3d points array of the shape [..., 3]

        Returns:
            normalized_coordinates: with z=1 of the shape [..., 3]
        """

    @abc.abstractmethod
    def inverse_projection(self, normalized_coords: np.ndarray)-> np.ndarray:
        """Iverse projection from normalized camera coordinates to 3d rays on the unit sphere.
        
        
        Args:
            normalized_coords: normmalized camera coordinates (z=1) of the shape [..., 3]
        
        Returns:
            rays3d: 3d rays on the unit sphere of the shape [..., 3]
        """

    @abc.abstractmethod
    def from_3d_to_2d(self, points3d: np.ndarray)-> np.ndarray:
        """The camera projection from 3D to 2D image.
        
        Args:
            points3d: the 3d points array of the shape [..., 3]
        
        Returns:
            points2d: the 2d pixels positions on the image [..., 2]
        """
    

    @abc.abstractmethod
    def from_2d_to_3d(self, pixels_coords: np.ndarray)-> np.ndarray:
        """The inverse projection from 2d image to 3d space.
        

        Args:
            pixels_coords: the pixels coordinates of the shape [..., 2]
        
        Returns:
            rays3d: the unit rays of the shape [..., 3]
        """


