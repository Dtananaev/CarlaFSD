#!/usr/bin/env python

# Copyright (c) 2024 Tananaev Denis
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
from typing import Tuple


class LenseDistortion:
    """Lense distortion.
    
    Note: see for mode details
        https://people.cs.rutgers.edu/~elgammal/classes/cs534/lectures/CameraCalibration-book-chapter.pdf


    Args:
        fx: focal length in x direction
        fy: focal length in y direction
        k0: radial distortion first coefficient
        k1: radial distortion second coefficient
        k2: tangential distortion (decentering distortion) first coefficient
        k3: tangential distortion (decentering distortion) second coefficient
        k4: radial distortion third coefficient
    """

    def __init__(self, fx: np.float64, fy: np.float64, k0: np.float64, k1: np.float64, k2: np.float64, k3: np.float64, k4: np.float64, max_iterations: int = 100, max_delta: np.float64 = 0.001)-> None:
        """Init."""
        self.k0 = k0
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4


        self.squared_fx = fx ** 2
        self.squared_fy = fy ** 2

        self.max_iterations = max_iterations
        self.max_delta = max_delta

    def distortion(self,  undist_x: np. ndarray, undist_y : np.ndarray)-> Tuple[np.ndarray,np.ndarray] :
        """Apply distortion on the undistorted set of rays.

        Args:
            undist_x: undistorted set of rays x coordinates in camera coordinates where z=1.
            undist_y:  undistorted set of rays y coordinates in camera coordinates where z=1.

        Returns:
            dist_x: distorted set of rays in x coordinates in camera coordinates where z=1.
            dist_y: distorted set of rays in y coordinates in camera coordinates where z=1.
        """

        undist_x = undist_x.astype(np.float64)
        undist_y = undist_y.astype(np.float64)

        x2=undist_x * undist_x
        y2=undist_y * undist_y
        r2=x2 + y2
        r4=r2*r2
        r6=r4*r2

        dist_x = undist_x * (1.0 + self.k0*r2 + self.k1*r4 + self.k4*r6) + 2.0 * self.k2 * undist_x * undist_y + self.k3 * (r2 + 2.0 * x2)
        dist_y = undist_y * (1.0 + self.k0*r2 + self.k1*r4 + self.k4*r6) + 2.0 * self.k3 * undist_x * undist_y + self.k2 * (r2 + 2.0 * y2) 


        return dist_x, dist_y
    
    def undistortion(self, dist_x:  np.ndarray, dist_y : np.ndarray)-> Tuple[np.ndarray,np.ndarray] :
        """Apply undistortion on distorted set of rays.
        
        Note: Due to complexity to derive close form solution from high order polynomial,
        here using Newton method instead.
        
        Args:
            dist_x: distorted set of rays in x coordinates in camera coordinates where z=1.
            dist_y: distorted set of rays in y coordinates in camera coordinates where z=1.
        
        Returns:
            undist_x: undistorted set of rays x coordinates in camera coordinates where z=1.
            undist_y:  undistorted set of rays y coordinates in camera coordinates where z=1.
        """
        dist_x = dist_x.astype(np.float64)
        dist_y = dist_y.astype(np.float64)
        
        undist_x, undist_y = dist_x.copy(), dist_y.copy()

        num_iterations = 0.0
        delta = np.ones_like(undist_x) +  self.max_delta

        while (num_iterations < self.max_iterations) & (delta > self.max_delta).any():
            mask = delta > self.max_delta

            updated_x, updated_y = self.distortion(undist_x, undist_y)

            delta_x = (updated_x - dist_x)
            delta_y = (updated_y - dist_y)


            undist_x[mask] -=  delta_x[mask] 
            undist_y[mask] -=  delta_y[mask] 

            # compute delta in pixels coordinates
            delta[mask] = (delta_x * delta_x * self.squared_fx + delta_y * delta_y  * self.squared_fy)[mask]
            num_iterations += 1.0
        return undist_x, undist_y