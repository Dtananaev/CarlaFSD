# Copyright (c) 2024 Tananaev Denis
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
import cv2
import open3d as o3d

def write_to_ply(filename: str, points3d: np.ndarray, colors3d: np.ndarray)-> None:
    """Saves 3d point clouds"""
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3d)
    pcd.colors = o3d.utility.Vector3dVector(colors3d)  # Normalize colors to [0, 1]
    o3d.io.write_point_cloud(filename, pcd)


def pinhole_projection(X: np.ndarray, K: np.ndarray):
    """Projects 3d points to the image.

    Note: Here K - is camera intrinsic camera parameters
            This parameters can be defined by calibration process
                    | fx 0.0  cx |
                K = | 0.0 fy  cy |
                    | 0.0 0.0 1.0|

    Args:
        X: the array of 3d points of the shape [num_points, 3]
        K: the intrinsic matrix of the shape 3x3
    
    Returns:
        x: homogeneos pixels coordinates of the shape [num_points, 3]
    """
    # We need Z values for normalization of the shape [1, num_points]
    Z = np.exnand_dims(X[..., -1], 0) # this is depth values which are equal Z coordinate of 3d point
    # Here we get tensor of the shape [3, num_points] with pixels coordinates
    x = (1.0 / Z) * (K @ X.T)
    
    return x.T


def inverse_pinhole_projection(pixels2d: np.ndarray, K: np.ndarray)-> np.ndarray:
    """Inverse projection from 2d pixels to 3d rays.

        Note: Here K - is camera intrinsic camera parameters
            This parameters can be defined by calibration process
                    | fx 0.0  cx |
                K = | 0.0 fy  cy |
                    | 0.0 0.0 1.0|
    Args:
      pixels2d: 2d pixels coordinates of the shape [num_points, 2 (x, y)]
      K: calibration matrix
    
    Returns:
        rays3d: 3d rays in camera coordinates of the shape [num_points, 3]
    """
    num_points, _  = pixels2d.shape

    ones = np.ones((num_points, 1))
    # Make homogeneous pixels coordinates of the shape [num_points, 3 (x, y, 1)]
    # in order to enable matrix multiplication for inverse projection
    homogeneos_coords = np.concatenate((pixels2d, ones), axis=-1)

    # Here inverse operation 
    rays3d = np.linalg.inv(K) @ homogeneos_coords.T

    
    # Here the 3d rays in camera coordinates of the shape [num_points, 3]
    rays3d = rays3d.T
    #  You may note that in order to get full 3d points we need Z coordinates (depth)
    # The depth information is lost during projection operation
    return rays3d

def main():
    """Main function."""

    image_path = "./images/rgb_pinhole_example.jpg"
    image = np.asarray(cv2.imread(image_path)[..., ::-1], dtype=np.float32)
    #  Get intrinsic calibration
    height, width, _ = image.shape
    fov = 90
    intrinsic = np.identity(3)
    intrinsic[0, 2] = float(width) / 2.0 
    intrinsic[1, 2] = float(height) / 2.0 
    intrinsic[0, 0] = intrinsic[1, 1] = float(width) / (2.0 * np.tan(float(fov) * np.pi / 360.0))


    # Get image coordinates
    y, x = np.meshgrid(range(height), range(width), indexing='ij')

    # Here pixels coords of the shape [height, width, 2]
    pixels2d = np.concatenate((x[..., None], y[..., None]), axis=-1)

    # Here pixels coords of the shape [height*width, 2]
    pixels2d = np.asarray(pixels2d, dtype=np.float32).reshape(-1, 2)

    rays3d = inverse_pinhole_projection(pixels2d=pixels2d, K=intrinsic)
    colors3d = image.reshape(-1, 3) / 255.0
    # Save point cloud in ply format
    filename = "./images/rgb_pinhole_example_rays3d.ply"
    write_to_ply(filename, rays3d, colors3d)


if __name__ == "__main__":
    main()