# Copyright (c) 2024 Tananaev Denis
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
import cv2
import open3d as o3d
import os
from carla_fsd.camera_fisheye.camera_models.equidistant_projection import EquidistantProjection

CURRENT_USER = os.environ['USER']

def write_to_ply(filename: str, points3d: np.ndarray, colors3d: np.ndarray)-> None:
    """Saves 3d point clouds"""
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3d)
    pcd.colors = o3d.utility.Vector3dVector(colors3d)  # Normalize colors to [0, 1]
    o3d.io.write_point_cloud(filename, pcd)


def main():
    """Main function."""

    image_path = f"/home/{CURRENT_USER}/carla_workspace/CarlaFSD/carla_fsd/camera_fisheye/tutorial/images/rgb_fisheye_example.jpg"
    image = np.asarray(cv2.imread(image_path)[..., ::-1], dtype=np.float32)
    #  Get intrinsic calibration
    height, width, _ = image.shape
    fov = 180

    fisheye_projetion = EquidistantProjection.from_fov(width=width, height=height, fov=fov, k0=0.0, k1=0.0, k2=0.0, k3=0.0, k4=0.0)

    # Get image coordinates
    y, x = np.meshgrid(range(height), range(width), indexing='ij')

    # Here pixels coords of the shape [height, width, 2]
    pixels2d = np.concatenate((x[..., None], y[..., None]), axis=-1)

    # Here pixels coords of the shape [height*width, 2]
    pixels2d = np.asarray(pixels2d, dtype=np.float32).reshape(-1, 2)

    rays3d = fisheye_projetion.from_2d_to_3d(pixels_coords=pixels2d)
    colors3d = image.reshape(-1, 3) / 255.0
    # Save point cloud in ply format
    filename = f"/home/{CURRENT_USER}/carla_workspace/CarlaFSD/carla_fsd/camera_fisheye/tutorial//images/rgb_fisheye_example_rays3d.ply"
    write_to_ply(filename, rays3d, colors3d)


if __name__ == "__main__":
    main()