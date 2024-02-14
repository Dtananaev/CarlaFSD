# Copyright (c) 2024 Tananaev Denis
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
import open3d as o3d
import cv2
import os
from carla_fsd.camera_fisheye.camera_models.equidistant_projection import EquidistantProjection
from scipy.spatial.transform import Rotation as R

CURRENT_USER = os.environ['USER']

def write_to_ply(filename: str, points3d: np.ndarray, colors3d: np.ndarray)-> None:
    """Saves 3d point clouds"""
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3d)
    pcd.colors = o3d.utility.Vector3dVector(colors3d)  # Normalize colors to [0, 1]
    o3d.io.write_point_cloud(filename, pcd)


def compute_fisheye_rays(fisheye_image_path:str)-> None:
    """Computes fisheye rays."""
    fisheye_image = np.asarray(cv2.imread(fisheye_image_path)[..., ::-1], dtype=np.float32)
    #  Get intrinsic calibration
    height, width, _ = fisheye_image.shape
    fov = 180

    fisheye_projetion = EquidistantProjection.from_fov(width=width, height=height, fov=fov, k0=0.0, k1=0.0, k2=0.0, k3=0.0, k4=0.0)

    # Get image coordinates
    y, x = np.meshgrid(range(height), range(width), indexing='ij')

    # Here pixels coords of the shape [height, width, 2]
    pixels2d = np.concatenate((x[..., None], y[..., None]), axis=-1)

    # Here pixels coords of the shape [height*width, 2]
    pixels2d = np.asarray(pixels2d, dtype=np.float32).reshape(-1, 2)

    rays3d = fisheye_projetion.from_2d_to_3d(pixels_coords=pixels2d)
    colors3d = fisheye_image.reshape(-1, 3) / 255.0
    # Save point cloud in ply format
    filename = f"/home/{CURRENT_USER}/carla_workspace/CarlaFSD/carla_fsd/camera_fisheye/tutorial//images/rgb_fisheye_rays3d.ply"
    write_to_ply(filename, rays3d, colors3d)


def compute_five_pinhole_rays(five_pinhole_path: str) -> None:
    """Computes five pinhole rays."""
    image = np.asarray(cv2.imread(five_pinhole_path)[..., ::-1], dtype=np.float32)
    #  Get intrinsic calibration
    height, width, _ = image.shape
    w = width // 5 # the width of single image
    fov = 90
    # Get calibration
    calibration = np.identity(3)
    calibration[0, 2] = float(w) / 2.0 
    calibration[1, 2] = float(height) / 2.0 
    calibration[0, 0] = calibration[1, 1] = float(w) / (2.0 * np.tan(float(fov) * np.pi / 360.0))

    inv_calib = np.linalg.inv(calibration)

    # Get images
    left_img = image[:, :w, :]
    top_img = image[:, w:2*w, :]
    front_img = image[:, 2*w:3*w, :]
    bottom_img = image[:, 3*w:4*w, :]
    right_img = image[:, 4*w:, :]


    # Get image coordinates
    y, x = np.meshgrid(range(height), range(w), indexing='ij')
    ones = np.ones_like(x)
    # Here pixels coords of the shape [height, w, 3]
    hom_pixels = np.asarray(np.concatenate((x[..., None], y[..., None], ones[..., None]), axis=-1), dtype=np.float32)
    hom_pixels = hom_pixels.reshape(-1, 3) # [height*w, 3]

    # Here rays for pihole camera of the shape [height*w, 3]
    rays  = (inv_calib @ hom_pixels.T).T
    # Get rays of front camera
    filename = f"/home/{CURRENT_USER}/carla_workspace/CarlaFSD/carla_fsd/camera_fisheye/tutorial//images/rgb_five_front_rays3d.ply"
    front_img = front_img.reshape(-1, 3) / 255.0

    write_to_ply(filename=filename, points3d=rays, colors3d=front_img)

    # Left camera
    cam_transform = R.from_euler('xyz',[0.0, -90, 0.0], degrees=True).as_matrix() # Here inverse angles because different chaining order
    left_rays = (cam_transform @ rays.T).T
    filename = f"/home/{CURRENT_USER}/carla_workspace/CarlaFSD/carla_fsd/camera_fisheye/tutorial//images/rgb_five_left_rays3d.ply"
    left_img = left_img.reshape(-1, 3) / 255.0
    write_to_ply(filename=filename, points3d=left_rays, colors3d=left_img)


   # Right camera
    cam_transform = R.from_euler('xyz',[0.0, 90, 0.0], degrees=True).as_matrix() # Here inverse angles because different chaining order
    right_rays = (cam_transform @ rays.T).T
    filename = f"/home/{CURRENT_USER}/carla_workspace/CarlaFSD/carla_fsd/camera_fisheye/tutorial//images/rgb_five_right_rays3d.ply"
    right_img = right_img.reshape(-1, 3) / 255.0
    write_to_ply(filename=filename, points3d=right_rays, colors3d=right_img)

    # Top
    
    cam_transform = R.from_euler('xyz',[90.0, 0.0, 0.0], degrees=True).as_matrix() # Here inverse angles because different chaining order
    top_rays = (cam_transform @ rays.T).T
    filename = f"/home/{CURRENT_USER}/carla_workspace/CarlaFSD/carla_fsd/camera_fisheye/tutorial//images/rgb_five_top_rays3d.ply"
    top_img = top_img.reshape(-1, 3) / 255.0
    write_to_ply(filename=filename, points3d=top_rays, colors3d=top_img)

    # Bottom
    cam_transform = R.from_euler('xyz',[-90.0, 0.0, 0.0], degrees=True).as_matrix() # Here inverse angles because different chaining order
    bottom_rays = (cam_transform @ rays.T).T
    filename = f"/home/{CURRENT_USER}/carla_workspace/CarlaFSD/carla_fsd/camera_fisheye/tutorial//images/rgb_five_bottom_rays3d.ply"
    bottom_img = bottom_img.reshape(-1, 3) / 255.0
    write_to_ply(filename=filename, points3d=bottom_rays, colors3d=bottom_img)


def main():
    """Main function."""

    #fisheye_image_path = f"/home/{CURRENT_USER}/carla_workspace/CarlaFSD/carla_fsd/camera_fisheye/tutorial/images/rgb_001.jpg"
    #compute_fisheye_rays(fisheye_image_path=fisheye_image_path)
    five_pinhole_path  = f"/home/{CURRENT_USER}/carla_workspace/CarlaFSD/carla_fsd/camera_fisheye/tutorial/images/five_rgb_001.jpg"
    compute_five_pinhole_rays(five_pinhole_path=five_pinhole_path)


if __name__ == "__main__":
    main()