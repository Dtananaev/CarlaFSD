#!/usr/bin/env python

# Copyright (c) 2024 Tananaev Denis
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


import cv2
import os
import numpy as np
import argparse
import glob
from carla_fsd.scene_flow.tools.optilca_flow_io import load_optical_flow_png
from carla_fsd.scene_flow.tools.optical_flow_visu import point_vec, flow_to_image

CURRENT_USER = os.environ['USER']


data_dir = f"/home/{CURRENT_USER}/carla_workspace/CarlaFSD/outputs"

def add_legend(image, legend, color_legend=[255, 255, 255], color_background=[0, 0, 0]):
    """Puts text in image."""
    image = image.copy()
    height, width,  _ = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale =1
    thickness_text = 2
    text_size, _ = cv2.getTextSize(legend, font, font_scale, thickness_text)
    text_w, text_h  = text_size
    center_x = width / 2

    x0 = int(center_x - text_w / 2.0)
    y0 = int(height - text_h - 5)
    x1 = int(center_x + text_w / 2.0)
    y1 = int(height -5)

    cv2.rectangle(image, (x0, y0), (x1, y1), color=color_background, thickness=-1)
    cv2.putText(image, legend, org=(x0, y1), fontFace=font, fontScale=font_scale, color=color_legend, thickness=thickness_text

    )
    return image


def warp_image_with_flow(image, flow):
    """Warp image with optical flow."""
    height, width, _ = image.shape
    xx, yy = np.meshgrid(range(width), range(height), indexing='xy')
    coords = np.asarray(np.stack((xx, yy), axis=-1), dtype=np.float32)
    maptable = coords + flow
    remapped_img = cv2.remap(image, maptable[..., 0], maptable[..., 1], interpolation=cv2.INTER_LINEAR)
    return remapped_img


def get_epipolar_lines(pts1, pts2, F):
    """Get epipolar lines from fundamental matrix."""
    idx1, idx2 = 1, 2
    lines1 = np.squeeze(cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), idx1, F), axis=1)
    lines2 = np.squeeze(cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), idx2, F), axis=1)
    return lines1, lines2

def visu_epilines(img, lines, points):
    """Draw lines."""

    _, width, _ = img.shape

    colors = np.random.randint(0, 255, [len(points),3]).tolist()
    point_radius= 3
    thikness =1
    for line, point, color in zip(lines, points, colors):
        # Y  = -(a*x +c) /b
        x0, y0 = 0, int(-line[2]/ (line[1]+1e-8) )
        x1, y1 = int(width-1), int( -(line[0]* (width-1)+ line[2])/(line[1]+1e-8))

        image = cv2.line(img, (x0, y0), (x1, y1), tuple(color), thikness)
        image = cv2.circle(img, tuple(point.astype(int)), point_radius, tuple(color), -1)
    return image

def compute_fundamental_matrix(flow):
    """Compute fundamental matrix.
    
    Args:
        flow: optical flow of the shape [height, width, 2 (dx, dy)]
    
    Returns:
        F:  3x3 fundamental matrix
    """

    # Get pixels coordinates of the image1
    height, width, _ = flow.shape
    xx, yy = np.meshgrid(range(width), range(height), indexing='xy')
    pts1 = np.asarray(np.stack((xx, yy), axis=-1), dtype=np.float32)

    # Compute corresponded coordinates in the image 2
    pts2 = pts1 + flow

    # Compute fundamental matrix by using 8 point algorithm and ransac
    pts1  = np.reshape(pts1, (-1, 2))
    pts2  = np.reshape(pts2, (-1, 2))
    inlier_tr = 3.0
    ransac_prob = 0.999
    # Here output F fundamental matrix
    F, valid_mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, inlier_tr, ransac_prob)


    return F, valid_mask


def sampson_error(F, pts1, pts2):
    """Estimate sampson error from coplanarity constraints.
    
    Args:
        F: fundamental matrix
        pts1: points on the first image of the shape [num_points, 2]
        pts2: corresponded points on the second image of the shape [num_points, 2]
    
    Returns:
        error: the coplanarity constraint error for each correspondence
        Note: for coplanar correspondence it should be zero (or near so)
    """

    num_points, _ = pts1.shape
    ones = np.ones((num_points, 1))

    hom_pts1 = np.concatenate((pts1, ones), axis=-1).T
    hom_pts2 = np.concatenate((pts2, ones), axis=-1).T
    Fp1 = F @ hom_pts1
    Fp2 = F.T @ hom_pts2
    p2Fp1 = (hom_pts2 * Fp1).sum(axis=0)

    error = (p2Fp1 **2) / (Fp1[0] **2 +Fp1[1]**2 + Fp2[0]**2 + Fp2[1]**2 + 1e-8)
    return error


def normalize(tensor):
    """Normalize tensor."""

    min = np.min(tensor)
    max = np.max(tensor)
    norm_tensor= (tensor - min) / (max - min)
    return norm_tensor

def homography_error(pts1, pts2, cam1_R_cam2, intrinsics):
    """Rotational homography error.
    
    Args:
        pts1: undistorted pixels coords [num_points, 2]
        pts2: undistorted pixels coords [num_points, 2]
        cam1_R_cam2: rotation from camera2 to camera1
        intrinsics: 3x3 intrinsics
    Returns:
        error: homography error
    """

    num_points, _ = pts1.shape
    ones = np.ones((num_points, 1))

    hom_pts1 = np.concatenate((pts1, ones), axis=-1).T
    hom_pts2 = np.concatenate((pts2, ones), axis=-1).T

    H =  intrinsics @ cam1_R_cam2 @ np.linalg.inv(intrinsics)

    Hp2 = H @ hom_pts2

    Hp1 = np.linalg.inv(H) @ hom_pts1

    error = np.linalg.norm( (hom_pts1 -Hp2)[:2], axis=0) **2 + np.linalg.norm( (hom_pts2 -Hp1)[:2], axis=0) **2
    return error

def compute_camera_motion(flow, intrinsics):
    """Compute camera motion from essential matrix."""
    height, width, _ = flow.shape
    xx, yy = np.meshgrid(range(width), range(height), indexing='xy')
    pts1 = np.asarray(np.stack((xx, yy), axis=-1), dtype=np.float32)
    pts2 = pts1 + flow

    pts1 = np.reshape(pts1, (-1, 2))
    pts2 = np.reshape(pts2, (-1, 2))


    ones = np.ones((height*width, 1))

    hom_pts1 = np.concatenate((pts1, ones), axis=-1)
    hom_pts2 = np.concatenate((pts2, ones), axis=-1)

    norm_points1 = (np.linalg.inv(intrinsics) @ hom_pts1.T).T[:, :2]
    norm_points2 = (np.linalg.inv(intrinsics) @ hom_pts2.T).T[:, :2]


    # Compute essential matrix, valid_mask -inliers
    E, valid_mask = cv2.findEssentialMat(norm_points1, norm_points2, focal=1.0, pp=(0.0, 0.0), method=cv2.RANSAC, prob=0.999, treshold=1.0)
    
    # Here outputs
    # retval: A scalar value indicating the success of the operation. If it's true, the decomposition was successful.
    # R: The rotation matrix representing the relative rotation between the two camera views.
    # t: The translation vector representing the relative translation between the two camera views.
    # mask: A mask indicating the inliers. It's optional and may not be returned if not provided
    # Recover pose (rotation and translation)
    retval, R, t,  mask = cv2.recoverPose(E, norm_points1, norm_points2, focal=1.0, pp=(0.0, 0.0))
    return E, R, t, valid_mask



def main(input_dir, output_dir):
    """The main script."""
    os.makedirs(output_dir, exist_ok=True)
    rgb_list = sorted(glob.glob(os.path.join(input_dir, "rgb", "*.jpg")))
    
    
    samples = np.random.randint(0, 600*800, [30])
    for idx in range(0, len(rgb_list)-1):
        img1_path = rgb_list[idx]
        img2_path = rgb_list[idx+1]


        basename = os.path.basename(img2_path)
        # optical flow
        flow_path = os.path.join(input_dir, "flow", basename.replace(".jpg", ".png"))
        optical_flow = np.asarray(load_optical_flow_png(flow_path), dtype=np.float32)

        img1 = np.asarray(cv2.imread(img1_path), dtype=np.float32)[..., ::-1]
        img2 = np.asarray(cv2.imread(img2_path), dtype=np.float32)[..., ::-1]

        flow_rgb = point_vec(img2, optical_flow,skip=20)
        flow_rgb = add_legend(flow_rgb, legend="backward flow")

        filename_to_save = os.path.join(output_dir, basename)
        flow = optical_flow[..., :2]


        
        F, valid_mask = compute_fundamental_matrix(flow)
        height, width, _  = flow.shape
        xx, yy = np.meshgrid(range(width), range(height), indexing='xy')
        pts1 = np.asarray(np.stack((xx, yy), axis=-1), dtype=np.float32)
        pts2 = pts1 + flow
        pts1  = np.reshape(pts1, (-1, 2))
        pts2  = np.reshape(pts2, (-1, 2))

        error = sampson_error(F, pts1, pts2)

        lines1, lines2 = get_epipolar_lines(pts1, pts2, F)
        print(f"samples {samples}")

        print(f"lines2 {lines2.shape}")
        epilines = visu_epilines(img2.copy(), lines2[samples], pts2[samples])
        epilines = add_legend(epilines, legend="epipolar lines")



        moseg =  normalize(error).reshape((height, width, 1))
        print(f" moseg {moseg.shape}")
        img2 = 0.6  *img2 + 0.4 * moseg * np.asarray([0, 255, 0]) 
        img2 = add_legend(img2, legend="motion segmentation")

        momask = 255 * moseg
        momask = np.concatenate((momask, momask, momask), axis=-1)
        momask = add_legend(momask, legend="motion mask")

        total_image1 = np.hstack((epilines, img2))
        total_image2 = np.hstack((flow_rgb, momask))
        total_image = np.vstack((total_image1, total_image2))
        filename_to_save = os.path.join(output_dir, "samson_"+basename)


        print(f"filename_to_save {filename_to_save}")
        cv2.imwrite(filename_to_save, total_image.astype("uint8")[..., ::-1])
        print(f"img1_path {img1_path}, img2_path {img2_path}")

        #input()


def get_args():
    """Gets arguments."""
    parser = argparse.ArgumentParser(description="Epipolar constraints.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=data_dir,
        help="Input dir.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{data_dir}/epipola_constrants",
        help="Ouptput dir.",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(input_dir = args.input_dir, output_dir=args.output_dir)



