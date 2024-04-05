#!/usr/bin/env python

# Copyright (c) 2024 Tananaev Denis
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import os
import numpy as np
import glob
import cv2
import copy

CURRENT_USER = os.environ['USER']

no_camera_motion_dir = f"/home/{CURRENT_USER}/carla_workspace/CarlaFSD/carla_fsd/scene_flow/tutorials/images/no_camera_motion"
with_camera_motion_dir = f"/home/{CURRENT_USER}/carla_workspace/CarlaFSD/carla_fsd/scene_flow/tutorials/images/with_camera_motion"

def background_subtraction(images_dir, video_name="./video.mp4", fps=2):
    """Make backround subtraction for motion segmentation."""

    images_list = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    frame = cv2.imread(os.path.join(images_list[0]))
    height, width, layers = frame.shape



    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for idx in range(len(images_list)-1):
        curr_image = np.asarray(cv2.imread(images_list[idx]), dtype=np.float32)[..., ::-1]
        next_image = np.asarray(cv2.imread(images_list[idx+1]), dtype=np.float32)[..., ::-1]
        
        
        moseg = copy.deepcopy(curr_image)

        curr_image = cv2.GaussianBlur(curr_image, (7, 7), 0)
        next_image = cv2.GaussianBlur(next_image, (7, 7), 0)

        diff = np.linalg.norm(curr_image - next_image, axis=-1) 
        motion_mask = diff > 15.0

        # Background substraction visu
        moseg[motion_mask, :] = 0.5 * moseg[motion_mask, :] + 0.5 * np.asarray([0, 255, 0])

        video.write(moseg.astype("uint8")[..., ::-1])
    cv2.destroyAllWindows()
    video.release()


if __name__=="__main__":

    background_subtraction(images_dir=with_camera_motion_dir, video_name="./video.mp4", fps=2)