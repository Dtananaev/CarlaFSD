# Copyright (c) 2024 Tananaev Denis
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
import cv2

def save_optical_flow_png(filename:str, optical_flow: np.ndarray)-> None:
    """Saves optical flow to filename.
    
    Args:
        optical_flow: array of the shape [height, width, 3(dx, dy, valid_mask)]

    """
    optical_flow = np.asarray(optical_flow, dtype=np.float64)
    height, width, channels = optical_flow.shape
    assert channels == 3
    scale_factor = 2**15 # this is half precision of the uint16
    # Normalize optical flow between [0: 65536]
    optical_flow[..., 0] =  scale_factor *  np.clip(optical_flow[..., 0] / width + 1.0, 0.0, 2.0)
    optical_flow[..., 1] =  scale_factor * np.clip(optical_flow[..., 1] / height + 1.0, 0.0, 2.0)

    optical_flow = np.asarray(optical_flow, dtype=np.uint16)
    cv2.imwrite(filename, optical_flow[..., ::-1])


def load_optical_flow_png(filename: str)-> np.ndarray:
    """Loads optical flow from filename."""
    optical_flow = np.asarray(cv2.imread(filename, -1), dtype=np.float64)[..., ::-1]

    height, width, channels = optical_flow.shape
    assert channels == 3
    scale_factor = 2**15  # this is half precision of the uint16

    optical_flow[..., 0] = width * (optical_flow[..., 0] / scale_factor - 1.0)
    optical_flow[..., 1] = height * (optical_flow[..., 1] / scale_factor - 1.0)

    return np.asarray(optical_flow, dtype=np.float32)