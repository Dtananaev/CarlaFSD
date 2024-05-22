#!/usr/bin/env python

# Copyright (c) 2024 Tananaev Denis
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import carla
from carla_fsd.camera_fisheye.fisheye_camera import FisheyeCamera, PinholeCamera
from carla_fsd.camera_fisheye.camera_models.equidistant_projection import EquidistantProjection
from carla_fsd.camera_fisheye.camera_models.stereographic_projection import StereographicProjection
from carla_fsd.camera_fisheye.camera_models.lense_distortion import LenseDistortion
import numpy as np
import random
import pygame
import cv2
import os
from pygame.locals import K_ESCAPE
from pygame.locals import K_SPACE
from pygame.locals import K_a
from pygame.locals import K_d
from pygame.locals import K_s
from pygame.locals import K_w

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FPS =3


def apply_distortion_on_pinhole(image, intrinsics, k0, k1, k2, k3, k4):
    """Apply distortion on pinhole image."""

    height, width, _ = image.shape

    fx, fy  = intrinsics[0,0], intrinsics[1, 1]
    lense_dist  = LenseDistortion(fx, fy, k0, k1, k2,k3, k4)
    # Get image coordinates
    y, x = np.meshgrid(range(height), range(width), indexing='ij')

    # Here pixels coords of the shape [height, width, 2]
    image_coords = np.concatenate((x[..., None], y[..., None], np.ones_like(y[..., None])), axis=-1)
    image_coords = np.asarray(image_coords, dtype=np.float32)
    print(f"image_coords {image_coords.shape}")

    image_coords = image_coords.reshape(-1,3)
    print(f"image_coords {image_coords.shape}")

    norm_coords  =  (np.linalg.inv(intrinsics) @ image_coords.T).T
    print(f"norm_coords {norm_coords.shape}")

    undist_x, undist_y, ones = norm_coords[..., 0], norm_coords[..., 1], norm_coords[..., 2]
    dist_x, dist_y = lense_dist.distortion(undist_x,undist_y)

    dist_norm_coords = np.concatenate((dist_x[..., None], dist_y[..., None], ones[..., None]), axis=-1  )
    print(f"dist_norm_coords {dist_norm_coords.shape}")
    pixels_coords = (intrinsics @ dist_norm_coords.T).T
    print(f"pixels_coords {pixels_coords.shape}")

    pixels_coords = pixels_coords.reshape((height, width, 3))
    pixels_coords = np.asarray(pixels_coords, dtype=np.float32)
    image = np.asarray(image, dtype=np.float32)
    print(f"pixels_coords {pixels_coords.shape}")

    # Rescale pixels for visu
    x_min, y_min  = np.min(pixels_coords[..., 0]), np.min(pixels_coords[..., 1])
    x_max, y_max  = np.max(pixels_coords[..., 0]), np.max(pixels_coords[..., 1])

    pixels_coords[..., 0] = width * (pixels_coords[..., 0] - x_min) / (x_max - x_min)
    pixels_coords[..., 1] = height * (pixels_coords[..., 1] - y_min) / (y_max - y_min)



    remapped_img = cv2.remap(image, pixels_coords[..., 0], pixels_coords[..., 1], cv2.INTER_NEAREST)

    return remapped_img


def spawn_random_vehicle(world):
    # Get a vehicle mercedes.
    vehicle_blueprints = world.get_blueprint_library().filter('vehicle.tesla.model3')
    spawn_points = world.get_map().get_spawn_points()
    return world.spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))

def set_synchronous_mode(world, synchronous_mode):
    """Sets synchronous mode."""
    settings = world.get_settings()
    settings.synchronous_mode = synchronous_mode
    world.apply_settings(settings)

def control(car):
    """
    Applies control to main car based on pygame pressed keys.
    Will return True If ESCAPE is hit to end game loop, otherwise False.
    """

    keys = pygame.key.get_pressed()
    if keys[K_ESCAPE]:
        return True

    control = car.get_control()
    control.throttle = 0
    if keys[K_w]:
        control.throttle = 1
        control.reverse = False
    elif keys[K_s]:
        control.throttle = 1
        control.reverse = True
    if keys[K_a]:
        control.steer = max(-1., min(control.steer - 0.05, 0))
    elif keys[K_d]:
        control.steer = min(1., max(control.steer + 0.05, 0))
    else:
        control.steer = 0
    control.hand_brake = keys[K_SPACE]

    car.apply_control(control)
    return False




def print_controls_help()->None:

    msg = """
    Welcome to Carla fisheye manual control.

    Use ARROWS or WASD keys for control.

        W            : throttle
        S            : brake
        AD           : steer
        Q            : toggle reverse
        Space        : hand-brake
        P            : toggle autopilot
        C            : change weather (Shift+C reverse)
        ESC          : quit
    """
    print(msg)


def main():
    """The main program."""
    try:
        # Initialize pygame and display for visualization
        pygame.init()
        display = pygame.display.set_mode((IMAGE_WIDTH, IMAGE_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame_clock = pygame.time.Clock()

        # Connect to client and get world
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(2.0)
        world = client.get_world()

        for bp in world.get_blueprint_library().filter('vehicle'):
            print(bp.id)
        # Set up actors
        ego_vehicle = spawn_random_vehicle(world)

        pinhole_camera = PinholeCamera(parent_actor=ego_vehicle, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, fov=90, tick=0.0, x=2.4, y=0.0, z=1.5,
                 roll=0.0, pitch=0.0, yaw=0.0,
                 camera_type ='sensor.camera.rgb')
        pinhole_intrinsics = pinhole_camera.sensor.calibration
        fisheye_camera_equidistant = FisheyeCamera(parent_actor=ego_vehicle, camera_model=EquidistantProjection, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, fov=180, tick=0.0,
                 x=2.40, y=0.0, z=1.5, roll=0, pitch=0, yaw=0, k0=0.0, k1=0.0, k2=0.0, k3=0.0, k4=0.0,  camera_type ='sensor.camera.rgb')
      
        actors_list = [ego_vehicle, fisheye_camera_equidistant, pinhole_camera]    
            
        set_synchronous_mode(world, True)
        print_controls_help()
        # The game loop
        dist_coeff = -0.001
        while True:
                world.tick()
                pygame_clock.tick_busy_loop(FPS)
                fisheye_camera_equidistant.create_fisheye_image()
                fisheye_camera = fisheye_camera_equidistant
                display.blit(pygame.surfarray.make_surface(fisheye_camera.image.swapaxes(0, 1)), (0, 0))
                pygame.display.flip()
                pygame.event.pump()
                dist_coeff*=2.0

                if control(ego_vehicle):
                    break
    finally:
        set_synchronous_mode(world, False)
        for actor in actors_list:
            actor.destroy()
        pygame.quit()



if __name__ == '__main__':
    main()
