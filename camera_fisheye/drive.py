#!/usr/bin/env python

# Copyright (c) 2024 Tananaev Denis
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Welcome to TSR manual control.

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


import carla
import cv2
from model.fisheye_camera import PinholeCamera, FisheyeCamera

import random
import pygame

from pygame.locals import K_ESCAPE
from pygame.locals import K_SPACE
from pygame.locals import K_a
from pygame.locals import K_d
from pygame.locals import K_s
from pygame.locals import K_w

FPS = 10

def spawn_random_vehicle(world):
    # Get a vehicle mercedes.
    blueprint_library = world.get_blueprint_library()
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



def main():
    """The main program."""
    try:
        # Initialize pygame and display for visualization
        pygame.init()
        display = pygame.display.set_mode((640, 480), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame_clock = pygame.time.Clock()

        # Connect to client and get world
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(2.0)
        world = client.get_world()

        for bp in world.get_blueprint_library().filter('vehicle'):
            print(bp.id)
        # Set up actors
        ego_vehicle = spawn_random_vehicle(world)
        camera = PinholeCamera(parent_actor=ego_vehicle, width=640, height=480, fov=90, tick=0.0,
                 x=0.0, y=0.0, z=4, roll=0, pitch=0, yaw=0, camera_type ='sensor.camera.rgb')
        fisheye_camera = FisheyeCamera(parent_actor=ego_vehicle, width=1024, height=512, horizontal_fov=180, vertical_fov=140, tick=0.0,
                 x=0.0, y=0.0, z=4, roll=0, pitch=0, yaw=0, camera_type ='sensor.camera.rgb')
        actors_list = [ego_vehicle, camera, fisheye_camera]    

            
        set_synchronous_mode(world, True)
        # The game loop
        while True:
                world.tick()
                pygame_clock.tick_busy_loop(FPS)
                display.blit(pygame.surfarray.make_surface(camera.image.swapaxes(0, 1)), (0, 0))
                pygame.display.flip()
                pygame.event.pump()

                fisheye_camera.create_fisheye_image()
                filename_to_save = f"/home/denis/carla_workspace/debug_output/box_{int(fisheye_camera.frame):04d}.jpg"
                print(f"writing image to {filename_to_save}")
                cv2.imwrite(filename_to_save,fisheye_camera.image[..., ::-1])
                if control(ego_vehicle):
                    break
    finally:
        set_synchronous_mode(world, False)
        for actor in actors_list:
            actor.destroy()
        pygame.quit()



if __name__ == '__main__':
    main()