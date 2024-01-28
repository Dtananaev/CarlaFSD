#!/usr/bin/env python

# Copyright (c) 2024 Tananaev Denis
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import carla
from camera_fisheye.fisheye_camera import FisheyeCamera
from camera_fisheye.camera_models.equidistant_projection import EquidistantProjection

import random
import pygame

from pygame.locals import K_ESCAPE
from pygame.locals import K_SPACE
from pygame.locals import K_a
from pygame.locals import K_d
from pygame.locals import K_s
from pygame.locals import K_w

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FPS = 4

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
        fisheye_camera = FisheyeCamera(parent_actor=ego_vehicle, camera_model=EquidistantProjection, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, fov=160, tick=0.0,
                 x=2.40, y=0.0, z=1.5, roll=0, pitch=0, yaw=0, camera_type ='sensor.camera.rgb')
        actors_list = [ego_vehicle, fisheye_camera]    
            
        set_synchronous_mode(world, True)
        print_controls_help()
        # The game loop
        while True:
                world.tick()
                pygame_clock.tick_busy_loop(FPS)
                fisheye_camera.create_fisheye_image()
                display.blit(pygame.surfarray.make_surface(fisheye_camera.image.swapaxes(0, 1)), (0, 0))
                pygame.display.flip()
                pygame.event.pump()

                if control(ego_vehicle):
                    break
    finally:
        set_synchronous_mode(world, False)
        for actor in actors_list:
            actor.destroy()
        pygame.quit()



if __name__ == '__main__':
    main()