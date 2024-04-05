#!/usr/bin/env python

# Copyright (c) 2024 Tananaev Denis
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import argparse
import glob
import os
import sys
import cv2
from carla_fsd.scene_flow.tools.optical_flow_visu import point_vec, flow_to_image
from carla_fsd.scene_flow.tools.optilca_flow_io import save_optical_flow_png, load_optical_flow_png
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random


try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

import pygame


IMAGE_WIDTH =800
IMAGE_HEIGHT =600


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def get_image(image_rgb):
    """Gets rgb image."""
    array = np.frombuffer(image_rgb.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image_rgb.height, image_rgb.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array

def get_optical_flow(image_flow, forward_flow =False):
    """Gets optical flow."""
    array = np.frombuffer(image_flow.raw_data, dtype=np.float32)
    array = np.reshape(array, (image_flow.height, image_flow.width, 2))
    array = array.copy()
    # Forward flow
    if forward_flow:
        array[..., 0] *= image_flow.width
        array[..., 1] *= -image_flow.height
    else:
        array[..., 0] *= -image_flow.width
        array[..., 1] *=  image_flow.height    
    
    return array
    
def get_depth(image_depth):
    """Gets depth."""
    def _decode_depth(depth):
        depth = np.asarray(depth, dtype = np.float32)
        d = depth[:,:,0] + depth[:,:,1]*256 + depth[:,:,2]*256*256
        # Normalize between 0 and 1
        d = d /  ( 256*256*256 - 1 )
        # The far plane set 1000 to restore meters
        return d * 1000 

    array = np.frombuffer(image_depth.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image_depth.height, image_depth.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return _decode_depth(array)


def get_arguments():
    parser = argparse.ArgumentParser(description="Scene flow dataset arguments.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Ouptput dir.",
    )
    args = parser.parse_args()
    return args
def main(output_dir, skip_first: int = 20):


    images_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    flow_dir = os.path.join(output_dir, "flow")
    visu_dir = os.path.join(output_dir, "visu")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(flow_dir, exist_ok=True)
    os.makedirs(visu_dir, exist_ok=True)

    actor_list = []
    pygame.init()

    counter = 0.0
    display = pygame.display.set_mode(
        (2* IMAGE_WIDTH, IMAGE_HEIGHT),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    try:
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.tesla.model3')),
            start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(False)
        vehicle.set_autopilot(True)

        x_pose = 2.40
        z_pose = 1.5

        camera_blueprint = blueprint_library.find('sensor.camera.rgb')
        camera_blueprint.set_attribute('image_size_x', str(IMAGE_WIDTH))
        camera_blueprint.set_attribute('image_size_y', str(IMAGE_HEIGHT))
        camera_blueprint.set_attribute('fov', "90")
        camera_rgb = world.spawn_actor(camera_blueprint
            ,
            carla.Transform(carla.Location(x=x_pose, z=z_pose), carla.Rotation(pitch=0)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)

        camera_blueprint =   blueprint_library.find('sensor.camera.optical_flow')
        camera_blueprint.set_attribute('image_size_x', str(IMAGE_WIDTH))
        camera_blueprint.set_attribute('image_size_y', str(IMAGE_HEIGHT))
        camera_blueprint.set_attribute('fov', "90")
        camera_optical_flow = world.spawn_actor(
          camera_blueprint,
            carla.Transform(carla.Location(x=x_pose, z=z_pose), carla.Rotation(pitch=0)),
            attach_to=vehicle)
        actor_list.append(camera_optical_flow)


        camera_blueprint =  blueprint_library.find('sensor.camera.depth')
        camera_blueprint.set_attribute('image_size_x', str(IMAGE_WIDTH))
        camera_blueprint.set_attribute('image_size_y', str(IMAGE_HEIGHT))
        camera_blueprint.set_attribute('fov', "90")
        camera_depth = world.spawn_actor(
            camera_blueprint,
            carla.Transform(carla.Location(x=x_pose, z=z_pose), carla.Rotation(pitch=0)),
            attach_to=vehicle)
        actor_list.append(camera_depth)


        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera_optical_flow, camera_depth, fps=30) as sync_mode:
            while True:
                
                
                if should_quit():
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_flow, image_depth = sync_mode.tick(timeout=2.0)
                image = get_image(image_rgb)
                optical_flow = get_optical_flow(image_flow, forward_flow=False)

                optical_flow_norm = np.linalg.norm(optical_flow, axis=-1)
                motion_mask = optical_flow_norm > 1.0

                optical_flow[~motion_mask, :] = 0.0

                optical_flow = np.concatenate((optical_flow, np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 1))), axis=-1)


                flow_rgb = point_vec(image, optical_flow, skip=20)
                colorflow = flow_to_image(optical_flow)
                # flow_rgb
                total_flow = np.hstack((flow_rgb, colorflow))
                depth = get_depth(image_depth)
                location = vehicle.get_transform()
                print(f"location {location}")
                #print(f"depth  min {np.min(depth)} max {np.max(depth)}")

                #print(f"optical_flow  min {np.min(optical_flow)} max {np.max(optical_flow)}")

                #print(f"depth {depth.shape}, optical_flow {optical_flow.shape}")

                # Choose the next waypoint and update the car location.
                #waypoint = random.choice(waypoint.next(1.5))
                #vehicle.set_transform(waypoint.transform)

                #if control(vehicle):
                #    break
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Draw the display.
                display.blit(pygame.surfarray.make_surface(total_flow.swapaxes(0, 1)), (0, 0))
                pygame.display.flip()
                pygame.event.pump()           
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()


                # Save 
                if counter > skip_first:
                    frame = int(counter)
                    # Save rgb
                    rgb_filename= os.path.join(images_dir, f"{frame:04d}.jpg")
                    cv2.imwrite(rgb_filename, image[...,::-1])

                    # Save depth
                    depth_filename = os.path.join(depth_dir, f"{frame:04d}.png")
                    cv2.imwrite(depth_filename, np.uint16(depth))

                    # Save flow
                    flow_filename = os.path.join(flow_dir, f"{frame:04d}.png")
                    save_optical_flow_png(filename=flow_filename, optical_flow=optical_flow)

                    # Save visu
                    visu_filename = os.path.join(visu_dir, f"{frame:04d}.jpg")
                    cv2.imwrite(visu_filename, total_flow[..., ::-1])
                counter+=1

    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':
    args = get_arguments()
    main(output_dir=args.output_dir)