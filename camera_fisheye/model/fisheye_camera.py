#!/usr/bin/env python

# Copyright (c) 2024 Tananaev Denis
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import carla
from carla import ColorConverter as cc
import numpy as np
import weakref
def process_image(image):
    ''' The callback function which gets raw image and convert it to array.'''
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    return array[:, :, ::-1] 

class PinholeCamera:
    """Simulate simple pinhole camera in carla.
 
    Args:
        parent_actor: vehicle actor to attach the camera
        width: width of the image
        height: height of the image
        fov: field of view in degrees
        tick: simulation seconds between sensor captures (ticks).
        x: x position with respect to the ego vehicle in meters
        y: y position with respect to the ego vehicle in meters
        z: z position with respect to the ego vehicle in meters
        camera_type: can be: 'sensor.camera.rgb', 'sensor.camera.semantic_segmentation' or 'sensor.camera.depth'
    """
    def __init__(self, parent_actor: carla.Actor, width: int, height: int, fov: int=90, tick: float=0.0,
                 x: float=-6.5, y:float=0.0, z:float=2.7,
                 roll:float=0, pitch:float=0, yaw:float=0,
                 camera_type: str ='sensor.camera.rgb')-> None:
        """Init."""
        if camera_type not in [ 'sensor.camera.rgb', 'sensor.camera.semantic_segmentation', 'sensor.camera.depth']:
            raise ValueError(f"Camera type {camera_type} is not supported!")
        # Carla related parameters
        self._parent = parent_actor
        # Visualization related parameters
        self.camera_type = camera_type
        self.image = None
        self.frame = 0
        # Set up the sensor
        blueprint = self._parent.get_world().get_blueprint_library().find(camera_type)
        # Modify the attributes of the blueprint to set image resolution and field of view.
        blueprint.set_attribute('image_size_x', str(width))
        blueprint.set_attribute('image_size_y', str(height))
        blueprint.set_attribute('fov', str(fov))
        # Set the time in seconds between sensor captures
        blueprint.set_attribute('sensor_tick', str(tick))

        # Provide the position of the sensor relative to the vehicle.
        transform  = carla.Transform(carla.Location(x=x, y=y, z=z),
                                        carla.Rotation(roll=roll, pitch=pitch, yaw=yaw))

        # Tell the world to spawn the sensor, don't forget to attach it to your vehicle actor.
        self.sensor = self._parent.get_world().spawn_actor(blueprint, transform, attach_to=self._parent)

        # Estimate intrinsic matrix for the camera
        calibration = np.identity(3)
        calibration[0, 2] = float(width) / 2.0 
        calibration[1, 2] = float(height) / 2.0 
        calibration[0, 0] = calibration[1, 1] = float(width) / (2.0 * np.tan(float(fov) * np.pi / 360.0))
        self.sensor.calibration = calibration

        # We need to pass the lambda a weak reference to self to avoid
        # circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: self._parse_image(weak_self, image))

    def destroy(self):
        """Destroys camera."""
        self.sensor.destroy()



    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.camera_type=='sensor.camera.depth':
            image.convert(cc.LogarithmicDepth) #  'Camera Depth (Logarithmic Gray Scale)'     
        elif self.camera_type== 'sensor.camera.semantic_segmentation':
            image.convert(cc.CityScapesPalette)
        else:         
            image.convert(cc.Raw)
            
        self.image = process_image(image)
        self.frame += 1