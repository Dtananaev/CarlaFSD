#!/usr/bin/env python

# Copyright (c) 2024 Tananaev Denis
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import carla
from carla import ColorConverter as cc
import numpy as np
import weakref
from scipy.spatial.transform import Rotation as R
import cv2

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
                 roll:float=0.0, pitch:float=0.0, yaw:float=0.0,
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
        # For pinhole camera the relation:
        # r = f * tan(theta)
        # theta is angle between principle point and incoming ray
        # r is the distance from principle point to incoming ray
        # r = width / 2.0; theta = np.deg2rad(FOV / 2.0) or FOV * pi /360
        # for more info see: 
        # https://www.researchgate.net/publication/6899685_A_Generic_Camera_Model_and_Calibration_Method_for_Conventional_Wide-Angle_and_Fish-Eye_Lenses
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




class FisheyeCamera:
    """ FisheyeCamera class that simulates equidistant projection fish eye camera."""
    def __init__(self, parent_actor: carla.Actor, width: int=1024, height: int=512, fov:int=180, tick:float=0.0,
                 x: float=-6.5, y: float=0.0, z:float=2.7, roll:float=0.0, pitch:float=0.0, yaw: float=0.0,
                 camera_type='sensor.camera.rgb')-> None:
        # Carla parameters
        self._parent = parent_actor # vehicle where camera will be attached
        self.image = None
        self._box_image = None
        self.frame = 0
        # Estimate calibration for the equidistant projection
        # The relation is r = f * theta
        # Here r = width / 2
        # theta = np.deg2rad(FOV / 2.0 ) or FOV * pi /360
        # for more info see: 
        # https://www.researchgate.net/publication/6899685_A_Generic_Camera_Model_and_Calibration_Method_for_Conventional_Wide-Angle_and_Fish-Eye_Lenses
        calibration = np.identity(3)
        calibration[0, 2] = float(width) / 2.0 
        calibration[1, 2] = float(height) / 2.0 
        calibration[0, 0] = calibration[1, 1] = float(width) / (2.0 * float(fov) * np.pi / 360.0)

        # Create cube from 5 pinhole cameras for reprojection to fish eye

        # We create pinhole with the same focal length as fish eye camera and FOV = 90
        # From the formula r = f * tan(theta) we can get 
        #  width / 2 = f * tan(FOV/2) = f * tan(45 deg); width = 2.0 * f (tan(45 deg) = 1), also we assume width = height
        pinhole_width = pinhole_height = 2.0 * calibration[0, 0]


        # initialize all cameras
        main_rot = R.from_euler('xyz',[roll, pitch, yaw], degrees=True).as_matrix()
        self._front_pinhole = PinholeCamera(self._parent, width=pinhole_width, height=pinhole_height, fov=90, tick=tick,
                 x=x, y=y, z=z,roll=roll, pitch=pitch, yaw=yaw,camera_type=camera_type)

        # First we rotate camera 90 degrees to the left
        # Then chain it to the main rotation to the vehicle
        left_local_rot =  R.from_euler('xyz',[0.0, 0.0, -90], degrees=True).as_matrix()
        left_rot = R.from_matrix( main_rot @ left_local_rot).as_euler('xyz', degrees=True)
        self._left_pinhole = PinholeCamera(self._parent, width=pinhole_width, height=pinhole_height, fov=90, tick=tick,
                 x=x, y=y, z=z,roll=left_rot[0], pitch=left_rot[1], yaw=left_rot[2],camera_type=camera_type)

        # Second we rotate camera 90 degrees to the right
        # Then chain it to the main rotation to the vehicle
        right_local_rot =  R.from_euler('xyz',[0.0, 0.0, 90], degrees=True).as_matrix()
        right_rot = R.from_matrix(main_rot @ right_local_rot).as_euler('xyz', degrees=True)
        self._right_pinhole = PinholeCamera(self._parent, width=pinhole_width, height=pinhole_height, fov=90, tick=tick,
                 x=x, y=y, z=z,roll=right_rot[0], pitch=right_rot[1], yaw=right_rot[2],camera_type=camera_type)

        # Third we rotate camera 90 degrees to the top
        # Then chain it to the main rotation to the vehicle
        top_local_rot =  R.from_euler('xyz',[0.0, 90.0, 0.0], degrees=True).as_matrix()
        top_rot = R.from_matrix(main_rot @ top_local_rot).as_euler('xyz', degrees=True)
        self._top_pinhole = PinholeCamera(self._parent, width=pinhole_width, height=pinhole_height, fov=90, tick=tick,
                 x=x, y=y, z=z,roll=top_rot[0], pitch=top_rot[1], yaw=top_rot[2],camera_type=camera_type)

        # Fourth we rotate camera 90 degrees to the bottom
        # Then chain it to the main rotation to the vehicle
        bottom_local_rot =  R.from_euler('xyz',[0.0, -90.0, 0.0], degrees=True).as_matrix()
        bottom_rot = R.from_matrix(main_rot @ bottom_local_rot).as_euler('xyz', degrees=True)
        self._bottom_pinhole = PinholeCamera(self._parent, width=pinhole_width, height=pinhole_height, fov=90, tick=tick,
                 x=x, y=y, z=z,roll=bottom_rot[0], pitch=bottom_rot[1], yaw=bottom_rot[2],camera_type=camera_type)

    def destroy(self):
        """Delete all cameras."""
        actors = [
            self._front_pinhole,
            self._left_pinhole,
            self._right_pinhole,
            self._top_pinhole,
            self._bottom_pinhole,
        ]
        for actor in actors:
            if actor is not None:
                actor.destroy()

    def create_fisheye_image(self):
        """Creates fisheye image.
        
        Note: this function should be called in order to update image for fish eye camera.
        """
        self._box_image = np.hstack((self._left_pinhole.image,
                            self._top_pinhole.image,
                            self._front_pinhole.image,
                            self._bottom_pinhole.image,                                
                            self._right_pinhole.image)).astype(np.float32)

        self.frame += 1.0

        # Run mapping
        #mapped_img = cv2.remap(box_image, self.map_x, self.map_y, cv2.INTER_NEAREST)

        #self.fish_eye_image = mapped_img.astype('uint8')



    # def add_cameras(self):

    #     if self.is_init == False:

    #         self.pinhole_width = int(2*self.fx)
    #         self.pinhole_height = int(2*self.fy)

    #         # Rotation over x - Roll
    #         # Rotation over y - Pitch
    #         # Rotation over z - Yaw
    #         Rm = eulerAnglesToRotationMatrix(np.deg2rad([ self.roll, self.pitch, self.yaw]))

    #         # initialize all cameras
    #         self._camera_front = CameraPinhole(self.vehicle, str(self.pinhole_width), str(self.pinhole_height),
    #                                            x=self.x, y=self.y, z=self.z,
    #                                            roll=self.roll, pitch=self.pitch, yaw=self.yaw,
    #                                            camera_type=self.camera_type,  pygame_disp = self._pygame_disp)
           
    #         Rl = eulerAnglesToRotationMatrix(np.deg2rad([0, 0, -90]))
    #         Rtl=Rm.dot(Rl)
    #         rot = rotationMatrixToEulerAngles(Rtl)
    #         rot= np.rad2deg(rot)
    #         print("Rl ", rot)

    #         self._camera_left = CameraPinhole(self.vehicle, str(self.pinhole_width), str(self.pinhole_height),
    #                                            x=self.x, y=self.y, z=self.z,
    #                                            roll=rot[0], pitch=rot[1], yaw=rot[2],
    #                                            camera_type=self.camera_type, pygame_disp = self._pygame_disp)

    #         Rr = eulerAnglesToRotationMatrix(np.deg2rad([0, 0, 90]))
    #         Rtr=Rm.dot(Rr)
    #         rot = rotationMatrixToEulerAngles(Rtr)
    #         rot= np.rad2deg(rot)
    #         print("Rl ", rot)

    #         self._camera_right = CameraPinhole(self.vehicle, str(self.pinhole_width), str(self.pinhole_height),
    #                                            x=self.x, y=self.y, z=self.z,
    #                                            roll=rot[0], pitch=rot[1], yaw=rot[2],
    #                                            camera_type=self.camera_type, pygame_disp = self._pygame_disp)
            

    #         Rt = eulerAnglesToRotationMatrix(np.deg2rad([0, 90, 0]))
    #         Rtt = Rm.dot(Rt)            
    #         rot = rotationMatrixToEulerAngles(Rtt)
    #         rot= np.rad2deg(rot)
    #         print("Rt ", rot)
    #         self._camera_top = CameraPinhole(self.vehicle, str(self.pinhole_width), str(self.pinhole_height),
    #                                            x=self.x, y=self.y, z=self.z,
    #                                            roll=rot[0], pitch=rot[1], yaw=rot[2],
    #                                            camera_type=self.camera_type, pygame_disp = self._pygame_disp)

    #         Rb = eulerAnglesToRotationMatrix(np.deg2rad([0, -90, 0]))
    #         Rbt = Rm.dot(Rb)            
    #         rot = rotationMatrixToEulerAngles(Rbt)
    #         rot= np.rad2deg(rot)
    #         print("Rb ", rot)
    #         self._camera_bottom = CameraPinhole(self.vehicle, str(self.pinhole_width), str(self.pinhole_height),
    #                                            x=self.x, y=self.y, z=self.z,
    #                                            roll=rot[0], pitch=rot[1], yaw=rot[2],
    #                                            camera_type=self.camera_type, pygame_disp = self._pygame_disp)
    #         self._compute_mapping_table()

    #         self.is_init = True
 
    #     else:
    #         Print("The camera sensor is already initialized")


    # def _compute_fish_eye(self):
    #     box_image = np.hstack((self._camera_left.image,
    #                            self._camera_top.image,
    #                            self._camera_front.image,
    #                            self._camera_bottom.image,                                
    #                            self._camera_right.image)).astype(np.float32) 

    #     # Run mapping
    #     mapped_img = cv2.remap(box_image, self.map_x, self.map_y, cv2.INTER_NEAREST)

    #     self.fish_eye_image = mapped_img.astype('uint8')
    #     if self.mask is not None:
    #        self.fish_eye_image[self.mask] = 0
    #     if self.recording:
    #        self.save_to_disk()


    # def save_image_to_disk(self):
    #     if self.fish_eye_image is not None:

    #       if self.camera_type == 'sensor.camera.rgb':
    #           if not os.path.exists("dataset/rgb"):
    #              os.makedirs("dataset/rgb")
    #           im = Image.fromarray(self.fish_eye_image)
    #           im.save('dataset/rgb/%08d.jpeg' % self._camera_front.frame)

    #       if self.camera_type == 'sensor.camera.depth':
    #           if not os.path.exists("dataset/depth"):
    #              os.makedirs("dataset/depth")
    #           im = Image.fromarray(self.fish_eye_image)
    #           im.save('dataset/depth/%08d.png' % self._camera_front.frame)

    # def save_pose_to_disk(self):
    #     if not os.path.exists("dataset/pose"):
    #        os.makedirs("dataset/pose")
    #     transform = self._camera_front.camera.get_transform()
    #     pose = np.array([transform.location.x, transform.location.y, transform.location.z,
    #                      transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw])
    #     np.savetxt('dataset/pose/%08d.txt' % self._camera_front.frame, pose) 


    # def _compute_mapping_table(self):

    #     ep = ErrorProp(self.fx, self.fy, self.cx, self.cy, 0, 0, 0, 0, 0)
        
    #     # Get calibration
    #     K = self._camera_front.camera.calibration

    #     # front
    #     R = np.eye(3)
    #     P_front = K.dot(R)
        
    #     # left
    #     R = RotationPitch(90)
    #     P_left = K.dot(R)

    #     # right
    #     R = RotationPitch(-90)
    #     P_right = K.dot(R)

    #     # top
    #     R = RotationRoll(-90)
    #     P_top = K.dot(R)

    #     # bottom
    #     R = RotationRoll(90)
    #     P_bottom = K.dot(R)

    #     self.map_x=np.zeros((self.height, self.width))
    #     self.map_y=np.zeros((self.height,self.width))

    #     idy=0
    #     for y in range(0,self.height):
    #         idx=0
    #         for x in range(0,self.width):

    #            x_z1=(x-self.cx)/self.fx
    #            y_z1=(y-self.cy)/self.fy 

    #            [X, Y, Z, theta, phi] = ep.invThetaProjection(x_z1,y_z1)
    #            ray = np.array([X,Y,Z])

    #            if x <self.width/2:
    #               coord_left = P_left.dot(ray)
    #               x_new = coord_left[0]/coord_left[2]
    #               y_new = coord_left[1]/coord_left[2]              
    #               if x_new>=0 and x_new<=self.pinhole_width and y_new>=0 and y_new<=self.pinhole_height:
    #                 x_c, y_c = self._get_coord_from_box_image(x_new, y_new, position = "left")
    #                 self.map_x[idy,idx]= x_c
    #                 self.map_y[idy,idx]= y_c

    #            if x >self.width/2:
    #               coord_right = P_right.dot(ray)
    #               x_new = coord_right[0]/coord_right[2]
    #               y_new = coord_right[1]/coord_right[2]              
    #               if x_new>=0 and x_new<=self.pinhole_width and y_new>=0 and y_new<=self.pinhole_height:
    #                 x_c, y_c = self._get_coord_from_box_image(x_new, y_new, position = "right")
    #                 self.map_x[idy,idx]= x_c
    #                 self.map_y[idy,idx]= y_c

    #            if y <self.height/2:
    #               coord_top = P_top.dot(ray)
    #               x_new = coord_top[0]/coord_top[2]
    #               y_new = coord_top[1]/coord_top[2]              
    #               if x_new>=0 and x_new<=self.pinhole_width and y_new>=0 and y_new<=self.pinhole_height:
    #                 x_c, y_c = self._get_coord_from_box_image(x_new, y_new, position = "top")
    #                 self.map_x[idy,idx]= x_c
    #                 self.map_y[idy,idx]= y_c

    #            if y >self.height/2:
    #               coord_bottom = P_bottom.dot(ray)
    #               x_new = coord_bottom[0]/coord_bottom[2]
    #               y_new = coord_bottom[1]/coord_bottom[2]              
    #               if x_new>=0 and x_new<=self.pinhole_width and y_new>=0 and y_new<=self.pinhole_height:
    #                 x_c, y_c = self._get_coord_from_box_image(x_new, y_new, position = "bottom")
    #                 self.map_x[idy,idx]= x_c
    #                 self.map_y[idy,idx]= y_c

    #            coord_new = P_front.dot(ray)
    #            x_new = coord_new[0]/coord_new[2]
    #            y_new = coord_new[1]/coord_new[2]

    #            if x_new>=0 and x_new<=self.pinhole_width and y_new>=0 and y_new<=self.pinhole_height:
    #               x_c, y_c = self._get_coord_from_box_image(x_new, y_new, position = "front")
    #               self.map_x[idy,idx]= x_c
    #               self.map_y[idy,idx]= y_c

    #            idx+=1
    #         idy+=1 
    #     self.map_x = self.map_x.astype(np.float32) 
    #     self.map_y = self.map_y.astype(np.float32) 
    #     print("Mapping table is initialized!")

    # def render(self, display=None):
    #     ''' The function gets display class from pygame window
    #         and render the image from the current camera. '''
    #     self._compute_fish_eye()
    #     if self.fish_eye_image is not None and self._pygame_disp and display is not None:
    #         array = self.fish_eye_image.swapaxes(0, 1)
    #         self._surface = pygame.surfarray.make_surface(array)
    #         display.blit(self._surface, (0, 0))

    # def _get_coord_from_box_image(self, x, y, position = "left"):
    #   x_new = 0
    #   y_new = 0
    #   '''
    #   This is fix for floating point in x for box image
    #   in case if x = width of image it ex 
    #   '''
    #   if x>=(self.pinhole_width-1):
    #     x=x-1
    #   if y>=(self.pinhole_height-1):
    #     y=y-1
    #   if position == "left":
    #      x_new = x
    #      y_new = y    
    #   if position == "top":

    #      x_new = x+self.pinhole_width
    #      y_new = y  

    #   if position == "front":
    #      x_new = x+2*self.pinhole_width
    #      y_new = y  
    #   if position == "bottom":
    #      x_new = x+3*self.pinhole_width
    #      y_new = y
    #   if position == "right":
    #      x_new = x+4*self.pinhole_width
    #      y_new = y
    #   return x_new, y_new


    # def getWorld2Camera(self):
    #     vehicle_world_matrix = CarlaHelpers.get_matrix_from_transform(self.vehicle.get_transform())
    #     camera_world_matrix = np.dot(vehicle_world_matrix, self.camera_vehicle_matrix)
    #     return camera_world_matrix

    # def destroySVSCamera(self):

    #     actors = [
    #         self._camera_front.camera,
    #         self._camera_left.camera,
    #         self._camera_top.camera,
    #         self._camera_bottom.camera,
    #         self._camera_right.camera,
    #     ]
    #     for actor in actors:
    #         if actor is not None:
    #             actor.destroy()