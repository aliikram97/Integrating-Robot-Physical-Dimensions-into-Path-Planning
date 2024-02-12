import carla
import math
import numpy as np
import cv2


class CarlaController:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)

        self.world = self.client.get_world()
        self.vehicle = None

    def spawn_vehicle(self):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.audi.tt')[0]

        spectator = self.world.get_spectator()
        transform = spectator.get_transform()

        self.vehicle = self.world.spawn_actor(vehicle_bp, transform)




    def get_camera_feed(self):

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_init_trans = carla.Transform(carla.Location(z=1.5))  # Adjust height as needed
        self.camera = self.world.spawn_actor(camera_bp, camera_init_trans, attach_to=self.vehicle)

        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        camera_data = {'image': np.zeros((image_h, image_w, 4))}

        self.camera.listen(lambda image: self.camera_callback(image, camera_data))

    # Inside the CarlaController class
    def camera_callback(self, image, data_dict):
        data_dict['image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

        # Example: Convert image data to a usable format (RGB)
        rgb_image = np.array(image.raw_data)
        rgb_image = rgb_image.copy()
        rgb_image = rgb_image.reshape((image.height, image.width, 4))
        rgb_image = rgb_image[:, :, :3]
        rgb_image = rgb_image.astype(np.uint8)

        # Display the image using OpenCV (you can replace this with your visualization code)
        cv2.imshow('Camera Feed', rgb_image)
        cv2.waitKey(60)  # Add a small delay to allow OpenCV to update the display

    def intensity_control(self):
        control = carla.VehicleControl()
        return control

    def get_speed(self):
        # Get the target speed from the user
        target_speed = float(input("Enter the target speed in km/h: "))
        # Print or return any relevant information based on your action policy
        print(f"Target Speed: {target_speed}")
        return target_speed

    def control_loop_steer(self,a):
        return a

    def action_policy(self,command):
        return command


    def control_loop_speed(self, target_speed):
        try:
            # Create control object outside the loop
            control = self.intensity_control()

            while True:
                # Print out the velocity information
                velocity = self.vehicle.get_velocity()
                speed_kmh = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
                print('Speed: %15.0f km/h' % speed_kmh)

                # Calculate the speed difference
                speed_difference = target_speed - speed_kmh

                # Calculate the intensity for the throttle
                throttle_intensity = speed_difference / target_speed

                print(f"Throttle Intensity: {throttle_intensity}")

                # Update the throttle value in the existing control object
                control.throttle = 1.0 if speed_kmh < target_speed else 0.0

                # Apply the control to the vehicle
                self.vehicle.apply_control(control)

        except KeyboardInterrupt:
            print("User interrupted the program.")
        finally:
            self.cleanup()

    # def actuator(self, throttle_intensity, target_speed, speed_kmh):
    #     if speed_kmh < target_speed:
    #         control = self.intensity_control()
    #         control.throttle = 1.0
    #         print(f"Throttle Intensity: {throttle_intensity}")
    #
    #         # Apply the control to the vehicle
    #         self.vehicle.apply_control(control)
    #     elif speed_kmh >= target_speed:
    #         # If current speed is greater than or equal to target speed, set throttle to 0
    #         control = self.intensity_control()
    #         control.throttle = 0.0
    #         print("Throttle Intensity: 0 (Speed greater than or equal to Target Speed)")
    #
    #         # Apply the control to the vehicle
    #         self.vehicle.apply_control(control)

    def cleanup(self):
        if self.vehicle is not None:
            if hasattr(self, 'camera'):
                self.camera.stop()
                self.camera.destroy()
            self.vehicle.destroy()

# if __name__ == '__main__':
#     controller = CarlaController()
#     controller.spawn_vehicle()
#     target_speed = controller.action_policy_control()  # Get the initial target speed from the user
#     controller.control_loop_speed(target_speed)