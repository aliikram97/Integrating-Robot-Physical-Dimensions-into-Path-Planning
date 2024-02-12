import sys
import select
import numpy as np
from gitup_point import CarlaController

control = CarlaController()

def is_key_pressed():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


def get_key():
    return sys.stdin.read(1)

control.spawn_vehicle()
speed = control.get_speed()
while True:
    # Check if a key has been pressed
    if is_key_pressed():
        key = get_key().lower()

        # Check if the pressed key is 'q'
        if key == 'q':
            print("Exiting the loop. Goodbye!")
            break

        if key == 's':
            speed = control.get_speed()

    # Your code for the main loop logic goes here
    control.control_loop_speed(speed)
    control.get_camera_feed()
# Additional cleanup code if needed
