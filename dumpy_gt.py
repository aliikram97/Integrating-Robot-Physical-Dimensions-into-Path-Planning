import carla

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)

# Get the world
world = client.get_world()

# Assuming you have already spawned a vehicle actor called 'vehicle'
vehicle = world.get_actor('vehicle')

# Define the desired velocity (for example, 10 m/s forward)
desired_velocity = carla.Vector3D(x=10.0, y=0.0, z=0.0)

# Set the velocity of the vehicle
vehicle.set_velocity(desired_velocity)
