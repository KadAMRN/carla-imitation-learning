# braking to correct, rear drive

# add interactive map, start and end point


#Dependencies
import glob
import os
import sys
import time
import numpy as np
import carla
from IPython.display import display, clear_output
import logging
import random
from datetime import datetime
import cv2

import argparse
import keyboard
import threading


#autopilot parser
def parse_arguments():
    parser = argparse.ArgumentParser(description='Autopilot Flag Parser')
    parser.add_argument('--autopilot', action='store_true', help='Enable autopilot')
    return parser.parse_args()

# Function to control the vehicle with keyboard inputs
def control_vehicle(vehicle):
    # Constants for controlling the vehicle
    STEER_INCREMENT = 0.05
    THROTTLE_INCREMENT = 0.05
    BRAKE_INCREMENT = 0.05

    # Initialize control values
    steer = 0.0
    throttle = 0.0
    brake = 0.0
    # drive_mode = "forward"  # Initial drive mode
    # reverse_mode = False  # Reverse mode status

    # Keyboard input handlers
    def on_key_release(key):
        nonlocal steer, throttle, brake

        if key.name == "left":
            steer=0#steer += STEER_INCREMENT # steer=0# steer += STEER_INCREMENT
        elif key.name == "right":
            steer=0#steer -= STEER_INCREMENT # steer=0# 
        if key.name == "up":
            throttle =0#throttle -= THROTTLE_INCREMENT # throttle =0# 
        if key.name == "down":
            brake=0#brake -= BRAKE_INCREMENT # brake=0# 

        # Update vehicle controls
        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = brake
        vehicle.apply_control(control)

    def on_key_press(key):
        nonlocal steer, throttle, brake

        if key.name == "left":
            steer -= STEER_INCREMENT
        elif key.name == "right":
            steer += STEER_INCREMENT
        if key.name == "up":
            throttle += THROTTLE_INCREMENT
        if key.name == "down":
            brake += BRAKE_INCREMENT
            throttle -=THROTTLE_INCREMENT
        # if key.name == "t":
        #     if drive_mode == "forward":
        #         drive_mode = "reverse"
        #         reverse_mode = True
        #     else:
        #         drive_mode = "forward"
        #         reverse_mode = False

        # Update vehicle controls
        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = brake
        # control.reverse = reverse_mode
        vehicle.apply_control(control)

    # Register keyboard event handlers
    keyboard.on_release_key("left", on_key_release)
    keyboard.on_release_key("right", on_key_release)
    keyboard.on_release_key("up", on_key_release)
    keyboard.on_release_key("down", on_key_release)
    keyboard.on_press_key("left", on_key_press)
    keyboard.on_press_key("right", on_key_press)
    keyboard.on_press_key("up", on_key_press)
    keyboard.on_press_key("down", on_key_press)




# Function to update the spectator view
def update_spectator_view():
    while True:
        # Get the location and rotation of the vehicle
        vehicle_location = vehicle.get_location()
        vehicle_rotation = vehicle.get_transform().rotation

        # Calculate the spectator view transform
        spectator_location = carla.Location(vehicle_location.x, vehicle_location.y, vehicle_location.z + 2.0)  # Adjust the height offset if needed
        spectator_rotation = carla.Rotation(vehicle_rotation.pitch - 15.0, vehicle_rotation.yaw, vehicle_rotation.roll)  # Adjust the pitch offset if needed
        spectator_transform = carla.Transform(spectator_location, spectator_rotation)

        # Set the spectator view transform
        spectator.set_transform(spectator_transform)

        time.sleep(0.005)




# Autopilot flag parsing
args = parse_arguments()




vehicle = None
cam = None

#enable logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

# Creating a client
client = carla.Client('localhost', 2000)
client.set_timeout(15.0)
client.load_world('Town10HD')
world = client.get_world()


#Create Folder to store data
today = datetime.now()
if today.hour < 10:
    h = "0"+ str(today.hour)
else:
    h = str(today.hour)
if today.minute < 10:
    m = "0"+str(today.minute)
else:
    m = str(today.minute)
directory = "C:\\Users\\Kader\\OneDrive\\Bureau\\WindowsNoEditor\\my work\\TestData" + today.strftime('%Y%m%d_')+ h + m + "_npy"

print(directory)

try:
    os.makedirs(directory)
except:
    print("Directory already exists")
try:
    inputs_file = open(directory + "/inputs.npy","ba+") 
    outputs_file = open(directory + "/outputs.npy","ba+")     
except:
    print("Files could not be opened")
    
#Spawn vehicle
#Get the blueprint concerning a tesla model 3 car
bp = world.get_blueprint_library().find('vehicle.tesla.model3')
#we attribute the role name brax to our blueprint
bp.set_attribute('role_name','brax')
#get a random color
color = random.choice(bp.get_attribute('color').recommended_values)
#put the selected color on our blueprint
bp.set_attribute('color',color)

#get all spawn points
spawn_points = world.get_map().get_spawn_points()
number_of_spawn_points = len(spawn_points)

#select a random spawn point
if 0 < number_of_spawn_points:
    random.shuffle(spawn_points)
    transform = spawn_points[0]
    #spawn our vehicle !
    vehicle = world.spawn_actor(bp,transform)
    print('\nVehicle spawned')
else: 
    #no spawn points 
    logging.warning('Could not found any spawn points')
     
#Adding a RGB camera sensor
WIDTH = 200
HEIGHT = 88
cam_bp = None
#Get blueprint of a camera
cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
#Set attributes 
cam_bp.set_attribute("image_size_x",str(WIDTH))
cam_bp.set_attribute("image_size_y",str(HEIGHT))
cam_bp.set_attribute("fov",str(105))
#Location to attach the camera on the car
cam_location = carla.Location(2,0,1)
cam_rotation = carla.Rotation(0,0,0)
cam_transform = carla.Transform(cam_location,cam_rotation)
#Spawn the camera and attach it to our vehicle 
cam = world.spawn_actor(cam_bp,cam_transform,attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)

#Gets the spectator view to where the vehicle is spawned
spectator = world.get_spectator()
spawn_points[0].location.z = spawn_points[0].location.z+1 #start_point was used to spawn the car but we move 1m up to avoid being on the floor
spectator.set_transform(spawn_points[0])

#Function to convert image to a numpy array
def process_image(image):
    #Get raw image in 8bit format
    raw_image = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    #Reshape image to RGBA
    raw_image = np.reshape(raw_image, (image.height, image.width, 4))


    #Taking only RGB
    processed_image = raw_image[:, :, :3]/255


    return processed_image

#Save required data
def save_image(carla_image):
    image = process_image(carla_image)

    control = vehicle.get_control()
    data = [control.steer, control.throttle, control.brake]
    np.save(inputs_file, image)
    np.save(outputs_file, data)
     

# Enable autopilot or keyboard control based on the flag
if args.autopilot:
    vehicle.set_autopilot(True)
else:
    control_vehicle(vehicle)


#Attach event listeners
cam.listen(save_image)




# Start the spectator view update loop
update_spectator_view_thread = threading.Thread(target=update_spectator_view)
update_spectator_view_thread.start()


# Main loop
try:
    i = 0
    #How much frames do we want to save
    while i < 2500:
        world_snapshot = world.wait_for_tick()
        clear_output(wait=True)
        display(f"{str(i)} frames saved")
        i += 1


except:
    print('\nSimulation error.')

#Destroy everything     
if vehicle is not None:
    if cam is not None:
        cam.stop()
        cam.destroy()
    vehicle.destroy()

#Close everything   
inputs_file.close()
outputs_file.close()
cv2.destroyAllWindows()
print("Data retrieval finished")
print(directory)




