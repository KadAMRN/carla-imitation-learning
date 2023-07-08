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

import argparse

import torch

#autopilot parser
def parse_arguments():
    parser = argparse.ArgumentParser(description='File Parser')
    parser.add_argument('--file', type=str, help='Path to the file')
    return parser.parse_args()

args=parse_arguments()
# Load your PyTorch model
file_path=args.file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(file_path)
model = model.to(device)
model.eval()  # Set the model to evaluation mode





# Function to control the vehicle with autopilot using PyTorch model prediction
def control_vehicle(image,vehicle):
    # Autopilot prediction loop
    # while True:
        #Get raw image in 8bit format
    raw_image = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    #Reshape image to RGBA
    raw_image = np.reshape(raw_image, (image.height, image.width, 4))


    #Taking only RGB
    processed_image = raw_image[:, :, :3]/255
    # Get input data for prediction (e.g., camera image)
    input_data = processed_image  # Replace with your own function to get the input data for prediction
    
    # Preprocess the input data if necessary

    # Convert the input data to a PyTorch tensor
    input_tensor = torch.from_numpy(input_data).unsqueeze(0)
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    # Perform prediction using your PyTorch model
    input_tensor = input_tensor.to(torch.float32)
    input_tensor = input_tensor.to(device)
    print("hello world")
    with torch.no_grad():
        prediction = model(input_tensor)

    print(prediction)

    # Convert the prediction to control values (e.g., steer, throttle, brake)
    steer = prediction[0][0]
    throttle = prediction[0][1]
    brake = prediction[0][2]

    # if brake < 0.1:
    #     brake = 0.0




    # global control
    
    # Update vehicle controls
    control = carla.VehicleControl()
    control.steer = steer.item()
    control.throttle = throttle.item()
    control.brake = brake.item()
    # control.brake = 0.
    if control.throttle > control.brake:
        control.brake = 0.

    control.hand_brake = 0
    control.reverse = 0
        
    print(f"control= {control}")
    # print(vehicle.get_velocity())
    # vehicle.set_autopilot(True)
    vehicle.apply_control(control)
    # return control


        
        # time.sleep(0.01)










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








vehicle = None
cam = None

#enable logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

# Creating a client
client = carla.Client('localhost', 2000)
client.set_timeout(15.0)
client.load_world('Town10HD')
world = client.get_world()


    
#Spawn vehicle
#Get the blueprint concerning a tesla model 3 car
bp = world.get_blueprint_library().find('vehicle.tesla.model3')
#we attribute the role name amrn to our blueprint
bp.set_attribute('role_name','amrn')
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


     




#Attach event listeners
cam.listen(lambda img : control_vehicle(img,vehicle))



# Start the spectator view update loop
update_spectator_view_thread = threading.Thread(target=update_spectator_view)
update_spectator_view_thread.start()

# Main loop
try:
    while True:
        
        
        world.tick()
        print(vehicle.get_control())
        display(f"world looping")

except:
    print('\nSimulation error.')


#Destroy everything     
if vehicle is not None:
    if cam is not None:
        cam.stop()
        cam.destroy()
    vehicle.destroy()

#Close everything   

cv2.destroyAllWindows()





