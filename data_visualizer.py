import numpy as np 
import cv2 
import matplotlib.pyplot as plt

DIRECTORY="C:\\Users\\Kader\\OneDrive\\Bureau\\WindowsNoEditor\\my work\\TestData20230615_0005_npy"





#We open the training data
INPUTS_FILE = open(DIRECTORY + "\\inputs.npy","br") 
OUTPUTS_FILE = open(DIRECTORY + "\\outputs.npy","br")  

#We get the data in
inputs = []
outputs = []

while True:
    try:
        input = np.load(INPUTS_FILE)
        inputs.append(input)
    except: 
        break
while True:
    try:
        output = np.load(OUTPUTS_FILE)
        outputs.append(output)
    except: 
        break


input_np = np.array(inputs)
output_np = np.array(outputs)

print(input_np.shape)    
print(output_np.shape)    
    
#we close everything
inputs = None
outputs = None

INPUTS_FILE.close()
OUTPUTS_FILE.close()

for i in range (input_np.shape[0]):

    image = input_np[i]
    # image = image[:, :, :3]*255
    plt.imshow(image)
    plt.axis('off')

# Adjust the spacing and layout

plt.show()