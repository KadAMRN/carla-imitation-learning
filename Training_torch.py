import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from xception import Xception
# Set device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
# Variables
DIRECTORY = "C:\\Users\\Kader\\OneDrive\\Bureau\\WindowsNoEditor\\my work\\TestData20230620_2330_npy"
WIDTH = 200
HEIGHT = 88
EPOCHS = 10
TRAINING_BATCH_SIZE = 2


INPUTS_FILE = open(DIRECTORY + "\\inputs.npy","br") 
OUTPUTS_FILE = open(DIRECTORY + "\\outputs.npy","br")
# Load the data
print("Data loading started")
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
print (len(inputs))
print (len(outputs))
input_np = np.array(inputs)
output_np = np.array(outputs)
print("Data loading finished")
print(input_np.shape)

# Take out the first 400 frames to avoid having the car idle
# input_np = input_np[400:, :, :]
# output_np = output_np[400:, :]
inputs = None
outputs = None

INPUTS_FILE.close()
OUTPUTS_FILE.close()
print("file closed")


# Transform the data
print("data transform started")
transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
])
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.ToTensor(),])
print("Defining model dataset")
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(input_np).float(),
    torch.tensor(output_np).float()
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAINING_BATCH_SIZE, shuffle=True)

# Create the InceptionV3 model
model = Xception()
# model = models.inception_v3(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)  # 3 output classes

print("model created")
# Move the model to the appropriate device
model = model.to(device)
print("model moved to appropriate device")
# Define the loss function
criterion = nn.MSELoss()
print("loss fct defined")
# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-6, nesterov=True)
print("OPT defined")
# Train the model
print("TRAINING STARTED")
for epoch in range(EPOCHS):
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        inputs = inputs.permute(0, 3, 1, 2).to(device)  # Transpose dimensions to [batch_size, 3, height, width]

        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {epoch_loss:.4f}")
print("TRAINING FINISHED")
# Save the trained model
model_file_path = f"C:\\Users\\Kader\\OneDrive\\Bureau\WindowsNoEditor\\my work\\Torch_trained\\test_model_2.pt"
torch.save(model.state_dict(), model_file_path)
print(f"Model saved to {model_file_path}")
