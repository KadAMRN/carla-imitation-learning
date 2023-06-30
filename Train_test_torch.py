import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from xception import Xception
from sklearn.model_selection import train_test_split

import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Process a file.')

# Add the file path argument
parser.add_argument('file', type=str, help='Path to the file')

# Parse the command-line arguments
args = parser.parse_args()

# Access the file path argument
model_file_path = args.file

# Use the file path in your code
# For example, print the file path
print("File path:", model_file_path)


# Set device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Variables
DIRECTORY = "C:\\Users\\Kader\\OneDrive\\Bureau\\WindowsNoEditor\\my work\\TestData20230620_2330_npy"
WIDTH = 200
HEIGHT = 88
EPOCHS = 10
TRAINING_BATCH_SIZE = 2
TEST_SIZE = 0.2  # Percentage of data to use for testing

INPUTS_FILE = open(DIRECTORY + "\\inputs.npy", "br") 
OUTPUTS_FILE = open(DIRECTORY + "\\outputs.npy", "br")

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

input_np = np.array(inputs)
output_np = np.array(outputs)
print("Data loading finished")
print(input_np.shape)

inputs = None
outputs = None

INPUTS_FILE.close()
OUTPUTS_FILE.close()
print("File closed")

# Split the data into training and testing sets
input_train, input_test, output_train, output_test = train_test_split(
    input_np, output_np, test_size=TEST_SIZE, random_state=42)

# Transform the data
print("Data transform started")
transform = transforms.Compose([
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor(),
])

print("Defining model dataset")
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(input_train).float(),
    torch.tensor(output_train).float()
)
test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(input_test).float(),
    torch.tensor(output_test).float()
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAINING_BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=TRAINING_BATCH_SIZE, shuffle=False)

# Create the model
model = Xception()
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)  # 3 output classes
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-6, nesterov=True)

# Training loop
print("Training started")
for epoch in range(EPOCHS):
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        inputs = inputs.permute(0, 3, 1, 2).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {epoch_loss:.4f}")

print("Training finished")

# Evaluate the model on the test set
model.eval()
test_loss = 0.0

with torch.no_grad():
    for test_inputs, test_labels in test_loader:
        test_inputs = test_inputs.to(device)
        test_inputs = test_inputs.permute(0, 3, 1, 2).to(device)
        test_labels = test_labels.to(device)

        test_outputs = model(test_inputs)
        test_loss += criterion(test_outputs, test_labels).item() * test_inputs.size(0)

test_loss /= len(test_dataset)
print(f"Testing Loss: {test_loss:.4f}")

# Save the trained model
# model_file_path = "C:\\Users\\Kader\\OneDrive\\Bureau\\WindowsNoEditor\\my work\\Torch_trained\\test_model_3.pt"
# torch.save(model.state_dict(), model_file_path)
torch.save(model, model_file_path)
print(f"Model saved to {model_file_path}")
