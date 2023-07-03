import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from xception import Xception
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description='Autopilot Flag Parser')
    parser.add_argument('--file', type=str, help='Path to the file')
    parser.add_argument('--data_path', type=str, help='Path to the training file')
    return parser.parse_args()




# Parse the command-line arguments
args = parse_arguments()

# Access the file path argument



# Use the file path in your code
# For example, print the file path


# Set device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Variables
DIRECTORY = args.data_path #"C:\\Users\\Kader\\OneDrive\\Bureau\\WindowsNoEditor\\my work\\TestData20230703_1654_npy" TestData20230618_2317_npy
model_file_path = args.file
print("File path:", model_file_path)
WIDTH = 200
HEIGHT = 88
EPOCHS = 15
TRAINING_BATCH_SIZE = 10
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
batch_losses = []
batch_accuracies = []
epoch_losses = []
epoch_accuracies = []

for epoch in range(EPOCHS):
    running_loss = 0.0
    total_samples = 0
    total_correct = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        inputs = inputs.permute(0, 3, 1, 2).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (outputs == labels).sum().item()

        batch_accuracy = (outputs == labels).sum().item() / labels.size(0)
        batch_accuracies.append(batch_accuracy)

        # Print batch loss and accuracy
        if batch_idx % 100 == 0:
            batch_loss = loss.item() * inputs.size(0)
            print(f"Epoch {epoch+1}/{EPOCHS} Batch {batch_idx}/{len(train_loader)} Loss: {batch_loss:.4f} Accuracy: {batch_accuracy:.4f}")

    epoch_loss = running_loss / len(train_dataset)
    epoch_accuracy = total_correct / total_samples
    epoch_losses.append(epoch_loss)
    epoch_accuracies.append(epoch_accuracy)

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f}")

print("Training finished")

# Plot batch loss
plt.plot(batch_losses)
plt.title("Batch Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.show()

# Plot batch accuracy
plt.plot(batch_accuracies)
plt.title("Batch Accuracy")
plt.xlabel("Batch")
plt.ylabel("Accuracy")
plt.show()

# Plot epoch loss
plt.plot(epoch_losses)
plt.title("Epoch Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Plot epoch accuracy
plt.plot(epoch_accuracies)
plt.title("Epoch Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

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
