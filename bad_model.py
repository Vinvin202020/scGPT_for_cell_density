import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

filename = r"C:\Users\Antoi\Documents\Master\ML\project2\cell_density_50.csv"  #load xenium data
output = pd.read_csv(filename)
densities = output.iloc[:, -1]

def bad_model(densities):
    avg_density = np.median(densities)
    N = len(densities)
    loss = np.sum(np.abs(densities-avg_density))/N
    return loss

loss = bad_model(densities)
print(f"loss={loss}")

filename = r"C:\Users\Antoi\Documents\Master\ML\project2\data_loaded.csv"  #load xenium data
data = pd.read_csv(filename)
# Assuming the input columns are 'x' and 'y' and the output column is 'density'
x = data.iloc[:, 7:-2].values  # Input features
n_features = x.shape[1]
y = densities   # Target (cell densities)

# 2. Data preprocessing: Standardize the input features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
# Convert y_train to a NumPy array and then to a tensor
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)

# Similarly for y_test
y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32).unsqueeze(1)

# 3. Define the neural network model
class CellDensityNN(nn.Module):
    def __init__(self):
        super(CellDensityNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, 32),   # Input layer: 2 features -> 32 neurons
            nn.ReLU(),          # Activation function
            nn.Linear(32, 16),  # Hidden layer: 32 -> 16 neurons
            nn.ReLU(),
            nn.Linear(16, 1)    # Output layer: 16 -> 1 output
        )
    
    def forward(self, x):
        return self.layers(x)

# Instantiate the model
model = CellDensityNN()

# 4. Define the loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. Train the model
epochs = 1000
train_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()       # Zero the gradients
    outputs = model(x_train_tensor)  # Forward pass
    loss = criterion(outputs, y_train_tensor)  # Compute loss
    loss.backward()             # Backpropagation
    optimizer.step()            # Update weights
    
    train_losses.append(loss.item())
    if (epoch+1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Plot training loss
plt.plot(range(epochs), train_losses)
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("Training Loss Curve")
plt.show()

# 6. Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_train = model(x_train_tensor)
    y_pred_test = model(x_test_tensor)

# Convert predictions to numpy arrays
y_pred_train = y_pred_train.squeeze().numpy()
y_pred_test = y_pred_test.squeeze().numpy()

# Calculate mean squared error
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")

train_mse = mean_absolute_error(y_train, y_pred_train)
test_mse = mean_absolute_error(y_test, y_pred_test)

print(f"Train MAE: {train_mse:.4f}")
print(f"Test MAE: {test_mse:.4f}")

# 7. Plot actual vs predicted densities
plt.scatter(y_test, y_pred_test, alpha=0.7)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Perfect fit line
plt.xlabel("Actual Densities")
plt.ylabel("Predicted Densities")
plt.title("Actual vs Predicted Densities")
#plt.show()