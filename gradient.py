import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("train.csv").dropna()
df["Type"] = LabelEncoder().fit_transform(df["Type"])
df["Date"] = pd.to_datetime(df["Date"]).map(pd.Timestamp.toordinal)

features = ['Store', 'Dept', 'Date', 'IsHoliday', 'Temperature', 'Fuel_Price',
            'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
            'CPI', 'Unemployment', 'Type', 'Size']
X = df[features]
y = df["Weekly_Sales"]

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Creating a Custom Dataset Class
class SalesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(SalesDataset(X_train, y_train), batch_size=32, shuffle=True)  # smaller batch
val_loader = DataLoader(SalesDataset(X_val, y_val), batch_size=64)

#defining the NN model
class SalesModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x): return self.net(x)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SalesModel(X.shape[1]).to(device)
criterion = nn.MSELoss() # Mean Squared Error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Gradient Accumulation Setup
accumulation_steps = 4  # Simulates 32 × 4 = 128 batch size

train_losses, val_losses = [], []

# Training Loop with Gradient Accumulation
for epoch in range(10):
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()

    for i, (batch_X, batch_y) in enumerate(train_loader):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss = loss / accumulation_steps  # Normalize loss
        loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            output = model(batch_X)
            loss = criterion(output, batch_y)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Gradient Accumulation Training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("gradient_accumulation_loss.png")
plt.show()
