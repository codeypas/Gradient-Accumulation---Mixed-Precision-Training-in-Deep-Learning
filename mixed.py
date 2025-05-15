import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("train.csv").fillna(0)
df["Type"] = LabelEncoder().fit_transform(df["Type"])
df["IsHoliday"] = df["IsHoliday"].astype(int)

features = [
    "Store", "Dept", "IsHoliday", "Temperature", "Fuel_Price", "CPI",
    "Unemployment", "Size", "Type"
]
target = "Weekly_Sales"

X_train, X_valid, y_train, y_valid = train_test_split(
    df[features], df[target], test_size=0.2, random_state=42
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)

# Creating a Custom Dataset
class SalesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(SalesDataset(X_train, y_train), batch_size=64, shuffle=True)
valid_loader = DataLoader(SalesDataset(X_valid, y_valid), batch_size=64)

# Defining Neural Network Model
class SalesModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x): return self.network(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SalesModel(X_train.shape[1]).to(device)
criterion = nn.MSELoss() #mean squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = GradScaler()

train_losses, val_losses = [], []

# Training with Mixed Precision
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device).view(-1, 1)

        optimizer.zero_grad()
        with autocast():
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in valid_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).view(-1, 1)
            with autocast():
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            valid_loss += loss.item()
    avg_val_loss = valid_loss / len(valid_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Mixed Precision Training (float16 + float32)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mixed_precision_loss.png")  # optional
plt.show()
