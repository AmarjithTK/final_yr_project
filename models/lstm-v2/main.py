from google.colab import drive
import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------------
# Google Drive Persistence Setup
# ---------------------------
def mount_drive():
    drive.mount('/content/drive')

# Set a base path in Google Drive
BASE_PATH = "/content/drive/My Drive/colab_persistent_storage"
os.makedirs(BASE_PATH, exist_ok=True)

def save_file(filename, content, mode="w"):
    path = os.path.join(BASE_PATH, filename)
    with open(path, mode) as f:
        f.write(content)
    print(f"✅ Saved at: {path}")

def load_file(filename, mode="r"):
    path = os.path.join(BASE_PATH, filename)
    if os.path.exists(path):
        with open(path, mode) as f:
            return f.read()
    else:
        print("⚠️ File not found")
        return None

def remove_file(filename):
    path = os.path.join(BASE_PATH, filename)
    if os.path.exists(path):
        os.remove(path)
        print(f"🗑️ Removed: {path}")
    else:
        print("⚠️ File not found")

def clear_storage():
    if os.path.exists(BASE_PATH):
        shutil.rmtree(BASE_PATH)
        os.makedirs(BASE_PATH, exist_ok=True)
        print("🧹 Storage cleared.")

# ---------------------------
# LSTM Model Training & Evaluation
# ---------------------------
# 🔧 Arguments (edit here)
CSV_FILE = "../../datasets/residential_3months.csv"
LOOKBACK = 24
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 1
TRAIN_SPLIT = 0.8
TARGET_COLS = ["P", "Q"]           # Predict both P and Q

MODEL_PATH = os.path.join(BASE_PATH, "lstm_model.pt")
SCALER_PATH = os.path.join(BASE_PATH, "scaler.npy")
VALDATA_PATH = os.path.join(BASE_PATH, "val_data.npz")

device = torch.device("cpu")  # force CPU for compatibility

class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    print("Loading saved model and scaler from Google Drive...")
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    scaler = MinMaxScaler()
    scaler.min_, scaler.scale_ = np.load(SCALER_PATH, allow_pickle=True)
    model.eval()
    skip_training = True
else:
    skip_training = False

if not skip_training:
    # Step 2: Load & preprocess
    df = pd.read_csv(CSV_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    data = df[TARGET_COLS].values  # Only P and Q

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    np.save(SCALER_PATH, [scaler.min_, scaler.scale_])  # Save scaler params

    # Step 3: Create sequences
    def create_sequences(data, lookback):
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback])
            y.append(data[i+lookback])
        return np.array(X), np.array(y)

    X, y = create_sequences(data_scaled, LOOKBACK)

    # Train/val split
    split_idx = int(len(X) * TRAIN_SPLIT)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Step 4: Torch Dataset
    class TimeSeriesDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_ds = TimeSeriesDataset(X_train, y_train)
    val_ds = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Step 5: LSTM Model
    model = LSTMModel(input_size=2, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Step 6: Training
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Save the model and scaler to Google Drive
    torch.save(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    np.save(SCALER_PATH, [scaler.min_, scaler.scale_])
    print(f"Scaler saved to {SCALER_PATH}")

    # Save validation data for later evaluation
    np.savez(VALDATA_PATH, X_val=X_val, y_val=y_val)
    print(f"Validation data saved to {VALDATA_PATH}")

else:
    # Load validation data for evaluation
    val_data = np.load(VALDATA_PATH)
    X_val, y_val = val_data["X_val"], val_data["y_val"]

    class TimeSeriesDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    val_ds = TimeSeriesDataset(X_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Step 7: Evaluation
model.eval()
preds, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds.extend(outputs.cpu().numpy())
        actuals.extend(y_batch.numpy())

preds = scaler.inverse_transform(preds)
actuals = scaler.inverse_transform(actuals)

# Calculate RMSE and MAE for each target
rmse = np.sqrt(mean_squared_error(actuals, preds, multioutput='raw_values'))
mae = mean_absolute_error(actuals, preds, multioutput='raw_values')
print(f"Validation RMSE (P, Q): {rmse}, MAE (P, Q): {mae}")

# Plot P vs t and Q vs t
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(actuals[:,0], label="True P")
plt.plot(preds[:,0], label="Predicted P")
plt.legend()
plt.title("True vs Predicted P (Validation)")

plt.subplot(2,1,2)
plt.plot(actuals[:,1], label="True Q")
plt.plot(preds[:,1], label="Predicted Q")
plt.legend()
plt.title("True vs Predicted Q (Validation)")
plt.tight_layout()
plt.show()
