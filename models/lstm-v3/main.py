import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

import config

# ---------------------------
# LSTM Model Definition
# ---------------------------
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

# ---------------------------
# Dataset Class
# ---------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------
# Sequence Creation Utility
# ---------------------------
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

# ---------------------------
# 1. Train and Evaluate Model
# ---------------------------
def train_and_evaluate():
    # Load and preprocess data
    df = pd.read_csv(config.CSV_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    data = df[config.TARGET_COLS].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    np.save(config.SCALER_PATH, [scaler.min_, scaler.scale_])

    X, y = create_sequences(data_scaled, config.LOOKBACK)
    split_idx = int(len(X) * config.TRAIN_SPLIT)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_ds = TimeSeriesDataset(X_train, y_train)
    val_ds = TimeSeriesDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    model = LSTMModel(
        input_size=2,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        output_size=2
    ).to(config.DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    # Training loop
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
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
                X_batch, y_batch = X_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{config.EPOCHS}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Save model, scaler, and validation data
    torch.save(model, config.MODEL_PATH)
    np.save(config.SCALER_PATH, [scaler.min_, scaler.scale_])
    np.savez(config.VALDATA_PATH, X_val=X_val, y_val=y_val)

    # Evaluation
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(config.DEVICE)
            outputs = model(X_batch)
            preds.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())
    preds = scaler.inverse_transform(preds)
    actuals = scaler.inverse_transform(actuals)
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

# ---------------------------
# 2. Predict P, Q for Future Timestamps
# ---------------------------
def predict_pq_for_timestamps(input_df):
    """
    input_df: DataFrame with at least 'timestamp', 'P', 'Q' columns for lookback window.
    Returns: DataFrame with predicted P, Q for each input timestamp.
    """
    # Load model and scaler
    model = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    scaler = MinMaxScaler()
    scaler.min_, scaler.scale_ = np.load(config.SCALER_PATH, allow_pickle=True)
    model.eval()

    # Prepare input data for prediction
    data = input_df[config.TARGET_COLS].values
    data_scaled = scaler.transform(data)
    # Only last LOOKBACK rows are used for prediction
    X_pred = []
    for i in range(len(data_scaled) - config.LOOKBACK + 1):
        X_pred.append(data_scaled[i:i+config.LOOKBACK])
    X_pred = np.array(X_pred)
    X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32).to(config.DEVICE)

    # Predict
    with torch.no_grad():
        preds = model(X_pred_tensor).cpu().numpy()
    preds = scaler.inverse_transform(preds)

    # Prepare output DataFrame
    pred_timestamps = input_df["timestamp"].iloc[config.LOOKBACK:].reset_index(drop=True)
    pred_df = pd.DataFrame(preds, columns=["Predicted_P", "Predicted_Q"])
    pred_df["timestamp"] = pred_timestamps
    return pred_df[["timestamp", "Predicted_P", "Predicted_Q"]]

# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    # To train and evaluate:
    train_and_evaluate()

    # To predict for new timestamps:
    # Create a sample future DataFrame with dummy data for demonstration
    import datetime

    # Generate 30 minutes of future timestamps (2 samples, 15 min interval)
    last_time = pd.Timestamp("2023-01-01 00:00:00")
    timestamps = [last_time + datetime.timedelta(minutes=15 * i) for i in range(26)]  # 26 for lookback + 1

    # Dummy P and Q values (replace with real data as needed)
    P_values = np.linspace(100, 120, 26)
    Q_values = np.linspace(50, 60, 26)

    future_df = pd.DataFrame({
        "timestamp": timestamps,
        "P": P_values,
        "Q": Q_values
    })

    # Call the prediction function
    pred_df = predict_pq_for_timestamps(future_df)
    print(pred_df)