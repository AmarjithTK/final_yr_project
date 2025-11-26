import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ---------------------------
# 1. HAND-TUNED CONFIGURATION
# ---------------------------
HYPERPARAMS = {
    'CSV_FILE': 'microgrid_data.csv',
    'FEATURE_COLS': ['P', 'Q', 'V', 'I', 'PF'],
    'TARGET_COLS': ['P', 'Q'],
    
    # Time Settings
    'LOOKBACK': 96,             # Past 24 hours (96 * 15min)
    'HORIZON': 96,              # Predict Next 24 hours
    
    # Model Architecture
    'HIDDEN_SIZE': 128,         
    'NUM_LAYERS': 3,            
    'DROPOUT': 0.2,             
    'BIDIRECTIONAL': True,      
    
    # Training
    'BATCH_SIZE': 32,
    'LR': 0.001,
    'EPOCHS': 100,
    'PATIENCE': 10,             
    
    # Uncertainty (Quantile Regression)
    'QUANTILES': [0.1, 0.5, 0.9] # 10th (Low), 50th (Median), 90th (High)
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------
# 2. FEATURE ENGINEERING
# ---------------------------
def add_cyclical_time_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    seconds_in_day = 24 * 60 * 60
    df['seconds'] = df['timestamp'].dt.hour * 3600 + df['timestamp'].dt.minute * 60
    
    df['sin_time'] = np.sin(2 * np.pi * df['seconds'] / seconds_in_day)
    df['cos_time'] = np.cos(2 * np.pi * df['seconds'] / seconds_in_day)
    return df

# ---------------------------
# 3. DATASET & LOSS FUNCTION
# ---------------------------
class MicrogridDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_multistep_sequences(input_data, target_data, lookback, horizon):
    X, y = [], []
    for i in range(len(input_data) - lookback - horizon + 1):
        X.append(input_data[i : i + lookback])
        y.append(target_data[i + lookback : i + lookback + horizon])
    return np.array(X), np.array(y)

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        loss = 0
        for i, q in enumerate(self.quantiles):
            # Slice prediction for specific quantile
            pred_slice = preds[:, :, :, i]
            errors = target - pred_slice
            loss += torch.max((q - 1) * errors, q * errors).mean()
        return loss

# ---------------------------
# 4. ADVANCED LSTM MODEL
# ---------------------------
class AdvancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, horizon, dropout, bidirectional, num_quantiles):
        super(AdvancedLSTM, self).__init__()
        self.horizon = horizon
        self.output_size = output_size
        self.num_quantiles = num_quantiles
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=bidirectional
        )
        
        self.fc = nn.Linear(
            hidden_size * self.num_directions, 
            horizon * output_size * num_quantiles
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        out = self.fc(last_step)
        
        # Reshape to (Batch, Horizon, Features, Quantiles)
        out = out.view(-1, self.horizon, self.output_size, self.num_quantiles)
        return out

# ---------------------------
# 5. TRAINING PIPELINE (FIXED)
# ---------------------------
def train_model():
    print("Loading and Preprocessing Data...")
    df = pd.read_csv(HYPERPARAMS['CSV_FILE'])
    df = add_cyclical_time_features(df)
    
    input_cols = HYPERPARAMS['FEATURE_COLS'] + ['sin_time', 'cos_time']
    target_cols = HYPERPARAMS['TARGET_COLS']
    
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    data_X_scaled = scaler_X.fit_transform(df[input_cols])
    data_y_scaled = scaler_y.fit_transform(df[target_cols])
    
    X, y = create_multistep_sequences(data_X_scaled, data_y_scaled, HYPERPARAMS['LOOKBACK'], HYPERPARAMS['HORIZON'])
    
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    train_loader = DataLoader(MicrogridDataset(X_train, y_train), batch_size=HYPERPARAMS['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(MicrogridDataset(X_val, y_val), batch_size=HYPERPARAMS['BATCH_SIZE'], shuffle=False)
    
    model = AdvancedLSTM(
        input_size=X_train.shape[2],
        hidden_size=HYPERPARAMS['HIDDEN_SIZE'],
        num_layers=HYPERPARAMS['NUM_LAYERS'],
        output_size=2, 
        horizon=HYPERPARAMS['HORIZON'],
        dropout=HYPERPARAMS['DROPOUT'],
        bidirectional=HYPERPARAMS['BIDIRECTIONAL'],
        num_quantiles=len(HYPERPARAMS['QUANTILES'])
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMS['LR'])
    criterion = QuantileLoss(HYPERPARAMS['QUANTILES'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print("Starting Training with Early Stopping...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(HYPERPARAMS['EPOCHS']):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                preds = model(batch_X)
                val_loss += criterion(preds, batch_y).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model, 'advanced_lstm.pth')
        else:
            patience_counter += 1
            if patience_counter >= HYPERPARAMS['PATIENCE']:
                print("Early stopping triggered.")
                break
                
    # --- FIX: Added weights_only=False to allow loading the custom class ---
    model = torch.load('advanced_lstm.pth', weights_only=False)
    return model, scaler_X, scaler_y, val_loader, data_X_scaled

# ---------------------------
# 6. PLOT 1: DAILY VALIDATION
# ---------------------------
def plot_daily_validation(model, val_loader, scaler_y):
    model.eval()
    batch_X, batch_y = next(iter(val_loader))
    batch_X = batch_X.to(DEVICE)
    
    with torch.no_grad():
        preds = model(batch_X).cpu().numpy()
        actuals = batch_y.numpy()
        
    idx = 0
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    titles = ['Active Power (P)', 'Reactive Power (Q)']
    
    for i in range(2):
        min_val = scaler_y.data_min_[i]
        scale_val = scaler_y.scale_[i]
        
        def unscale(x): return x / scale_val + min_val 
        
        p10 = unscale(preds[idx, :, i, 0])
        p50 = unscale(preds[idx, :, i, 1])
        p90 = unscale(preds[idx, :, i, 2])
        act = unscale(actuals[idx, :, i])
        
        ax = axes[i]
        ax.plot(act, 'k-', linewidth=2, label='Actual')
        ax.plot(p50, 'b--', linewidth=2, label='Predicted (Median)')
        ax.fill_between(range(len(p50)), p10, p90, color='blue', alpha=0.2, label='10-90% Uncertainty')
        
        ax.set_title(f"Daily Forecast: {titles[i]}")
        ax.set_xlabel("Time (15-min intervals)")
        ax.set_ylabel("Power (kW / kVAR)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.suptitle("VALIDATION: 24-Hour Look-Ahead with Uncertainty", fontsize=14)
    plt.tight_layout()
    plt.show()

# ---------------------------
# 7. PLOT 2: WEEKLY FORECAST
# ---------------------------
def plot_weekly_forecast(model, data_X_scaled, scaler_y):
    print("Generating 7-Day Recursive Forecast...")
    model.eval()
    
    current_input = torch.tensor(data_X_scaled[-HYPERPARAMS['LOOKBACK']:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    weekly_preds = []
    
    for _ in range(7):
        with torch.no_grad():
            pred_raw = model(current_input)
            pred_median = pred_raw[:, :, :, 1]
            weekly_preds.append(pred_median.cpu().numpy())
            
            next_input = current_input.cpu().clone().numpy()
            next_input[0, :, 0:2] = pred_median.cpu().numpy()[0]
            current_input = torch.tensor(next_input, dtype=torch.float32).to(DEVICE)
            
    weekly_flat = np.concatenate(weekly_preds, axis=1).reshape(-1, 2)
    weekly_real = scaler_y.inverse_transform(weekly_flat)
    
    fig, ax = plt.subplots(figsize=(15, 6))
    time_steps = range(len(weekly_real))
    
    ax.plot(time_steps, weekly_real[:, 0], color='tab:blue', label='Active Power (P)')
    ax.plot(time_steps, weekly_real[:, 1], color='tab:red', alpha=0.7, label='Reactive Power (Q)')
    
    for i in range(0, len(weekly_real), 96):
        ax.axvline(x=i, color='gray', linestyle=':', alpha=0.5)
        
    ax.set_title("PROJECTION: 7-Day Recursive Forecast (P & Q)")
    ax.set_xlabel("Time (15-min intervals over 7 days)")
    ax.set_ylabel("Power")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

# ---------------------------
# MAIN EXECUTION
# ---------------------------
if __name__ == "__main__":
    try:
        # 1. Train
        model, scaler_X, scaler_y, val_loader, data_X_scaled = train_model()
        
        # 2. Daily Plot
        plot_daily_validation(model, val_loader, scaler_y)
        
        # 3. Weekly Plot
        plot_weekly_forecast(model, data_X_scaled, scaler_y)
        
    except FileNotFoundError:
        print("Error: 'microgrid_data.csv' not found. Please ensure CSV exists.")