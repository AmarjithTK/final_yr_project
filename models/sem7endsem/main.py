import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import copy

# ---------------------------
# 1. CONFIGURATION
# ---------------------------
CONFIG = {
    'FROM_DRIVE': True,
    'CSV_FILE': 'kerala_datasetv2.csv',
    'DRIVE_CSV_PATH': '/content/drive/MyDrive/final_yr_project/kerala_datasetv2.csv',
    'COLS': ['P', 'Q'],
    'TIME_RESAMPLE': '15min',
    'HIDDEN_SIZE': 64,
    'LAYERS': 2,
    'BATCH_SIZE': 128,
    'LR': 0.005,
    'EPOCHS': 50,
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'QUANTILE_Q': 0.90,
    'WIENER_WINDOW': 5,
    'PLOT_DAY_NUM': 3  # The specific day (index) to plot predictions for
}

# ---------------------------
# 2. WIENER FILTER
# ---------------------------
def numpy_wiener(x, mysize=None, noise=None):
    x = np.asarray(x)
    if mysize is None: mysize = 3
    N = mysize
    lMean = np.convolve(x, np.ones(N)/N, mode='same')
    lVar = np.convolve(x**2, np.ones(N)/N, mode='same') - lMean**2
    if noise is None: noise = np.mean(lVar)
    res = x - lMean
    res *= (1 - noise / (lVar + 1e-6))
    res += lMean
    return np.where(lVar < noise, lMean, res)

# ---------------------------
# 3. DATA PREPARATION
# ---------------------------
def calculate_window_size(resample_str):
    if 'min' in resample_str:
        mins = int(resample_str.replace('min', ''))
        steps = int((24 * 60) / mins)
    elif 'H' in resample_str:
        hours = int(resample_str.replace('H', ''))
        steps = int(24 / hours)
    else:
        steps = 96
    return steps

def prepare_data():
    if CONFIG['FROM_DRIVE']:
        print("Mounting Google Drive...")
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            csv_path = CONFIG['DRIVE_CSV_PATH']
            print(f"Using dataset from Drive: {csv_path}")
        except:
            print("Error: Cannot mount Drive.")
            return None, None, None, None, None
    else:
        csv_path = CONFIG['CSV_FILE']
        print(f"Using local dataset: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except:
        print("Error: CSV file not found.")
        return None, None, None, None, None

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    df = df[CONFIG['COLS']]
    df = df.resample(CONFIG['TIME_RESAMPLE']).mean().ffill().bfill()

    print(f"Applying Wiener Filter (Window={CONFIG['WIENER_WINDOW']})...")
    for col in CONFIG['COLS']:
        df[col] = numpy_wiener(df[col].values, mysize=CONFIG['WIENER_WINDOW'])

    data = df.values
    scaler = MinMaxScaler((0, 1))
    data_scaled = scaler.fit_transform(data)

    steps_per_day = calculate_window_size(CONFIG['TIME_RESAMPLE'])
    CONFIG['LOOKBACK'] = steps_per_day
    CONFIG['HORIZON'] = steps_per_day

    print(f"Time Step: {CONFIG['TIME_RESAMPLE']} | Steps/Day: {steps_per_day}")

    X, y = [], []
    for i in range(len(data_scaled) - CONFIG['LOOKBACK'] - CONFIG['HORIZON'] + 1):
        X.append(data_scaled[i : i + CONFIG['LOOKBACK']])
        y.append(data_scaled[i + CONFIG['LOOKBACK'] : i + CONFIG['LOOKBACK'] + CONFIG['HORIZON']])

    X, y = np.array(X), np.array(y)

    if len(X) == 0: return None, None, None, None, None

    split = int(len(X) * 0.8)
    return X[:split], y[:split], X[split:], y[split:], scaler

# ---------------------------
# 4. MODEL & TRAINING UTILS
# ---------------------------
class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.9):
        super().__init__()
        self.q = quantile
    def forward(self, preds, target):
        errors = target - preds
        loss = torch.max((self.q - 1) * errors, self.q * errors)
        return torch.mean(loss)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, CONFIG['HIDDEN_SIZE'], batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(CONFIG['HIDDEN_SIZE'], CONFIG['HORIZON'] * output_dim)
        self.output_dim = output_dim
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out.view(-1, CONFIG['HORIZON'], self.output_dim)

def train_model(X_train, y_train, X_val, y_val, loss_type='mse'):
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
    
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    val_loader = DataLoader(val_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False)

    model = LSTMModel(input_dim=2, output_dim=2).to(CONFIG['DEVICE'])
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG['LR'], weight_decay=1e-5)

    if loss_type == 'mae': crit = nn.L1Loss()
    elif loss_type == 'mse': crit = nn.MSELoss()
    elif loss_type == 'huber': crit = nn.HuberLoss(delta=0.1)
    elif loss_type == 'quantile': crit = QuantileLoss(quantile=CONFIG['QUANTILE_Q'])
    
    print(f"Training LSTM ({loss_type})...")
    
    # History storage
    history = {'train': [], 'val': []}
    
    best_loss = float('inf')
    patience = 8
    patience_counter = 0
    best_state = None
    
    for epoch in range(CONFIG['EPOCHS']):
        # --- Training Loop ---
        model.train()
        train_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(CONFIG['DEVICE']), by.to(CONFIG['DEVICE'])
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        history['train'].append(avg_train_loss)

        # --- Validation Loop ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(CONFIG['DEVICE']), by.to(CONFIG['DEVICE'])
                loss = crit(model(bx), by)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        history['val'].append(avg_val_loss)
        
        # Early Stopping Check
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"  -> Early stop at epoch {epoch}")
            break
            
    if best_state: model.load_state_dict(best_state)
    return model, history

def predict(model, X):
    model.eval()
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    preds = []
    with torch.no_grad():
        for b in loader: preds.append(model(b[0].to(CONFIG['DEVICE'])).cpu().numpy())
    return np.concatenate(preds, axis=0)

# ---------------------------
# 5. PLOTTING FUNCTIONS
# ---------------------------
def plot_loss_curves(histories_dict):
    """
    Plots a 2x2 grid of loss curves for the 4 models.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Define plot order
    model_names = ['LSTM (MAE)', 'LSTM (MSE)', 'LSTM (Huber)', 'LSTM (Quantile)']
    
    for ax, name in zip(axes, model_names):
        if name in histories_dict:
            hist = histories_dict[name]
            epochs = range(1, len(hist['train']) + 1)
            
            ax.plot(epochs, hist['train'], 'b-', label='Train Loss', linewidth=2)
            ax.plot(epochs, hist['val'], 'r--', label='Val Loss', linewidth=2)
            
            ax.set_title(f"{name} Training Curve")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compute_metrics(y_true, y_pred):
    y_true_f, y_pred_f = y_true.flatten(), y_pred.flatten()
    mse = mean_squared_error(y_true_f, y_pred_f)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_f, y_pred_f)
    r2 = r2_score(y_true_f, y_pred_f)
    data_range = np.max(y_true_f) - np.min(y_true_f)
    if data_range == 0: data_range = 1e-6
    return {
        "NRMSE (%)": (rmse / data_range) * 100,
        "NMAE (%)": (mae / data_range) * 100,
        "MAPE (%)": np.mean(np.abs((y_true_f - y_pred_f) / (y_true_f + 1e-6))) * 100,
        "R2 Score": r2
    }

def bulk_inverse_scale(data, scaler):
    N, H, F = data.shape
    flat = data.reshape(-1, F)
    flat_inv = scaler.inverse_transform(flat)
    return flat_inv.reshape(N, H, F)

# ---------------------------
# 6. MAIN
# ---------------------------
def main():
    X_train, y_train, X_val, y_val, scaler = prepare_data()
    if X_train is None: return

    # --- Train Models & Collect History ---
    model_mae, hist_mae = train_model(X_train, y_train, X_val, y_val, 'mae')
    model_mse, hist_mse = train_model(X_train, y_train, X_val, y_val, 'mse')
    model_huber, hist_huber = train_model(X_train, y_train, X_val, y_val, 'huber')
    model_quant, hist_quant = train_model(X_train, y_train, X_val, y_val, 'quantile')

    # --- Plot Loss Curves ---
    print("\nPlotting Training/Validation Loss Curves...")
    all_histories = {
        'LSTM (MAE)': hist_mae,
        'LSTM (MSE)': hist_mse,
        'LSTM (Huber)': hist_huber,
        'LSTM (Quantile)': hist_quant
    }
    plot_loss_curves(all_histories)

    # --- Predict ---
    print("Generating Predictions...")
    pred_base = X_val 
    pred_mae = predict(model_mae, X_val)
    pred_mse = predict(model_mse, X_val)
    pred_huber = predict(model_huber, X_val)
    pred_quant = predict(model_quant, X_val)

    # --- Inverse Scale ---
    y_real = bulk_inverse_scale(y_val, scaler)
    p_base_real = bulk_inverse_scale(pred_base, scaler)
    p_mae_real = bulk_inverse_scale(pred_mae, scaler)
    p_mse_real = bulk_inverse_scale(pred_mse, scaler)
    p_huber_real = bulk_inverse_scale(pred_huber, scaler)
    p_quant_real = bulk_inverse_scale(pred_quant, scaler)

    # --- Metrics Table ---
    targets = ['Active Power (P)', 'Reactive Power (Q)']
    results = []
    models_dict = {
        'Baseline': p_base_real,
        'LSTM (MAE)': p_mae_real,
        'LSTM (MSE)': p_mse_real,
        'LSTM (Huber)': p_huber_real,
        'LSTM (Quantile)': p_quant_real
    }

    for i, name in enumerate(targets):
        for model_name, pred_data in models_dict.items():
            metrics = compute_metrics(y_real[:,:,i], pred_data[:,:,i])
            for metric, value in metrics.items():
                results.append({'Target': name, 'Model': model_name, 'Metric': metric, 'Value': value})

    df_res = pd.DataFrame(results)
    pivot = df_res.pivot_table(index=['Target', 'Metric'], columns='Model', values='Value')
    pivot = pivot[['Baseline', 'LSTM (MAE)', 'LSTM (MSE)', 'LSTM (Huber)', 'LSTM (Quantile)']]

    print("\n" + "="*100)
    print(f"{'FINAL METRICS':^100}")
    print("="*100)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(pivot)
    print("-" * 100)

    # --- PLOTTING PREDICTIONS (Deterministic) ---
    steps_per_day = CONFIG['HORIZON']
    day_num = CONFIG['PLOT_DAY_NUM']
    idx = (day_num - 1) * steps_per_day
    
    if idx >= len(X_val): idx = len(X_val) - 1

    print(f"\nPlotting Predictions for Day {day_num} (Validation Index: {idx})")

    color_map = {
        'LSTM (MAE)': 'blue', 'LSTM (MSE)': 'green', 
        'LSTM (Huber)': 'orange', 'LSTM (Quantile)': 'purple'
    }

    comparison_models = [
        ('LSTM (MAE)', models_dict['LSTM (MAE)']),
        ('LSTM (MSE)', models_dict['LSTM (MSE)']),
        ('LSTM (Huber)', models_dict['LSTM (Huber)']),
        ('LSTM (Quantile)', models_dict['LSTM (Quantile)'])
    ]

    for i, name in enumerate(targets):
        actual = y_real[idx, :, i]
        baseline = models_dict['Baseline'][idx, :, i]
        
        # Generate separate plots for each model comparison
        for model_name, model_pred in comparison_models:
            pred = model_pred[idx, :, i]
            c = color_map.get(model_name, 'gray')
            
            plt.figure(figsize=(12, 6))
            plt.plot(actual, 'k-', linewidth=2.5, label='Actual', alpha=0.9)
            plt.plot(baseline, 'r:', linewidth=2, label='Baseline')
            plt.plot(pred, linestyle='--', linewidth=2, label=model_name, color=c, alpha=0.9)
            
            plt.title(f"{name} - {model_name} vs Baseline - Day {day_num}")
            plt.ylabel("Value")
            plt.xlabel("Time Steps")
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()