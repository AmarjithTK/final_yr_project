import os
import torch

# Set this to True if running locally, False if running in Colab/Drive
LOCAL_RUN = True

# Google Drive base path for persistent storage

# Data and model configuration
if LOCAL_RUN:
    BASE_PATH = "./"

    CSV_FILE = "../../datasets/residential_3months.csv"
else:
    BASE_PATH = "/content/drive/My Drive/colab_persistent_storage"

    CSV_FILE = "datasets/residential_3months.csv"


os.makedirs(BASE_PATH, exist_ok=True)



LOOKBACK = 24
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 1
TRAIN_SPLIT = 0.8
TARGET_COLS = ["P", "Q"]           # Predict both P and Q

# File paths for model, scaler, and validation data
MODEL_PATH = os.path.join(BASE_PATH, "lstm_model.pt")
SCALER_PATH = os.path.join(BASE_PATH, "scaler.npy")
VALDATA_PATH = os.path.join(BASE_PATH, "val_data.npz")

# Device configuration
DEVICE = torch.device("cpu")  # Use CPU for compatibility

# Utility: Google Drive mount function
def mount_drive(drive_module):
    drive_module.mount('/content/drive')