"""
SafeWatch Configuration
=======================
All tunable parameters for the SafeWatch pipeline live here.
No magic numbers scattered across files.
"""

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "safewatch_lstm.pt")
SCALER_PATH = os.path.join(MODELS_DIR, "safewatch_scaler.pkl")

# --- LSTM Model Architecture ---
# These MUST match the architecture used during training.
INPUT_SIZE = 18         # Features per frame: 9 spatial cells x 2 stats
HIDDEN_SIZE = 64        # LSTM hidden state dimensionality
NUM_LAYERS = 2          # Stacked LSTM layers
DROPOUT = 0.3           # Dropout between LSTM layers

# --- Optical Flow Features ---
FEATURE_NAMES = [
    f"cell{i+1}_{stat}"
    for i in range(9)
    for stat in ["mean_mag", "std_mag"]
]

# --- Sliding Window ---
WINDOW_SIZE = 30        # Number of frames per sequence (6 seconds at 5 fps)
STRIDE = 5             # Frames to advance between predictions (1 second at 5 fps)
FPS = 5                # Target frame rate pulled from RTSP stream

# --- Alert Thresholds ---
ALERT_THRESHOLD = 0.7   # Score >= 0.7 → RED alert (fight detected)
WARNING_THRESHOLD = 0.5 # Score >= 0.5 and < 0.7 → YELLOW warning (suspicious)

# --- Score Fusion Weights ---
# LSTM (primary) + Autoencoder (secondary) weighted combination.
# These are untuned initial values — revisit after autoencoder is built.
LSTM_WEIGHT = 0.7       # Primary model weight        # TODO: NOT YET WIRED — reserved for score fusion
AUTOENCODER_WEIGHT = 0.3  # Secondary model weight    # TODO: NOT YET WIRED — reserved for score fusion

# --- Motion Gate ---
# Minimum pixel-difference threshold to consider a frame as "has motion".
# Frames below this are skipped entirely before any AI processing.
MOTION_THRESHOLD = 2   # Pixel intensity difference threshold
MOTION_MIN_AREA = 500   # Minimum contiguous changed pixels  # TODO: NOT YET WIRED — reserved for contour-based gate

# --- Server ---
HTTP_PORT = 8080        # Dashboard + MJPEG feeds served on this port
WS_PORT = 8765          # WebSocket alert server port
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

# --- Cameras ---
# Define your camera sources here. Set source to None for inactive cameras.
# source can be: a local file path, an RTSP URL, or a webcam index (0, 1, ...)
CAMERAS = [
    {"id": "Camera 1", "source": "1MVS2QPWbHc_2.avi"},
    {"id": "Camera 2", "source": None},
    {"id": "Camera 3", "source": None},
    {"id": "Camera 4", "source": None},
]
