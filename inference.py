"""
SafeWatch Inference Module
==========================
Loads the trained LSTM fight-detection model and runs inference
on optical flow feature sequences.

This module is the bridge between the trained model and the live
stream pipeline. It does NOT train anything — inference only.

Usage:
    from inference import SafeWatchPredictor

    predictor = SafeWatchPredictor()

    # Single sequence — shape (30, 18)
    score = predictor.predict(flow_features)

    # Batch of sequences — shape (batch, 30, 18)
    scores = predictor.predict(batch_of_features)
"""

import numpy as np
import torch
import torch.nn as nn
import joblib

from config import (
    LSTM_MODEL_PATH,
    SCALER_PATH,
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    WINDOW_SIZE,
    ALERT_THRESHOLD,
    WARNING_THRESHOLD,
)


class FightDetectorLSTM(nn.Module):
    """
    LSTM classifier for fight detection from optical flow features.

    Architecture (must match training):
        - Input:  (batch, 30, 18)  — 30 timesteps, 18 features each
        - 2-layer LSTM, hidden_size=64, dropout=0.3
        - Fully connected layer: 64 → 1
        - Sigmoid activation → fight probability [0, 1]
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, input_size)

        Returns:
            Tensor of shape (batch, 1) — fight probabilities
        """
        # lstm_out shape: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Take the output from the last timestep only
        last_timestep = lstm_out[:, -1, :]  # (batch, hidden_size)

        # Project to single score and squash to [0, 1]
        out = self.fc(last_timestep)        # (batch, 1)
        out = self.sigmoid(out)

        return out


class SafeWatchPredictor:
    """
    High-level inference wrapper. Loads the trained LSTM model once
    and provides a clean predict() interface for the stream pipeline.

    Handles:
        - Automatic device selection (GPU if available, else CPU)
        - Single sequence and batch input
        - Input validation
        - Score interpretation (alert / warning / normal)
    """

    def __init__(self, model_path: str = LSTM_MODEL_PATH, scaler_path: str = SCALER_PATH, device: str = None):
        """
        Load the trained model weights and feature scaler.

        Args:
            model_path:  Path to the saved .pt model file.
            scaler_path: Path to the saved .pkl scaler file.
            device:      Force a specific device ('cuda' / 'cpu').
                         If None, auto-selects GPU when available.
        """
        # Resolve device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Instantiate model architecture and load trained weights
        self.model = FightDetectorLSTM()
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()  # Lock in inference mode — no dropout, no gradient tracking

        # Load scaler
        self.scaler = joblib.load(scaler_path)
        print(f"[SafeWatch] Scaler loaded from {scaler_path}")

        print(f"[SafeWatch] Model loaded from {model_path}")
        print(f"[SafeWatch] Running on {self.device}")

    def predict(self, flow_features: np.ndarray) -> np.ndarray:
        """
        Run fight detection on one or more optical flow feature sequences.

        Accepts:
            - Single sequence:  np.ndarray of shape (30, 18)
            - Batch:            np.ndarray of shape (batch, 30, 18)

        Each row in the sequence contains 18 features extracted from
        one RAFT optical flow field:
            9 spatial cells x 2 stats (mean_mag, std_mag)

        Returns:
            np.ndarray of fight probabilities, shape (batch,).
            Each value is a float in [0, 1].

        Raises:
            ValueError: If input shape is invalid or sequence length
                        doesn't match WINDOW_SIZE.
        """
        # --- Input validation ---
        if not isinstance(flow_features, np.ndarray):
            raise TypeError(
                f"Expected numpy array, got {type(flow_features).__name__}"
            )

        # Handle single sequence: (30, 18) → (1, 30, 18)
        if flow_features.ndim == 2:
            flow_features = np.expand_dims(flow_features, axis=0)

        if flow_features.ndim != 3:
            raise ValueError(
                f"Expected input shape (30, 18) or (batch, 30, 18), "
                f"got shape {flow_features.shape}"
            )

        batch_size, seq_len, num_features = flow_features.shape

        if seq_len != WINDOW_SIZE:
            raise ValueError(
                f"Sequence length must be {WINDOW_SIZE}, got {seq_len}"
            )

        if num_features != INPUT_SIZE:
            raise ValueError(
                f"Expected {INPUT_SIZE} features per frame, got {num_features}."
            )

        # --- Inference ---
        flat = flow_features.reshape(-1, num_features)
        flat_scaled = self.scaler.transform(flat)
        flow_features = flat_scaled.reshape(batch_size, seq_len, num_features)

        tensor = torch.FloatTensor(flow_features).to(self.device)

        with torch.no_grad():
            scores = self.model(tensor)  # (batch, 1)

        # Flatten to 1D numpy array: (batch,)
        return scores.squeeze(-1).cpu().numpy()

    def predict_with_status(self, flow_features: np.ndarray) -> list[dict]:
        """
        Run prediction and return scores with human-readable alert status.

        Returns a list of dicts, one per input sequence:
            {
                "score": 0.82,
                "status": "ALERT",       # "ALERT" | "WARNING" | "NORMAL"
                "threshold": 0.7
            }
        """
        scores = self.predict(flow_features)

        results = []
        for score in scores:
            score_val = float(score)

            if score_val >= ALERT_THRESHOLD:
                status = "ALERT"
            elif score_val >= WARNING_THRESHOLD:
                status = "WARNING"
            else:
                status = "NORMAL"

            results.append({
                "score": round(score_val, 4),
                "status": status,
                "threshold": ALERT_THRESHOLD,
            })

        return results


# ---------------------------------------------------------------------------
# Quick self-test: run `python inference.py` to verify the module loads
# and the forward pass works with dummy data.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    print("=" * 50)
    print("SafeWatch Inference — Self Test")
    print("=" * 50)

    # Check if model file exists
    if not os.path.exists(LSTM_MODEL_PATH):
        print(f"\n[!] Model file not found at: {LSTM_MODEL_PATH}")
        print("    Place your trained safewatch_lstm.pt in the models/ folder.")
        print("    Running a dry test with randomly initialized weights...\n")

        # Create a temporary model for testing the pipeline
        dummy_model = FightDetectorLSTM()
        os.makedirs(os.path.dirname(LSTM_MODEL_PATH), exist_ok=True)
        torch.save(dummy_model.state_dict(), LSTM_MODEL_PATH)
        print(f"    [OK] Saved dummy model to {LSTM_MODEL_PATH}")

    # Check if scaler exists
    if not os.path.exists(SCALER_PATH):
        print(f"\n[!] Scaler file not found at: {SCALER_PATH}")
        print("    Running a dry test with dummy StandardScaler...\n")
        import joblib
        from sklearn.preprocessing import StandardScaler
        dummy_scaler = StandardScaler()
        dummy_scaler.fit(np.random.rand(100, INPUT_SIZE))
        joblib.dump(dummy_scaler, SCALER_PATH)
        print(f"    [OK] Saved dummy scaler to {SCALER_PATH}")

    # Load predictor
    predictor = SafeWatchPredictor()

    # Test 1: Single sequence (30, 18)
    print("\n--- Test 1: Single sequence (30, 18) ---")
    single = np.random.rand(WINDOW_SIZE, INPUT_SIZE).astype(np.float32)
    result = predictor.predict(single)
    print(f"  Input shape:  {single.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  Score:        {result[0]:.4f}")

    # Test 2: Batch of 4 sequences (4, 30, 18)
    print("\n--- Test 2: Batch of 4 sequences (4, 30, 18) ---")
    batch = np.random.rand(4, WINDOW_SIZE, INPUT_SIZE).astype(np.float32)
    results = predictor.predict(batch)
    print(f"  Input shape:  {batch.shape}")
    print(f"  Output shape: {results.shape}")
    for i, s in enumerate(results):
        print(f"  Sequence {i}: score={s:.4f}")

    # Test 3: predict_with_status
    print("\n--- Test 3: predict_with_status ---")
    status_results = predictor.predict_with_status(batch)
    for i, r in enumerate(status_results):
        print(f"  Sequence {i}: {r}")

    # Test 4: Input validation
    print("\n--- Test 4: Input validation ---")
    try:
        bad_input = np.random.rand(15, 3).astype(np.float32)
        predictor.predict(bad_input)
    except ValueError as e:
        print(f"  Caught expected error: {e}")

    try:
        bad_features = np.random.rand(30, 5).astype(np.float32)
        predictor.predict(bad_features)
    except ValueError as e:
        print(f"  Caught expected error: {e}")

    print("\n" + "=" * 50)
    print("[OK] All tests passed. inference.py is ready.")
    print("=" * 50)
