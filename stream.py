"""
SafeWatch Stream Pipeline
=========================
FrameBuffer: Thread-safe single-slot buffer for MJPEG streaming.
CameraPipeline: Per-camera thread that reads video, writes frames
to the buffer (for display), and runs the AI pipeline (for detection).

Display and AI are decoupled — the MJPEG server reads from the buffer
at its own pace, never blocked by RAFT/LSTM inference.
"""

import cv2
import torch
import numpy as np
import threading
import time
from collections import deque
from datetime import datetime
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

from config import (
    FPS,
    MOTION_THRESHOLD,
    WINDOW_SIZE,
    STRIDE,
    INPUT_SIZE,
)
from inference import SafeWatchPredictor


class FrameBuffer:
    """
    Thread-safe single-slot frame buffer.
    Only the latest frame is kept — old frames are discarded.
    No queue bloat, no torn frames, O(1) memory.
    """

    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()

    def write(self, frame):
        with self._lock:
            self._frame = frame.copy()

    def read(self):
        with self._lock:
            return self._frame


class CameraPipeline:
    """
    Processes a single camera: reads frames, writes to buffer for
    MJPEG display, runs motion gate + RAFT + LSTM for fight detection.
    """

    def __init__(self, camera_id, source, frame_buffer, alert_callback=None, loop=True):
        self.camera_id = camera_id
        self.source = source
        self.frame_buffer = frame_buffer
        self.alert_callback = alert_callback
        self.loop = loop
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load models (each pipeline gets its own copy for thread safety)
        predictor = SafeWatchPredictor(device=device)

        weights = Raft_Small_Weights.DEFAULT
        raft_transforms = weights.transforms()
        raft_model = raft_small(weights=weights, progress=False).to(device)
        raft_model.eval()

        while self._running:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                print(f"[{self.camera_id}] Cannot open: {self.source}")
                break

            source_fps = cap.get(cv2.CAP_PROP_FPS)
            if source_fps <= 0:
                source_fps = 30
            frame_skip_ratio = max(1, int(round(source_fps / FPS)))
            is_file = not str(self.source).startswith("rtsp")

            print(f"[{self.camera_id}] Started | {self.source} | {source_fps:.0f} fps -> {FPS} fps")

            prev_gray = None
            prev_raft_frame = None
            window = deque(maxlen=WINDOW_SIZE)
            frame_count = 0
            processed_count = 0

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break

                # Write EVERY frame to buffer for smooth MJPEG display
                self.frame_buffer.write(frame)

                frame_count += 1

                # Pace video file playback to real-time
                if is_file:
                    time.sleep(1 / source_fps)

                # AI only processes at target FPS
                if frame_count % frame_skip_ratio != 0:
                    continue

                # --- AI Pipeline below this line ---
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_resized = cv2.resize(frame, (256, 256))
                rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                raft_frame = (rgb / 255.0).astype(np.float32)

                if prev_gray is None:
                    prev_gray = gray
                    prev_raft_frame = raft_frame
                    continue

                # Motion Gate
                diff = cv2.absdiff(prev_gray, gray)
                mean_diff = np.mean(diff)
                prev_gray = gray

                if mean_diff < MOTION_THRESHOLD:
                    prev_raft_frame = raft_frame
                    continue

                # RAFT Optical Flow
                t1 = torch.from_numpy(prev_raft_frame).permute(2, 0, 1).unsqueeze(0)
                t2 = torch.from_numpy(raft_frame).permute(2, 0, 1).unsqueeze(0)
                t1, t2 = raft_transforms(t1, t2)
                t1, t2 = t1.to(device), t2.to(device)
                prev_raft_frame = raft_frame

                with torch.no_grad():
                    flow = raft_model(t1, t2)[-1][0]

                dx = flow[0].cpu().numpy()
                dy = flow[1].cpu().numpy()
                magnitude = np.sqrt(dx**2 + dy**2)

                # 3x3 Spatial Grid
                h, w = magnitude.shape
                cell_h, cell_w = h // 3, w // 3
                features = []
                for i in range(3):
                    for j in range(3):
                        cell = magnitude[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                        features.append(np.mean(cell))
                        features.append(np.std(cell))

                window.append(features)
                processed_count += 1

                # Inference trigger
                if len(window) == WINDOW_SIZE and processed_count % STRIDE == 0:
                    self._predict_and_alert(predictor, np.array(window, dtype=np.float32))

            # End-of-stream inference for short clips
            if len(window) >= 15 and processed_count < WINDOW_SIZE:
                window_np = np.array(window, dtype=np.float32)
                if len(window) < WINDOW_SIZE:
                    pad = np.zeros((WINDOW_SIZE - len(window), INPUT_SIZE), dtype=np.float32)
                    window_np = np.vstack([window_np, pad])
                self._predict_and_alert(predictor, window_np)

            cap.release()

            if not self.loop or not self._running:
                break
            print(f"[{self.camera_id}] Looping video...")

        print(f"[{self.camera_id}] Pipeline stopped.")

    def _predict_and_alert(self, predictor, window_np):
        result = predictor.predict_with_status(window_np)[0]
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {self.camera_id} | Score: {result['score']:.2f} | Status: {result['status']}")
        if self.alert_callback:
            self.alert_callback(self.camera_id, result['score'], result['status'])
