"""
SafeWatch — Unified Launcher
=============================
Single command to start the entire system:
    python safewatch.py

Starts: HTTP server (dashboard + MJPEG feeds + REST API),
WebSocket alerts, and all camera pipelines.
Opens the browser automatically.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import asyncio
import threading
import webbrowser
import time
import json
import uuid
import re
import cv2
import numpy as np
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from config import CAMERAS, HTTP_PORT, WS_PORT, FPS, DASHBOARD_DIR, BASE_DIR
from stream import FrameBuffer, CameraPipeline
from alert import AlertServer


# ================================================================
# Camera Manager — thread-safe runtime camera registry
# ================================================================
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
REGISTRY_PATH = os.path.join(BASE_DIR, "cameras.json")

ALLOWED_VIDEO_EXTENSIONS = {".avi", ".mp4", ".mkv", ".mov", ".webm"}
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500MB


class CameraManager:
    """
    Thread-safe runtime camera registry.
    Manages FrameBuffers, CameraPipelines, and JSON persistence.
    """

    def __init__(self, alert_callback):
        self._cameras = {}          # {index: dict}
        self._next_index = 0        # Monotonic, never reused
        self._lock = threading.RLock()
        self._alert_callback = alert_callback

    # --- Persistence ---

    def _save_unlocked(self):
        """Write cameras.json. Caller must hold self._lock."""
        data = {
            "next_index": self._next_index,
            "cameras": [
                {
                    "index": c["index"],
                    "id": c["id"],
                    "source": c["source"],
                    "source_type": c["source_type"],
                }
                for c in self._cameras.values()
            ],
        }
        with open(REGISTRY_PATH, "w") as f:
            json.dump(data, f, indent=2)

    def load_or_seed(self, config_cameras):
        """Load from cameras.json if it exists, otherwise seed from config.py."""
        with self._lock:
            if os.path.exists(REGISTRY_PATH):
                with open(REGISTRY_PATH, "r") as f:
                    data = json.load(f)
                self._next_index = data.get("next_index", 0)
                for cam in data.get("cameras", []):
                    self._cameras[cam["index"]] = {
                        "index": cam["index"],
                        "id": cam["id"],
                        "source": cam["source"],
                        "source_type": cam.get("source_type", "none"),
                        "pipeline": None,
                        "buffer": None,
                    }
                print(f"[SafeWatch] Loaded {len(self._cameras)} cameras from cameras.json")
                print(f"[SafeWatch] (To re-seed from config.py, delete cameras.json)")
            else:
                for i, cam in enumerate(config_cameras):
                    source = cam.get("source")
                    self._cameras[i] = {
                        "index": i,
                        "id": cam["id"],
                        "source": source,
                        "source_type": self._detect_source_type(source),
                        "pipeline": None,
                        "buffer": None,
                    }
                self._next_index = len(config_cameras)
                self._save_unlocked()
                print(f"[SafeWatch] First run — seeded {len(self._cameras)} cameras from config.py")

    @staticmethod
    def _detect_source_type(source):
        if source is None:
            return "none"
        if isinstance(source, int):
            return "webcam"
        if str(source).startswith(("rtsp://", "http://", "https://")):
            return "cctv"
        if str(source).startswith("uploads"):
            return "video_upload"
        return "video_path"

    # --- Camera Operations ---

    def add_camera(self, camera_id):
        """Create a new offline camera tile. Returns camera info."""
        with self._lock:
            # Enforce unique names — append (2), (3)... if name already exists
            existing_names = {c["id"] for c in self._cameras.values()}
            unique_id = camera_id
            suffix = 2
            while unique_id in existing_names:
                unique_id = f"{camera_id} ({suffix})"
                suffix += 1

            index = self._next_index
            self._next_index += 1
            self._cameras[index] = {
                "index": index,
                "id": unique_id,
                "source": None,
                "source_type": "none",
                "pipeline": None,
                "buffer": None,
            }
            self._save_unlocked()
        return self._entry_to_dict(self._cameras[index])

    def connect_camera(self, index, source, source_type):
        """Validate source, create FrameBuffer + CameraPipeline, start thread."""
        with self._lock:
            entry = self._cameras.get(index)
            if entry is None:
                return {"error": f"Camera index {index} not found"}
            if entry["pipeline"] is not None:
                return {"error": f"{entry['id']} is already connected. Disconnect first."}

        # Validate source (outside lock — can be slow for RTSP)
        ok, msg = self._validate_source(source)
        if not ok:
            return {"error": msg}

        # Determine if this is a file that should loop
        is_file_source = source_type in ("video_path", "video_upload")

        with self._lock:
            # Re-check after validation (could have changed)
            entry = self._cameras.get(index)
            if entry is None:
                return {"error": f"Camera index {index} not found"}
            if entry["pipeline"] is not None:
                return {"error": f"{entry['id']} is already connected. Disconnect first."}

            buf = FrameBuffer()
            pipeline = CameraPipeline(
                camera_id=entry["id"],
                source=source,
                frame_buffer=buf,
                alert_callback=lambda cid, s, st, idx=index: self._alert_callback(cid, idx, s, st),
                loop=is_file_source,
            )
            entry["source"] = source
            entry["source_type"] = source_type
            entry["buffer"] = buf
            entry["pipeline"] = pipeline
            self._save_unlocked()

        pipeline.start()
        print(f"[SafeWatch] Connected: {entry['id']} -> {source}")
        return {"camera": self._entry_to_dict(entry)}

    def disconnect_camera(self, index):
        """Stop pipeline, keep tile as OFFLINE."""
        pipeline = None
        with self._lock:
            entry = self._cameras.get(index)
            if entry is None:
                return {"error": f"Camera index {index} not found"}
            if entry["pipeline"] is None:
                return {"error": f"{entry['id']} is not connected"}
            pipeline = entry["pipeline"]
            entry["pipeline"] = None
            entry["buffer"] = None
            entry["source"] = None
            entry["source_type"] = "none"
            self._save_unlocked()

        # Stop OUTSIDE lock — can block up to 5s
        pipeline.stop()
        print(f"[SafeWatch] Disconnected: {entry['id']}")
        return {"camera": self._entry_to_dict(entry)}

    def remove_camera(self, index):
        """Stop pipeline if running, remove from registry entirely."""
        pipeline = None
        with self._lock:
            entry = self._cameras.get(index)
            if entry is None:
                return {"error": f"Camera index {index} not found"}
            pipeline = entry.get("pipeline")
            del self._cameras[index]
            self._save_unlocked()

        # Stop OUTSIDE lock
        if pipeline:
            pipeline.stop()
        print(f"[SafeWatch] Removed: {entry['id']}")
        return {"cameras": self.list_cameras()}

    def list_cameras(self):
        """Return JSON-safe list of all cameras with status."""
        with self._lock:
            return [self._entry_to_dict(c) for c in self._cameras.values()]

    def get_buffer(self, index):
        """Get frame buffer for MJPEG serving. Returns None if offline."""
        with self._lock:
            entry = self._cameras.get(index)
            if entry is None:
                return None
            return entry.get("buffer")

    def auto_connect_saved(self):
        """On startup, reconnect cameras that had a source saved in JSON."""
        with self._lock:
            to_connect = [
                (entry["index"], entry["source"], entry["source_type"])
                for entry in self._cameras.values()
                if entry["source"] is not None
            ]

        for index, source, source_type in to_connect:
            result = self.connect_camera(index, source, source_type)
            if "error" in result:
                with self._lock:
                    entry = self._cameras.get(index)
                if entry:
                    print(f"[SafeWatch] Warning: {entry['id']} failed to auto-connect: {result['error']}")

    @staticmethod
    def _entry_to_dict(entry):
        return {
            "index": entry["index"],
            "id": entry["id"],
            "source": entry["source"],
            "source_type": entry["source_type"],
            "active": entry.get("pipeline") is not None,
        }

    @staticmethod
    def _validate_source(source, timeout=5):
        """Try to open source and read one frame. Returns (success, message)."""
        result = [False, "Timeout"]

        def _try():
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    result[0] = True
                    result[1] = "OK"
                else:
                    result[1] = "Source opened but no frames readable"
            else:
                result[1] = f"Cannot open source: {source}"
            cap.release()

        t = threading.Thread(target=_try, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            return False, f"Connection timed out after {timeout}s"
        return result[0], result[1]


# ================================================================
# HTTP Handler — dashboard, MJPEG feeds, and REST API
# ================================================================
class SafeWatchHandler(SimpleHTTPRequestHandler):
    """
    Routes:
        GET  /feed/<index>                   → MJPEG stream
        GET  /api/cameras                    → List cameras + ws_port
        POST /api/cameras                    → Create offline tile
        POST /api/cameras/<idx>/connect      → Connect source
        POST /api/cameras/<idx>/disconnect   → Stop pipeline
        DELETE /api/cameras/<idx>            → Remove tile
        POST /api/videos                     → Upload video file
        /*                                   → Static files from dashboard/
    """
    camera_manager = None   # Set by launcher
    serve_directory = "dashboard"
    ws_port = WS_PORT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=self.serve_directory, **kwargs)

    # --- Routing ---

    def do_GET(self):
        if self.path.startswith("/feed/"):
            self._serve_mjpeg()
        elif self.path.startswith("/api/cameras"):
            self._handle_get_cameras()
        else:
            super().do_GET()

    def do_POST(self):
        path = self.path.split("?")[0]  # strip query params

        if path == "/api/cameras":
            self._handle_create_camera()
        elif re.match(r"^/api/cameras/\d+/connect$", path):
            self._handle_connect_camera(path)
        elif re.match(r"^/api/cameras/\d+/disconnect$", path):
            self._handle_disconnect_camera(path)
        elif self.path.startswith("/api/videos"):
            self._handle_upload_video()
        else:
            self.send_error(404)

    def do_DELETE(self):
        path = self.path.split("?")[0]
        if re.match(r"^/api/cameras/\d+$", path):
            self._handle_delete_camera(path)
        else:
            self.send_error(404)

    # --- API Handlers ---

    def _handle_get_cameras(self):
        payload = {
            "cameras": self.camera_manager.list_cameras(),
            "ws_port": self.ws_port,
        }
        self._send_json(200, payload)

    def _handle_create_camera(self):
        body = self._read_json_body()
        if body is None:
            return
        camera_id = body.get("id", "").strip()
        if not camera_id:
            return self._send_json(400, {"error": "Camera name is required"})
        result = self.camera_manager.add_camera(camera_id)
        self._send_json(200, {"camera": result, "cameras": self.camera_manager.list_cameras()})

    def _handle_connect_camera(self, path):
        index = int(path.split("/")[3])
        body = self._read_json_body()
        if body is None:
            return
        source = body.get("source", "").strip()
        source_type = body.get("source_type", "").strip()
        if not source:
            return self._send_json(400, {"error": "Source is required"})
        if source_type not in ("cctv", "video_path", "video_upload"):
            return self._send_json(400, {"error": f"Invalid source_type: {source_type}"})
        result = self.camera_manager.connect_camera(index, source, source_type)
        if "camera" in result:
            self._send_json(200, result)
        elif "not found" in result.get("error", ""):
            self._send_json(404, result)
        elif "already connected" in result.get("error", ""):
            self._send_json(409, result)
        else:
            self._send_json(400, result)

    def _handle_disconnect_camera(self, path):
        index = int(path.split("/")[3])
        result = self.camera_manager.disconnect_camera(index)
        if "camera" in result:
            self._send_json(200, result)
        elif "not found" in result.get("error", ""):
            self._send_json(404, result)
        else:
            self._send_json(400, result)

    def _handle_delete_camera(self, path):
        index = int(path.split("/")[3])
        result = self.camera_manager.remove_camera(index)
        if "cameras" in result:
            self._send_json(200, result)
        else:
            self._send_json(404, result)

    def _handle_upload_video(self):
        qs = parse_qs(urlparse(self.path).query)
        filename = qs.get("filename", [None])[0]
        if not filename:
            return self._send_json(400, {"error": "filename query param required"})

        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            return self._send_json(400, {"error": f"Unsupported format: {ext}"})

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            return self._send_json(400, {"error": "No file data received"})
        if content_length > MAX_UPLOAD_SIZE:
            return self._send_json(413, {"error": "File too large (max 500MB)"})

        # Sanitize filename
        safe_base = re.sub(r"[^a-zA-Z0-9_.-]", "_", os.path.splitext(filename)[0])
        safe_name = f"{uuid.uuid4().hex[:8]}_{safe_base}{ext}"
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        save_path = os.path.join(UPLOADS_DIR, safe_name)

        with open(save_path, "wb") as f:
            remaining = content_length
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 65536))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)

        self._send_json(200, {"filename": save_path})

    # --- MJPEG ---

    def _serve_mjpeg(self):
        try:
            index = int(self.path.split("/")[2])
        except (IndexError, ValueError):
            self.send_error(404)
            return

        buf = self.camera_manager.get_buffer(index)
        if buf is None:
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            black = cv2.imencode(".jpg", self._black_frame())[1].tobytes()
            self.wfile.write(black)
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()

        try:
            while True:
                # Re-check buffer each frame — if camera disconnected, stop
                buf = self.camera_manager.get_buffer(index)
                if buf is None:
                    break

                frame = buf.read()
                if frame is None:
                    time.sleep(0.1)
                    continue

                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                data = jpeg.tobytes()

                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(data)}\r\n".encode())
                self.wfile.write(b"\r\n")
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                self.wfile.flush()

                time.sleep(1 / FPS)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    # --- Helpers ---

    def _read_json_body(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            return json.loads(body) if body else {}
        except (json.JSONDecodeError, ValueError):
            self._send_json(400, {"error": "Invalid JSON body"})
            return None

    def _send_json(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _black_frame():
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def log_message(self, format, *args):
        # Suppress per-request HTTP logs
        pass


# ================================================================
# Main launcher
# ================================================================
async def main():
    print("=" * 55)
    print("  SafeWatch — AI-Powered CCTV Anomaly Detection")
    print("=" * 55)

    # 1. Start WebSocket alert server
    alert_server = AlertServer(port=WS_PORT)
    await alert_server.start()

    # 2. Create async-safe alert callback for camera threads
    loop = asyncio.get_running_loop()

    def send_alert_from_thread(camera_id, camera_index, score, status):
        asyncio.run_coroutine_threadsafe(
            alert_server.send_alert(camera_id, camera_index, score, status),
            loop,
        )

    # 3. Create camera manager and load/seed cameras
    manager = CameraManager(alert_callback=send_alert_from_thread)
    manager.load_or_seed(CAMERAS)

    # 4. Start HTTP + MJPEG + API server
    SafeWatchHandler.camera_manager = manager
    SafeWatchHandler.serve_directory = DASHBOARD_DIR
    SafeWatchHandler.ws_port = WS_PORT
    http_server = ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), SafeWatchHandler)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()
    print(f"[SafeWatch] Dashboard:  http://127.0.0.1:{HTTP_PORT}")
    print(f"[SafeWatch] Alerts:    ws://127.0.0.1:{WS_PORT}")

    # 5. Auto-connect cameras that had a saved source
    manager.auto_connect_saved()

    # 6. Open browser
    print()
    time.sleep(1)
    webbrowser.open(f"http://127.0.0.1:{HTTP_PORT}")
    print(f"[SafeWatch] Browser opened. System ready.")
    print(f"[SafeWatch] Press Ctrl+C to stop.\n")

    # 7. Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n[SafeWatch] Shutting down...")
        # Disconnect all cameras
        for index in list(manager._cameras.keys()):
            entry = manager._cameras.get(index)
            if entry and entry.get("pipeline"):
                entry["pipeline"].stop()
        await alert_server.stop()
        http_server.shutdown()
        print("[SafeWatch] Goodbye.")


if __name__ == "__main__":
    asyncio.run(main())
