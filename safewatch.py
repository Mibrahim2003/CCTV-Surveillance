"""
SafeWatch — Unified Launcher
=============================
Single command to start the entire system:
    python safewatch.py

Starts: HTTP server (dashboard + MJPEG feeds), WebSocket alerts,
and all camera pipelines. Opens the browser automatically.
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import asyncio
import threading
import webbrowser
import time
import cv2
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from functools import partial

from config import CAMERAS, HTTP_PORT, WS_PORT, FPS, DASHBOARD_DIR
from stream import FrameBuffer, CameraPipeline
from alert import AlertServer


# ================================================================
# Combined HTTP handler: serves dashboard AND MJPEG feeds
# ================================================================
class SafeWatchHandler(SimpleHTTPRequestHandler):
    """
    Routes:
        /feed/<index>  → MJPEG stream for camera at that index
        /*             → Static files from dashboard/ directory
    """
    frame_buffers = {}  # Populated by the launcher before server starts
    serve_directory = "dashboard"  # Set by launcher before server starts

    camera_config = []  # Set by launcher: [{"id": "Camera 1", "active": true}, ...]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=self.serve_directory, **kwargs)

    def do_GET(self):
        if self.path.startswith("/feed/"):
            self._serve_mjpeg()
        elif self.path == "/api/cameras":
            self._serve_camera_config()
        else:
            super().do_GET()

    def _serve_camera_config(self):
        import json
        data = json.dumps(self.camera_config).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_mjpeg(self):
        try:
            index = int(self.path.split("/")[2])
        except (IndexError, ValueError):
            self.send_error(404)
            return

        buf = self.frame_buffers.get(index)
        if buf is None:
            # Camera inactive — send a single black frame
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
            pass  # Client disconnected

    @staticmethod
    def _black_frame():
        return __import__("numpy").zeros((480, 640, 3), dtype=__import__("numpy").uint8)

    def log_message(self, format, *args):
        # Suppress per-request HTTP logs to keep terminal clean
        pass


# ================================================================
# Main launcher
# ================================================================
async def main():
    print("=" * 55)
    print("  SafeWatch — AI-Powered CCTV Anomaly Detection")
    print("=" * 55)

    # 1. Create frame buffers for active cameras
    frame_buffers = {}
    for i, cam in enumerate(CAMERAS):
        if cam["source"] is not None:
            frame_buffers[i] = FrameBuffer()

    # 2. Start HTTP + MJPEG server
    SafeWatchHandler.frame_buffers = frame_buffers
    SafeWatchHandler.serve_directory = DASHBOARD_DIR
    SafeWatchHandler.camera_config = [
        {"id": cam["id"], "active": cam["source"] is not None, "index": i}
        for i, cam in enumerate(CAMERAS)
    ]
    http_server = ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), SafeWatchHandler)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()
    print(f"[SafeWatch] Dashboard:  http://127.0.0.1:{HTTP_PORT}")

    # 3. Start WebSocket alert server
    alert_server = AlertServer()
    await alert_server.start()
    print(f"[SafeWatch] Alerts:    ws://127.0.0.1:{WS_PORT}")

    # 4. Create async-safe alert callback for camera threads
    loop = asyncio.get_running_loop()

    def send_alert_from_thread(camera_id, score, status):
        asyncio.run_coroutine_threadsafe(
            alert_server.send_alert(camera_id, score, status),
            loop,
        )

    # 5. Start camera pipelines
    pipelines = []
    for i, cam in enumerate(CAMERAS):
        if cam["source"] is not None:
            pipeline = CameraPipeline(
                camera_id=cam["id"],
                source=cam["source"],
                frame_buffer=frame_buffers[i],
                alert_callback=send_alert_from_thread,
                loop=True,
            )
            pipeline.start()
            pipelines.append(pipeline)
            print(f"[SafeWatch] Pipeline:  {cam['id']} -> {cam['source']}")
        else:
            print(f"[SafeWatch] Pipeline:  {cam['id']} -> offline")

    # 6. Open browser
    print()
    time.sleep(1)  # Let servers bind before opening browser
    webbrowser.open(f"http://127.0.0.1:{HTTP_PORT}")
    print(f"[SafeWatch] Browser opened. System ready.")
    print(f"[SafeWatch] Press Ctrl+C to stop.\n")

    # 7. Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n[SafeWatch] Shutting down...")
        for p in pipelines:
            p.stop()
        await alert_server.stop()
        http_server.shutdown()
        print("[SafeWatch] Goodbye.")


if __name__ == "__main__":
    asyncio.run(main())
