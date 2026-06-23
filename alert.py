"""
SafeWatch Alert Server
======================
WebSocket server that broadcasts fight detection alerts
to all connected dashboard clients.

Usage:
    # Start the server standalone:
    python alert.py

    # Or import and use programmatically:
    from alert import AlertServer
    server = AlertServer()
    await server.start()
    await server.send_alert("Camera 1", camera_index=0, score=0.82, status="ALERT")
"""

import asyncio
import json
import os
import sys
from datetime import datetime

# Force line-buffered output so prints appear immediately when run in background
sys.stdout.reconfigure(line_buffering=True)

try:
    import websockets
    from websockets.asyncio.server import serve
except ImportError:
    print("[!] websockets library not found. Install it with:")
    print("    pip install websockets")
    raise

from config import ALERT_THRESHOLD, WARNING_THRESHOLD, WS_PORT, BASE_DIR

# Absolute path so the log lands beside the code, not in the launch CWD.
ALERTS_LOG_PATH = os.path.join(BASE_DIR, "alerts.log")


class AlertServer:
    """
    WebSocket server that maintains connections to dashboard clients
    and broadcasts alert messages.
    """

    def __init__(self, host="127.0.0.1", port=WS_PORT):
        self.host = host
        self.port = port
        self.clients = set()
        self.server = None
        self._log_file = None   # Opened once in start(), line-buffered append

    async def _handler(self, websocket):
        """Handle a new dashboard client connection."""
        self.clients.add(websocket)
        remote = websocket.remote_address
        print(f"[Alert] Dashboard connected: {remote[0]}:{remote[1]} ({len(self.clients)} total)")

        try:
            # Keep connection alive by listening (dashboard doesn't send, but we need to hold)
            async for _ in websocket:
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"[Alert] Dashboard disconnected: {remote[0]}:{remote[1]} ({len(self.clients)} total)")

    async def send_alert(self, camera_id: str, camera_index: int, score: float, status: str):
        """
        Broadcast an alert to all connected dashboard clients.

        Args:
            camera_id:    Camera name, e.g. "Camera 1"
            camera_index: Camera index for tile matching
            score:        Fight probability score (0.0 to 1.0)
            status:       "ALERT", "WARNING", or "NORMAL"
        """
        message = json.dumps({
            "camera": camera_id,
            "camera_index": camera_index,
            "score": round(score, 4),
            "status": status,
            "timestamp": datetime.now().isoformat(),
        })

        # Persist to disk for auditable monitoring.
        # Skip NORMAL — it fires ~1/sec/camera and would grow the log unbounded.
        if status != "NORMAL" and self._log_file is not None:
            try:
                self._log_file.write(
                    f"{datetime.now().isoformat()} | {camera_id} | {status} | {score}\n"
                )
            except OSError as e:
                print(f"[Alert] Failed to write alerts.log: {e}")

        if not self.clients:
            return

        # Broadcast to all connected dashboards
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)

        # Clean up dead connections
        self.clients -= disconnected

    async def start(self):
        """Start the WebSocket server."""
        # Open the audit log once, line-buffered, instead of per-alert.
        try:
            self._log_file = open(ALERTS_LOG_PATH, "a", buffering=1)
        except OSError as e:
            print(f"[Alert] Could not open {ALERTS_LOG_PATH}: {e}")
            self._log_file = None

        self.server = await serve(
            self._handler, self.host, self.port,
            reuse_address=True,
        )
        print(f"[Alert] WebSocket server running on ws://{self.host}:{self.port}")
        return self.server

    async def stop(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            print("[Alert] Server stopped.")
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


# ---------------------------------------------------------------
# Standalone mode: run as a demo server with test alerts
# ---------------------------------------------------------------
async def _demo():
    """Run the server and send demo alerts every few seconds."""
    import random

    server = AlertServer()
    await server.start()

    cameras = ["Camera 1", "Camera 2", "Camera 3", "Camera 4"]
    print("[Alert] Demo mode: sending test alerts every 3 seconds...")
    print("[Alert] Open dashboard/index.html in a browser to see live updates.")

    try:
        while True:
            await asyncio.sleep(3)

            # Pick a random camera and generate a random score
            cam = random.choice(cameras)
            score = random.uniform(0.0, 1.0)

            if score >= ALERT_THRESHOLD:
                status = "ALERT"
            elif score >= WARNING_THRESHOLD:
                status = "WARNING"
            else:
                status = "NORMAL"

            camera_index = cameras.index(cam)
            print(f"[Alert] {cam} | Score: {score:.2f} | Status: {status}")
            await server.send_alert(cam, camera_index, score, status)

    except KeyboardInterrupt:
        print("\n[Alert] Shutting down...")
        await server.stop()


if __name__ == "__main__":
    asyncio.run(_demo())
