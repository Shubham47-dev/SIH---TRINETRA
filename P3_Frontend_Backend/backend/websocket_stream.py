import asyncio
import random
import time
from typing import List
from fastapi import WebSocket


# 1. Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[SYSTEM] Screen connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"[SYSTEM] Screen disconnected. Total: {len(self.active_connections)}")

    async def broadcast_json(self, data: dict):
        for connection in self.active_connections:
            await connection.send_json(data)


# Create a single instance to be used across the app
manager = ConnectionManager()

# A global dictionary to let us change the simulation state later via REST API
sim_state = {
    "current_target": "Scanning..."
}


# 2. Background Task
async def stream_radar_data():
    """Runs forever in the background, pushing data to the frontend."""
    while True:
        if len(manager.active_connections) > 0:

            # If we haven't forced a specific target, pick a random one
            if sim_state["current_target"] == "Scanning...":
                target = random.choice(["Bird", "Civilian Drone", "Unknown"])
            else:
                target = sim_state["current_target"]

            # Construct the TRINETRA payload
            payload = {
                "timestamp": time.strftime("%H:%M:%S"),
                "classification": target,
                "confidence": round(random.uniform(75.0, 99.9), 1),
                "range_meters": random.randint(50, 1000),
                "velocity_mps": random.randint(0, 30)
            }

            await manager.broadcast_json(payload)

        # 2 frames per second
        await asyncio.sleep(0.5)