import asyncio
import random
import time
from typing import List
from fastapi import WebSocket


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


manager = ConnectionManager()

sim_state = {
    "current_target": "Scanning..."
}


#Background Task
async def stream_radar_data():
    """Runs forever in the background, pushing data to the frontend."""
    while True:
        if len(manager.active_connections) > 0:

            if sim_state["current_target"] == "Scanning...":
                target = random.choice(["Bird", "Civilian Drone", "Unknown"])
            else:
                target = sim_state["current_target"]

            payload = {
                "timestamp": time.strftime("%H:%M:%S"),
                "classification": target,
                "confidence": round(random.uniform(75.0, 99.9), 1),
                "range_meters": random.randint(50, 1000),
                "velocity_mps": random.randint(0, 30)
            }

            await manager.broadcast_json(payload)

        await asyncio.sleep(0.5)