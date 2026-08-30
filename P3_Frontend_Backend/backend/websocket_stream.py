import asyncio
import json
import random
from datetime import datetime
from typing import List
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

# Global state for threat injection
sim_state = {
    "override_threat": None
}

def generate_radar_frame():
    threat_classes = ["drone", "human", "vehicle", "bird"]
    selected_class = sim_state["override_threat"] or random.choice(threat_classes)

    # 20x20 Range-Doppler matrix
    range_doppler = [
        [round(random.uniform(0.05, 0.95), 3) for _ in range(20)]
        for _ in range(20)
    ]

    # 30x40 Spectrogram matrix
    spectrogram = [
        [round(random.uniform(0.01, 0.85), 3) for _ in range(40)]
        for _ in range(30)
    ]

    threat_data = {
        "class": selected_class,
        "confidence": round(random.uniform(0.82, 0.99), 2),
        "bearing": round(random.uniform(10.0, 350.0), 1),
        "range_km": round(random.uniform(0.5, 4.8), 2),
        "velocity_mps": round(random.uniform(-15.0, 35.0), 1),
    }

    return {
        "range_doppler": range_doppler,
        "spectrogram": spectrogram,
        "threat": threat_data,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }

async def stream_radar_data():
    while True:
        frame = generate_radar_frame()
        await manager.broadcast(json.dumps(frame))
        await asyncio.sleep(0.5)