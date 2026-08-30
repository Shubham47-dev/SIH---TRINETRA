import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from websocket_stream import manager, stream_radar_data, sim_state

app = FastAPI(title="TRINETRA Backend API")

# Enable CORS for external dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(stream_radar_data())

@app.websocket("/ws/radar")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection open to listen for client heartbeats
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

@app.post("/api/inject-threat")
async def inject_threat():
    sim_state["override_threat"] = "drone"
    return {"status": "Threat injected: DRONE"}

@app.post("/api/reset-radar")
async def reset_radar():
    sim_state["override_threat"] = None
    return {"status": "Radar reset to auto sweep"}