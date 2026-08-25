from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio

# Import your tools from the other file in the same folder
from websocket_stream import manager, stream_radar_data, sim_state

app = FastAPI(title="TRINETRA Backend API")



# 1. Start the background stream when the server boots
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(stream_radar_data())

# 2. The WebSocket Route (For the React Frontend to connect to)
@app.websocket("/ws/radar")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text() # Keeps connection open
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# 3. REST API Endpoint (Hackathon Command Button)
# A POST request here changes the behavior of the background stream!
@app.post("/api/inject-threat")
async def inject_threat():
    sim_state["current_target"] = "ARMED PAYLOAD DRONE"
    return {"message": "Threat injected into radar stream successfully"}

@app.post("/api/reset-radar")
async def reset_radar():
    sim_state["current_target"] = "Scanning..."
    return {"message": "Radar reset to normal scanning mode"}

# 4. Dummy Test Page
html_test_page = """
<!DOCTYPE html>
<html>
    <head><style>body{background:#0a0e14;color:#00ff66;font-family:monospace;padding:20px;} button{padding:10px; margin:5px;}</style></head>
    <body>
        <h2>TRINETRA LIVE FEED</h2>
        <button onclick="fetch('/api/inject-threat', {method: 'POST'})">Inject Threat (POST)</button>
        <button onclick="fetch('/api/reset-radar', {method: 'POST'})">Reset Scan (POST)</button>
        <pre id="radar-data" style="font-size:16px; margin-top:20px;"></pre>
        <script>
            var ws = new WebSocket("ws://127.0.0.1:8000/ws/radar");
            ws.onmessage = function(event) {
                var data = JSON.parse(event.data);
                document.getElementById('radar-data').textContent = JSON.stringify(data, null, 2);
            };
        </script>
    </body>
</html>
"""

@app.get("/")
async def test_page():
    return HTMLResponse(html_test_page)