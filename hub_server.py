import io
import cv2
import uvicorn
import librosa
import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from fastapi import FastAPI, UploadFile, File
from datetime import datetime

VOLUME_THRESHOLD = 0.015   
CONF_THRESHOLD = 75.0      

app = FastAPI(title="TRINETRA ONNX Command Hub")

global_threat_state = {
    "target_identified": "Scanning...",
    "confidence": 0.0,
    "timestamp": "N/A",
    "details": {}
}

print("[SYSTEM] Initializing ONNX Runtime Engine...")
ort_session = ort.InferenceSession("spectra_eca_model.onnx")
input_name = ort_session.get_inputs()[0].name

classes = ["Bird", "Human", "Drone (Unarmed)", "Drone (Payload)", "Ambient / Noise"]
print(f"[SYSTEM] Engine Online. Bound to input layer: '{input_name}'")

@app.post("/sensor-input")
async def receive_audio(file: UploadFile = File(...)):
    global global_threat_state
    
    try:
        audio_bytes = await file.read()
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, duration=1.0)
       
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude < VOLUME_THRESHOLD:
            global_threat_state = {
                "target_identified": "Clear (Ambient)",
                "confidence": 100.0,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "details": {c: 0.0 for c in classes}
            }
            return {"status": "success"}
        
        if len(audio) < sr:
            audio = np.pad(audio, (0, sr - len(audio)), mode='constant')
            
        stft_matrix = librosa.stft(audio, n_fft=256, hop_length=172)
        spectrogram_db = librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max)
        
        spec_min, spec_max = spectrogram_db.min(), spectrogram_db.max()
        if (spec_max - spec_min) > 0:
            spec_norm = (spectrogram_db - spec_min) / (spec_max - spec_min)
        else:
            spec_norm = (spectrogram_db - spec_min)
            
        resized_matrix = cv2.resize(spec_norm, (128, 128), interpolation=cv2.INTER_CUBIC)
        input_array = np.expand_dims(np.expand_dims(resized_matrix, axis=0), axis=0).astype(np.float32)
        
        outputs = ort_session.run(None, {input_name: input_array})
        probabilities = softmax(outputs[0][0]) * 100
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]
        predicted_class = classes[predicted_idx]
        
        if predicted_class == "Ambient / Noise":
            target_display = "Clear (Ambient)"
        elif confidence < CONF_THRESHOLD:
            target_display = "Acoustic Anomaly (Unverified)"
        else:
            target_display = predicted_class
            
        details = {classes[i]: round(float(probabilities[i]), 2) for i in range(len(classes))}
        
        global_threat_state = {
            "target_identified": target_display,
            "confidence": round(float(confidence), 2),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "details": details
        }
        
        return {"status": "success"}
        
    except Exception as e:
        print(f"[!] Processing Error: {e}")
        return {"status": "error"}

@app.get("/latest-threat")
def get_latest_threat():
    return global_threat_state

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)