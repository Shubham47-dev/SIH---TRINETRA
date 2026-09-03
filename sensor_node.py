import io
import time
import requests
import numpy as np
import sounddevice as sd
import soundfile as sf

# --- CONFIGURATION ---
HUB_URL = "http://127.0.0.1:8000/sensor-input"
SAMPLE_RATE = 22050
DURATION = 1.0  # 1-second audio chunks
CHANNELS = 1    # Mono audio

def start_sensor():
    print("""
    ========================================
       TRINETRA EDGE SENSOR NODE [ONLINE]
    ========================================
    """)
    print(f"[SYSTEM] Initializing Microphone Capture at {SAMPLE_RATE}Hz...")
    
    try:
        while True:
            # 1. Record 1 second of audio from the laptop mic
            audio_chunk = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
            sd.wait()  # Block execution until the 1 second is fully recorded
            
            # 2. Encode to WAV format entirely in RAM (Bypasses SSD read/write bottleneck)
            wav_io = io.BytesIO()
            sf.write(wav_io, audio_chunk, SAMPLE_RATE, format='WAV')
            wav_io.seek(0) # Reset the pointer to the beginning of the memory stream
            
            # 3. Fire the payload to the Command Hub
            try:
                files = {'file': ('live_capture.wav', wav_io, 'audio/wav')}
                response = requests.post(HUB_URL, files=files, timeout=1.5)
                
                if response.status_code == 200:
                    print(f" [📡] Packet Transmitted | Status: {response.json().get('status', 'OK')}")
                else:
                    print(f" [!] Hub rejected packet. Status: {response.status_code}")
                    
            except requests.exceptions.RequestException:
                print(" [!] Connection Lost: Hub unreachable. Retrying...")
                time.sleep(1) # Wait a second before hammering the network again
                
    except KeyboardInterrupt:
        print("\n[SYSTEM] Sensor Node gracefully shut down.")
    except Exception as e:
        print(f"\n[!] HARDWARE FATAL ERROR: {e}")
        print(" -> Is your microphone plugged in and permitted in Windows Privacy Settings?")

if __name__ == "__main__":
    start_sensor()