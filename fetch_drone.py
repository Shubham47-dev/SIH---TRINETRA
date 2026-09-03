import os
import requests
import pandas as pd

# 1. Configuration
OUTPUT_FOLDER = "ai_data/data/train/2_drone_unarmed"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
TEMP_FILE = "temp_chunk.parquet"

# We bypass the API and hit the direct URLs of the lightweight 75MB chunks
CHUNKS = [
    "https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples/resolve/main/data/train-00010-of-00039.parquet",
    "https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples/resolve/main/data/train-00020-of-00039.parquet",
    "https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples/resolve/main/data/train-00025-of-00039.parquet"
]

TARGET_COUNT = 5000
global_count = 0

print("[SYSTEM] Bypassing the Hugging Face library entirely...")

for i, url in enumerate(CHUNKS):
    if global_count >= TARGET_COUNT:
        break
        
    print(f"\n[SYSTEM] Downloading data chunk {i+1}/3 (approx 75MB)... please wait.")
    
    # Download the chunk directly
    response = requests.get(url)
    with open(TEMP_FILE, "wb") as f:
        f.write(response.content)
        
    print(f"[+] Chunk {i+1} downloaded! Extracting audio files...")
    
    # Read the parquet file using pandas
    df = pd.read_parquet(TEMP_FILE)
    
    for index, row in df.iterrows():
        if row['label'] == 1:
            # HuggingFace Parquet files store the audio as a dictionary
            audio_bytes = row['audio']['bytes']
            
            file_path = os.path.join(OUTPUT_FOLDER, f"hf_drone_{global_count:04d}.wav")
            
            # Write the raw bytes directly to a .wav file
            with open(file_path, "wb") as f:
                f.write(audio_bytes)
            
            global_count += 1
            
            if global_count % 500 == 0:
                print(f" [+] Extracted {global_count}/{TARGET_COUNT} drone files...")
                
            if global_count >= TARGET_COUNT:
                break

# Clean up the temporary download file
if os.path.exists(TEMP_FILE):
    os.remove(TEMP_FILE)

print(f"\n[+] SUCCESS! {global_count} files extracted straight to your folder.")