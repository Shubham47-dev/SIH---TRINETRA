import os
import pandas as pd
import librosa
import soundfile as sf

# 1. Configuration - Ensure these match your folder names
ESC50_AUDIO = "ESC-50-master/audio"
ESC50_CSV = "ESC-50-master/meta/esc50.csv"
BASE_TRAIN = "ai_data/data/train"

# 2. Map the exact ESC-50 category strings to your TRINETRA classes
BIRD_CATS = ['chirping_birds', 'crow']
HUMAN_CATS = ['footsteps', 'laughing', 'sneezing', 'coughing', 'breathing']
AMBIENT_CATS = ['wind', 'rain', 'thunderstorm', 'engine', 'vacuum_cleaner', 'washing_machine']

dirs = {
    "bird": os.path.join(BASE_TRAIN, "0_bird"),
    "human": os.path.join(BASE_TRAIN, "1_human"),
    "ambient": os.path.join(BASE_TRAIN, "4_ambient_noise")
}

# Ensure destination folders exist
for d in dirs.values():
    os.makedirs(d, exist_ok=True)

print("[SYSTEM] Reading ESC-50 Metadata...")
df = pd.read_csv(ESC50_CSV)

count = {"bird": 0, "human": 0, "ambient": 0}
print("[SYSTEM] Segregating and chopping ESC-50 into 1-second chunks. This will take a moment...")

# 3. Process the dataset
for index, row in df.iterrows():
    cat = row['category']
    file_name = row['filename']
    src_path = os.path.join(ESC50_AUDIO, file_name)
    
    # Check which TRINETRA class this file belongs to
    target_key = None
    if cat in BIRD_CATS: target_key = "bird"
    elif cat in HUMAN_CATS: target_key = "human"
    elif cat in AMBIENT_CATS: target_key = "ambient"
        
    if target_key and os.path.exists(src_path):
        # Load the 5-second audio
        y, sr = librosa.load(src_path, sr=22050)
        
        # Calculate exactly how many frames make up 1 second
        chunk_length = sr * 1 
        
        # Slice the 5-second wave into 5 separate 1-second chunks
        for i in range(5):
            start = i * chunk_length
            end = start + chunk_length
            chunk = y[start:end]
            
            # Save the new 1-second chunk straight to the training folder
            out_name = f"{file_name.replace('.wav', '')}_chunk{i}.wav"
            out_path = os.path.join(dirs[target_key], out_name)
            
            sf.write(out_path, chunk, sr)
            count[target_key] += 1

print(f"\n[+] SUCCESS! Data generated:")
print(f"    -> {count['bird']} Bird clips")
print(f"    -> {count['human']} Human clips")
print(f"    -> {count['ambient']} Ambient Noise clips")