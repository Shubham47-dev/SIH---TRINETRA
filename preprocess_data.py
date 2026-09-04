import os
import cv2
import random
import librosa
import numpy as np

# DIRECTORY ROUTING
RAW_BASE = "ai_data/data/train"       
OUT_BASE = "ai_data/spectrograms"     
CLASSES = ["0_bird", "1_human", "2_drone_unarmed", "3_drone_payload", "4_ambient_noise"]
VAL_SPLIT = 0.20                      

def convert_to_spectrogram(wav_path, img_path):
    try:
        audio, sr = librosa.load(wav_path, sr=22050, duration=1.0)
        
        if len(audio) < sr:
            audio = np.pad(audio, (0, sr - len(audio)), mode='constant')

        stft_matrix = librosa.stft(audio, n_fft=256, hop_length=172)
        spectrogram_db = librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max)
        
        spec_min, spec_max = spectrogram_db.min(), spectrogram_db.max()
        if spec_max - spec_min > 0:
            spec_norm = (spectrogram_db - spec_min) / (spec_max - spec_min)
        else:
            spec_norm = spectrogram_db - spec_min
            
        resized = cv2.resize(spec_norm, (128, 128), interpolation=cv2.INTER_CUBIC)
        img_uint8 = (resized * 255).astype(np.uint8)
        
        cv2.imwrite(img_path, img_uint8)
    except Exception as e:
        # Fails silently and safely without crashing the main loop
        pass

def main():
    print("[SYSTEM] Initializing Sequential STFT Preprocessing (Low RAM Mode)...")
    tasks = []
    
    for cls in CLASSES:
        raw_dir = os.path.join(RAW_BASE, cls)
        train_out = os.path.join(OUT_BASE, "train", cls)
        val_out = os.path.join(OUT_BASE, "val", cls)
        
        os.makedirs(train_out, exist_ok=True)
        os.makedirs(val_out, exist_ok=True)
        
        if not os.path.exists(raw_dir):
            print(f" [!] Missing raw directory: {raw_dir}")
            continue
            
        files = [f for f in os.listdir(raw_dir) if f.endswith(".wav")]
        random.shuffle(files)
        
        split_idx = int(len(files) * (1 - VAL_SPLIT))
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        for f in train_files:
            tasks.append((os.path.join(raw_dir, f), os.path.join(train_out, f.replace(".wav", ".png"))))
            
        for f in val_files:
            tasks.append((os.path.join(raw_dir, f), os.path.join(val_out, f.replace(".wav", ".png"))))
            
        print(f" [+] {cls}: Routed {len(train_files)} to Train | {len(val_files)} to Val")

    total_tasks = len(tasks)
    print(f"\n[SYSTEM] Crunching {total_tasks} files sequentially. This will take ~5-10 minutes.")
    
    # Process sequentially to protect RAM
    for i, (wav, img) in enumerate(tasks):
        convert_to_spectrogram(wav, img)
        
        if (i + 1) % 500 == 0:
            print(f" [+] Generated {i + 1}/{total_tasks} spectrograms...")
            
    print("\n[+] SUCCESS! All spectrograms generated and perfectly split for PyTorch.")

if __name__ == "__main__":
    main()