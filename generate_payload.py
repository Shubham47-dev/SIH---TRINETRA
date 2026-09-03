import os
import librosa
import soundfile as sf

input_folder = "ai_data/data/train/2_drone_unarmed"
output_folder = "ai_data/data/train/3_drone_payload"
os.makedirs(output_folder, exist_ok=True)

print("[SYSTEM] Starting synthetic payload generation (Pitch-Shifting)...")

# Grab all the newly downloaded drone files
files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]
total_files = len(files)

print(f"[SYSTEM] Found {total_files} unarmed drone files. Processing...")

for i, file_name in enumerate(files):
    file_path = os.path.join(input_folder, file_name)
    
    try:
        # Load the original high-RPM audio
        y, sr = librosa.load(file_path, sr=22050)
        
        # Shift pitch DOWN by 3 semitones (simulates heavy payload/lower RPM)
        y_payload = librosa.effects.pitch_shift(y, sr=sr, n_steps=-3)
        
        # Save the new synthetic payload audio
        output_path = os.path.join(output_folder, f"payload_{file_name}")
        sf.write(output_path, y_payload, sr)
        
        # Print progress every 500 files
        if (i + 1) % 500 == 0:
            print(f" [+] Generated {i + 1}/{total_files} synthetic payload files...")
            
    except Exception as e:
        print(f" [!] Skipped {file_name} due to error: {e}")

print("\n[+] SUCCESS! Synthetic payload dataset created.")