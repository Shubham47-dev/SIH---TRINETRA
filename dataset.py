import os
import cv2
import torch
import librosa
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class RealRadarSpectrogramDataset(Dataset):
    def __init__(self, data_dir, time_steps=128, freq_bins=128):
        self.data_dir = Path(data_dir)
        self.time_steps = time_steps
        self.freq_bins = freq_bins
        
        self.filepaths = list(self.data_dir.rglob("*.wav")) + list(self.data_dir.rglob("*.WAV"))
        
        self.class_map = {
            "0_bird": 0,
            "1_human": 1,
            "2_drone_unarmed": 2,
            "3_drone_payload": 3
        }

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        try:
            wav_path = self.filepaths[idx]
            
            # 1. Extract Label from parent folder name
            parent_folder = wav_path.parent.name
            label = self.class_map[parent_folder]
            
            # 2. Load Audio (Force 22050Hz, max 1 second)
            audio, sr = librosa.load(str(wav_path), sr=22050, duration=1.0)
            
            # Standardize length: Pad with zeros if less than 1 second
            if len(audio) < sr:
                pad_width = sr - len(audio)
                audio = np.pad(audio, (0, pad_width), mode='constant')

            # 3. Apply the STFT Math
            stft_matrix = librosa.stft(audio, n_fft=256, hop_length=172)
            magnitude = np.abs(stft_matrix)
            
            # 4. Decibel Scaling
            spectrogram_db = librosa.amplitude_to_db(magnitude, ref=np.max)
            
            # 5. Normalize (0 to 1)
            spec_min, spec_max = spectrogram_db.min(), spectrogram_db.max()
            if spec_max - spec_min > 0:
                spectrogram_normalized = (spectrogram_db - spec_min) / (spec_max - spec_min)
            else:
                spectrogram_normalized = spectrogram_db - spec_min
                
            # 6. Resize to 128x128 for the CNN
            resized_matrix = cv2.resize(spectrogram_normalized, (self.time_steps, self.freq_bins), interpolation=cv2.INTER_CUBIC)
            
            # 7. Convert to Tensor -> Shape (1, 128, 128)
            tensor_matrix = torch.tensor(resized_matrix, dtype=torch.float32).unsqueeze(0)
            
            return tensor_matrix, label
            
        except Exception as e:
            # If the file is corrupted or empty, intercept the crash
            print(f"\n[!] Skipping corrupted file: {self.filepaths[idx].name} | Error: {e}")
            
            # Pick a completely random index from the dataset
            new_idx = random.randint(0, len(self.filepaths) - 1)
            
            # Recursively try again with the new valid file
            return self.__getitem__(new_idx)