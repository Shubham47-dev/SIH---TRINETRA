import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model_cnn import MicroDopplerECACNN

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 20
NUM_CLASSES = 5  
MODEL_SAVE_PATH = "spectra_eca_best_weights.pt"

def train_network():
    # 1. HARDWARE ACCELERATION
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Engine Booting. Training on device: {device}")
    if torch.cuda.is_available():
        print(f"[SYSTEM] GPU Detected: {torch.cuda.get_device_name(0)}")

    # 2. DATASET ROUTING
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Pointing to the new spectrogram folders
    train_dir = os.path.join("ai_data", "spectrograms", "train")
    val_dir = os.path.join("ai_data", "spectrograms", "val")

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

    # num_workers=0 prevents Windows multi-processing memory crashes
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"[DATA] Classes locked: {train_dataset.classes}")
    print(f"[DATA] Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    # 3. INITIALIZE AI
    model = MicroDopplerECACNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Anti-hallucination smoothing
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0.0

    # 4. THE LEARNING LOOP
    print("\n--- STARTING TRAINING PIPELINE ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100 * correct_train / total_train

        # --- VALIDATION PHASE ---
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = 100 * correct_val / total_val if total_val > 0 else 0.0

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Save weights only if the AI gets smarter
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f" [+] Checkpoint! Saved new best weights to {MODEL_SAVE_PATH}")

    print(f"\n[SYSTEM] Training complete. Peak Validation Accuracy: {best_val_accuracy:.2f}%")

if __name__ == "__main__":
    train_network()