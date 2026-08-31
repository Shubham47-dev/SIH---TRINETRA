import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model_cnn import MicroDopplerECACNN, RadarSpectrogramDataset

def train_model():
    print("Loading Radar Dataset...")
    train_dataset = RadarSpectrogramDataset(num_samples=1000)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MicroDopplerECACNN(num_classes=4).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
 
    epochs = 5
    print(f"Starting Training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (spectrograms, labels) in enumerate(train_loader):
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            optimizer.zero_grad()
 
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
    
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    torch.save(model.state_dict(), "spectra_eca_weights.pth")
    print("Model weights saved to spectra_eca_weights.pth")

if __name__ == "__main__":
    train_model()