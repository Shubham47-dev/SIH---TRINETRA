import torch
import torch.nn as nn

class ECABlock(nn.Module):
    def __init__(self, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class MicroDopplerECACNN(nn.Module):
    def __init__(self, num_classes=4):
        super(MicroDopplerECACNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.eca1 = ECABlock(k_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.eca2 = ECABlock(k_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.pool1(self.eca1(torch.relu(self.bn1(self.conv1(x)))))
        x = self.pool2(self.eca2(torch.relu(self.bn2(self.conv2(x)))))
        return self.classifier(x)