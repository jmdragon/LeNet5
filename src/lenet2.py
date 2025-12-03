# lenet2.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet2(nn.Module):
    """
    Modified LeNet-style CNN for MNIST with:
      - ReLU activations
      - Max pooling instead of average subsampling
      - Standard linear 10-way classifier (logits)
    """

    def __init__(self):
        super().__init__()
        # Input: (B, 1, 32, 32)

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)   # 32x32 -> 28x28
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)  # 14x14 -> 10x10

        # Pooling (max pool gives some translational invariance)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)       # 28 -> 14, 14 -> 7

        # Fully connected layers
        # After conv+pool: 16 feature maps of size 5x5? Let's check:
        # 32 -> conv5 -> 28 -> pool2 -> 14
        # 14 -> conv5 -> 10 -> pool2 -> 5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)   # 10 classes

    def forward(self, x):
        # x: (B, 1, 32, 32), pixels in [0,255]

        # Optional simple normalization (not as aggressive as LeNet1’s)
        x = x / 255.0

        x = self.pool(F.relu(self.conv1(x)))  # (B, 6, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 16, 5, 5)

        x = x.view(x.size(0), -1)            # (B, 16*5*5)
        x = F.relu(self.fc1(x))              # (B, 120)
        x = F.relu(self.fc2(x))              # (B, 84)
        logits = self.fc3(x)                 # (B, 10)

        return logits
