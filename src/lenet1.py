# lenet1.py
import torch
import torch.nn as nn
import numpy as np

class LeNet5RBF(nn.Module):
    """
    LeNet-5 with:
      - C1 (6x5x5), S2 (2x2 avg pool)
      - C3 (16x5x5), S4 (2x2 avg pool)
      - C5 (120x5x5), F6 (84 units)
      - Output: 10 RBF units (squared Euclidean distance to fixed prototypes)

    The output is a (batch_size, 10) matrix of penalties y_k; 
    lower y_k = better match for class k.
    """
    def __init__(self, prototypes: np.ndarray):
        super().__init__()

        # Convolution + subsampling layers (simplified connectivity: full connection)
        self.c1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)   # 32x32 -> 28x28
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)      # 28x28 -> 14x14

        self.c3 = nn.Conv2d(6, 16, kernel_size=5, stride=1)  # 14x14 -> 10x10
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)      # 10x10 -> 5x5

        self.c5 = nn.Conv2d(16, 120, kernel_size=5, stride=1)  # 5x5 -> 1x1
        self.f6 = nn.Linear(120, 84)  # 84 = 7 x 12 bitmap for RBFs

        # RBF prototypes μ_k (10 x 84), fixed (not trainable)
        proto = prototypes.astype(np.float32)
        self.register_buffer("prototypes", torch.from_numpy(proto))  # (10, 84)

        # scaled tanh parameters (Appendix A in the paper)
        self.a = 1.7159
        self.b = 2.0 / 3.0

    def scaled_tanh(self, x):
        return self.a * torch.tanh(self.b * x)

    def forward(self, x):
        """
        x: (B,1,32,32), pixel values in [0, 255]
        Returns:
          penalties: (B,10) where smaller is better.
        """

        # Map [0,255] -> roughly [-0.1, 1.175] as in the paper
        x = -0.1 + (x / 255.0) * 1.275

        x = self.c1(x)
        x = self.scaled_tanh(x)
        x = self.s2(x)

        x = self.c3(x)
        x = self.scaled_tanh(x)
        x = self.s4(x)

        x = self.c5(x)
        x = self.scaled_tanh(x)

        x = x.view(x.size(0), -1)   # (B, 120)
        x = self.f6(x)
        f6 = self.scaled_tanh(x)    # (B, 84)

        # RBF output: squared Euclidean distance to each prototype
        # f6: (B,84); prototypes: (10,84)
        diff = f6.unsqueeze(1) - self.prototypes.unsqueeze(0)  # (B, 10, 84)
        penalties = torch.sum(diff * diff, dim=2)              # (B, 10)

        return penalties
