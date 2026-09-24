"""FP32 reference model for the CIFAR-10 cuda-convnet-style workload.

Architecture (5x5 valid convs, ReLU, 2x2 max-pool, FC):
    input [N,3,32,32]
      Conv2d(3->32,  5x5 valid) -> [N,32,28,28]  ReLU  MaxPool(2x2/2) -> [N,32,14,14]
      Conv2d(32->32, 5x5 valid) -> [N,32,10,10]  ReLU  MaxPool(2x2/2) -> [N,32,5,5]
      Conv2d(32->64, 5x5 valid) -> [N,64,1,1]    ReLU  Flatten -> [N,64]
      Linear(64->10)             -> [N,10] logits
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m: nn.Module) -> None:
    """Explicit init (reproduces PyTorch's documented default)."""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in = m.weight.size(1)
            for s in m.weight.shape[2:]:
                fan_in *= s
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(m.bias, -bound, bound)


class Cifar10Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0, bias=True)
        self.fc = nn.Linear(64, 10, bias=True)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_layers(x)["logits"]

    def forward_layers(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        c1 = self.conv1(x)                        # [N,32,28,28]
        r1 = F.relu(c1)
        p1 = F.max_pool2d(r1, 2, 2)               # [N,32,14,14]
        c2 = self.conv2(p1)                       # [N,32,10,10]
        r2 = F.relu(c2)
        p2 = F.max_pool2d(r2, 2, 2)               # [N,32,5,5]
        c3 = self.conv3(p2)                       # [N,64,1,1]
        r3 = F.relu(c3)
        flat = r3.view(r3.size(0), -1)            # [N,64]
        logits = self.fc(flat)                    # [N,10]
        return {
            "input": x, "c1": c1, "relu1": r1, "pool1": p1,
            "c2": c2, "relu2": r2, "pool2": p2,
            "c3": c3, "relu3": r3, "logits": logits,
        }
