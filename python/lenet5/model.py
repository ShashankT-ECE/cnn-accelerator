"""L1 FP32 reference model for the locked LeNet-5 workload.

Architecture (``docs/LENET5_SPEC.md`` §2), exact:

    input [N,1,32,32]
      Conv2d(1->6,   5x5, valid) -> [N,6,28,28]
      ReLU
      MaxPool(2x2/2)             -> [N,6,14,14]
      Conv2d(6->16,  5x5, valid) -> [N,16,10,10]
      ReLU
      MaxPool(2x2/2)             -> [N,16,5,5]
      Conv2d(16->120,5x5, valid) -> [N,120,1,1]
      ReLU
      Flatten                    -> [N,120]
      Linear(120->84)            -> [N,84]
      ReLU
      Linear(84->10)             -> [N,10]  (logits, no ReLU)

This is the FP32 "L1" reference: the accuracy ground truth. It is deliberately
independent of the future INT8 (L2) hardware reference — no quantization here.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m: nn.Module) -> None:
    """Explicit weight init (reproduces PyTorch 2.1.2's documented default).

    Conv2d and Linear:
      weight ~ kaiming_uniform_(a=sqrt(5))
      bias   ~ uniform_(-1/sqrt(fan_in), +1/sqrt(fan_in))

    Applied after construction so the scheme is visible and version-robust,
    rather than relying on ``nn.Module``'s implicit ``reset_parameters()``.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in = m.weight.size(1)
            for s in m.weight.shape[2:]:
                fan_in *= s
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(m.bias, -bound, bound)


class LeNet5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=True)
        self.conv5 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0, bias=True)
        self.fc1 = nn.Linear(120, 84, bias=True)
        self.fc2 = nn.Linear(84, 10, bias=True)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits [N,10]."""
        return self.forward_layers(x)["output"]

    def forward_layers(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Layer-by-layer activations (the L1 golden reference).

        Keys map to the spec's layer names. Conv/FC outputs are exposed
        pre-ReLU (the values the future INT8 golden compares against).
        """
        c1 = self.conv1(x)                       # [N,6,28,28]
        relu1 = F.relu(c1)
        pool1 = F.max_pool2d(relu1, 2, 2)        # [N,6,14,14]
        c3 = self.conv3(pool1)                   # [N,16,10,10]
        relu3 = F.relu(c3)
        pool2 = F.max_pool2d(relu3, 2, 2)        # [N,16,5,5]
        c5 = self.conv5(pool2)                   # [N,120,1,1]
        relu5 = F.relu(c5)
        flat = relu5.view(relu5.size(0), -1)     # [N,120]
        f6 = self.fc1(flat)                      # [N,84]
        relu6 = F.relu(f6)
        output = self.fc2(relu6)                 # [N,10] logits
        return {
            "input": x,
            "c1": c1, "relu1": relu1, "pool1": pool1,
            "c3": c3, "relu3": relu3, "pool2": pool2,
            "c5": c5, "relu5": relu5,
            "f6": f6, "relu6": relu6,
            "output": output,
        }
