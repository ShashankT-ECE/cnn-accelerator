"""CIFAR-10 cuda-convnet-style workload (Step 6.2).

Independent of the LeNet-5 model. Implements the FP32 reference for the
secondary CNN workload (docs/CIFAR10_WORKLOAD_SPEC.md).
"""
from . import config  # noqa: F401
from .model import Cifar10Net, init_weights  # noqa: F401
