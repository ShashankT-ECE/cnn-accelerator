"""L1 FP32 LeNet-5 reference package (Step 1B).

Independent from the Phase-1/Phase-2 reference code and from the future INT8
(L2) hardware reference. Implements the locked workload in
``docs/LENET5_SPEC.md``. Nothing here quantizes.
"""
from . import config  # noqa: F401
from .model import LeNet5, init_weights  # noqa: F401
