"""Frozen configuration for the L1 FP32 LeNet-5 reference (Step 1B).

Every reproducibility-relevant choice is pinned here — no hidden defaults.
Architecture matches ``docs/LENET5_SPEC.md`` (LOCKED) exactly.

The INT8 (L2) hardware reference is deliberately NOT part of this module;
nothing here quantizes.
"""
from __future__ import annotations

# ---- Architecture (docs/LENET5_SPEC.md §2) --------------------------------
IMG_H = 32
IMG_W = 32
NUM_CLASSES = 10

# Per-layer channel counts (5x5 valid convolutions, then fully-connected).
C1_IC, C1_OC = 1, 6
C3_IC, C3_OC = 6, 16
C5_IC, C5_OC = 16, 120
F6_IN, F6_OUT = 120, 84
OUT_IN, OUT_OUT = 84, 10

KERNEL = 5           # 5x5 valid (no padding), stride 1
POOL_KERNEL = 2      # max-pool 2x2
POOL_STRIDE = 2

# Derived totals (params include biases; verified against the spec).
TOTAL_PARAMS = 61706
TOTAL_MACS = 416520

# ---- Training (all frozen; no hidden defaults) ----------------------------
SEED = 42
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
OPTIMIZER = "SGD"          # SGD(momentum=0.9), weight_decay=0
LR_SCHEDULER = "none"      # constant learning rate, no scheduling
LOSS = "CrossEntropyLoss"  # reduction='mean' (PyTorch default)
DEVICE = "cpu"             # torch 2.1.2+cpu — no CUDA available

# Weight init: explicit kaiming_uniform_(a=sqrt(5)) for weights,
# uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) for biases — see model.init_weights.

# ---- Data split (fixed indices, deterministic — no random split) ---------
TRAIN_START, TRAIN_END = 0, 50000     # MNIST train[0:50000]    -> train
VAL_START, VAL_END = 50000, 60000     # MNIST train[50000:60000] -> val
TEST_N = 10000                        # MNIST test[0:10000]     -> test

# ---- Checkpoint paths (relative to repo root) -----------------------------
CKPT_DIR = "data/checkpoint"
CKPT_FILE = "lenet5_fp32.pt"
META_FILE = "lenet5_fp32_meta.json"
