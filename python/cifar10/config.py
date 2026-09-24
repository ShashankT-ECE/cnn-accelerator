"""Frozen configuration for the CIFAR-10 cuda-convnet-style workload (Step 6.2).

Independent of the LeNet-5 config. Architecture and training choices are pinned
here (no hidden defaults).
"""
from __future__ import annotations

# ---- Architecture -----------------------------------------------------------
IMG_C = 3
IMG_H = 32
IMG_W = 32
NUM_CLASSES = 10

# Per-layer channel counts (5x5 valid convolutions, then FC).
C1_IC, C1_OC = 3, 32
C2_IC, C2_OC = 32, 32
C3_IC, C3_OC = 32, 64
FC_IN, FC_OUT = 64, 10

KERNEL = 5          # 5x5 valid (no padding), stride 1
POOL_KERNEL = 2     # max-pool 2x2
POOL_STRIDE = 2

# Derived totals (params include biases).
TOTAL_PARAMS = 79978
TOTAL_MACS = 4493440

# ---- Training (frozen) ------------------------------------------------------
SEED = 42
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
OPTIMIZER = "SGD"
LR_SCHEDULER = "none"
LOSS = "CrossEntropyLoss"
DEVICE = "cpu"

# ---- Data split (fixed indices, deterministic) -----------------------------
TRAIN_START, TRAIN_END = 0, 45000     # CIFAR-10 train[0:45000]    -> train
VAL_START, VAL_END = 45000, 50000     # CIFAR-10 train[45000:50000] -> val
TEST_N = 10000                         # CIFAR-10 test[0:10000]     -> test

# ---- Checkpoint paths -------------------------------------------------------
CKPT_DIR = "data/checkpoint"
CKPT_FILE = "cifar10_fp32.pt"
META_FILE = "cifar10_fp32_meta.json"
