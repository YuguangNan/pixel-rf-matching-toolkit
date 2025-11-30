"""
Global configuration file for the Pixel-RF-Matching-Toolkit.

This file centralizes:
- ADS project paths
- Frequency definitions (auto-detected from CTI when possible)
- Pixel grid resolution
- Default file names
- Model paths
- Common simulation parameters
"""

import os

# -------------------------------------------------------------
# Project Root
# -------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# -------------------------------------------------------------
# ADS Paths (Modify these according to your local workspace)
# -------------------------------------------------------------
ADS_EM_SETUP_DIR = r"D:\emSimulation\emSetup_MoM"  # Folder containing proj_a, proj.cti, etc.

PROJ_A_PATH = os.path.join(ADS_EM_SETUP_DIR, "proj_a")   # Layout file
CTI_PATH    = os.path.join(ADS_EM_SETUP_DIR, "proj.cti")
PRT_PATH    = os.path.join(ADS_EM_SETUP_DIR, "proj.prt")

# -------------------------------------------------------------
# Pixel Layout Configuration
# -------------------------------------------------------------
PIXEL_SIZE = 15   # Default grid resolution (15×15)

# Ports are located at left/right center rows for 15×15
PORT_START = (PIXEL_SIZE // 2, 0)
PORT_END   = (PIXEL_SIZE // 2, PIXEL_SIZE - 1)

# -------------------------------------------------------------
# Dataset & Model Paths
# -------------------------------------------------------------
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "datasets")
MODEL_DIR   = os.path.join(PROJECT_ROOT, "models")

FORWARD_MODEL_PATH = os.path.join(MODEL_DIR, "dnn_forward_best.h5")

# -------------------------------------------------------------
# Inverse Design Parameters
# -------------------------------------------------------------
TARGET_FREQ_IDX = 10   # Example target frequency index (will match CTI)
TARGET_S11_MAX_DB = -10
TARGET_S22_MAX_DB = -10
TARGET_S21_MIN_DB = -1

# Source / Load impedance for inverse matching
Z_SOURCE = 50 + 0j
Z_LOAD   = 50 + 0j

# -------------------------------------------------------------
# GA Search Parameters
# -------------------------------------------------------------
POP_SIZE = 30
N_GENERATIONS = 40
CROSS_RATE = 0.7
MUT_RATE = 0.02
ELITE_K = 2
