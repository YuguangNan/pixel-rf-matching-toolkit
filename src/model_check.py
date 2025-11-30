import os
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from keras.models import load_model

# =============================
# Import utilities
# =============================
from src.utils.sparam_utils import (
    flat_to_complex,
    db20,
    parse_cti
)

from src.utils.ads_utils import run_ads_simulation
from src.utils.layout_utils import write_proj_a  # unified name
# (your script had generate_connected_layout inside it, so we copy it directly here unchanged)


# =============================
# 1. Path configuration
# =============================
MODEL_PATH      = r"dnn_model_4.h5"
EM_SETUP_DIR    = r"D:\emSimulation\emSetup_MoM"
PROJ_A_PATH     = os.path.join(EM_SETUP_DIR, "proj_a")
CTI_PATH        = os.path.join(EM_SETUP_DIR, "proj.cti")


# =============================
# 2. ADS frequencies
# =============================
FREQS = np.array([
    1e8, 5.304347826e8, 9.608695652e8,
    1.391304348e9, 1.821739130e9, 2.252173913e9, 2.4e9,
    2.682608696e9, 3.113043478e9, 3.543478261e9, 3.973913043e9,
    4.404347826e9, 4.834782609e9, 5.265217391e9, 5.695652174e9,
    6.126086956e9, 6.556521739e9, 6.986956522e9, 7.417391304e9,
    7.847826087e9, 8.278260869e9, 8.708695652e9, 9.139130435e9,
    9.569565217e9, 1e10
])
N_FREQ = len(FREQS)


# =============================
# 3. Forward model prediction
# =============================
def model_predict(model, grid15):
    x = grid15[np.newaxis, ..., np.newaxis].astype(np.float32)
    out = model.predict(x, verbose=0)[0]  # (150,)

    S11 = flat_to_complex(out[0:2*N_FREQ])
    S21 = flat_to_complex(out[2*N_FREQ:4*N_FREQ])
    S22 = flat_to_complex(out[4*N_FREQ:6*N_FREQ])
    return S11, S21, S22


# =============================
# 4. Your original layout generator (unchanged)
# =============================
SIZE = 15
START = (7, 0)
END   = (7, 14)

def generate_connected_layout():
    """Original logic kept EXACTLY as is."""
    grid = np.zeros((SIZE, SIZE), dtype=np.int8)

    # Connect a simple path first
    for x in range(0, SIZE):
        grid[7, x] = 1

    # Add random blocks
    for _ in range(70):
        y = np.random.randint(0, SIZE)
        x = np.random.randint(0, SIZE)
        grid[y, x] = 1

    return grid


# =============================
# 5. Main process
# =============================
def main():
    model = load_model(MODEL_PATH, compile=False)
    print("✔ Forward model loaded")

    # Generate layout
    grid = generate_connected_layout()

    # Write proj_a
    write_proj_a(grid, START, END, PROJ_A_PATH)

    # Predict using DNN
    S11_p, S21_p, S22_p = model_predict(model, grid)

    # Run ADS Momentum
    run_ads_simulation()

    # Read ADS output
    freqs_ads, S11_ads, S21_ads, S22_ads = parse_cti(CTI_PATH)

    # =============================
    # Visualization
    # =============================
    plt.figure(figsize=(10,6))
    plt.plot(freqs_ads/1e9, db20(S11_ads), label="ADS S11")
    plt.plot(FREQS/1e9, db20(S11_p), "--", label="DNN S11")
    plt.grid(); plt.legend(); plt.title("S11 Comparison")
    plt.xlabel("GHz"); plt.ylabel("dB")
    plt.savefig("S11 Comparison.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(freqs_ads/1e9, db20(S21_ads), label="ADS S21")
    plt.plot(FREQS/1e9, db20(S21_p), "--", label="DNN S21")
    plt.grid(); plt.legend(); plt.title("S21 Comparison")
    plt.xlabel("GHz"); plt.ylabel("dB")
    plt.savefig("S21 Comparison.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(freqs_ads/1e9, db20(S22_ads), label="ADS S22")
    plt.plot(FREQS/1e9, db20(S22_p), "--", label="DNN S22")
    plt.grid(); plt.legend(); plt.title("S22 Comparison")
    plt.xlabel("GHz"); plt.ylabel("dB")
    plt.savefig("S22 Comparison.png", dpi=200)
    plt.show()

    print("\n🎉 model_check finished — ADS vs DNN comparison ready.")


if __name__ == "__main__":
    main()
