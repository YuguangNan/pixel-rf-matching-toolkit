import os
import random
import shutil
import subprocess

# =========================
# Import utilities
# =========================
from src.utils.layout_utils import (
    generate_connected_network,
    save_proj_a,
    save_txt
)
from src.utils.ads_utils import run_ads_simulation


# =========================
# Main simulation loop
# =========================
size = 15
em_setup_path = r"D:\emSimulation\emSetup_MoM"
output_base   = r"D:\emResults"

os.makedirs(em_setup_path, exist_ok=True)
os.makedirs(output_base,   exist_ok=True)


def run_simulation_and_save(proj_folder, output_base):
    """
    Wrapper around adsMomWrapper + moving outputs.
    This keeps your original logic unchanged.
    """
    os.chdir(proj_folder)

    # Call ADS Momentum
    run_ads_simulation()

    # Save results if CTI exists
    if os.path.exists("proj.cti"):
        count = len(os.listdir(output_base))
        result_path = os.path.join(output_base, f"result_{count}")
        os.makedirs(result_path, exist_ok=True)

        shutil.copy("proj.cti",  os.path.join(result_path, "proj.cti"))
        shutil.copy("proj_a",     os.path.join(result_path, "proj_a"))
        shutil.copy("proj_a.txt", os.path.join(result_path, "proj_a.txt"))


# =========================
# Loop multiple simulations
# =========================
for i in range(1000):
    grid, start, end = generate_connected_network(size)
    save_proj_a(grid, start, end, os.path.join(em_setup_path, "proj_a"))
    save_txt(grid, os.path.join(em_setup_path, "proj_a.txt"))

    run_simulation_and_save(em_setup_path, output_base)
