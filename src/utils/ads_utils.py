import os
import shutil
import subprocess
import numpy as np


# ============================================================
# ============ ADS Momentum Simulation Utilities ==============
# ============================================================


def run_ads(em_setup_path):
    """
    Run ADS Momentum using adsMomWrapper.
    Equivalent to:
        adsMomWrapper -O -3D proj proj
    Working directory is the EM setup folder.
    """
    cmd = "adsMomWrapper -O -3D proj proj"
    print("[INFO] Running ADS:", cmd)
    subprocess.run(cmd, cwd=em_setup_path, shell=True, check=True)
    print("✔ ADS Momentum simulation completed.")



def run_simulation_and_save(proj_folder, output_base):
    """
    Execute adsMomWrapper inside the ADS EM setup folder.
    After simulation, copy proj.cti and proj_a/proj_a.txt into
    a new result folder under output_base.
    """
    os.chdir(proj_folder)
    subprocess.run(["adsMomWrapper", "-O", "-3D", "proj", "proj"], shell=True)

    if os.path.exists("proj.cti"):
        count = len(os.listdir(output_base))
        result_path = os.path.join(output_base, f"result_{count}")
        os.makedirs(result_path, exist_ok=True)

        shutil.copy("proj.cti", os.path.join(result_path, "proj.cti"))
        shutil.copy("proj_a", os.path.join(result_path, "proj_a"))
        shutil.copy("proj_a.txt", os.path.join(result_path, "proj_a.txt"))

        print(f"✔ Saved ADS results into: {result_path}")



# ============================================================
# ===================== CTI File Parser ========================
# ============================================================

def parse_cti(path):
    """
    Parse proj.cti file:
      - Read frequency list
      - Extract BEGIN/END complex S parameter blocks
      - Return S11, S21, S22（block 0, 1, 3）
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CTI file not found: {path}")

    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    freqs = []
    in_var = False

    # -------------------------
    # Parse frequency list
    # -------------------------
    for ln in lines:
        if ln.startswith("VAR_LIST_BEGIN"):
            in_var = True
            continue
        if ln.startswith("VAR_LIST_END"):
            in_var = False
            continue
        if in_var:
            freqs.append(float(ln))

    freqs = np.array(freqs)

    # -------------------------
    # Parse S-parameter blocks
    # -------------------------
    blocks = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("BEGIN"):
            vals = []
            i += 1
            while i < len(lines) and not lines[i].startswith("END"):
                r, im = lines[i].split(',')
                vals.append(float(r) + 1j * float(im))
                i += 1
            blocks.append(np.array(vals, dtype=np.complex128))
        i += 1

    if len(blocks) < 4:
        raise ValueError("CTI file does not contain sufficient S-parameter blocks (expected >=4).")

    S11 = blocks[0]
    S21 = blocks[1]
    S22 = blocks[3]

    return freqs, S11, S21, S22
