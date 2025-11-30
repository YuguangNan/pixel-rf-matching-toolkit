import os
import numpy as np

# ======= Base directory containing result_xxxx folders =======
base_dir = r"D:\20GHz_dataset"

# ======= Find all result_xxxx subfolders =======
subfolders = [
    os.path.join(base_dir, f)
    for f in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, f)) and f.startswith("result_")
]

# =====================================================
# Function: Parse proj.cti and extract S11, S21, S22
# =====================================================
def parse_cti(file_path):
    """Parse proj.cti and extract S11, S21, S22 (real + imag)."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    data_blocks = []
    current_block = []
    in_block = False

    for line in lines:
        line = line.strip()
        if line.startswith('BEGIN'):
            current_block = []
            in_block = True
        elif line.startswith('END'):
            in_block = False
            if len(current_block) > 0:
                data_blocks.append(np.array(current_block, dtype=np.float32))
        elif in_block and ',' in line:
            try:
                re, im = [float(x) for x in line.split(',')[:2]]
                current_block.append([re, im])
            except ValueError:
                continue

    # Must have at least 4 data blocks
    if len(data_blocks) < 4:
        print(f"⚠️ Not enough data blocks: {file_path}")
        return None

    # Extract S parameters
    S11 = np.array(data_blocks[0])
    S21 = np.array(data_blocks[2])  # Note: S21 is the 3rd block
    S22 = np.array(data_blocks[3])

    # Ensure each block has 51 freq points × 2 columns (real + imag)
    if S11.shape != (51, 2) or S21.shape != (51, 2) or S22.shape != (51, 2):
        print(f"⚠️ Invalid frequency point count: {file_path}")
        return None

    return S11, S21, S22


# =====================================================
# Function: Parse proj_a.txt layout
# =====================================================
def parse_layout(txt_path):
    """Read proj_a.txt and convert to 15×15 binary matrix."""
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        grid = []
        for line in lines:
            line = line.strip()
            if line:
                nums = [int(ch) for ch in line.split() if ch in ['0', '1']]
                if len(nums) == 15:
                    grid.append(nums)

        grid = np.array(grid)
        if grid.shape == (15, 15):
            return grid
        else:
            return None
    except Exception:
        return None


# =====================================================
# Main extraction loop
# =====================================================
X_list, S11_list, S21_list, S22_list = [], [], [], []
valid_count = 0

for folder in subfolders:
    cti_path = os.path.join(folder, "proj.cti")
    txt_path = os.path.join(folder, "proj_a.txt")

    if os.path.exists(cti_path) and os.path.exists(txt_path):
        layout = parse_layout(txt_path)
        sparams = parse_cti(cti_path)

        if layout is not None and sparams is not None:
            S11, S21, S22 = sparams
            X_list.append(layout)
            S11_list.append(S11)
            S21_list.append(S21)
            S22_list.append(S22)
            valid_count += 1

print(f"✅ Successfully parsed {valid_count} samples, saving dataset...")

# =====================================================
# Convert to NumPy arrays and save
# =====================================================
X = np.array(X_list, dtype=np.int8)
S11 = np.array(S11_list, dtype=np.float32)
S21 = np.array(S21_list, dtype=np.float32)
S22 = np.array(S22_list, dtype=np.float32)

save_path = os.path.join(base_dir, "sparam_20GHz_dataset.npz")
np.savez(save_path, X=X, S11=S11, S21=S21, S22=S22)

print(f"✅ Dataset extraction completed: {valid_count} samples")
print(f"✅ Saved as: {save_path}")
print(f"📊 Shapes -> X={X.shape}, S11={S11.shape}, S21={S21.shape}, S22={S22.shape}")
