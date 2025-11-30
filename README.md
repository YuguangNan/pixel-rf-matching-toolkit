# Pixel-RF-Matching-Toolkit  
### Pixel Layout Automation · ADS Momentum Simulation · Forward/Inverse DL Models

This toolkit provides a complete workflow for **pixel-based RF matching network design**, including:

- Pixel layout generation (15×15 by default)
- Automatic export to DXF or `proj_a` (ADS layout)
- Automated ADS Momentum EM simulation via `adsMomWrapper`
- Dataset extraction (.npz)
- Deep-learning forward model
- GA-based inverse design

The toolkit is designed to be **extendable**, allowing future researchers to expand to 20×20, 25×25, or arbitrary pixel resolutions.

---

# ✨ Features

### ✔ Pixel Layout Generator  
- Generates connected pixel patterns  
- Avoids isolated islands  
- Supports random skeleton + random filling  
- Default grid size: **15×15 (extendable)**

### ✔ ADS Momentum Automation  
The toolkit assumes that ADS has already been configured with:

- Input/output ports (left and right pixel rows)
- Frequency sweep points  
- Mesh density (mesh quality)
- Substrate definition (material stack)
- Boundary conditions  
- Momentum settings  
- Reference impedance (typically 50 Ω)

These settings must be configured **in the ADS project**.  
The Python scripts only replace:

proj_a (layout geometry)


and run:



adsMomWrapper -O -3D proj proj


ADS then generates the simulation outputs:



proj.cti
proj.prt


### ✔ Dataset Extraction  
- Reads S11 / S21 / S22 from CTI  
- Automatically follows **the frequency points defined in ADS**  
- No assumption about frequency count  
- Saves dataset to `.npz`

### ✔ Forward Model (Deep Learning)  
- Accepts N×N inputs (default 15×15)
- Output dimension automatically matches:



6 × N_freq (Re/Im × S11/S21/S22)


- Multi-scale CNN + SE attention  
- Trains on your ADS dataset

### ✔ Genetic Algorithm Inverse Design  
- Targets S11, S22 (return loss)  
- Targets S21 (gain)  
- Supports arbitrary source/load impedance  
- Can include fill-factor regularization  
- Calls ADS again to validate predicted layout

---

# 📡 ADS Configuration Requirements

Before using this toolkit, **ADS must be properly configured** inside the corresponding project folder.

Required manual settings:

### ✔ Frequency Sweep  
Set the frequency points in ADS Momentum setup.  
Python will **read whatever ADS outputs**—no need to modify the script.

### ✔ Input/Output Ports  
For 15×15 layouts:

- Input port = left side center pixel  
- Output port = right side center pixel  

If you change to 20×20 or 25×25, update:



start = (N//2, 0)
end = (N//2, N-1)


### ✔ Substrate  
Define substrate stack (e.g., Rogers, FR4, Si/SiO2, IHP SG13G2 stack, etc.)  
Python does NOT modify material definitions.

### ✔ Mesh Quality  
Must be configured in ADS (Momentum mesh density, edge mesh, etc.)

### ✔ Momentum Settings  
- Reference impedance  
- Solver type  
- Convergence criteria  
Must also be configured inside ADS.

⚠ Python scripts **do not overwrite these parameters** — they reuse the existing ADS project settings.

---



🚀 Workflow Summary

Define ADS simulation settings
(freq sweep, substrate, mesh, ports, Momentum settings)

Generate random layouts
→ proj_a

Run ADS automatically
→ produces proj.cti

Extract dataset
→ .npz

Train forward model

Run GA inverse design
→ produces layout + ADS validation
