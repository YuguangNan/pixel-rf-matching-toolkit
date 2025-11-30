import os
import numpy as np
from src.utils.layout_utils import (
    generate_connected_network,
    export_to_dxf
)

# =========================
# Basic parameters
# =========================
SIZE = 15

if __name__ == "__main__":
    # Generate a connected layout (grid, start, end)
    grid, start, end = generate_connected_network(size=SIZE, max_path_factor=2)

    # Export to DXF
    export_to_dxf(
        grid,
        start,
        end,
        cell_size=1,
        filename="layout.dxf"
    )
