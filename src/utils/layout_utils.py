import os
import random
import numpy as np
import ezdxf
from collections import deque


# ============================================================
# =============== from generate_layout.py =====================
# ============================================================

# ===== Basic parameters =====
SIZE  = 15
START = (SIZE // 2, 0)
END   = (SIZE // 2, SIZE - 1)


# ===== BFS shortest path length =====
def shortest_path_length(grid, start, end):
    H, W = grid.shape
    if grid[start] == 0 or grid[end] == 0:
        return None
    q = deque([(start[0], start[1], 0)])  # (x, y, path length)
    seen = {start}
    while q:
        x, y, d = q.popleft()
        if (x, y) == end:
            return d
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx, ny] == 1 and (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append((nx, ny, d + 1))
    return None


# ===== Count isolated conductive islands (excluding main connected path) =====
def count_islands_excluding_path(grid, start, end):
    H, W = grid.shape
    visited = set()
    q = deque([start])
    visited.add(start)
    while q:
        x, y = q.popleft()
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = x+dx, y+dy
            if 0<=nx<H and 0<=ny<W and grid[nx,ny]==1 and (nx,ny) not in visited:
                visited.add((nx,ny))
                q.append((nx,ny))

    seen = set(visited)
    islands = 0
    for i in range(H):
        for j in range(W):
            if grid[i,j]==1 and (i,j) not in seen:
                islands += 1
                dq = deque([(i,j)])
                seen.add((i,j))
                while dq:
                    x,y = dq.popleft()
                    for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nx,ny = x+dx, y+dy
                        if 0<=nx<H and 0<=ny<W and grid[nx,ny]==1 and (nx,ny) not in seen:
                            seen.add((nx,ny))
                            dq.append((nx,ny))
    return islands


# ===== Random generation (with path tortuosity constraint) =====
def generate_connected_network(size=SIZE, fill_lo=0.25, fill_hi=0.55,
                               max_islands=3, max_tries=5000, max_path_factor=1.5):
    for _ in range(max_tries):
        p_on = random.uniform(fill_lo, fill_hi)
        g = (np.random.rand(size, size) < p_on).astype(int)
        g[START] = 1
        g[END]   = 1

        # theoretical minimal Manhattan distance
        L_min = abs(END[0]-START[0]) + abs(END[1]-START[1])
        L_actual = shortest_path_length(g, START, END)

        if L_actual is None:
            continue
        if L_actual > max_path_factor * L_min:
            continue
        if count_islands_excluding_path(g, START, END) > max_islands:
            continue

        return g, START, END

    raise RuntimeError("Too many attempts, failed to generate a valid structure!")


# ===== Export DXF =====
def export_to_dxf(grid, start, end, cell_size=1, filename="layout.dxf"):
    doc = ezdxf.new(dxfversion='R2000')
    doc.header['$INSUNITS'] = 4  # unit: mm
    msp = doc.modelspace()
    size = grid.shape[0]

    if 'cond' not in doc.layers:
        doc.layers.new(name='cond', dxfattribs={'color': 1})

    for i in range(size):
        for j in range(size):
            if grid[i, j] == 1:
                x = j * cell_size
                y = (size - i - 1) * cell_size
                msp.add_lwpolyline([
                    (x, y),
                    (x + cell_size, y),
                    (x + cell_size, y + cell_size),
                    (x, y + cell_size),
                    (x, y)
                ], close=True, dxfattribs={'layer': 'cond'})

    # input port block
    in_x = -cell_size
    in_y = (size - start[0] - 1) * cell_size
    msp.add_lwpolyline([
        (in_x, in_y),
        (in_x + cell_size, in_y),
        (in_x + cell_size, in_y + cell_size),
        (in_x, in_y + cell_size),
        (in_x, in_y)
    ], close=True, dxfattribs={'layer': 'cond'})

    # output port block
    out_x = size * cell_size
    out_y = (size - end[0] - 1) * cell_size
    msp.add_lwpolyline([
        (out_x, out_y),
        (out_x + cell_size, out_y),
        (out_x + cell_size, out_y + cell_size),
        (out_x, out_y + cell_size),
        (out_x, out_y)
    ], close=True, dxfattribs={'layer': 'cond'})

    doc.saveas(filename)



# ============================================================
# =============== from inverse_design: proj_a writer ==========
# ============================================================

def save_proj_a(grid, start, end, filepath):
    """
    Same as original proj_a generator; only reorganized into utils.
    """
    with open(filepath, 'w') as f:
        f.write("UNITS MM,1000;\n")
        f.write("EDIT proj;\n")

        size = len(grid)
        cell_size = 1.0

        def add_block(x, y):
            x1 = x * cell_size
            y1 = y * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            f.write(
                f"ADD P1 :W0.000000 {x1:.6f},{y1:.6f} "
                f"{x2:.6f},{y1:.6f} {x2:.6f},{y2:.6f} {x1:.6f},{y2:.6f};\n"
            )
            f.write("  BEGIN_ASSOC\n")
            f.write("    ADD N1 :F1.000000 :R0 :T12345 'net=P1' 0.00, 0.00;\n")
            f.write("  END_ASSOC\n")

        # flip y-direction to match ADS coordinate usage
        for i in range(size):
            for j in range(size):
                if grid[i, j] == 1:
                    add_block(j, size - i - 1)

        # input/output ports (same as original)
        in_x  = -1
        in_y  = size - start[0] - 1
        out_x = size
        out_y = size - end[0] - 1
        add_block(in_x, in_y)
        add_block(out_x, out_y)

        f.write("SAVE;\n")

    print("✅ proj_a written:", filepath)
