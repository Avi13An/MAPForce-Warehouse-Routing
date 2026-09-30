"""
Procedural layout generator for narrow-aisle warehouse stress benchmarks.
"""
from typing import List, Tuple
from .grid import WarehouseGrid

def build_custom_narrow_aisle_warehouse(
    rows: int = 35, 
    cols: int = 50, 
    aisle_width: int = 1,
    shelf_block_height: int = 5,
    shelf_block_width: int = 2
) -> WarehouseGrid:
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    env = WarehouseGrid(grid, rows, cols)

    for y in range(3, rows - 3):
        if (y - 3) % (shelf_block_height + aisle_width) < shelf_block_height:
            for x in range(3, cols - 3):
                if (x - 3) % (shelf_block_width + aisle_width) < shelf_block_width:
                    env.static_grid[y][x] = 1
                    env.register_shelf(x, y)

    for x in range(4, cols - 4, 3):
        env.register_packing_station(x, rows - 2)

    return env
