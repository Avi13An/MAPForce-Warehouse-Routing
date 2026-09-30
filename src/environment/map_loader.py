"""
MovingAI map parser for warehouse benchmark layouts.
Automatically registers shelf pickup locations and boundary packing stations.
"""
from typing import List, Tuple
from .grid import WarehouseGrid

def load_movingai_map(filepath: str) -> WarehouseGrid:
    """
    Parses a MovingAI .map file and instantiates a configured WarehouseGrid.
    """
    with open(filepath, 'r') as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]

    height, width = 0, 0
    map_start_idx = 0

    for idx, line in enumerate(raw_lines):
        if line.startswith("height"):
            height = int(line.split()[1])
        elif line.startswith("width"):
            width = int(line.split()[1])
        elif line.startswith("map"):
            map_start_idx = idx + 1
            break

    if height == 0 or width == 0:
        raise ValueError(f"Invalid MovingAI header detected in {filepath}")

    grid: List[List[int]] = []
    traversable_symbols = {'.', 'G', 'S'}

    for row_str in raw_lines[map_start_idx : map_start_idx + height]:
        row = [0 if char in traversable_symbols else 1 for char in row_str]
        grid.append(row)

    env = WarehouseGrid(grid, height, width)

    # Automatically identify shelves (cells adjacent to obstacles '@')
    for y in range(height):
        for x in range(width):
            if env.static_grid[y][x] == 1:
                # Find adjacent traversable spots where robots can stop and pick
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if env.in_bounds(nx, ny) and env.static_grid[ny][nx] == 0:
                        env.register_shelf(nx, ny)

    # Designate bottom border as default packing stations
    for x in range(2, width - 2, 4):
        if env.is_traversable(x, height - 2):
            env.register_packing_station(x, height - 2)

    return env