"""
MovingAI map parser for warehouse benchmark layouts.
"""
from typing import List, Tuple

def load_movingai_map(filepath: str) -> Tuple[List[List[int]], int, int]:
    """
    Parses a MovingAI format .map file.
    Returns:
        grid: 2D list of integers (0 = traversable, 1 = static obstacle)
        height: Total rows in the grid
        width: Total columns in the grid
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

    return grid, height, width
