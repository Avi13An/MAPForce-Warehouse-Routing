"""
Warehouse Grid State & Topology Manager.
"""
from typing import List, Tuple, Set

class WarehouseGrid:
    def __init__(self, grid: List[List[int]], height: int, width: int):
        self.height = height
        self.width = width
        self.static_grid = [row[:] for row in grid]
        self.dynamic_obstacles: Set[Tuple[int, int]] = set()
        self.shelves: Set[Tuple[int, int]] = set()
        self.packing_stations: Set[Tuple[int, int]] = set()

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_traversable(self, x: int, y: int) -> bool:
        if not self.in_bounds(x, y):
            return False
        if self.static_grid[y][x] == 1:
            return False
        if (x, y) in self.dynamic_obstacles:
            return False
        return True

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        candidate_actions = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
        valid_neighbors = []
        for dx, dy in candidate_actions:
            nx, ny = x + dx, y + dy
            if self.is_traversable(nx, ny):
                valid_neighbors.append((nx, ny))
        return valid_neighbors

    def register_shelf(self, x: int, y: int):
        if self.in_bounds(x, y):
            self.shelves.add((x, y))

    def register_packing_station(self, x: int, y: int):
        if self.in_bounds(x, y):
            self.packing_stations.add((x, y))

    def block_aisle(self, cells: List[Tuple[int, int]]):
        for cell in cells:
            self.dynamic_obstacles.add(cell)

    def clear_disruptions(self):
        self.dynamic_obstacles.clear()
