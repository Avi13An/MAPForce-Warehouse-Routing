"""Regression: ECBS must not identify search nodes by memory address.

Until 28 Oct, ecbs() tracked expanded nodes with id(node). CPython reuses the
address of a freed object, so a newly created child could share an address
with an already-expanded node, be taken as expanded, and be skipped. ECBS then
reported "constraint tree exhausted" on solvable instances, and whether it
happened depended on whatever had run earlier in the same process.

The test forces the worst case -- every object reports the same id -- which
the old code could not survive and the fixed code never consults.
"""

import src.planners.ecbs as ecbs_module
from src.planners.common import SimpleGrid, validate_solution
from src.planners.ecbs import ecbs

# Two robots must swap ends of a corridor; the pockets below let one wait.
GRID = SimpleGrid([".....", ".#.#."])
STARTS = {0: (0, 0), 1: (4, 0)}
GOALS = {0: (4, 0), 1: (0, 0)}


def test_needs_several_expansions():
    res = ecbs(GRID, STARTS, GOALS, w=1.5)
    assert res.solved and res.expansions > 1
    assert validate_solution(res.solution, GRID) == []


def test_solution_does_not_depend_on_memory_addresses(monkeypatch):
    honest = ecbs(GRID, STARTS, GOALS, w=1.5)
    monkeypatch.setattr(ecbs_module, "id", lambda obj: 0, raising=False)
    colliding = ecbs(GRID, STARTS, GOALS, w=1.5)
    assert colliding.solved, colliding.failure
    assert colliding.cost == honest.cost
    assert colliding.expansions == honest.expansions
