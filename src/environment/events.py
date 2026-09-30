"""
Dynamic Environment Disruption & Event Manager.
Handles scheduled mid-run corridor blockages and obstacle events.
"""
from typing import List, Set, Tuple
from .grid import WarehouseGrid

class TimedBlockedAisleEvent:
    def __init__(self, trigger_timestep: int, duration: int, blocked_cells: List[Tuple[int, int]]):
        self.trigger_timestep = trigger_timestep
        self.duration = duration
        self.blocked_cells = blocked_cells
        self.is_active = False

    def active_at(self, timestep: int) -> bool:
        """Blocked from `trigger_timestep` up to, not including, trigger + duration."""
        return self.trigger_timestep <= timestep < self.trigger_timestep + self.duration

class DisruptionManager:
    def __init__(self, env: WarehouseGrid):
        self.env = env
        self.scheduled_events: List[TimedBlockedAisleEvent] = []
        # Cells this manager has put into env.dynamic_obstacles, so it only
        # ever removes its own blocks, never ones added some other way.
        self._applied: Set[Tuple[int, int]] = set()

    def schedule_event(self, trigger_timestep: int, duration: int, blocked_cells: List[Tuple[int, int]]):
        """Schedules a future blocked-aisle incident."""
        self.scheduled_events.append(TimedBlockedAisleEvent(trigger_timestep, duration, blocked_cells))

    def update(self, current_timestep: int):
        """Called at every simulation step by the lifelong execution loop.

        The closed cells are recomputed from the whole schedule each call
        (fixed 30 Sep). The earlier version toggled cells only on the exact
        trigger and end steps, so a skipped call missed an event entirely,
        and when the first of two overlapping events ended it reopened a cell
        the second event still blocked.
        """
        active: Set[Tuple[int, int]] = set()
        for event in self.scheduled_events:
            event.is_active = event.active_at(current_timestep)
            if event.is_active:
                active.update(tuple(c) for c in event.blocked_cells)

        for cell in self._applied - active:
            self.env.dynamic_obstacles.discard(cell)
        self.env.block_aisle(list(active))
        self._applied = active
