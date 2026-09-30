"""
Dynamic Environment Disruption & Event Manager.
Handles scheduled mid-run corridor blockages and obstacle events.
"""
from typing import Dict, List, Tuple
from .grid import WarehouseGrid

class TimedBlockedAisleEvent:
    def __init__(self, trigger_timestep: int, duration: int, blocked_cells: List[Tuple[int, int]]):
        self.trigger_timestep = trigger_timestep
        self.duration = duration
        self.blocked_cells = blocked_cells
        self.is_active = False

class DisruptionManager:
    def __init__(self, env: WarehouseGrid):
        self.env = env
        self.scheduled_events: List[TimedBlockedAisleEvent] = []

    def schedule_event(self, trigger_timestep: int, duration: int, blocked_cells: List[Tuple[int, int]]):
        """Schedules a future blocked-aisle incident."""
        self.scheduled_events.append(TimedBlockedAisleEvent(trigger_timestep, duration, blocked_cells))

    def update(self, current_timestep: int):
        """Called at every simulation step by the lifelong execution loop."""
        for event in self.scheduled_events:
            # Activate disruption
            if current_timestep == event.trigger_timestep:
                self.env.block_aisle(event.blocked_cells)
                event.is_active = True
            
            # Remove disruption after duration expires
            elif current_timestep == event.trigger_timestep + event.duration and event.is_active:
                for cell in event.blocked_cells:
                    if cell in self.env.dynamic_obstacles:
                        self.env.dynamic_obstacles.remove(cell)
                event.is_active = False