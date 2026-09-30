"""
Demonstration script showing how all team modules interact with Module 1.
"""
import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.map_loader import load_movingai_map
from src.environment.layouts import build_custom_narrow_aisle_warehouse
from src.environment.events import DisruptionManager

def main():
    print("--- 1. Testing MovingAI Benchmark 1 ---")
    env1 = load_movingai_map("data/maps/warehouse-10-20-10-2-1.map")
    print(f"Loaded Map 1: {env1.width}x{env1.height}")
    print(f"Total Pick/Shelf locations indexed: {len(env1.shelves)}")
    print(f"Total Packing Stations indexed: {len(env1.packing_stations)}")

    print("\n--- 2. Testing Custom Narrow-Aisle Layout ---")
    env_custom = build_custom_narrow_aisle_warehouse(rows=35, cols=50)
    print(f"Custom Map dimensions: {env_custom.width}x{env_custom.height}")

    print("\n--- 3. Testing Timed Blocked-Aisle Disruption ---")
    disruptor = DisruptionManager(env_custom)
    
    # Schedule corridor blockage at timestep t=5 for 10 timesteps at (5, 10).
    # (5, 10) is an aisle cell between two shelf columns. The demo used to
    # block (10, 15), which is a shelf wall, so it printed False before,
    # during and after the event and never showed anything.
    disruptor.schedule_event(trigger_timestep=5, duration=10, blocked_cells=[(5, 10)])
    
    # Timestep 4: traversable
    disruptor.update(current_timestep=4)
    print(f"Timestep 4 traversable at (5, 10): {env_custom.is_traversable(5, 10)}")
    
    # Timestep 5: blocked
    disruptor.update(current_timestep=5)
    print(f"Timestep 5 traversable at (5, 10): {env_custom.is_traversable(5, 10)} (Blocked!)")
    
    # Timestep 15: unblocked
    disruptor.update(current_timestep=15)
    print(f"Timestep 15 traversable at (5, 10): {env_custom.is_traversable(5, 10)} (Cleared!)")

    print("\nModule 1 fully functional and ready for downstream integration!")

if __name__ == "__main__":
    main()