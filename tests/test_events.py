"""Gates for DisruptionManager (fixed 28 Oct)."""

from src.environment.events import DisruptionManager
from src.environment.grid import WarehouseGrid


def _env():
    return WarehouseGrid([[0, 0, 0, 0]], height=1, width=4)


def test_a_skipped_trigger_step_does_not_miss_the_event():
    env = _env(); dm = DisruptionManager(env)
    dm.schedule_event(trigger_timestep=5, duration=4, blocked_cells=[(1, 0)])
    dm.update(4)
    dm.update(7)                       # step 5 and 6 never reported
    assert not env.is_traversable(1, 0)
    dm.update(9)
    assert env.is_traversable(1, 0)


def test_overlapping_events_keep_a_shared_cell_closed():
    env = _env(); dm = DisruptionManager(env)
    dm.schedule_event(0, 5, [(1, 0), (2, 0)])
    dm.schedule_event(3, 10, [(2, 0)])
    for t in range(6):
        dm.update(t)
    assert env.is_traversable(1, 0)            # first event over
    assert not env.is_traversable(2, 0)        # second still holds it
    dm.update(13)
    assert env.is_traversable(2, 0)


def test_manual_blocks_are_left_alone():
    env = _env(); dm = DisruptionManager(env)
    env.block_aisle([(3, 0)])                  # blocked by someone else
    dm.schedule_event(0, 2, [(1, 0)])
    dm.update(0); dm.update(2)
    assert not env.is_traversable(3, 0)


def test_is_active_flags_track_the_window():
    env = _env(); dm = DisruptionManager(env)
    dm.schedule_event(2, 3, [(0, 0)])
    dm.update(1); assert not dm.scheduled_events[0].is_active
    dm.update(4); assert dm.scheduled_events[0].is_active
    dm.update(5); assert not dm.scheduled_events[0].is_active
