"""Lifelong execution: the rolling-horizon loop that produces throughput.

Owner per the proposal: Jwalant (lifelong execution). Written here because
throughput is the headline metric and nothing else produces it.

The scheme follows RHCR (Li et al., 2021), as Section 4.4 says: orders keep
arriving, so planning is a *sequence* of windowed MAPF instances rather than
one problem. Every `replan_every` timesteps all robots replan from where they
actually are; only the first `replan_every` steps of each plan are executed,
then the window slides.

Two things make this harder than looping a MAPF solver:

* **Execution drift.** Each robot may be delayed with probability `delay_prob`
  per timestep. Once one robot stalls, the others' plans are no longer
  conflict-free with respect to where it actually is -- so collisions are
  *possible during execution even though every plan was collision-free when
  issued*. That is the whole point of the delay condition in Section 4.5, and
  `collisions` counts exactly those.
* **Planner failure.** If the planner cannot solve the window inside its time
  limit, robots hold position and the failure is recorded. Section 4.4 requires
  this to be a measured result, not a crash.

Task assignment is greedy nearest-idle-robot and is deliberately identical
across planners, so that the only thing varying between experimental
conditions is the path planner.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set

from ..demand.orders import OrderStream, Task
from ..planners.common import (Cell, GridView, PlanResult, Solution, cell_at,
                               solution_cost)
from .disruptions import BlockedGridView, Reachability, active_blocked_cells

# A planner is anything that maps (grid, starts, goals) to a PlanResult.
Planner = Callable[[GridView, Dict[int, Cell], Dict[int, Cell]], PlanResult]


@dataclass
class Robot:
    rid: int
    pos: Cell
    task: Optional[Task] = None
    carrying: bool = False        # True once the shelf has been reached
    dock: Optional[Cell] = None   # the specific drop cell this robot is using
    park: Optional[Cell] = None   # waiting, but told to step off another's goal

    @property
    def goal(self) -> Cell:
        """Where this robot is currently trying to get to.

        An idle robot's goal is where it stands -- it must still appear in the
        MAPF instance, because it occupies a cell and others must route around
        it.

        A carrying robot heads for its assigned *dock* cell rather than the
        station itself. Several robots delivering to one station would
        otherwise share a goal cell, and MAPF requires an agent to be able to
        park on its goal, so a shared goal makes the whole window unsolvable.

        A carrying robot with no dock (all taken, or cut off by a blocked
        aisle) waits where it is. A waiting robot with a `park` cell has been
        told to step off a cell another robot needs.
        """
        if self.waiting:
            return self.park if self.park is not None else self.pos
        if self.carrying:
            return self.dock
        return self.task.shelf

    @property
    def waiting(self) -> bool:
        """No destination right now: idle, or carrying with no usable
        drop-off cell. A waiting robot stays put (or steps to `park`)."""
        return self.task is None or (self.carrying and self.dock is None)


def dock_cells(grid: GridView, stations: Sequence[Cell]) -> List[Cell]:
    """Drop-off cells: each station plus its passable neighbours.

    A small queue of distinct cells per station, so that concurrent deliveries
    get distinct MAPF goals.
    """
    docks: List[Cell] = []
    for s in stations:
        if grid.passable(s):
            docks.append(s)
        for n in grid.neighbours(s):
            if n not in docks:
                docks.append(n)
    return docks


@dataclass
class SimulationResult:
    """Everything Section 4.5 asks for, measured rather than asserted."""

    completed_orders: int
    timesteps: int
    throughput: float             # orders per 1000 timesteps (headline metric)
    planning_calls: int
    planning_failures: int
    success_rate: float
    total_plan_time: float        # CPU seconds spent planning
    mean_plan_time: float         # CPU seconds per planning call
    collisions: int               # raw count of vertex + edge collisions
    collision_rate: float         # collisions per 1000 robot-timesteps (Section 4.5)
    total_expansions: int
    mean_flowtime: float          # mean order completion latency
    congestion_exposure: float
    # Path cost (added 10 Nov; Section 4.5 of the proposal). Over solved
    # planning calls only:
    #   mean_path_cost   sum of path costs of the returned plan, per call
    #   path_cost_ratio  total path cost / total of each robot's own shortest
    #                    distance to its goal. That sum is a lower bound on
    #                    the optimal (CBS) cost of the same call, so the ratio
    #                    is an upper bound on "cost relative to CBS", available
    #                    for every planner and every call, including the many
    #                    calls on which CBS itself times out.
    mean_path_cost: float = 0.0
    path_cost_ratio: float = 0.0
    # Blocked-aisle disruptions (all zero when the run has none):
    disruption_replans: int = 0   # extra replans because the map changed mid-window
    disruption_stalls: int = 0    # moves cancelled because the next cell was closed
    tasks_deferred: int = 0       # tasks handed back because their shelf was cut off

    def as_row(self) -> Dict[str, float]:
        return {
            "completed_orders": self.completed_orders,
            "timesteps": self.timesteps,
            "throughput": round(self.throughput, 3),
            "planning_calls": self.planning_calls,
            "planning_failures": self.planning_failures,
            "success_rate": round(self.success_rate, 4),
            "mean_plan_time": round(self.mean_plan_time, 4),
            "collisions": self.collisions,
            "collision_rate": round(self.collision_rate, 3),
            "total_expansions": self.total_expansions,
            "mean_flowtime": round(self.mean_flowtime, 2),
            "congestion_exposure": round(self.congestion_exposure, 4),
            "mean_path_cost": round(self.mean_path_cost, 2),
            "path_cost_ratio": round(self.path_cost_ratio, 4),
            "disruption_replans": self.disruption_replans,
            "disruption_stalls": self.disruption_stalls,
            "tasks_deferred": self.tasks_deferred,
        }


def _manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class _ShortestDistances:
    """Each robot's own shortest distance to its goal on the static grid.

    The sum over robots is a lower bound on the optimal cost of a planning
    call: CBS can do no better than every robot taking its own shortest path.
    Distances are measured on the undisrupted grid, which can only make them
    shorter, so the bound still holds while an aisle is blocked. One backward
    BFS per goal, kept in a small LRU cache because goals (shelves and docks)
    persist across many consecutive calls.
    """

    def __init__(self, grid: GridView, size: int = 512) -> None:
        from collections import OrderedDict
        from ..planners.low_level import TrueDistanceHeuristic
        self._grid = grid
        self._make = TrueDistanceHeuristic
        self._cache: "OrderedDict[Cell, object]" = OrderedDict()
        self._size = size

    def __call__(self, start: Cell, goal: Cell) -> float:
        h = self._cache.get(goal)
        if h is None:
            h = self._make(self._grid, goal)
            self._cache[goal] = h
            if len(self._cache) > self._size:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(goal)
        return h(start)


def assign_tasks(
    robots: Sequence[Robot],
    pending: List[Task],
    timestep: int,
    can_reach: Optional[Callable[[Cell, Cell], bool]] = None,
) -> None:
    """Greedy nearest-idle assignment, identical across every planner.

    Deliberately simple: the project is comparing path planners, so task
    assignment is held fixed. A smarter assigner would confound the comparison.

    `can_reach(robot_cell, shelf)`, when given, skips shelves a blocked aisle
    has cut off from that robot; those orders wait for the aisle to reopen.
    """
    # Two robots may not be given the same destination: in MAPF an agent must
    # be able to park on its goal, so a shared goal cell makes the instance
    # unsolvable and every planning call in the window would fail.
    claimed = {r.goal for r in robots}
    idle = [r for r in robots if r.task is None]
    for robot in idle:
        if not pending:
            return
        candidates = [i for i, t in enumerate(pending)
                      if t.shelf not in claimed
                      and (can_reach is None or can_reach(robot.pos, t.shelf))]
        if not candidates:
            return
        best_i = min(candidates,
                     key=lambda i: _manhattan(robot.pos, pending[i].shelf))
        task = pending.pop(best_i)
        task.assigned_at = timestep
        robot.task = task
        robot.carrying = False
        robot.park = None
        # Recount rather than `claimed.discard(robot.pos)`: the cell this robot
        # stood on may also be another robot's destination, and discarding it
        # would let a third robot be sent to the same shelf.
        claimed = {r.goal for r in robots}


def _claim_dock(
    robot: Robot,
    robots: Sequence[Robot],
    docks: Sequence[Cell],
    closed: Set[Cell],
    reach: Optional["Reachability"],
) -> None:
    """Give a carrying robot the nearest drop-off cell nobody else is using.

    If every one is taken (or closed, or cut off), the robot gets no dock and
    waits until one frees up. The earlier fallback -- send it to the station
    cell anyway -- could give two robots the same goal, which no MAPF planner
    can satisfy, so every planning call in that window failed.
    """
    claimed = {o.goal for o in robots if o is not robot}
    free = [d for d in docks if d not in claimed and d not in closed
            and (reach is None or reach.connected(robot.pos, d))]
    robot.dock = min(free, key=lambda d: _manhattan(robot.pos, d)) if free \
        else None


def invalidate_cut_off_goals(
    robots: Sequence[Robot],
    pending: List[Task],
    docks: Sequence[Cell],
    reach: Reachability,
) -> int:
    """Re-target robots whose destination a blockage has made unreachable.

    Handing the planner an unreachable goal makes the *whole* call fail and
    freezes every robot, so this runs before each planning call while any
    aisle is blocked:

    * heading for a shelf it can no longer reach -> the task goes back to the
      front of the queue and the robot is free for another order;
    * carrying, but its drop-off cell is cut off -> switch to the nearest
      reachable free drop-off cell, or release the claim and wait if there is
      none (it retries every step).

    Returns how many tasks were handed back.
    """
    deferred = 0
    for r in robots:
        if r.task is None:
            continue
        if not r.carrying:
            if not reach.connected(r.pos, r.task.shelf):
                r.task.assigned_at = -1
                pending.insert(0, r.task)
                r.task = None
                deferred += 1
            continue
        if r.dock is None or reach.connected(r.pos, r.dock):
            continue                # still looking for a dock, or fine as is
        claimed = {o.goal for o in robots if o is not r}
        options = [d for d in docks
                   if d not in claimed and reach.connected(r.pos, d)]
        # No reachable dock: give the claim up entirely. Keeping it while the
        # robot waits would hide the cell from everyone else's "is it taken?"
        # check, and two robots would want it once the aisle reopens.
        r.dock = (min(options, key=lambda d: _manhattan(r.pos, d))
                  if options else None)
    return deferred


def clear_claimed_cells(robots: Sequence[Robot], grid: GridView) -> None:
    """Send idle robots off cells that another robot is heading for.

    An idle robot's goal is "wherever it is now". Planners do move idle robots
    aside to let others pass, so an idle robot can end a step on another
    robot's shelf; a disruption can also make a robot drop its task mid-route.
    Either way two robots then share a goal cell, which no MAPF planner can
    satisfy, and the call fails for everyone. So each such robot gets a `park` cell:
    the nearest reachable cell that no robot is heading for. The same applies
    to a carrying robot waiting for a free drop-off cell.
    """
    for r in robots:
        if r.waiting and r.park is not None and r.pos == r.park:
            r.park = None                       # arrived; it waits here now
    for r in robots:
        if not r.waiting:
            r.park = None
            continue
        others = {o.goal for o in robots if o is not r}
        if r.goal not in others:
            continue
        taken = others | {o.pos for o in robots if o is not r}
        seen = {r.pos}
        frontier = [r.pos]
        r.park = None
        while frontier and r.park is None:
            nxt = []
            for c in frontier:
                for n in grid.neighbours(c):
                    if n in seen:
                        continue
                    seen.add(n)
                    if n not in taken:
                        r.park = n
                        break
                    nxt.append(n)
                if r.park is not None:
                    break
            frontier = nxt


def count_execution_collisions(
    before: Dict[int, Cell], after: Dict[int, Cell]
) -> int:
    """Vertex and edge collisions in one executed timestep.

    Zero while plans are followed exactly; non-zero once delays make robots
    drift off plan. This is the collision-rate metric.
    """
    collisions = 0
    rids = sorted(after)
    for i, a in enumerate(rids):
        for b in rids[i + 1:]:
            if after[a] == after[b]:
                collisions += 1
            elif after[a] == before[b] and after[b] == before[a]:
                collisions += 1
    return collisions


def run_lifelong(
    grid: GridView,
    planner: Planner,
    order_stream: OrderStream,
    start_positions: Sequence[Cell],
    horizon: int = 200,
    replan_every: int = 5,
    delay_prob: float = 0.0,
    seed: int = 0,
    congestion=None,
    verbose: bool = False,
    disruptions: Sequence = (),
    on_step: Optional[Callable[[int, Dict[int, Cell], frozenset], None]] = None,
    replan_at: Sequence[int] = (),
    on_frame: Optional[Callable[[int, Sequence[Robot], Dict], None]] = None,
) -> SimulationResult:
    """Run the warehouse for `horizon` timesteps and measure what came out.

    `congestion`, if given, is updated from each issued plan so that a
    congestion-aware planner sees current traffic, and is also used to report
    how much congested ground the fleet actually covered.

    `disruptions` is a schedule of blocked-aisle events (Avi's
    `TimedBlockedAisleEvent`, or anything with `trigger_timestep`, `duration`
    and `blocked_cells`). While an event is active:

    * the planner is given a `BlockedGridView` with the closed cells removed
      (cells a robot is standing on stay open until it leaves);
    * a robot never moves into a closed cell -- if its plan says to, it holds
      and the stall is counted;
    * goals the closure cuts off are re-targeted before planning;
    * the map changing (an aisle closing or reopening) triggers an immediate
      replan instead of waiting for the next window boundary.

    With no disruptions every one of those steps is skipped, so results are
    identical to a run without the feature.

    `on_step(t, positions, closed_cells)`, if given, is called after every
    executed timestep. The visualiser and the tests use it to watch the run.

    `on_frame(t, robots, info)`, if given, is called after every timestep
    with the live robot objects and running totals (`done`, `collisions`,
    `failures`, `replanned`, `closed`). The visualiser records from it, so a
    replay is this exact loop, not a copy of it that can drift.

    `replan_at` lists extra timesteps at which to replan. It exists for fair
    comparisons: replanning more often raises throughput on its own, so a
    no-disruption control run should replan at the same moments the disrupted
    run is forced to (see `stress_test.py`).
    """
    rng = random.Random(seed)
    robots = [Robot(rid=i, pos=p) for i, p in enumerate(start_positions)]
    docks = dock_cells(grid, order_stream.demand.stations)
    pending: List[Task] = []
    completed: List[Task] = []

    plans: Solution = {}
    plan_step = 0                 # how far into the current plan we are
    planning_calls = 0
    planning_failures = 0
    total_plan_time = 0.0
    total_expansions = 0
    collisions = 0
    exposure_samples: List[float] = []
    shortest = _ShortestDistances(grid)
    cost_total = 0
    lb_total = 0.0
    costed_calls = 0

    disruptions = list(disruptions)
    forced_replans = set(replan_at)
    blocked: Set[Cell] = set()    # closed cells the planner must avoid
    closed_now: Set[Cell] = set() # every cell closed this step, occupied or not
    disruption_replans = 0
    disruption_stalls = 0
    tasks_deferred = 0

    for t in range(horizon):
        pending.extend(order_stream.release(t))

        # --- disruptions: what is closed right now? -------------------------
        view: GridView = grid
        reach: Optional[Reachability] = None
        map_changed = False
        if disruptions:
            closed_now = active_blocked_cells(disruptions, t)
            occupied = {r.pos for r in robots}
            now_blocked = closed_now - occupied
            map_changed = now_blocked != blocked
            blocked = now_blocked
            if blocked:
                view = BlockedGridView(grid, blocked)
                reach = Reachability(view)
                tasks_deferred += invalidate_cut_off_goals(
                    robots, pending, docks, reach)

        assign_tasks(robots, pending, t,
                     can_reach=reach.connected if reach is not None else None)
        clear_claimed_cells(robots, view)

        # --- replan at the window boundary, or at once if the map changed ---
        boundary = t % replan_every == 0 or t in forced_replans
        replanned = boundary or map_changed
        if replanned:
            if map_changed and not boundary:
                disruption_replans += 1
            starts = {r.rid: r.pos for r in robots}
            goals = {r.rid: r.goal for r in robots}
            t0 = time.process_time()   # CPU seconds, like the planners' limit
            result = planner(view, starts, goals)
            total_plan_time += time.process_time() - t0
            planning_calls += 1
            total_expansions += result.expansions

            if result.solved and result.solution:
                plans = result.solution
                plan_step = 0
                # Path cost of this call against its shortest-path lower bound.
                # Measurement only: nothing here feeds back into the run.
                lb = sum(shortest(starts[a], goals[a]) for a in plans)
                if lb != float("inf"):
                    cost_total += solution_cost(plans)
                    lb_total += lb
                    costed_calls += 1
                if congestion is not None:
                    congestion.update_planned(plans, window=replan_every)
                    exposure_samples.append(congestion.solution_score(plans))
            else:
                # Section 4.4: robots hold position, the failure is recorded.
                planning_failures += 1
                plans = {r.rid: [r.pos] for r in robots}
                plan_step = 0
                if verbose:
                    print(f"  t={t}: planning failed ({result.failure})")

        # --- execute one timestep -------------------------------------------
        # Each robot tries to take its planned step; a delayed one stays put.
        # Once any robot stalls, the others' plans no longer match where it
        # actually is, so intended moves can collide. We count those collisions
        # -- that is the metric Section 4.5 asks for -- and then resolve them,
        # because two robots cannot physically share a cell and the next
        # planning window would otherwise get an infeasible start state (two
        # agents on one cell has no MAPF solution at all).
        before = {r.rid: r.pos for r in robots}
        intended: Dict[int, Cell] = {}
        for r in robots:
            path = plans.get(r.rid)
            on_plan = bool(path) and cell_at(path, plan_step) == r.pos
            if not on_plan or (delay_prob > 0 and rng.random() < delay_prob):
                # Off-plan robots hold until the next replan rather than
                # teleporting back onto their path -- a robot that was delayed
                # or pushed aside is physically where it is, not where the plan
                # says, and the window slides soon enough to re-route it.
                intended[r.rid] = r.pos
            else:
                intended[r.rid] = cell_at(path, plan_step + 1)

        # A closed aisle is closed to new entries, whatever the plan says. The
        # plan can disagree only for a cell that was open at planning time
        # because a robot stood on it; the robot stops in front instead.
        if closed_now:
            for r in robots:
                if intended[r.rid] != r.pos and intended[r.rid] in closed_now:
                    intended[r.rid] = r.pos
                    disruption_stalls += 1

        collisions += count_execution_collisions(before, intended)

        # Resolve to a physically valid joint move, by fixpoint.
        #
        # A robot may advance only if its target cell is genuinely vacated. It
        # is NOT enough to revert a blocked robot to its own cell: another
        # robot may already have moved into that cell, which would put two
        # robots on one square and hand the next planning window an instance
        # with no solution at all. So instead of reverting individually, we
        # iteratively cancel moves until no cancellation triggers another:
        #
        #   * two robots targeting one cell -> all but the lowest id cancel;
        #   * a robot targeting a cell whose occupant is not moving -> cancel;
        #   * two robots swapping -> both cancel.
        #
        # A cancelled robot stays on `before[r]`, which is collision-free by
        # induction, so the fixpoint is always reachable and always valid.
        moving = {r.rid for r in robots if intended[r.rid] != before[r.rid]}
        occupant = {before[r.rid]: r.rid for r in robots}

        changed = True
        while changed:
            changed = False

            targets: Dict[Cell, List[int]] = {}
            for rid in moving:
                targets.setdefault(intended[rid], []).append(rid)
            for cell, claimants in targets.items():
                if len(claimants) > 1:
                    for rid in sorted(claimants)[1:]:
                        moving.discard(rid)
                        changed = True

            for rid in list(moving):
                target = intended[rid]
                holder = occupant.get(target)
                if holder is not None and holder != rid and holder not in moving:
                    moving.discard(rid)       # cell will not be vacated
                    changed = True

            for rid in list(moving):
                other = occupant.get(intended[rid])
                if (other is not None and other in moving
                        and intended[other] == before[rid]):
                    moving.discard(rid)       # head-on swap
                    moving.discard(other)
                    changed = True

        for r in robots:
            r.pos = intended[r.rid] if r.rid in moving else before[r.rid]
        plan_step += 1

        positions = [r.pos for r in robots]
        assert len(set(positions)) == len(positions), (
            "two robots on one cell after resolution -- the next planning "
            "window would be unsolvable"
        )

        # --- task progress ---------------------------------------------------
        for r in robots:
            if r.task is None:
                continue
            if not r.carrying and r.pos == r.task.shelf:
                r.carrying = True
                r.dock = None
                _claim_dock(r, robots, docks, closed_now, reach)
            elif r.carrying and r.dock is None:
                # Every drop-off cell was taken: keep trying, one step at a time.
                _claim_dock(r, robots, docks, closed_now, reach)
            # Compare with the dock itself, not `r.goal`: a waiting robot's goal
            # is its own cell and it must not deliver where it waits.
            elif r.carrying and r.pos == r.dock:
                r.task.completed_at = t
                completed.append(r.task)
                r.task = None
                r.carrying = False
                r.dock = None

        if on_step is not None:
            on_step(t, {r.rid: r.pos for r in robots}, frozenset(closed_now))
        if on_frame is not None:
            on_frame(t, robots, {"done": len(completed), "collisions": collisions,
                                 "failures": planning_failures,
                                 "replanned": replanned, "closed": sorted(closed_now)})

    n = len(completed)
    flowtimes = [c.completed_at - c.released_at for c in completed]
    # Normalised so that runs with different fleet sizes and horizons are
    # comparable: 10 collisions among 5 robots is far worse than among 40.
    robot_timesteps = len(robots) * horizon
    return SimulationResult(
        completed_orders=n,
        timesteps=horizon,
        throughput=1000.0 * n / horizon if horizon else 0.0,
        planning_calls=planning_calls,
        planning_failures=planning_failures,
        success_rate=((planning_calls - planning_failures) / planning_calls)
        if planning_calls else 0.0,
        total_plan_time=total_plan_time,
        mean_plan_time=total_plan_time / planning_calls if planning_calls else 0.0,
        collisions=collisions,
        collision_rate=1000.0 * collisions / robot_timesteps
        if robot_timesteps else 0.0,
        total_expansions=total_expansions,
        mean_flowtime=sum(flowtimes) / len(flowtimes) if flowtimes else 0.0,
        congestion_exposure=(sum(exposure_samples) / len(exposure_samples))
        if exposure_samples else 0.0,
        mean_path_cost=cost_total / costed_calls if costed_calls else 0.0,
        path_cost_ratio=cost_total / lb_total if lb_total else 0.0,
        disruption_replans=disruption_replans,
        disruption_stalls=disruption_stalls,
        tasks_deferred=tasks_deferred,
    )
