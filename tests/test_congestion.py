"""Gates for the congestion map -- the demand signal CA-ECBS steers by."""

from src.planners.common import SimpleGrid
from src.planners.congestion import (
    CongestionMap,
    clustered_demand,
    uniform_demand,
    zipf_demand,
)

OPEN = SimpleGrid([
    ".....",
    ".....",
    ".....",
    ".....",
    ".....",
])

TWO_ROUTES = SimpleGrid([
    ".....",
    ".###.",
    ".....",
])


def test_demand_distributions_are_normalised():
    shelves = [(x, 0) for x in range(5)]
    for demand in (uniform_demand(shelves),
                   zipf_demand(shelves),
                   clustered_demand(shelves, centre=(0, 0))):
        assert abs(sum(demand.values()) - 1.0) < 1e-9


def test_zipf_is_actually_skewed():
    """The 'real' skew level must concentrate demand, or the control and the
    treatment conditions are the same experiment run twice."""
    shelves = [(x, 0) for x in range(10)]
    d = zipf_demand(shelves, exponent=1.0)
    ranked = sorted(d.values(), reverse=True)
    assert ranked[0] > 3 * ranked[-1]


def test_static_map_is_normalised_to_unit_peak():
    """A single lambda has to mean the same thing on every map size."""
    cong = CongestionMap(OPEN, shelves=[(4, 4)], stations=[(0, 0)])
    assert cong.static
    assert abs(max(cong.static.values()) - 1.0) < 1e-9
    assert min(cong.static.values()) >= 0.0


def test_only_cells_on_a_shortest_route_carry_traffic():
    """The distance-field identity d(s,c) + d(c,p) == d(s,p) is the whole test.

    Going (0,1) -> (4,1) round the block, the top and bottom corridors are both
    6 steps, so both are shortest routes and both must carry load. The wall
    cells carry none, and so must the blocked interior.
    """
    cong = CongestionMap(TWO_ROUTES, shelves=[(0, 1)], stations=[(4, 1)])
    assert cong.score((2, 0)) > 0, "top corridor is a shortest route"
    assert cong.score((2, 2)) > 0, "bottom corridor is equally short"
    # Endpoints lie on every route, so they are at least as hot as the middle.
    assert cong.score((0, 1)) >= cong.score((2, 0))


def test_cells_off_every_shortest_route_carry_nothing():
    """A detour that is strictly longer must score zero, not 'a little'."""
    cong = CongestionMap(TWO_ROUTES, shelves=[(0, 0)], stations=[(4, 0)])
    # The direct top row is the unique shortest route here, so the bottom
    # corridor is off-route entirely.
    assert cong.score((2, 0)) > 0
    assert cong.score((2, 2)) == 0.0


def test_unreachable_station_contributes_nothing():
    walled = SimpleGrid(["..#..", "..#..", "..#.."])
    cong = CongestionMap(walled, shelves=[(0, 0)], stations=[(4, 0)])
    assert cong.score((0, 0)) == 0.0


def test_planned_occupancy_tracks_the_fleet():
    cong = CongestionMap(OPEN, shelves=[(4, 4)], stations=[(0, 0)])
    cong.clear_planned()
    assert cong.planned == {}

    solution = {
        0: [(0, 0), (1, 0), (2, 0)],
        1: [(0, 1), (1, 0), (2, 1)],   # both pass through (1, 0)
    }
    cong.update_planned(solution)
    assert cong.planned[(1, 0)] == 1.0            # the peak cell
    assert cong.planned[(0, 0)] < cong.planned[(1, 0)]


def test_planned_occupancy_respects_the_window():
    cong = CongestionMap(OPEN, shelves=[(4, 4)], stations=[(0, 0)])
    solution = {0: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]}
    cong.update_planned(solution, window=2)
    assert (1, 0) in cong.planned
    assert (4, 0) not in cong.planned


def test_path_score_is_a_mean_not_a_sum():
    """Otherwise lambda quietly becomes a second length penalty.

    A path twice as long through equally busy cells must not score twice as
    congested -- its length is already paid for in the cost term.
    """
    cong = CongestionMap(OPEN, shelves=[(4, 4)], stations=[(0, 0)])
    short = [(0, 0), (1, 0)]
    long = [(0, 0), (1, 0), (0, 0), (1, 0)]
    assert abs(cong.path_score(short) - cong.path_score(long)) < 0.35


def test_hottest_returns_descending_scores():
    cong = CongestionMap(OPEN, shelves=[(4, 4), (0, 4)], stations=[(0, 0)])
    top = cong.hottest(5)
    scores = [s for _, s in top]
    assert scores == sorted(scores, reverse=True)


def test_numpy_and_python_builds_are_bit_identical():
    """The fast numpy build (28 Oct) must give exactly the map the original
    loop gives, rounding included, or CA-ECBS's focal order could change."""
    from src.planners import congestion as C
    from src.demand.orders import build_demand_model
    from src.tuning.sweep import narrow_aisle_grid, shelves_and_stations
    if C._np is None:
        import pytest
        pytest.skip("numpy not installed; only the pure-Python path exists")
    g = narrow_aisle_grid(rows=13, cols=17)
    shelves, stations = shelves_and_stations(g)
    for skew in ("uniform", "real", "extreme"):
        w = build_demand_model(shelves, stations, skew=skew, seed=2).shelf_weights
        fast = C.CongestionMap(g, shelves, stations, w).static
        saved, C._np = C._np, None
        try:
            slow = C.CongestionMap(g, shelves, stations, w).static
        finally:
            C._np = saved
        assert fast == slow
