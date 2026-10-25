"""Gates for parallel shards (merge) and CPU-time planner limits."""

import types

import src.planners.cbs as cbs_module
import src.planners.ecbs as ecbs_module
from src.experiments import runner
from src.planners.common import SimpleGrid

GRID = SimpleGrid([".....", ".#.#."])
STARTS = {0: (0, 0), 1: (4, 0)}
GOALS = {0: (4, 0), 1: (0, 0)}


def _rows():
    base = {"layout": "x", "planner": "CBS", "robots": 10, "skew": "real",
            "delay": 0.05, "seed": 0, "throughput": 12.5, "success_rate": 1.0,
            "congestion_exposure": 0.4, "demand_source": "synthetic"}
    return [base, dict(base, seed=1, throughput=7)]


def test_csv_round_trip_restores_numbers(tmp_path):
    path = str(tmp_path / "a.csv")
    runner.write_csv(_rows(), path)
    back = runner.read_csv(path)
    assert back[0]["robots"] == 10 and back[0]["delay"] == 0.05
    assert back[1]["throughput"] == 7 and back[0]["demand_source"] == "synthetic"


def test_merge_combines_shards_and_rejects_overlap(tmp_path):
    a, b, out = (str(tmp_path / n) for n in ("a.csv", "b.csv", "out.csv"))
    rows = _rows()
    runner.write_csv(rows[:1], a)
    runner.write_csv(rows[1:], b)
    assert runner.main(["--merge", a, b, "--out", out]) == 0
    assert len(runner.read_csv(out)) == 2
    assert runner.main(["--merge", a, a, "--out", out]) == 1


def _clock(cpu_step):
    """A fake `time` module: wall clock frozen, CPU clock jumps `cpu_step`."""
    state = {"cpu": 0.0}

    def process_time():
        state["cpu"] += cpu_step
        return state["cpu"]

    return types.SimpleNamespace(process_time=process_time, perf_counter=lambda: 0.0)


def test_time_limits_count_cpu_time_not_wall_clock(monkeypatch):
    for module, call in ((cbs_module, lambda: cbs_module.cbs(GRID, STARTS, GOALS, time_limit=1.0)),
                         (ecbs_module, lambda: ecbs_module.ecbs(GRID, STARTS, GOALS, w=1.5, time_limit=1.0))):
        monkeypatch.setattr(module, "time", _clock(cpu_step=10.0))
        res = call()
        assert not res.solved and res.failure == "time limit"
        monkeypatch.setattr(module, "time", _clock(cpu_step=0.0))
        assert call().solved
