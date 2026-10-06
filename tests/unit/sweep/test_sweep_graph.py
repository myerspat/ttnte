import torch

from ttnte.sweep import EdgeState, SweepGraph, SweepInterface


def _itf(first, second, fwd, bwd):
    return SweepInterface(
        first,
        second,
        torch.tensor(fwd, dtype=torch.float64),
        torch.tensor(bwd, dtype=torch.float64),
        1e-12,
    )


def _is_acyclic(graph, angle_set):
    n = graph.num_nodes
    up = {i: set(graph.upstream(angle_set, i)) for i in range(n)}
    done = set()
    while len(done) < n:
        ready = [i for i in range(n) if i not in done and up[i] <= done]
        if not ready:
            return False
        done.update(ready)
    return True


def test_chain_has_no_lag():
    # 0 -> 1 -> 2 for direction 0, reversed for direction 1
    g = SweepGraph(
        3, [_itf(0, 1, [1.0, 0.0], [0.0, 1.0]), _itf(1, 2, [1.0, 0.0], [0.0, 1.0])], 2
    )
    assert len(g.angle_sets) == 2
    assert g.num_weight_lagged == 0 and g.num_dfs_lagged == 0
    s0 = int(g.direction_set[0])
    assert (
        g.upstream(s0, 0) == []
        and g.upstream(s0, 1) == [0]
        and g.upstream(s0, 2) == [1]
    )


def test_two_way_face_keeps_larger_weight():
    """Owens' rule: a face with flow both ways keeps the larger upwind integral as the
    dependency and lags the other."""
    g = SweepGraph(2, [_itf(0, 1, [0.3], [0.1])], 1)
    s = g.angle_sets[0]
    assert s.forward[0] == EdgeState.DEPENDENCY and s.backward[0] == EdgeState.LAGGED
    assert g.num_weight_lagged == 1 and _is_acyclic(g, 0)


def test_three_cycle_is_broken_by_dfs():
    # 0 -> 1 -> 2 -> 0 with one-way faces: only the DFS can break it
    g = SweepGraph(
        3,
        [_itf(0, 1, [1.0], [0.0]), _itf(1, 2, [1.0], [0.0]), _itf(0, 2, [0.0], [1.0])],
        1,
    )
    assert g.num_dfs_lagged == 1
    assert _is_acyclic(g, 0)
    states = list(g.angle_sets[0].forward) + list(g.angle_sets[0].backward)
    assert states.count(EdgeState.LAGGED) == 1


def test_identical_directions_share_an_angle_set():
    g = SweepGraph(2, [_itf(0, 1, [1.0, 2.0, 0.0], [0.0, 0.0, 1.0])], 3)
    assert len(g.angle_sets) == 2
    assert g.direction_set.tolist()[0] == g.direction_set.tolist()[1]


def test_topological_order_respects_dependencies():
    g = SweepGraph(
        4,
        [_itf(2, 0, [1.0], [0.0]), _itf(0, 1, [1.0], [0.0]), _itf(3, 1, [1.0], [0.0])],
        1,
    )
    order = g.topological_order(0)
    pos = {u: k for k, u in enumerate(order)}
    assert sorted(order) == [0, 1, 2, 3]
    assert pos[2] < pos[0] < pos[1] and pos[3] < pos[1]


def test_schedule_unit_and_subdomain():
    g = SweepGraph(3, [_itf(0, 1, [1.0], [0.0]), _itf(1, 2, [1.0], [0.0])], 1)
    per_unit = g.schedule(0, [0, 1, 2])
    assert [c.units for c in per_unit] == [[0], [1], [2]]
    assert [c.upstream for c in per_unit] == [[], [0], [1]]
    whole = g.schedule(0, [7, 7, 7])
    assert len(whole) == 1 and whole[0].units == [0, 1, 2] and whole[0].upstream == []


def test_schedule_merges_chunk_cycles():
    """Units 0 -> 1 -> 2 -> 3 with chunks {0, 2} and {1, 3}: the chunk graph is cyclic
    although the unit graph is not, so both chunks must merge into one task that still
    sweeps the units in upwind order."""
    g = SweepGraph(
        4,
        [_itf(0, 1, [1.0], [0.0]), _itf(1, 2, [1.0], [0.0]), _itf(2, 3, [1.0], [0.0])],
        1,
    )
    tasks = g.schedule(0, [0, 1, 0, 1])
    assert len(tasks) == 1
    assert tasks[0].units == [0, 1, 2, 3]
    # A grouping that keeps the chunk graph acyclic stays split
    split = g.schedule(0, [0, 0, 1, 1])
    assert [t.units for t in split] == [[0, 1], [2, 3]]
    assert split[1].upstream == [0]
