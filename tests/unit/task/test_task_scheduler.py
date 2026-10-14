"""TaskScheduler execution modes: dependency order, exactly-once execution,
re-execution and error handling."""

import random

import pytest

from ttnte.task import (
    DeviceTarget,
    ExecutionMode,
    TaskGraph,
    TaskScheduler,
    TaskStatus,
)

MODES = [ExecutionMode.POLL, ExecutionMode.PUSH]


def _random_dag(num_tasks, max_deps, seed):
    """TaskGraph whose payloads log their index; returns (graph, deps, log)."""
    rng = random.Random(seed)
    graph, log, deps = TaskGraph(), [], []
    tasks = []
    for i in range(num_tasks):
        task = graph.create_task(DeviceTarget.CPU_ASYNC, f"t{i}")
        task.set_payload(lambda i=i: (log.append(i), TaskStatus.COMPLETED)[1])
        d = rng.sample(range(i), min(i, rng.randint(0, max_deps)))
        for j in d:
            graph.add_dependency(task, tasks[j])
        deps.append(d)
        tasks.append(task)
    return graph, deps, log


def _check(deps, log):
    assert sorted(log) == list(range(len(deps))), "every task exactly once"
    position = {t: k for k, t in enumerate(log)}
    for i, d in enumerate(deps):
        for j in d:
            assert position[j] < position[i], f"task {i} ran before dependency {j}"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("num_threads", [1, 4, 16])
def test_random_dag_order(mode, num_threads):
    graph, deps, log = _random_dag(400, 4, seed=num_threads)
    TaskScheduler(num_threads).execute(graph, mode)
    _check(deps, log)
    assert all(t.status == TaskStatus.COMPLETED for t in graph.get_tasks())


@pytest.mark.parametrize("mode", MODES)
def test_reexecution_and_new_dependency(mode):
    graph, deps, log = _random_dag(200, 3, seed=7)
    scheduler = TaskScheduler(8)
    for _ in range(3):
        log.clear()
        graph.reset()
        scheduler.execute(graph, mode)
        _check(deps, log)

    # A dependency added between executions through the Task itself (not the
    # graph) must be honoured; edges to earlier tasks keep the graph acyclic
    tasks = graph.get_tasks()
    last = len(tasks) - 1
    k = next(j for j in range(last) if j not in deps[last])
    tasks[last].add_dependency(tasks[k])
    deps[last].append(k)
    log.clear()
    graph.reset()
    scheduler.execute(graph, mode)
    _check(deps, log)


@pytest.mark.parametrize("mode", MODES)
def test_empty_graph(mode):
    TaskScheduler(2).execute(TaskGraph(), mode)


def test_push_rejects_non_cpu_async():
    graph = TaskGraph()
    task = graph.create_task(DeviceTarget.CPU_SYNC, "sync")
    task.set_payload(lambda: TaskStatus.COMPLETED)
    with pytest.raises(RuntimeError, match="CPU_ASYNC"):
        TaskScheduler(2).execute(graph, ExecutionMode.PUSH)


def test_push_rejects_incomplete_payload():
    graph = TaskGraph()
    a = graph.create_task(DeviceTarget.CPU_ASYNC, "a")
    b = graph.create_task(DeviceTarget.CPU_ASYNC, "b")
    a.set_payload(lambda: TaskStatus.POLLING)
    b.set_payload(lambda: TaskStatus.COMPLETED)
    graph.add_dependency(b, a)
    with pytest.raises(RuntimeError, match="COMPLETED"):
        TaskScheduler(2).execute(graph, ExecutionMode.PUSH)


@pytest.mark.parametrize("mode", MODES)
def test_dependency_added_after_execution(mode):
    """A stale cached plan would run the fast task before the slow one."""
    import time

    graph, log = TaskGraph(), []
    slow = graph.create_task(DeviceTarget.CPU_ASYNC, "slow")
    fast = graph.create_task(DeviceTarget.CPU_ASYNC, "fast")
    slow.set_payload(
        lambda: (time.sleep(0.05), log.append("slow"), TaskStatus.COMPLETED)[2]
    )
    fast.set_payload(lambda: (log.append("fast"), TaskStatus.COMPLETED)[1])
    scheduler = TaskScheduler(4)
    scheduler.execute(graph, mode)
    assert log == ["fast", "slow"]

    fast.add_dependency(slow)
    log.clear()
    graph.reset()
    scheduler.execute(graph, mode)
    assert log == ["slow", "fast"]


def _polling_payload(name, log, threads, polls):
    """Network-like payload: POLLING `polls` times, then COMPLETED."""
    import threading

    state = {"n": 0}

    def payload():
        threads.add(threading.get_ident())
        if state["n"] < polls:
            state["n"] += 1
            return TaskStatus.POLLING
        state["n"] = 0  # reusable after graph.reset()
        log.append(name)
        return TaskStatus.COMPLETED

    return payload


@pytest.mark.parametrize("num_threads", [1, 4])
def test_push_network_tasks(num_threads):
    """Mixed CPU / NETWORK graph: recv -> cpu -> send chains and a network root; network
    payloads run only on the calling thread."""
    import threading

    graph, log, net_threads = TaskGraph(), [], set()
    deps = {}
    tasks = {}

    def cpu(name):
        t = graph.create_task(DeviceTarget.CPU_ASYNC, name)
        t.set_payload(lambda: (log.append(name), TaskStatus.COMPLETED)[1])
        tasks[name] = t
        return t

    def net(name, polls):
        t = graph.create_task(DeviceTarget.NETWORK_ASYNC, name)
        t.set_payload(_polling_payload(name, log, net_threads, polls))
        tasks[name] = t
        return t

    for c in range(6):
        recv = net(f"recv{c}", polls=c % 3)
        work = cpu(f"work{c}")
        send = net(f"send{c}", polls=2)
        after = cpu(f"after{c}")
        graph.add_dependency(work, recv)
        graph.add_dependency(send, work)
        graph.add_dependency(after, send)
        deps[f"work{c}"] = [f"recv{c}"]
        deps[f"send{c}"] = [f"work{c}"]
        deps[f"after{c}"] = [f"send{c}"]
        if c:
            graph.add_dependency(work, tasks[f"work{c - 1}"])
            deps[f"work{c}"].append(f"work{c - 1}")

    scheduler = TaskScheduler(num_threads)
    for _ in range(3):
        log.clear()
        net_threads.clear()
        graph.reset()
        scheduler.execute(graph, ExecutionMode.PUSH)
        assert sorted(log) == sorted(tasks), "every task exactly once"
        pos = {n: k for k, n in enumerate(log)}
        for name, ds in deps.items():
            for d in ds:
                assert pos[d] < pos[name], f"{name} ran before {d}"
        assert net_threads == {threading.get_ident()}
