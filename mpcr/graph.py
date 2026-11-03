#Dylan Kenneth Eliot

"""Dependency graph definitions."""
from dataclasses import dataclass, field
from typing import Callable, Any

@dataclass
class Task:
    task_id: str
    fn: Callable
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    deps: tuple = ()

class TaskGraph:
    def __init__(self):
        self._tasks = {}

    def add(self, task_id, fn, *args, deps=(), **kwargs):
        if task_id in self._tasks:
            raise ValueError(f"duplicate task id: {task_id}")
        if not callable(fn):
            raise TypeError("fn must be callable")
        self._tasks[task_id] = Task(task_id, fn, args, kwargs, tuple(deps))
        return self

    @property
    def tasks(self):
        return dict(self._tasks)

    def validate(self):
        for task in self._tasks.values():
            missing = set(task.deps) - self._tasks.keys()
            if missing:
                raise ValueError(f"{task.task_id}: unknown dependencies {sorted(missing)}")
        visiting, visited = set(), set()
        def visit(node):
            if node in visiting:
                raise ValueError(f"dependency cycle detected at {node}")
            if node in visited:
                return
            visiting.add(node)
            for dep in self._tasks[node].deps:
                visit(dep)
            visiting.remove(node)
            visited.add(node)
        for node in self._tasks:
            visit(node)
        return True

    def ready_waves(self, completed=()):
        self.validate()
        done = set(completed)
        remaining = set(self._tasks) - done
        waves = []
        while remaining:
            ready = sorted(
                tid for tid in remaining
                if set(self._tasks[tid].deps).issubset(done)
            )
            if not ready:
                raise RuntimeError("graph cannot make progress")
            waves.append(tuple(ready))
            done.update(ready)
            remaining.difference_update(ready)
        return tuple(waves)

    def __len__(self):
        return len(self._tasks)
