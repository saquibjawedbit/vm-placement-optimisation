"""Shared result type and helpers for the optimisers."""

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Result:
    name: str
    best_x: np.ndarray
    best_assignment: np.ndarray
    best_fitness: float
    trace: list          # best fitness after every fitness evaluation
    iterations: int
    evals: int
    runtime: float


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start


def random_positions(rng, n, problem):
    return rng.uniform(0, problem.n_pms, size=(n, problem.n_vms))


def clip(x, problem):
    # stay strictly below n_pms so floor() never leaves the valid range
    return np.clip(x, 0.0, problem.n_pms - 1e-9)
