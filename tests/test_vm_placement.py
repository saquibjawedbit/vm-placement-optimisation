import numpy as np
import pytest

from vm_placement import ALGORITHMS, Problem
from vm_placement.problem import PENALTY


@pytest.fixture
def problem():
    return Problem()


def test_decode_maps_box_to_pm_indices(problem):
    assert list(problem.decode([0.0, 1.9, 2.5, 4.999, 7.0])) == [0, 1, 2, 4, 4]
    assert Problem.to_display([2, 2, 4, 2, 0]) == [3, 3, 5, 3, 1]


def test_fitness_counts_servers_with_power_tie_breaker(problem):
    # VM4 on PM4, everything else on PM2: feasible, 2 servers, 850 W
    ev = problem.evaluate_assignment([1, 1, 1, 3, 1])
    assert ev.feasible and ev.servers_used == 2 and ev.power == 850
    assert 2 < ev.fitness < 3


def test_overload_is_penalised_above_any_feasible_solution(problem):
    ev = problem.evaluate_assignment([3, 3, 3, 3, 3])   # everything on the small PM4
    assert not ev.feasible and ev.violations == 3
    assert ev.fitness > PENALTY > problem.evaluate_assignment([0, 1, 2, 3, 4]).fitness


def test_brute_force_optimum(problem):
    _, ev = problem.brute_force()
    assert ev.servers_used == 2 and ev.power == 850


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_algorithms_respect_budget_and_find_feasible_two_server_solution(problem, name):
    r = ALGORITHMS[name](problem, 20, max_evals=1000, seed=3)
    ev = problem.evaluate_assignment(r.best_assignment)
    assert r.evals == 1000 and len(r.trace) == 1000
    assert ev.feasible and ev.servers_used == 2
    assert np.isclose(r.best_fitness, ev.fitness)
    assert all(a >= b for a, b in zip(r.trace, r.trace[1:]))   # best-so-far is monotone
