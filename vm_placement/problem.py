"""Encoding, decoding and fitness for the VM placement problem.

A solution is a list with one entry per VM saying which PM it runs on, using
1-based PM numbers: [3, 3, 5, 3, 1] puts VM1, VM2 and VM4 on PM3, VM3 on PM5
and VM5 on PM1.

The optimisers search a continuous box [0, n_pms) per VM; ``decode`` floors each
coordinate to a PM index, so every algorithm shares one fitness function.
"""

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from . import data

# Any capacity violation costs at least this much, which is more than the worst
# feasible fitness (all 5 servers on plus the power tie-breaker), so a feasible
# placement always beats an infeasible one.
PENALTY = 100.0


@dataclass
class Evaluation:
    fitness: float
    servers_used: int
    power: float
    violations: int
    overload: float
    feasible: bool


@dataclass
class Problem:
    pm_names: list = field(default_factory=lambda: list(data.PMS))
    vm_names: list = field(default_factory=lambda: list(data.VMS))

    def __post_init__(self):
        pms = np.array([data.PMS[n] for n in self.pm_names], dtype=float)
        self.capacity = pms[:, :3]            # (n_pms, 3)
        self.pm_power = pms[:, 3]             # (n_pms,)
        self.demand = np.array([data.VMS[n] for n in self.vm_names], dtype=float)
        self.n_pms = len(self.pm_names)
        self.n_vms = len(self.vm_names)
        self.total_power = self.pm_power.sum()
        self.reset_counter()

    # --- evaluation budget / convergence tracking ---------------------------
    def reset_counter(self):
        self.evals = 0
        self.best_so_far = np.inf
        self.trace = []                       # best fitness after each evaluation

    # --- encoding -----------------------------------------------------------
    def decode(self, x):
        """Continuous vector -> 0-based PM index per VM."""
        idx = np.floor(np.asarray(x, dtype=float)).astype(int)
        return np.clip(idx, 0, self.n_pms - 1)

    @staticmethod
    def to_display(assignment):
        """0-based indices -> the 1-based list used in the question, e.g. [3, 3, 5, 3, 1]."""
        return [int(a) + 1 for a in assignment]

    def loads(self, assignment):
        load = np.zeros_like(self.capacity)
        np.add.at(load, np.asarray(assignment), self.demand)
        return load

    # --- fitness ------------------------------------------------------------
    def evaluate_assignment(self, assignment):
        assignment = np.asarray(assignment)
        load = self.loads(assignment)
        active = np.zeros(self.n_pms, dtype=bool)
        active[np.unique(assignment)] = True

        excess = np.maximum(0.0, load - self.capacity) / self.capacity
        violations = int((excess > 0).sum())
        overload = float(excess.sum())

        servers = int(active.sum())
        power = float(self.pm_power[active].sum())
        # servers used + power tie-breaker (always < 1) + capacity penalty
        fitness = servers + power / (self.total_power + 1.0)
        fitness += PENALTY * (violations + overload)
        return Evaluation(float(fitness), servers, power, violations, overload, violations == 0)

    def __call__(self, x):
        """Fitness of a continuous position. Counts towards the evaluation budget."""
        f = self.evaluate_assignment(self.decode(x)).fitness
        self.evals += 1
        self.best_so_far = min(self.best_so_far, f)
        self.trace.append(self.best_so_far)
        return f

    # --- reporting helpers --------------------------------------------------
    def utilisation(self, assignment):
        """Mean CPU/RAM/BW utilisation (%) over the servers that are switched on."""
        load = self.loads(assignment)
        active = np.unique(assignment)
        util = load[active] / self.capacity[active] * 100.0
        per_resource = util.mean(axis=0)
        return {"CPU": per_resource[0], "RAM": per_resource[1], "BW": per_resource[2],
                "mean": per_resource.mean()}

    def brute_force(self):
        """Exhaustively check all n_pms ** n_vms placements (3125 here) for the true optimum."""
        best = None
        for a in product(range(self.n_pms), repeat=self.n_vms):
            ev = self.evaluate_assignment(a)
            if best is None or ev.fitness < best[1].fitness:
                best = (np.array(a), ev)
        return best
