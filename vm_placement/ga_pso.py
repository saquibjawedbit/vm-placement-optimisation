"""Hybrid GA-PSO.

Each iteration the swarm first moves like PSO:
    v = w v + c1 r1 (pbest - x) + c2 r2 (gbest - x),   x = x + v
with inertia w decreasing linearly from w_max to w_min. Then every moved
particle goes through GA operators before it is evaluated:
  * crossover - with probability pc, uniform crossover with the personal best
    of a tournament-selected particle;
  * mutation  - each gene is reset to a random PM with probability pm.
Personal and global bests are updated after evaluation as in plain PSO.
"""

import numpy as np

from .base import Result, Timer, clip, random_positions


def ga_pso(problem, swarm_size=20, max_evals=2000, w_max=0.9, w_min=0.4, c1=1.5, c2=1.5,
           pc=0.7, pm=0.1, tournament=3, seed=None):
    rng = np.random.default_rng(seed)
    problem.reset_counter()
    D = problem.n_vms
    v_max = problem.n_pms / 2
    max_iter = max(1, max_evals // swarm_size)

    with Timer() as t:
        x = random_positions(rng, swarm_size, problem)
        v = rng.uniform(-v_max, v_max, size=x.shape)
        fit = np.array([problem(p) for p in x])
        pbest, pbest_f = x.copy(), fit.copy()
        g = int(pbest_f.argmin())
        gbest, gbest_f = pbest[g].copy(), pbest_f[g]

        it = 0
        while problem.evals < max_evals:
            it += 1
            w = w_max - (w_max - w_min) * min(it / max_iter, 1.0)

            # PSO move
            r1, r2 = rng.random(x.shape), rng.random(x.shape)
            v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
            v = np.clip(v, -v_max, v_max)
            x = clip(x + v, problem)

            for i in range(swarm_size):
                if problem.evals >= max_evals:
                    break
                # GA crossover with a tournament-selected personal best
                if rng.random() < pc:
                    cand = rng.choice(swarm_size, tournament, replace=False)
                    mate = pbest[cand[pbest_f[cand].argmin()]]
                    mask = rng.random(D) < 0.5
                    x[i] = np.where(mask, mate, x[i])
                # GA mutation: random reset
                mut = rng.random(D) < pm
                x[i, mut] = rng.uniform(0, problem.n_pms, size=mut.sum())

                f = problem(x[i])
                if f < pbest_f[i]:
                    pbest[i], pbest_f[i] = x[i].copy(), f
                    if f < gbest_f:
                        gbest, gbest_f = x[i].copy(), f

    return Result("GA-PSO", gbest, problem.decode(gbest), float(gbest_f),
                  list(problem.trace), it, problem.evals, t.elapsed)
