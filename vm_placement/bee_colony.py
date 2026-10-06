"""Artificial Bee Colony (Karaboga, 2005).

Food sources are candidate placements. Each cycle has three phases:
  * employed bees - one per source, try a neighbour v_j = x_j + phi * (x_j - x_kj)
    on one random dimension j and keep it if it is better;
  * onlooker bees - pick sources with probability proportional to their quality
    and search around them the same way;
  * scout bees   - a source that has not improved for ``limit`` trials is
    abandoned and replaced by a random one.
"""

import numpy as np

from .base import Result, Timer, clip, random_positions


def artificial_bee_colony(problem, colony_size=20, max_evals=2000, limit=None, seed=None):
    rng = np.random.default_rng(seed)
    problem.reset_counter()
    D = problem.n_vms
    SN = colony_size // 2                 # number of food sources = employed bees = onlookers
    limit = limit if limit is not None else SN * D

    with Timer() as t:
        foods = random_positions(rng, SN, problem)
        fit = np.array([problem(x) for x in foods])
        trials = np.zeros(SN, dtype=int)
        best_x, best_f = foods[fit.argmin()].copy(), fit.min()

        def try_neighbour(i):
            nonlocal best_x, best_f
            k = rng.choice([s for s in range(SN) if s != i])
            j = rng.integers(D)
            v = foods[i].copy()
            v[j] = v[j] + rng.uniform(-1, 1) * (foods[i, j] - foods[k, j])
            # when both sources agree on dimension j the step is zero, so also
            # allow a small random jump to keep the discrete search moving
            if np.floor(v[j]) == np.floor(foods[i, j]) and rng.random() < 0.5:
                v[j] = rng.uniform(0, problem.n_pms)
            v = clip(v, problem)
            fv = problem(v)
            if fv <= fit[i]:
                foods[i], fit[i], trials[i] = v, fv, 0
                if fv < best_f:
                    best_x, best_f = v.copy(), fv
            else:
                trials[i] += 1

        it = 0
        while problem.evals < max_evals:
            it += 1
            # employed bee phase
            for i in range(SN):
                if problem.evals >= max_evals:
                    break
                try_neighbour(i)

            # onlooker bee phase: roulette wheel on quality 1 / (1 + fitness)
            quality = 1.0 / (1.0 + fit)
            prob = quality / quality.sum()
            for _ in range(SN):
                if problem.evals >= max_evals:
                    break
                try_neighbour(int(rng.choice(SN, p=prob)))

            # scout bee phase
            worn = int(trials.argmax())
            if trials[worn] > limit and problem.evals < max_evals:
                foods[worn] = random_positions(rng, 1, problem)[0]
                fit[worn] = problem(foods[worn])
                trials[worn] = 0
                if fit[worn] < best_f:
                    best_x, best_f = foods[worn].copy(), fit[worn]

    return Result("ABC", best_x, problem.decode(best_x), float(best_f),
                  list(problem.trace), it, problem.evals, t.elapsed)
