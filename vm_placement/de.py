"""Differential Evolution (DE/rand/1/bin).

For every target vector a mutant is built from the scaled difference of two
random population members added to a third: v = x_r1 + F * (x_r2 - x_r3).
Binomial crossover mixes the mutant with the target, and the trial replaces
the target only if it is at least as fit (greedy selection).
"""

import numpy as np

from .base import Result, Timer, clip, random_positions


def differential_evolution(problem, pop_size=20, max_evals=2000, F=0.5, CR=0.9, seed=None):
    rng = np.random.default_rng(seed)
    problem.reset_counter()
    D = problem.n_vms

    with Timer() as t:
        pop = random_positions(rng, pop_size, problem)
        fit = np.array([problem(x) for x in pop])
        it = 0
        while problem.evals < max_evals:
            it += 1
            for i in range(pop_size):
                if problem.evals >= max_evals:
                    break
                r1, r2, r3 = rng.choice([j for j in range(pop_size) if j != i], 3, replace=False)
                mutant = clip(pop[r1] + F * (pop[r2] - pop[r3]), problem)

                cross = rng.random(D) < CR
                cross[rng.integers(D)] = True     # at least one gene from the mutant
                trial = np.where(cross, mutant, pop[i])

                f_trial = problem(trial)
                if f_trial <= fit[i]:
                    pop[i], fit[i] = trial, f_trial

    b = int(fit.argmin())
    return Result("DE", pop[b], problem.decode(pop[b]), float(fit[b]),
                  list(problem.trace), it, problem.evals, t.elapsed)
