from .bee_colony import artificial_bee_colony
from .de import differential_evolution
from .ga_pso import ga_pso
from .problem import Problem

ALGORITHMS = {
    "DE": differential_evolution,
    "ABC": artificial_bee_colony,
    "GA-PSO": ga_pso,
}

__all__ = ["ALGORITHMS", "Problem", "artificial_bee_colony", "differential_evolution", "ga_pso"]
