from dataclasses import dataclass
import numpy as np
from algorithms import greedy_d
from load_balancing.simulation import (
    Simulation,
)


@dataclass
class WeightedResult:
    mixing_steps: int
    maximum_load: float


def sorting_steps(repeats: int, n: int):
    """Run the greedy_d simulation for a reverse sorted weight distribution that is progressively less sorted."""
    np.random.seed(42)
    # Initialise simulation
    sim = Simulation(greedy_d)

    balls = np.random.uniform(0, 1, size=n)
    balls = -np.sort(-balls)  # reverse sorted weights

    print("Number of balls and bins: ", n)
    result = []
    for sorting_step in range(100):
        result.append(
            WeightedResult(
                sorting_step,
                sim(n, n, 2, ball_weights=balls, repeats=repeats).maximum_load,
            )
        )
        # Perform random swap
        index1, index2 = np.random.randint(0, n, 2)
        balls[index1], balls[index2] = balls[index2], balls[index1]
    return result
