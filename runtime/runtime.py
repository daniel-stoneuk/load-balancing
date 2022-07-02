from contextlib import contextmanager
import json
import time
from typing import Callable, Generator, List, Tuple
import load_balancing.core as lb
from load_balancing.simulation import Simulation


@contextmanager
def timer() -> Generator[Callable[[], float], None, None]:
    """Performance timer context manager
    :yield: runtime: Method that returns the runtime of the function
    :rtype: Callable[[], float]
    """
    start = time.perf_counter()
    runtime = 0
    yield lambda: runtime
    runtime = time.perf_counter() - start


def test_runtime(
    min_balls: int,
    max_balls: int,
    step: int,
    method: Callable,
    method_args: Tuple,
    repeats=10,
):
    x: List[int] = []
    y: List[float] = []

    # Force Numba to jit compile
    method(min_balls, min_balls, *method_args)

    for balls in range(min_balls, max_balls, step):
        average = 0
        print(balls)

        for i in range(repeats):
            with timer() as time:
                method(balls, balls // 10, *method_args)
            average += time()

        x.append(balls)
        y.append(average / repeats)

    return {"x": x, "y": y}


from balancer import python_greedy_d, numba_greedy_d

if __name__ == "__main__":

    def k_greedy_spec(ball: lb.Ball, bins: lb.Bins, k: int):
        chosen_bins = bins.choose(k)
        return chosen_bins.min()

    sim_k_greedy = Simulation(k_greedy_spec)

    step = 10000
    min_balls = 10
    max_balls = 200000
    method_args = (2,)

    python_greedy_d_runtimes = test_runtime(
        min_balls, max_balls, step, python_greedy_d, method_args
    )
    numba_greedy_d_runtimes = test_runtime(
        min_balls, max_balls, step, numba_greedy_d, method_args
    )
    c_greedy_d_runtimes = test_runtime(
        min_balls, max_balls, step, numba_greedy_d, method_args
    )
    sim_greedy_d_runtimes = test_runtime(
        min_balls, max_balls, step, sim_k_greedy, method_args
    )

    with open("python_runtime.json", "w") as f:
        json.dump(python_greedy_d_runtimes, f)

    with open("numba_runtime.json", "w") as f:
        json.dump(numba_greedy_d_runtimes, f)

    with open("c_runtime.json", "w") as f:
        json.dump(c_greedy_d_runtimes, f)

    with open("sim_runtime.json", "w") as f:
        json.dump(sim_greedy_d_runtimes, f)
