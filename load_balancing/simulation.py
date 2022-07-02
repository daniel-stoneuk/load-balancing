from collections import defaultdict
from functools import partial
from operator import attrgetter
import random
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any, Dict, List, Protocol, Tuple

import numpy as np
from numba import njit
from numpy.typing import NDArray
from tqdm.auto import tqdm

import load_balancing.core as core
from load_balancing.utils import available_cores


class BalancerCountError(Exception):
    pass


class MissingStepParameter(Exception):
    pass


class NumberOfRepeatsError(Exception):
    pass


class NoJobError(Exception):
    pass


@njit
def init_bins(n: int):
    return core.Bins(
        n,
        np.arange(0, n),
        np.zeros(n, dtype=np.float64),
        core.Counters(),
    )


@njit
def parallel_load_balancer(
    balancers: int, ball_weights: NDArray[np.float64], n: int, algorithm, args
):
    m = len(ball_weights)
    bins = init_bins(n)
    chosen_bins = np.zeros(balancers, dtype=np.int64)
    chosen_weights = np.zeros(balancers, dtype=np.float64)
    for ball_index in range(m):
        # Commit load updates
        balancer_index = ball_index % balancers
        chosen_bin = algorithm(
            core.Ball(m, (ball_index // balancers) + 1, ball_weights[ball_index]),
            bins,
            *args,
        )
        chosen_bins[balancer_index] = chosen_bin.index
        chosen_weights[balancer_index] = ball_weights[ball_index]
        if balancer_index == balancers - 1:
            for i in range(balancers):
                bin_index = chosen_bins[i]
                bins.load[bin_index] = bins.load[bin_index] + chosen_weights[i]

    # Commit final load updates
    for i in range(m % balancers):
        bin_index = chosen_bins[i]
        bins.load[bin_index] = bins.load[bin_index] + chosen_weights[i]

    return bins.load, bins.counters


@njit
def load_balancer(ball_weights: NDArray[np.float64], n: int, algorithm, args):
    m = len(ball_weights)
    bins = init_bins(n)
    for ball_index in range(m):
        chosen_bin = algorithm(
            core.Ball(m, ball_index + 1, ball_weights[ball_index]), bins, *args
        )
        bins.load[chosen_bin.index] = (
            bins.load[chosen_bin.index] + ball_weights[ball_index]
        )

    return bins.load, bins.counters


@dataclass
class SimulationResult:
    m: int
    n: int
    maximum_load: float
    minimum_load: float
    std_load: float
    load_accesses: int


def get_steps(min_balls: int, max_balls: int, step: int | None, num: int | None):
    if step is not None:
        return np.arange(min_balls, max_balls, step)
    elif num is not None:
        return np.geomspace(min_balls, max_balls, num, dtype=np.int64)
    else:
        raise MissingStepParameter


class Algorithm(Protocol):
    def __call__(self, ball: core.Ball, bins: core.Bins, *args: Any) -> core.Bin:
        ...

    @property
    def __name__(self) -> str:
        ...


class Simulation:
    def __init__(self, algorithm: Algorithm) -> None:
        self.name = algorithm.__name__
        self.algorithm = njit(algorithm)

    def __call__(
        self,
        m: int,
        n: int,
        *args,
        ball_weights: NDArray[np.float64] | None = None,
        repeats: int = 1,
        balancers: int = 1,
    ) -> SimulationResult:

        if ball_weights is None:
            ball_weights = np.ones(m, dtype=np.float64)

        if balancers < 1:
            raise BalancerCountError("Requires at least 1 balancer")
        elif balancers == 1:
            load_balancer_callable = load_balancer
        else:
            load_balancer_callable = partial(parallel_load_balancer, balancers)

        result = SimulationResult(m, n, 0, 0, 0, 0)

        if repeats < 1:
            raise NumberOfRepeatsError("Requires at least 1 repeat")

        for _ in range(repeats):
            load, counters = load_balancer_callable(
                ball_weights, n, self.algorithm, args
            )
            result.maximum_load += np.amax(load)
            result.minimum_load += np.amin(load)
            result.std_load += np.std(load)
            result.load_accesses += counters.load

        if repeats > 1:
            result.maximum_load /= repeats
            result.minimum_load /= repeats
            result.std_load /= repeats
            result.load_accesses //= repeats

        return result

    def run(
        self,
        min_balls: int,
        max_balls: int,
        step: int | None = None,
        num: int | None = None,
        *args,
        repeats: int = 1,
        bins: int | None = None,
    ):
        results = []
        steps = get_steps(min_balls, max_balls, step, num)
        for balls in tqdm(steps):
            results.append(
                self(balls, max_balls if bins is None else bins, *args, repeats=repeats)
            )

        return results


@dataclass
class SimulationJob:
    algorithm: Algorithm
    algorithm_id: str
    args: List
    balls: int
    bins: int
    repeats: int
    balancers: int


@dataclass
class SimulationJobResult:
    algorithm_id: str
    result: SimulationResult


simulation_cache = {}


def parallel_runner(job: SimulationJob):
    if job.algorithm_id not in simulation_cache:
        simulation_cache[job.algorithm_id] = Simulation(job.algorithm)

    return SimulationJobResult(
        job.algorithm_id,
        simulation_cache[job.algorithm_id](
            job.balls, job.bins, *job.args, repeats=job.repeats, balancers=job.balancers
        ),
    )


class ParallelSimulation:
    @staticmethod
    def algorithm_id(algorithm: Algorithm, args: List):
        if args:
            args_string = "_".join(str(x) for x in args)
            return f"{algorithm.__name__}_{args_string}"
        else:
            return algorithm.__name__

    def __init__(self, *strategies: Tuple[Algorithm, List]) -> None:
        self.strategies = [
            (algorithm, self.algorithm_id(algorithm, args), args)
            for (algorithm, args) in strategies
        ]

        if hasattr(sys, "ps1"):
            print(
                "Warning: Parallel Simulations may hang if functions are defined in interactive mode."
            )

    def _run_jobs(self, jobs) -> Dict[str, List[SimulationResult]]:

        results: Dict[str, List[SimulationResult]] = defaultdict(list)

        if len(jobs) == 0:
            raise NoJobError

        cpu_count = min(len(jobs), available_cores())
        print(f"Using {cpu_count} processes to run {len(jobs)} jobs")

        with Pool(cpu_count) as p:
            result: SimulationJobResult
            for result in tqdm(
                p.imap_unordered(func=parallel_runner, iterable=jobs),
                total=len(jobs),
            ):
                results[result.algorithm_id].append(result.result)

        return dict(results)

    def __call__(
        self,
        m: int,
        n: int,
        repeats=1,
        balancers: int = 1,
    ) -> Dict[str, List[SimulationResult]]:

        jobs = []

        # Create a job for each algorithm
        for algorithm, algorithm_id, args in self.strategies:
            jobs.append(
                SimulationJob(algorithm, algorithm_id, args, m, n, repeats, balancers)
            )

        return self._run_jobs(jobs)

    def run(
        self,
        min_balls: int,
        max_balls: int,
        step: int | None = None,
        num: int | None = None,
        repeats=1,
        shuffle=True,
        bins: int | None = None,
        balancers: int = 1,
    ) -> Dict[str, List[SimulationResult]]:

        jobs = []
        results = {}

        steps = get_steps(min_balls, max_balls, step, num)

        # Create jobs for each algorithm
        for algorithm, algorithm_id, args in self.strategies:
            results[algorithm_id] = []
            for balls in steps:
                jobs.append(
                    SimulationJob(
                        algorithm,
                        algorithm_id,
                        args,
                        balls,
                        max_balls if bins is None else bins,
                        repeats,
                        balancers,
                    )
                )

        # Shuffle to provide accurate time estimate
        if shuffle:
            random.shuffle(jobs)

        results = self._run_jobs(jobs)

        # Always sort since jobs may be returned in the wrong order
        for _, algorithm_id, _ in self.strategies:
            results[algorithm_id] = sorted(results[algorithm_id], key=attrgetter("m"))

        return results
