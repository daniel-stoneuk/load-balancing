import load_balancing.core as lb
from load_balancing.utils import timer
from load_balancing.simulation import (
    ParallelSimulation,
    Simulation,
)


def uniform(ball: lb.Ball, bins: lb.Bins):
    """Select a bin at random"""
    return bins.choose_one()


def greedy_n(ball: lb.Ball, bins: lb.Bins):
    """Select the bin with the lowest load"""
    return bins.min()


def greedy_d(ball: lb.Ball, bins: lb.Bins, d: int):
    """Choose two bins at random, and select the bin with the lowest load"""
    chosen_bins = bins.choose(d)
    return chosen_bins.min()


def always_go_left(ball: lb.Ball, bins: lb.Bins):
    """Divide the bins into two halves, and choose a random bin from each. Select the lowest loaded. Tiebreaks: Choose left."""
    left_bins, right_bins = bins.split(2)
    left_bin = left_bins.choose_one()
    right_bin = right_bins.choose_one()
    if left_bin.load <= right_bin.load:
        return left_bin
    else:
        return right_bin


def fair_tiebreak(ball: lb.Ball, bins: lb.Bins):
    """Divide the bins into two halves, and choose a random bin from each. Select the lowest loaded. Tiebreaks: Fair."""
    left_bins, right_bins = bins.split(2)
    left_bin = left_bins.choose_one()
    right_bin = right_bins.choose_one()
    if left_bin.load < right_bin.load:
        return left_bin
    elif left_bin.load > right_bin.load:
        return right_bin
    else:
        if lb.probability(0.5):
            return left_bin
        else:
            return right_bin


def one_plus_beta(ball: lb.Ball, bins: lb.Bins, p: float):
    """With probability p choose two bins, otherwise choose one at random"""
    if lb.probability(p):
        return bins.choose(2).min()
    else:
        return bins.choose_one()


def threshold(ball: lb.Ball, bins: lb.Bins):
    threshold = ball.m / bins.n + 1
    while True:
        chosen_bin = bins.choose_one()
        if chosen_bin.load < threshold:
            return chosen_bin


def adaptive(ball: lb.Ball, bins: lb.Bins):
    n = bins.n
    while True:
        chosen_bin = bins.choose_one()
        if chosen_bin.load < ball.position / n + 1:
            return chosen_bin


def main():

    m = 10000
    n = 1000
    print(f"Running simulation for all algorithms. {m=} {n=}")

    with timer() as time:
        uniform_sim = Simulation(uniform)
        print("Uniform: ", uniform_sim(m, n))

        greedy_n_sim = Simulation(greedy_n)
        print("Greedy N: ", greedy_n_sim(m, n))

        greedy_d_sim = Simulation(greedy_d)
        print("Greedy D: ", greedy_d_sim(m, n, 2))

        one_plus_beta_sim = Simulation(one_plus_beta)
        print("One Plus Beta (0.1): ", one_plus_beta_sim(m, n, 0.1))
        print("One Plus Beta (0.2): ", one_plus_beta_sim(m, n, 0.2))
        print("One Plus Beta (0.3): ", one_plus_beta_sim(m, n, 0.3))
        print("One Plus Beta (0.4): ", one_plus_beta_sim(m, n, 0.4))
        print("One Plus Beta (0.5): ", one_plus_beta_sim(m, n, 0.5))
        print("One Plus Beta (0.6): ", one_plus_beta_sim(m, n, 0.6))
        print("One Plus Beta (0.7): ", one_plus_beta_sim(m, n, 0.7))
        print("One Plus Beta (0.8): ", one_plus_beta_sim(m, n, 0.8))
        print("One Plus Beta (0.9): ", one_plus_beta_sim(m, n, 0.9))

        always_go_left_sim = Simulation(always_go_left)
        print("Always Go Left: ", always_go_left_sim(m, n))

        threshold_sim = Simulation(threshold)
        print("Threshold: ", threshold_sim(m, n))

        adaptive_sim = Simulation(adaptive)
        print("Adaptive: ", adaptive_sim(m, n))

    print("Finished in", time(), "seconds")

    print(f"Running parallel simulation for uniform and greedy_d")

    with timer() as time:

        parallel_simulation = ParallelSimulation((uniform, []), (greedy_d, [2]))
        results = parallel_simulation.run(1000, 10000, 1000)
        for algorithm in ["uniform", "greedy_d"]:
            for result in results[algorithm]:
                print(algorithm, f"{m=} Maximum Load:", result.maximum_load)

    print("Finished in", time(), "seconds")


if __name__ == "__main__":
    main()
