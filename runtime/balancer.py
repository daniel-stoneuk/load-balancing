import numpy as np
from numba import njit


def balance(balls: int, bins: int):
    probabilities = np.full(bins, 1 / bins)
    load = np.random.multinomial(balls, probabilities)
    return load


def python_greedy_d(m: int, n: int, d: int):
    load = np.zeros(n, dtype="int64")
    for ball in range(m):
        lowest_bin = np.random.randint(0, n)
        for i in range(d - 1):
            choice = np.random.randint(0, n)
            if load[choice] < load[lowest_bin]:
                lowest_bin = choice
        load[lowest_bin] += 1
    return np.amax(load)


@njit
def numba_greedy_d(balls: int, bins: int, d: int):
    load = np.zeros(bins, dtype="int64")
    for ball in range(balls):
        lowest_bin = np.random.randint(0, bins)
        for i in range(d - 1):
            choice = np.random.randint(0, bins)
            if load[choice] < load[lowest_bin]:
                lowest_bin = choice
        load[lowest_bin] += 1
    return np.amax(load)


# https://stackoverflow.com/a/64084123
import ctypes

handle = ctypes.CDLL("lib.so")
handle.K_Greedy.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]


def C_K_Greedy(balls: int, bins: int, k: int):
    return handle.K_Greedy(balls, bins, k)
