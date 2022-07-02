"""Core API for creating a load balancing algorithm
"""
from __future__ import annotations

import numpy as np
from numba import boolean, float64, int64, njit
from numba.experimental import jitclass


@jitclass(
    [
        ("load", int64),
    ]
)
class Counters:
    load: int

    def __init__(self):
        self.load = 0


Counters_Type = Counters.class_type.instance_type  # type: ignore


@jitclass(
    [
        ("weight", float64),
        ("position", int64),
    ]
)
class Ball:
    m: int
    weight: int
    position: int

    def __init__(self, m: int, position: int, weight: int):
        self.m = m
        self.position = position
        self.weight = weight


@jitclass(
    [
        ("index", int64),
        ("_load", float64),
        ("counters", Counters_Type),
    ]
)
class Bin:
    index: int
    _load: float
    counters: Counters

    def __init__(self, index: int, load: float, counters: Counters):
        self.index = index
        self._load = load
        self.counters = counters

    @property
    def load(self) -> float:
        self.counters.load += 1
        return self._load


@jitclass(
    [
        ("indices", int64[:]),
        ("load", float64[:]),
        ("counters", Counters_Type),
    ]
)
class Bins:
    n: int
    indices: np.ndarray
    load: np.ndarray
    counters: Counters

    def __init__(
        self, n: int, indices: np.ndarray, load: np.ndarray, counters: Counters
    ):
        self.n = n
        self.indices = indices
        self.load = load
        self.counters = counters

    def choose(self, d: int):
        return Bins(
            self.n,
            np.random.choice(self.indices, size=d, replace=True),
            self.load,
            self.counters,
        )

    def choose_one(self):
        chosen_index = np.random.choice(self.indices, size=1, replace=True)[0]
        return Bin(chosen_index, self.load[chosen_index], self.counters)

    def split(self, n: int):
        return [
            Bins(self.n, indices, self.load, self.counters)
            for indices in np.array_split(self.indices, n)
        ]

    def min(self):
        min_index = self.indices[0]
        for i in range(1, self.indices.size):
            index = self.indices[i]
            if self.load[index] < self.load[min_index]:
                min_index = index

        self.counters.load += self.indices.size
        return Bin(min_index, self.load[min_index], self.counters)


@njit(boolean(float64))
def probability(p: float):
    return np.random.random() < p
