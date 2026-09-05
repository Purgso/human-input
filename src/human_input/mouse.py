from collections.abc import Sequence
from time import perf_counter, sleep

import numpy as np
from mouse import (
    DOUBLE,
    DOWN,
    LEFT,
    MIDDLE,
    RIGHT,
    UP,
    X2,
    X,
    click,
    double_click,
    get_position,
    move,
    press,
    release,
    right_click,
    wheel,
)

from .trajectory import generate_trajectory

__all__ = [
    "DOUBLE",
    "DOWN",
    "LEFT",
    "MIDDLE",
    "RIGHT",
    "UP",
    "X2",
    "X",
    "click",
    "double_click",
    "path_to",
    "press",
    "release",
    "right_click",
    "wheel",
]

Point = tuple[int, int]


def path_to(
    target: Point | Sequence[Point],
    target_size: float | tuple[float, float],
):
    target_points = np.asarray(target)
    if target_points.shape == (2,):
        target_points = target_points.reshape(1, 2)
    if target_points.ndim != 2 or target_points.shape[1] != 2:
        raise ValueError("Targets must be a point or a non-empty series of points.")
    if len(target_points) == 0:
        raise ValueError("Targets must contain at least one point.")

    timestamps, path = generate_trajectory(
        get_position(), target_points.tolist(), target_size
    )
    start = perf_counter()
    while True:
        now = perf_counter()
        elapsed = now - start
        if elapsed >= timestamps[-1]:
            break
        position = np.stack(
            [
                np.interp(elapsed, timestamps, path[:, 0]),
                np.interp(elapsed, timestamps, path[:, 1]),
            ]
        )
        move(int(position[0]), int(position[1]))
        sleep(0.001)
    final_target = path[-1]
    move(int(final_target[0]), int(final_target[1]))
