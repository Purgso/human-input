from mouse import get_position, move, press, release, click, double_click, right_click, wheel, LEFT, RIGHT, MIDDLE, X, X2, UP, DOWN, DOUBLE
from time import perf_counter, sleep
import numpy as np

from .trajectory import generate_trajectory

def path_to(target: tuple[int, int], target_size: float | tuple[float, float]):
    timestamps, path = generate_trajectory(get_position(), target, target_size)
    start = perf_counter()
    while True:
        now = perf_counter()
        elapsed = now - start
        if elapsed >= timestamps[-1]:
            break
        position = np.stack([
            np.interp(elapsed, timestamps, path[:, 0]),
            np.interp(elapsed, timestamps, path[:, 1]),
        ])
        move(int(position[0]), int(position[1]))
        sleep(0.001)
    move(target[0], target[1])