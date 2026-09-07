from time import perf_counter, sleep
from typing import Optional, Sequence, Tuple, Union
from math import exp

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
    get_position,
    move,
    press,
    release,
    wheel,
)

from .trajectory import generate_trajectory, Point, _rng
from .speed import speed

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
    "right_click",
    "double_click",
    "path_to",
    "visualize_path",
    "press",
    "release",
    "wheel",
    "Point",
]


def path_to(
    target: Union[Point, Sequence[Point]],
    target_size: Union[float, Tuple[float, float]],
) -> None:
    if not isinstance(target, (tuple, list)):
        raise ValueError("Targets must be a point or a non-empty series of points.")
    if isinstance(target, Sequence) and len(target) == 0:
        raise ValueError("Targets must contain at least one point.")

    xs, ys, timestamps = generate_trajectory(
        get_position(), target, target_size
    )
    start = perf_counter()
    while True:
        now = perf_counter()
        elapsed = now - start
        if elapsed >= timestamps[-1]:
            break
        position = np.stack(
            [
                np.interp(elapsed, timestamps, xs),
                np.interp(elapsed, timestamps, ys),
            ]
        )
        move(int(position[0]), int(position[1]))
        sleep(0.001)
    final_target = (xs[-1], ys[-1])
    move(int(final_target[0]), int(final_target[1]))


def visualize_path(
    target: Union[Point, Sequence[Point]],
    target_size: Union[float, Tuple[float, float]],
    *,
    start: Optional[Point] = None,
) -> None:
    """Display a generated path and its velocity profile.

    Matplotlib is an optional dependency.  When it is unavailable, this
    function prints installation guidance and returns without generating a
    path.  ``start`` defaults to the current mouse position.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("visualize_path requires matplotlib; install it with `pip install matplotlib`.")
        return

    if not isinstance(target, (tuple, list)):
        raise ValueError("Targets must be a point or a non-empty series of points.")
    if len(target) == 0:
        raise ValueError("Targets must contain at least one point.")

    if isinstance(target[0], (int, float)):
        waypoints = [target]
    else:
        waypoints = list(target)

    if isinstance(target_size, (int, float)):
        target_width = target_height = target_size
    else:
        target_width, target_height = target_size

    path_start = get_position() if start is None else start
    xs, ys, timestamps = generate_trajectory(path_start, target, target_size)
    velocity_x = np.gradient(xs, timestamps)
    velocity_y = np.gradient(ys, timestamps)
    velocity = np.hypot(velocity_x, velocity_y)

    figure, (path_axis, speed_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        layout="constrained",
    )

    points = path_axis.scatter(xs, ys, c=timestamps, cmap="viridis", s=12)
    path_axis.plot(xs, ys, color="black", alpha=0.35)
    path_axis.scatter(xs[0], ys[0], color="limegreen", s=80, label="Start", zorder=3)
    path_axis.scatter(xs[-1], ys[-1], color="red", s=80, label="End", zorder=3)

    for index, waypoint in enumerate(waypoints):
        region = plt.Rectangle(
            (
                waypoint[0] - target_width / 2,
                waypoint[1] - target_height / 2,
            ),
            target_width,
            target_height,
            fill=False,
            edgecolor="tab:red",
            linewidth=2,
            label="Target region" if index == 0 else None,
        )
        path_axis.add_patch(region)

    figure.colorbar(points, ax=path_axis, label="Time (s)")
    path_axis.set_title("Mouse path")
    path_axis.set_xlabel("Screen X (px)")
    path_axis.set_ylabel("Screen Y (px)")
    path_axis.set_aspect("equal")
    path_axis.invert_yaxis()
    path_axis.legend()

    speed_axis.plot(timestamps, velocity, color="tab:blue")
    speed_axis.fill_between(timestamps, velocity, color="tab:blue", alpha=0.2)
    speed_axis.set_title("Velocity profile")
    speed_axis.set_xlabel("Time (s)")
    speed_axis.set_ylabel("Speed (px/s)")
    speed_axis.set_xlim(timestamps[0], timestamps[-1])
    speed_axis.set_ylim(bottom=0)
    speed_axis.grid(alpha=0.3)

    plt.show()


def click(button: str = LEFT) -> None:
    press(button)
    sleep(min(exp(speed.single_click_mu + speed.single_click_sigma * _rng.normalvariate(0.0, 1.0)), speed.single_click_max))
    release(button)
    sleep(min(exp(speed.single_click_mu + speed.single_click_sigma * _rng.normalvariate(0.0, 1.0)), speed.single_click_max))

def double_click(button: str = LEFT) -> None:
    press(button)
    sleep(min(exp(speed.first_click_mu + speed.first_click_sigma * _rng.normalvariate(0.0, 1.0)), speed.first_click_max))
    release(button)
    sleep(min(exp(speed.second_click_mu + speed.second_click_sigma * _rng.normalvariate(0.0, 1.0)), speed.second_click_max))
    press(button)
    sleep(min(exp(speed.second_click_mu + speed.second_click_sigma * _rng.normalvariate(0.0, 1.0)), speed.second_click_max))
    release(button)

def right_click() -> None:
    click(RIGHT)
