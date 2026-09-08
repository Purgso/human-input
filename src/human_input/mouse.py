from __future__ import annotations

from math import exp
from time import perf_counter, sleep
from typing import Sequence

import mouse as _mouse
import numpy as np

from . import settings
from .trajectory import Point, _rng, generate_trajectory

__all__ = [
    "DOUBLE",
    "DOWN",
    "LEFT",
    "MIDDLE",
    "RIGHT",
    "UP",
    "X2",
    "ButtonEvent",
    "MoveEvent",
    "Point",
    "WheelEvent",
    "X",
    "click",
    "double_click",
    "drag",
    "get_position",
    "hold",
    "hook",
    "is_pressed",
    "on_button",
    "on_click",
    "on_double_click",
    "on_middle_click",
    "on_right_click",
    "path_to",
    "play",
    "press",
    "record",
    "release",
    "replay",
    "right_click",
    "unhook",
    "unhook_all",
    "visualize_path",
    "wait",
    "wheel",
]


# Public passthroughs to the underlying mouse package. The click helpers below
# are intentionally humanized; movement and event APIs keep upstream behavior.
ButtonEvent = _mouse.ButtonEvent
DOUBLE = _mouse.DOUBLE
DOWN = _mouse.DOWN
LEFT = _mouse.LEFT
MIDDLE = _mouse.MIDDLE
MoveEvent = _mouse.MoveEvent
RIGHT = _mouse.RIGHT
UP = _mouse.UP
WheelEvent = _mouse.WheelEvent
X2 = _mouse.X2
X = _mouse.X
drag = _mouse.drag
get_position = _mouse.get_position
hold = _mouse.hold
hook = _mouse.hook
is_pressed = _mouse.is_pressed
on_button = _mouse.on_button
on_click = _mouse.on_click
on_double_click = _mouse.on_double_click
on_middle_click = _mouse.on_middle_click
on_right_click = _mouse.on_right_click
play = _mouse.play
press = _mouse.press
record = _mouse.record
release = _mouse.release
replay = _mouse.replay
unhook = _mouse.unhook
unhook_all = _mouse.unhook_all
wait = _mouse.wait
wheel = _mouse.wheel


def path_to(
    target: Point | Sequence[Point],
    target_size: float | tuple[float, float],
) -> None:
    if not isinstance(target, (tuple, list)):
        raise TypeError("Targets must be a point or a non-empty series of points.")
    if isinstance(target, Sequence) and len(target) == 0:
        raise ValueError("Targets must contain at least one point.")

    xs, ys, timestamps = generate_trajectory(get_position(), target, target_size)
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
        _mouse.move(int(position[0]), int(position[1]))
        sleep(0.001)
    final_target = (xs[-1], ys[-1])
    _mouse.move(int(final_target[0]), int(final_target[1]))


def visualize_path(
    target: Point | Sequence[Point],
    target_size: float | tuple[float, float],
    *,
    start: Point | None = None,
) -> None:
    """Display a generated path and its velocity profile.

    Matplotlib is an optional dependency.  When it is unavailable, this
    function prints installation guidance and returns without generating a
    path.  ``start`` defaults to the current mouse position.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print(
            "visualize_path requires matplotlib; install it with `pip install matplotlib`."
        )
        return

    if not isinstance(target, (tuple, list)):
        raise TypeError("Targets must be a point or a non-empty series of points.")
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
        region = plt.Rectangle(  # type: ignore
            (
                waypoint[0] - target_width / 2,  # type: ignore
                waypoint[1] - target_height / 2,  # type: ignore
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


def _click_delay(mu: float, sigma: float, maximum: float) -> None:
    duration = min(exp(mu + sigma * _rng.normalvariate(0.0, 1.0)), maximum)
    sleep(duration / settings.speed.mouse_click_scaling)


def click(button: str = LEFT) -> None:
    press(button)
    _click_delay(
        settings.settings.single_click_mu,
        settings.settings.single_click_sigma,
        settings.settings.single_click_max,
    )
    release(button)
    _click_delay(
        settings.settings.single_click_mu,
        settings.settings.single_click_sigma,
        settings.settings.single_click_max,
    )


def double_click(button: str = LEFT) -> None:
    press(button)
    _click_delay(
        settings.settings.first_click_mu,
        settings.settings.first_click_sigma,
        settings.settings.first_click_max,
    )
    release(button)
    _click_delay(
        settings.settings.second_click_mu,
        settings.settings.second_click_sigma,
        settings.settings.second_click_max,
    )
    press(button)
    _click_delay(
        settings.settings.second_click_mu,
        settings.settings.second_click_sigma,
        settings.settings.second_click_max,
    )
    release(button)


def right_click() -> None:
    click(RIGHT)
