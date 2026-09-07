from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class Speed:
    # Mouse Movement Speed
    fitts_a: float = 0.07
    fitts_b: float = 0.105
    lognormal_sigma: float = 0.42
    velocity_peak: float = 0.38

    # Mouse Path Geometry
    bend_fraction: float = 0.02
    curvature_slowdown: float = 0.33
    correction_probability: Tuple[float, ...] = (0.50, 0.30, 0.15, 0.05)
    initial_error_scale: float = 0.65
    correction_decay: float = 0.35
    endpoint_sigma: float = 0.22
    impulse_overlap: float = 0.2
    tension: float = 0.5

    # Travel Time Limits
    minimum_time: float = 0.01
    maximum_time: float = 5.0

    # Mouse Click Timings
    single_click_mu: float = -2.0
    single_click_sigma: float = 0.6
    single_click_max: float = 0.5
    first_click_mu: float = -2.0
    first_click_sigma: float = 0.5
    first_click_max: float = 0.5
    second_click_mu: float = -1.5
    second_click_sigma: float = 0.6
    second_click_max: float = 0.75

    # Typing Speed
    key_scaling: float = 1.0

SUPERHUMAN = ...
VERY_FAST = ...
FAST = ...
NORMAL = Speed()
SLOW = ...
VERY_SLOW = ...

speed: Speed = NORMAL
