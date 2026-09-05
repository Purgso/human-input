from dataclasses import dataclass

@dataclass(frozen=True)
class Speed:
    # Mouse Movement Speed
    fitts_a: float = 0.05
    fitts_b: float = 0.105
    lognormal_sigma: float = 0.42
    velocity_peak: float = 0.38

    # Mouse Path Geometry
    bend_fraction: float = 0.08
    curvature_slowdown: float = 1.5
    correction_probability: tuple[float, ...] = (0.50, 0.30, 0.15, 0.05)
    initial_error_scale: float = 0.65
    correction_decay: float = 0.35
    endpoint_sigma: float = 0.22

    # Limits
    minimum_time: float = 0.05
    maximum_time: float = 2.0

    # Typing Speed
    key_scaling: float = 1.0

SUPERHUMAN = ...
VERY_FAST = ...
FAST = ...
NORMAL = Speed()
SLOW = ...
VERY_SLOW = ...

speed: Speed = NORMAL