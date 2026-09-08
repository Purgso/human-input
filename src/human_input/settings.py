from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Settings:
    # Mouse Movement Speed
    fitts_a: float = 0.07
    fitts_b: float = 0.105
    lognormal_sigma: float = 0.42
    velocity_peak: float = 0.38

    # Mouse Path Geometry
    bend_fraction: float = 0.02
    curvature_slowdown: float = 0.33
    correction_probability: tuple[float, ...] = (0.50, 0.30, 0.15, 0.05)
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

    # Typing Timings
    default_float_mu: float = 5.378254224
    default_float_sigma: float = 0.713738600
    float_max: float = 2.0

    default_hold_mu: float = 4.489425935
    default_hold_sigma: float = 0.349903662
    hold_max: float = 1.0


@dataclass(frozen=True)
class Speed:
    mouse_move_scaling: float
    mouse_click_scaling: float
    keyboard_scaling: float

    def __post_init__(self) -> None:
        if (
            min(
                self.mouse_move_scaling,
                self.mouse_click_scaling,
                self.keyboard_scaling,
            )
            <= 0
        ):
            raise ValueError("Speed scaling values must be positive.")


SUPERHUMAN = Speed(
    mouse_move_scaling=5.0, mouse_click_scaling=5.0, keyboard_scaling=5.0
)
VERY_FAST = Speed(mouse_move_scaling=2.0, mouse_click_scaling=2.0, keyboard_scaling=2.0)
FAST = Speed(mouse_move_scaling=1.5, mouse_click_scaling=1.5, keyboard_scaling=1.5)
NORMAL = Speed(mouse_move_scaling=1.0, mouse_click_scaling=1.0, keyboard_scaling=1.0)
SLOW = Speed(mouse_move_scaling=0.5, mouse_click_scaling=0.5, keyboard_scaling=0.5)
VERY_SLOW = Speed(
    mouse_move_scaling=0.25, mouse_click_scaling=0.25, keyboard_scaling=0.25
)

speed: Speed = NORMAL
settings: Settings = Settings()
