# human-input

`human-input` is built on top of the Python
[`mouse`](https://github.com/boppreh/mouse) and
[`keyboard`](https://github.com/boppreh/keyboard) libraries. It keeps their familiar
input-control APIs while adding human-like mouse movement, clicking, and typing.

## Installation

```console
pip install human-input
```

The underlying libraries may require additional operating-system permissions to
control or monitor global input.

## Mouse movement

Move to one target using its center coordinate and clickable size:

```python
from human_input import mouse

mouse.path_to((800, 450), target_size=40)
```

Pass several waypoints to move through them in one continuous trajectory. Target
size can be a single value for a square or a `(width, height)` pair:

```python
mouse.path_to(
    [(300, 200), (600, 300), (800, 450)],
    target_size=(80, 40),
)
```

## Clicking

The click helpers use sampled human-like press, release, and inter-click delays:

```python
from human_input import mouse

mouse.click()
mouse.double_click()
mouse.right_click()
mouse.click(mouse.MIDDLE)
```

Other major functions from the underlying `mouse` library, including `move`,
`drag`, `wheel`, hooks, and recording/playback, are available as passthroughs.

## Typing

```python
from human_input import keyboard

keyboard.write("Hello, world!")
```

Each key-down and key-up event is scheduled independently using sampled flight
and hold times. Shift is handled explicitly for uppercase letters and shifted
symbols. Major `keyboard` functions such as `press`, `release`, `add_hotkey`,
`wait`, hooks, and recording/playback are also available as passthroughs.

## Execution speed

Set `settings.speed` to one of the included profiles. Larger scaling values mean
faster execution:

```python
from human_input import keyboard, mouse, settings

settings.speed = settings.FAST

# These calls use the new speed profile.
mouse.path_to((800, 450), target_size=40)
keyboard.write("This is faster.")
```

The predefined profiles are `SUPERHUMAN`, `VERY_FAST`, `FAST`, `NORMAL`, `SLOW`,
and `VERY_SLOW`. A custom profile can control each input category separately:

```python
settings.speed = settings.Speed(
    mouse_move_scaling=1.25,
    mouse_click_scaling=0.9,
    keyboard_scaling=1.1,
)
```

## Trajectory visualization

Matplotlib is optional and is not installed with `human-input`. If it is
available, `visualize_path` plots the generated path, every waypoint region, and
the velocity profile:

```console
pip install matplotlib
```

```python
from human_input import mouse

mouse.visualize_path(
    [(300, 200), (600, 300), (800, 450)],
    target_size=(80, 40),
    start=(100, 100),  # Optional; defaults to the current mouse position.
)
```

## How the human-like behavior is generated

### Mouse movement

Each requested movement is split into target-to-target segments. Segment duration
is estimated with [Fitts's law](https://www2.psychology.uiowa.edu/faculty/mordkoff/infoproc/pdfs/Fitts%201954.pdf),
using movement distance and effective target width to model the speed-accuracy
tradeoff.

The path generator randomizes control points within the target regions, adds
small corrective movements near the destination, and joins the points with
smooth spline curves. Its overall trajectory construction, overlapping movement
segments, and lognormal velocity profiles were informed by the
[Sigma-Lognormal model](https://doi.org/10.1016/j.patcog.2008.10.017) of rapid
human movement. The implementation is inspired by that model rather than a full
biomechanical reproduction of it.

### Mouse clicks

Click press and release delays are sampled from lognormal distributions. Their
parameters were derived using click-duration observations from the
[Mouse Dynamics Dataset for Behavioral User Substitution Detection in Electronic Testing](https://doi.org/10.17632/3rfsbcmgfw.1).

### Keyboard input

Typing uses separate flight-time (key-down to following key-down) and hold-time
(key-down to key-up) distributions for individual character codes. The bundled
parameters were created from the
[Timing distributions in free text keystroke dynamics profiles](https://doi.org/10.17632/sjk7kz35nh.1)
dataset and its companion study,
[On the shape of timings distributions in free-text keystroke dynamics profiles](https://doi.org/10.1016/j.heliyon.2021.e08413).
Unknown characters fall back to general timing distributions, and all generated
press/release events are ordered before execution so natural rollover can occur.
