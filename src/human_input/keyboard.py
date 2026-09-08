"""Human-like typing built on top of the ``keyboard`` package."""

import csv
import random
from math import exp
from pathlib import Path
from time import perf_counter, sleep
from typing import Dict, List, NamedTuple, Optional, Tuple

import keyboard as _keyboard

from .settings import settings, speed

__all__ = ["write"]


# Rounded means of the corresponding columns in keypress_timings_ascii.csv.
# These are used for characters that do not have observations in the table.
_TIMINGS_PATH = Path(__file__).with_name("data") / "keypress_timings_ascii.csv"
_rng = random.Random()

_SHIFTED_KEYS = {
    "~": "`",
    "!": "1",
    "@": "2",
    "#": "3",
    "$": "4",
    "%": "5",
    "^": "6",
    "&": "7",
    "*": "8",
    "(": "9",
    ")": "0",
    "_": "-",
    "+": "=",
    "{": "[",
    "}": "]",
    "|": "\\",
    ":": ";",
    '"': "'",
    "<": ",",
    ">": ".",
    "?": "/",
}


class _KeyEvent(NamedTuple):
    timestamp: float
    pressed: bool
    key: str


def _load_timings() -> Dict[str, Dict[int, Tuple[float, float]]]:
    timings: Dict[str, Dict[int, Tuple[float, float]]] = {
        "FT": {},
        "HT": {},
    }
    with _TIMINGS_PATH.open(newline="", encoding="utf-8-sig") as timing_file:
        for row in csv.DictReader(timing_file):
            timings[row["feature"]][int(row["key"])] = (
                float(row["mu_log_ms"]),
                float(row["sigma_log"]),
            )
    return timings


_TIMINGS = _load_timings()


def _keyboard_key(character: str) -> str:
    """Translate control characters to names understood by ``keyboard``."""
    return {
        "\n": "enter",
        "\r": "enter",
        "\t": "tab",
        "\b": "backspace",
    }.get(character, character)


def _character_code(character: str) -> int:
    if character in ("\n", "\r"):
        return 13
    return ord(character)


def _sample_seconds(feature: str, character_code: int) -> float:
    default = (
        (settings.default_float_mu, settings.default_float_sigma)
        if feature == "FT"
        else (settings.default_hold_mu, settings.default_hold_sigma)
    )
    maximum = settings.float_max if feature == "FT" else settings.hold_max
    mu, sigma = _TIMINGS[feature].get(character_code, default)
    sampled = min(exp(_rng.normalvariate(mu, sigma)) / 1000.0, maximum)
    return sampled * speed.keyboard_scaling


def _shifted_key(character: str) -> Optional[str]:
    if "A" <= character <= "Z":
        return character.lower()
    return _SHIFTED_KEYS.get(character)


def _generate_events(text: str) -> List[_KeyEvent]:
    events: List[_KeyEvent] = []
    press_time = 0.0
    latest_release = 0.0
    index = 0

    while index < len(text):
        character = text[index]
        shifted_key = _shifted_key(character)

        if shifted_key is not None:
            # Shift boundaries never overlap neighboring character events.
            shift_press = max(press_time, latest_release)
            events.append(_KeyEvent(shift_press, True, "shift"))

            leading_hold = _sample_seconds("HT", -1)
            shifted_press = shift_press + leading_hold

            while shifted_key is not None:
                character_code = _character_code(character)
                hold_time = _sample_seconds("HT", character_code)
                flight_time = _sample_seconds("FT", character_code)
                release_time = shifted_press + hold_time

                events.append(_KeyEvent(shifted_press, True, shifted_key))
                events.append(_KeyEvent(release_time, False, shifted_key))
                latest_release = release_time
                index += 1

                if index >= len(text):
                    break

                character = text[index]
                shifted_key = _shifted_key(character)
                if shifted_key is not None:
                    # Shifted characters never roll over into one another.
                    shifted_press += max(flight_time, hold_time)

            trailing_hold = _sample_seconds("HT", -1)
            shift_release = latest_release + trailing_hold
            events.append(_KeyEvent(shift_release, False, "shift"))
            latest_release = shift_release

            if index < len(text):
                default_flight = _sample_seconds("FT", -1)
                press_time = shift_release + max(0.0, default_flight - trailing_hold)
            continue

        character_code = _character_code(character)
        hold_time = _sample_seconds("HT", character_code)
        release_time = press_time + hold_time
        key = _keyboard_key(character)
        events.append(_KeyEvent(press_time, True, key))
        events.append(_KeyEvent(release_time, False, key))
        latest_release = max(latest_release, release_time)
        press_time += _sample_seconds("FT", character_code)
        index += 1

    events.sort(key=lambda event: event.timestamp)
    return events


def write(text: str) -> None:
    """Type *text* using sampled flight and hold times for every key event."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    events = _generate_events(text)
    start_time = perf_counter()
    held_keys: Dict[str, int] = {}

    try:
        for event in events:
            delay = event.timestamp - (perf_counter() - start_time)
            if delay > 0:
                sleep(delay)

            if event.pressed:
                _keyboard.press(event.key)
                held_keys[event.key] = held_keys.get(event.key, 0) + 1
            else:
                _keyboard.release(event.key)
                remaining = held_keys.get(event.key, 1) - 1
                if remaining > 0:
                    held_keys[event.key] = remaining
                else:
                    held_keys.pop(event.key, None)
    finally:
        for key in held_keys:
            _keyboard.release(key)
