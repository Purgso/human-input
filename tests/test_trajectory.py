import pytest

from human_input import settings
from human_input.trajectory import _rng, generate_trajectory


def test_trajectory_starts_at_requested_position() -> None:
    start = (100.0, 200.0)
    _rng.seed(7)

    xs, ys, timestamps = generate_trajectory(start, (500.0, 300.0), 40.0)

    assert (xs[0], ys[0]) == pytest.approx(start)
    assert len(xs) == len(ys) == len(timestamps)
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == 0.0


def test_trajectory_rejects_empty_targets() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        generate_trajectory((0.0, 0.0), [], 10.0)


def test_speed_profile_changes_duration(monkeypatch) -> None:
    monkeypatch.setattr(settings, "speed", settings.NORMAL)
    _rng.seed(7)
    normal_duration = generate_trajectory((0.0, 0.0), (500.0, 250.0), 40.0)[2][-1]

    monkeypatch.setattr(settings, "speed", settings.FAST)
    _rng.seed(7)
    fast_duration = generate_trajectory((0.0, 0.0), (500.0, 250.0), 40.0)[2][-1]

    assert fast_duration < normal_duration
