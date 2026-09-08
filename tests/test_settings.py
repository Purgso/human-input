import pytest

from human_input import settings


def test_speed_scaling_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        settings.Speed(0.0, 1.0, 1.0)


def test_fast_profile_uses_larger_multipliers() -> None:
    assert settings.FAST.mouse_move_scaling > settings.NORMAL.mouse_move_scaling
    assert settings.FAST.mouse_click_scaling > settings.NORMAL.mouse_click_scaling
    assert settings.FAST.keyboard_scaling > settings.NORMAL.keyboard_scaling
