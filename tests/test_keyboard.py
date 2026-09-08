from human_input import keyboard


def test_timing_data_is_loaded() -> None:
    assert len(keyboard._TIMINGS["FT"]) == 124
    assert len(keyboard._TIMINGS["HT"]) == 122


def test_shifted_run_has_no_rollover(monkeypatch) -> None:
    def sample(feature: str, character_code: int) -> float:
        if character_code == -1:
            return 0.3 if feature == "FT" else 0.1
        return 0.15 if feature == "FT" else 0.2

    monkeypatch.setattr(keyboard, "_sample_seconds", sample)

    events = keyboard._generate_events("aBCd")
    actual = [(round(event.timestamp, 2), event.pressed, event.key) for event in events]

    assert actual == [
        (0.0, True, "a"),
        (0.2, False, "a"),
        (0.2, True, "shift"),
        (0.3, True, "b"),
        (0.5, False, "b"),
        (0.5, True, "c"),
        (0.7, False, "c"),
        (0.8, False, "shift"),
        (1.0, True, "d"),
        (1.2, False, "d"),
    ]


def test_write_executes_sorted_events(monkeypatch) -> None:
    calls = []
    events = [
        keyboard._KeyEvent(0.0, True, "a"),
        keyboard._KeyEvent(0.0, False, "a"),
    ]
    monkeypatch.setattr(keyboard, "_generate_events", lambda text: events)
    monkeypatch.setattr(keyboard, "sleep", lambda delay: None)
    monkeypatch.setattr(
        keyboard._keyboard, "press", lambda key: calls.append(("down", key))
    )
    monkeypatch.setattr(
        keyboard._keyboard, "release", lambda key: calls.append(("up", key))
    )

    keyboard.write("a")

    assert calls == [("down", "a"), ("up", "a")]
