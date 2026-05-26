"""Lightweight tests that don't require a trained model or the data download."""

from src.features.elo import (
    EloTable,
    expected_score,
    goal_margin_multiplier,
)


def test_expected_score_equal_ratings() -> None:
    # Equal ratings, no home advantage → 0.5
    assert abs(expected_score(1500, 1500) - 0.5) < 1e-9


def test_expected_score_home_advantage() -> None:
    # Home advantage should increase home expectation
    base = expected_score(1500, 1500, home_adv=0)
    boosted = expected_score(1500, 1500, home_adv=65)
    assert boosted > base


def test_goal_margin_multiplier_monotone() -> None:
    # Bigger wins → bigger K multiplier
    multipliers = [goal_margin_multiplier(m) for m in range(1, 6)]
    assert multipliers == sorted(multipliers)


def test_elo_table_default() -> None:
    table = EloTable()
    assert table.get("Made Up FC") == 1500.0
    table.set("Made Up FC", 1700.0)
    assert table.get("Made Up FC") == 1700.0


if __name__ == "__main__":
    test_expected_score_equal_ratings()
    test_expected_score_home_advantage()
    test_goal_margin_multiplier_monotone()
    test_elo_table_default()
    print("All sanity tests passed.")
