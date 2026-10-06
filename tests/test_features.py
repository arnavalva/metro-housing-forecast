"""Tests that targets look forward and features only look backward."""

import pandas as pd
import pytest

from src.features import add_baseline_features, add_target


@pytest.fixture
def toy_panel() -> pd.DataFrame:
    """One metro whose value rises by 1 each month: 100, 101, 102, ..."""
    dates = pd.date_range("2010-01-01", periods=36, freq="MS")
    return pd.DataFrame({"region_id": 1, "date": dates, "zhvi": range(100, 136)})


def test_target_is_growth_over_next_12_months(toy_panel):
    panel = add_target(toy_panel)
    first = panel.iloc[0]
    assert first["target"] == pytest.approx(112 / 100 - 1)


def test_target_missing_when_future_unknown(toy_panel):
    panel = add_target(toy_panel)
    assert panel["target"].tail(12).isna().all()


def test_growth_uses_only_past_values(toy_panel):
    panel = add_baseline_features(toy_panel)
    assert panel["growth_12m"].head(12).isna().all()
    row = panel.iloc[12]  # value 112, twelve months after 100
    assert row["growth_12m"] == pytest.approx(112 / 100 - 1)


def test_growth_unaffected_by_future_data(toy_panel):
    full = add_baseline_features(toy_panel)
    truncated = add_baseline_features(toy_panel.iloc[:24])
    cols = ["growth_12m", "hist_mean_growth"]
    pd.testing.assert_frame_equal(
        full[cols].iloc[:24].reset_index(drop=True),
        truncated[cols].reset_index(drop=True),
    )