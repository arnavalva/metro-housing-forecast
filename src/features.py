"""Target and feature construction for the metro housing panel.

Every column built here uses only information available at that row's date,
except the target, which is deliberately forward-looking.
"""

import pandas as pd

HORIZON_MONTHS = 12


def _value_at_offset(panel: pd.DataFrame, months: int, col: str = "zhvi") -> pd.Series:
    """Look up each metro's value `months` away from each row's date.

    Positive months look forward, negative look back. Matching on actual dates
    (not row positions) keeps this correct even if a metro has missing months.
    """
    lookup = panel[["region_id", "date", col]].copy()
    lookup["date"] = lookup["date"] - pd.DateOffset(months=months)
    merged = panel[["region_id", "date"]].merge(
        lookup, on=["region_id", "date"], how="left"
    )
    return pd.Series(merged[col].to_numpy(), index=panel.index)


def add_target(panel: pd.DataFrame, horizon: int = HORIZON_MONTHS) -> pd.DataFrame:
    """Target: percent change in home value over the next `horizon` months.

    The target for a row dated t is only known at t + horizon. The backtest
    relies on this when deciding which rows a model may train on.
    """
    panel = panel.copy()
    future = _value_at_offset(panel, horizon)
    panel["target"] = future / panel["zhvi"] - 1
    return panel


def add_baseline_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Backward-looking growth measures used by the baseline models."""
    panel = panel.sort_values(["region_id", "date"]).copy()

    past = _value_at_offset(panel, -12)
    panel["growth_12m"] = panel["zhvi"] / past - 1

    # Each metro's average 12-month growth from its first observation up to
    # and including the current date (an expanding mean, so no future data).
    has_value = panel["growth_12m"].notna()
    running_sum = panel["growth_12m"].fillna(0).groupby(panel["region_id"]).cumsum()
    running_count = has_value.groupby(panel["region_id"]).cumsum()
    panel["hist_mean_growth"] = running_sum / running_count.where(running_count > 0)

    return panel