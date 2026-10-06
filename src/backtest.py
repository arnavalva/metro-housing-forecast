"""Expanding-window backtest for 12-month-ahead home value forecasts.

For each test year Y, a model is trained only on rows whose 12-month target
was already known by January of Y, then it forecasts every month of Y. This
mimics forecasting in real time and prevents look-ahead leakage.

Run from the project root with:
    python -m src.backtest
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features import HORIZON_MONTHS, add_baseline_features, add_target
from src.models import HistoricalMean, Momentum, NoChange

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "panel.parquet"
RESULTS_DIR = PROJECT_ROOT / "results"

TOP_N_METROS = 400
FIRST_TEST_YEAR = 2012

# Forecast periods, labeled by the year the forecast is made.
PERIODS = {
    "2012-2019 (steady growth)": (2012, 2019),
    "2020-2021 (pandemic boom)": (2020, 2021),
    "2022+ (rate shock)": (2022, 2100),
}


def load_panel(top_n: int = TOP_N_METROS) -> pd.DataFrame:
    """Load the processed panel, keep the largest metros, add target and features."""
    panel = pd.read_parquet(PANEL_PATH)

    # size_rank: 1 = largest metro. Keep the top_n largest.
    ranks = panel.groupby("region_id")["size_rank"].first()
    keep = ranks.nsmallest(top_n).index
    panel = panel[panel["region_id"].isin(keep)]

    panel = add_target(panel)
    panel = add_baseline_features(panel)
    return panel


def backtest(model, panel: pd.DataFrame, first_year: int = FIRST_TEST_YEAR) -> pd.DataFrame:
    """Run an expanding-window backtest and return one row per forecast."""
    labeled = panel[panel["target"].notna()]
    last_year = labeled["date"].max().year

    results = []
    for year in range(first_year, last_year + 1):
        test_start = pd.Timestamp(year=year, month=1, day=1)
        test_end = pd.Timestamp(year=year, month=12, day=1)

        # A row dated s has its target revealed at s + 12 months, so only rows
        # with s + 12 months <= test_start may be used for training.
        train_cutoff = test_start - pd.DateOffset(months=HORIZON_MONTHS)
        train = labeled[labeled["date"] <= train_cutoff]
        test = labeled[labeled["date"].between(test_start, test_end)]
        if test.empty:
            continue

        model.fit(train)
        out = test[["region_id", "metro", "date", "target"]].copy()
        out["prediction"] = model.predict(test)
        out["model"] = model.name
        results.append(out)

    return pd.concat(results, ignore_index=True)


def score(forecasts: pd.DataFrame) -> pd.DataFrame:
    """MAE and RMSE in percentage points, overall and by period."""
    errors = forecasts.assign(
        error=(forecasts["prediction"] - forecasts["target"]) * 100,
        year=forecasts["date"].dt.year,
    )

    def metrics(df: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "MAE": df["error"].abs().mean(),
            "RMSE": np.sqrt((df["error"] ** 2).mean()),
            "n": len(df),
        })

    rows = []
    for model, group in errors.groupby("model"):
        rows.append({"model": model, "period": "All years", **metrics(group)})
        for label, (start, end) in PERIODS.items():
            subset = group[group["year"].between(start, end)]
            if not subset.empty:
                rows.append({"model": model, "period": label, **metrics(subset)})
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading panel...")
    panel = load_panel()
    print(f"  {panel['region_id'].nunique()} metros, "
          f"{panel['date'].min().date()} to {panel['date'].max().date()}")

    models = [NoChange(), Momentum(), HistoricalMean()]
    forecasts = pd.concat([backtest(m, panel) for m in models], ignore_index=True)

    # Compare every model on exactly the same metro-months.
    complete = forecasts.pivot_table(
        index=["region_id", "date"], columns="model", values="prediction"
    ).dropna().index
    forecasts = forecasts.set_index(["region_id", "date"]).loc[complete].reset_index()
    forecasts = forecasts.dropna(subset=["prediction"])

    results = score(forecasts)
    results.to_csv(RESULTS_DIR / "baseline_results.csv", index=False)
    forecasts.to_parquet(PROJECT_ROOT / "data" / "processed" / "baseline_forecasts.parquet")

    pd.set_option("display.width", 120)
    print("\nForecast error for 12-month home value growth (percentage points):\n")
    table = results.pivot(index="period", columns="model", values="MAE").round(2)
    print("MAE")
    print(table.to_string())
    table = results.pivot(index="period", columns="model", values="RMSE").round(2)
    print("\nRMSE")
    print(table.to_string())
    print(f"\nSaved {(RESULTS_DIR / 'baseline_results.csv').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()