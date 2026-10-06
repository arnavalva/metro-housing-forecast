"""Export a small snapshot of data and forecasts for the Streamlit app.

The full data/ folder is gitignored, so the deployed app can't see it. This
script writes compact files to app/snapshot/, which IS committed, so the app
works on Streamlit Community Cloud.

Run after the backtest, from the project root:
    python -m src.export_app_data
"""

import shutil
from pathlib import Path

import pandas as pd

from src.backtest import RESULTS_DIR, load_panel
from src.features import HORIZON_MONTHS
from src.models import HistoricalMean, Momentum

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKTEST_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_forecasts.parquet"
SNAPSHOT_DIR = PROJECT_ROOT / "app" / "snapshot"


def latest_forecasts(panel: pd.DataFrame) -> pd.DataFrame:
    """Forecast each metro's value 12 months after its most recent month."""
    latest = panel.sort_values("date").groupby("region_id").tail(1)

    rows = []
    for model in [Momentum(), HistoricalMean()]:
        out = latest[["region_id", "metro", "date", "zhvi"]].copy()
        out["predicted_growth"] = model.predict(latest)
        out["forecast_date"] = out["date"] + pd.DateOffset(months=HORIZON_MONTHS)
        out["forecast_value"] = out["zhvi"] * (1 + out["predicted_growth"])
        out["model"] = model.name
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading panel...")
    panel = load_panel()

    history = panel[["region_id", "metro", "state", "size_rank", "date", "zhvi", "growth_12m"]]
    history = history.astype({"zhvi": "float32", "growth_12m": "float32"})
    history.to_parquet(SNAPSHOT_DIR / "history.parquet", index=False)

    latest_forecasts(panel).to_parquet(SNAPSHOT_DIR / "forecasts.parquet", index=False)

    backtest = pd.read_parquet(BACKTEST_PATH)
    backtest = backtest[["region_id", "date", "target", "prediction", "model"]]
    backtest = backtest.astype({"target": "float32", "prediction": "float32"})
    backtest.to_parquet(SNAPSHOT_DIR / "backtest.parquet", index=False)

    shutil.copy(RESULTS_DIR / "baseline_results.csv", SNAPSHOT_DIR / "results.csv")

    for f in sorted(SNAPSHOT_DIR.iterdir()):
        print(f"Saved {f.relative_to(PROJECT_ROOT)} ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()