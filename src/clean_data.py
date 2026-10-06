"""Clean and merge the raw data into one analysis-ready panel.

Reads the raw files saved by fetch_data.py and produces
data/processed/panel.parquet with one row per metro per month:

    region_id | metro | state | size_rank | date | zhvi | mortgage_rate_30y | unemployment_rate | cpi

Run from the project root with:
    python -m src.clean_data
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ZILLOW_ID_COLS = ["RegionID", "SizeRank", "RegionName", "RegionType", "StateName"]


# --- Zillow ----------------------------------------------------------------

def clean_zillow(path: Path) -> pd.DataFrame:
    """Reshape Zillow's wide file (one column per month) into long format."""
    raw = pd.read_csv(path)

    # Keep metro areas only; the file also has a single national ("country") row.
    raw = raw[raw["RegionType"].str.lower() == "msa"]

    long = raw.melt(id_vars=ZILLOW_ID_COLS, var_name="date", value_name="zhvi")

    # Zillow dates are month-end (e.g. 2000-01-31). Convert to month-start so
    # they line up with FRED's monthly dates.
    long["date"] = pd.to_datetime(long["date"]).dt.to_period("M").dt.to_timestamp()

    long = long.rename(columns={
        "RegionID": "region_id",
        "SizeRank": "size_rank",
        "RegionName": "metro",
        "StateName": "state",
    }).drop(columns="RegionType")

    # Metros where Zillow coverage starts later have empty early months.
    long = long.dropna(subset=["zhvi"])
    return long


# --- FRED ------------------------------------------------------------------

def clean_fred(path: Path) -> pd.DataFrame:
    """Convert all FRED series to one value per month."""
    fred = pd.read_csv(path, parse_dates=["date"], index_col="date")

    # The mortgage rate is weekly; averaging within each month makes it monthly.
    # Monthly series are unchanged by this.
    monthly = fred.resample("MS").mean()
    monthly.index.name = "date"
    return monthly.reset_index()


# --- Validation ------------------------------------------------------------

def validate(panel: pd.DataFrame) -> None:
    """Fail loudly if the panel looks wrong."""
    dupes = panel.duplicated(subset=["region_id", "date"]).sum()
    if dupes:
        raise ValueError(f"Found {dupes} duplicate metro-month rows.")

    if (panel["zhvi"] <= 0).any():
        raise ValueError("Found zero or negative home values.")

    if panel["date"].min() > pd.Timestamp("2001-01-01"):
        raise ValueError(f"Data starts too late: {panel['date'].min().date()}")

    if panel["region_id"].nunique() < 100:
        raise ValueError(f"Only {panel['region_id'].nunique()} metros; expected hundreds.")


# --- Main ------------------------------------------------------------------

def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Cleaning Zillow data...")
    zillow = clean_zillow(RAW_DIR / "zillow_zhvi_metro.csv")

    print("Cleaning FRED data...")
    fred = clean_fred(RAW_DIR / "fred_macro.csv")

    # Left join: keep every metro-month, attach the national macro values.
    panel = zillow.merge(fred, on="date", how="left")
    panel = panel.sort_values(["region_id", "date"]).reset_index(drop=True)

    validate(panel)

    out_path = PROCESSED_DIR / "panel.parquet"
    panel.to_parquet(out_path, index=False)

    print(f"Saved {out_path.relative_to(PROJECT_ROOT)}")
    print(f"  {len(panel):,} rows, {panel['region_id'].nunique()} metros")
    print(f"  Dates {panel['date'].min().date()} to {panel['date'].max().date()}")
    print("  Missing values per column:")
    print(panel.isna().sum().to_string())


if __name__ == "__main__":
    main()