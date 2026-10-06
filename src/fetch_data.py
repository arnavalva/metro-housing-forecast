"""Download raw data for the metro housing forecast project.

Pulls two sources and saves them unchanged to data/raw/:
  1. Zillow Home Value Index (ZHVI) for U.S. metro areas, monthly.
  2. Macroeconomic series from FRED (mortgage rates, unemployment, CPI).

Run from the project root with:
    python -m src.fetch_data
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

# --- Configuration ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

ZILLOW_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zhvi/"
    "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
)

# FRED series ID -> readable column name
FRED_SERIES = {
    "MORTGAGE30US": "mortgage_rate_30y",  # weekly, percent
    "UNRATE": "unemployment_rate",        # monthly, percent
    "CPIAUCSL": "cpi",                    # monthly, index
}

FRED_START_DATE = "1995-01-01"


# --- Download functions ----------------------------------------------------

def fetch_zillow() -> pd.DataFrame:
    """Download the Zillow metro ZHVI file and check it looks right."""
    print("Downloading Zillow ZHVI metro data...")
    df = pd.read_csv(ZILLOW_URL)

    required = {"RegionID", "RegionName", "RegionType"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Zillow file is missing expected columns: {missing}")
    if len(df) < 100:
        raise ValueError(f"Zillow file has only {len(df)} rows; expected hundreds.")

    print(f"  Got {len(df)} regions and {df.shape[1]} columns.")
    return df


def fetch_fred() -> pd.DataFrame:
    """Download each FRED series and combine them into one table."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY not found. Make sure .env exists in the project root "
            "and contains a line like FRED_API_KEY=your_key"
        )

    fred = Fred(api_key=api_key)
    series = {}
    for series_id, name in FRED_SERIES.items():
        print(f"Downloading FRED series {series_id}...")
        s = fred.get_series(series_id, observation_start=FRED_START_DATE)
        if s.empty:
            raise ValueError(f"FRED series {series_id} came back empty.")
        series[name] = s
        print(f"  Got {len(s)} observations, latest {s.index.max().date()}.")

    df = pd.concat(series, axis=1)
    df.index.name = "date"
    return df


# --- Main ------------------------------------------------------------------

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    zillow = fetch_zillow()
    zillow_path = RAW_DIR / "zillow_zhvi_metro.csv"
    zillow.to_csv(zillow_path, index=False)
    print(f"Saved {zillow_path.relative_to(PROJECT_ROOT)}")

    fred = fetch_fred()
    fred_path = RAW_DIR / "fred_macro.csv"
    fred.to_csv(fred_path)
    print(f"Saved {fred_path.relative_to(PROJECT_ROOT)}")

    print("Done.")


if __name__ == "__main__":
    main()