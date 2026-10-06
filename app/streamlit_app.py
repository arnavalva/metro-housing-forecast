"""Metro home value forecaster: Streamlit app.

Run locally from the project root with:
    streamlit run app/streamlit_app.py
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

SNAPSHOT = Path(__file__).resolve().parent / "snapshot"

HISTORY_COLOR = "#1F5F8B"
MODEL_COLORS = {"Momentum (last 12m)": "#B45309", "Historical mean": "#6B7A8C"}

st.set_page_config(page_title="Metro home value forecaster", page_icon="🏠", layout="wide")


@st.cache_data
def load_data():
    history = pd.read_parquet(SNAPSHOT / "history.parquet")
    forecasts = pd.read_parquet(SNAPSHOT / "forecasts.parquet")
    backtest = pd.read_parquet(SNAPSHOT / "backtest.parquet")
    results = pd.read_csv(SNAPSHOT / "results.csv")
    return history, forecasts, backtest, results


history, forecasts, backtest, results = load_data()

# --- Metro picker (largest metros first) -----------------------------------

metros = (
    history.groupby(["region_id", "metro"])["size_rank"].first()
    .sort_values().reset_index()
)

st.title("Where are home prices headed?")
st.write(
    "12-month home value forecasts for the 400 largest U.S. metro areas, "
    "built from Zillow and Federal Reserve (FRED) data."
)

choice = st.selectbox("Metro area", metros["metro"], index=0)
region_id = metros.loc[metros["metro"] == choice, "region_id"].iloc[0]

metro_hist = history[history["region_id"] == region_id].sort_values("date")
metro_fc = forecasts[forecasts["region_id"] == region_id]
latest = metro_hist.iloc[-1]

# --- Headline numbers ------------------------------------------------------

momentum = metro_fc[metro_fc["model"] == "Momentum (last 12m)"].iloc[0]
hist_mean = metro_fc[metro_fc["model"] == "Historical mean"].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Typical home value, {latest['date']:%b %Y}", f"${latest['zhvi']:,.0f}")
c2.metric("Change over past 12 months", f"{latest['growth_12m']:+.1%}")
c3.metric("Momentum forecast, next 12 months", f"{momentum['predicted_growth']:+.1%}")
c4.metric("Historical-average forecast", f"{hist_mean['predicted_growth']:+.1%}")

# --- Price history with forecasts ------------------------------------------

start_year = st.slider("Show history from", 2000, int(latest["date"].year) - 1, 2012)
shown = metro_hist[metro_hist["date"].dt.year >= start_year]

history_line = alt.Chart(shown).mark_line(color=HISTORY_COLOR, strokeWidth=2.5).encode(
    x=alt.X("date:T", title=None),
    y=alt.Y("zhvi:Q", title="Typical home value ($)", scale=alt.Scale(zero=False),
            axis=alt.Axis(format="$,.0f")),
    tooltip=[alt.Tooltip("date:T", format="%b %Y"), alt.Tooltip("zhvi:Q", format="$,.0f")],
)

# Each forecast is a dashed segment from the latest value to the 12-month forecast.
segments = pd.concat([
    pd.DataFrame({
        "model": row["model"],
        "date": [row["date"], row["forecast_date"]],
        "value": [row["zhvi"], row["forecast_value"]],
    })
    for _, row in metro_fc.iterrows()
])
forecast_lines = alt.Chart(segments).mark_line(strokeDash=[6, 4], strokeWidth=2.5, point=True).encode(
    x="date:T",
    y="value:Q",
    color=alt.Color("model:N", title="Forecast",
                    scale=alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values())),
                    legend=alt.Legend(orient="top-left")),
    tooltip=["model:N", alt.Tooltip("date:T", format="%b %Y"),
             alt.Tooltip("value:Q", format="$,.0f", title="Forecast value")],
)

st.altair_chart((history_line + forecast_lines).properties(height=420), width="stretch")

# --- How well did each forecast do in the past? ----------------------------

st.subheader(f"How past forecasts for {choice} turned out")
st.write(
    "Each point is a 12-month forecast made in that month, compared with what "
    "actually happened. Points below the actual line were too pessimistic; "
    "points above were too optimistic."
)

metro_bt = backtest[(backtest["region_id"] == region_id) & backtest["model"].isin(MODEL_COLORS)]
actual = (
    metro_bt.drop_duplicates("date")[["date", "target"]]
    .assign(model="Actual", value=lambda d: d["target"])
)
predicted = metro_bt.assign(value=metro_bt["prediction"])[["date", "model", "value"]]
bt_long = pd.concat([actual[["date", "model", "value"]], predicted])

domain = ["Actual", *MODEL_COLORS]
colors = ["#1C2733", *MODEL_COLORS.values()]
bt_chart = alt.Chart(bt_long).mark_line(strokeWidth=2).encode(
    x=alt.X("date:T", title="Month forecast was made"),
    y=alt.Y("value:Q", title="12-month growth", axis=alt.Axis(format="%")),
    color=alt.Color("model:N", title=None, scale=alt.Scale(domain=domain, range=colors),
                    legend=alt.Legend(orient="top-left")),
    tooltip=["model:N", alt.Tooltip("date:T", format="%b %Y"), alt.Tooltip("value:Q", format="+.1%")],
)
st.altair_chart(bt_chart.properties(height=320), width="stretch")

# --- Accuracy across all metros --------------------------------------------

st.subheader("Average forecast error across all 400 metros")
st.write(
    "Mean absolute error in percentage points of 12-month growth. Lower is better. "
    "Momentum works well in steady markets but misses turning points like the "
    "2022 mortgage rate shock."
)
mae = results.pivot(index="period", columns="model", values="MAE").round(2)
st.dataframe(mae, width="stretch")

st.caption(
    "Data: Zillow Home Value Index (ZHVI, all homes, smoothed and seasonally adjusted) "
    "and FRED. Forecasts are simple baselines; a machine learning model is in progress. "
    "Not financial advice."
)