"""Forecasting models. Each has fit(train) and predict(test).

The baselines ignore the training data, but sharing one interface means the
backtest can evaluate them and the machine learning models the same way.
"""

import numpy as np
import pandas as pd


class NoChange:
    """Predicts 0% growth: next year's price equals today's."""

    name = "No change"

    def fit(self, train: pd.DataFrame) -> "NoChange":
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(test))


class Momentum:
    """Predicts that the next 12 months repeat the last 12 months."""

    name = "Momentum (last 12m)"

    def fit(self, train: pd.DataFrame) -> "Momentum":
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return test["growth_12m"].to_numpy()


class HistoricalMean:
    """Predicts the metro's long-run average annual growth to date."""

    name = "Historical mean"

    def fit(self, train: pd.DataFrame) -> "HistoricalMean":
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return test["hist_mean_growth"].to_numpy()