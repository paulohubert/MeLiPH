from __future__ import annotations

from abc import ABC, abstractmethod
from statistics import NormalDist
from typing import Dict, Optional, Tuple, Type
import warnings

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None


class BaseForecastModel(ABC):
    name: str = "base"

    @staticmethod
    def _validate_inputs(y: pd.DataFrame, th: int, conf: float) -> pd.DataFrame:
        if not isinstance(y, pd.DataFrame):
            raise TypeError("`y` must be a pandas DataFrame with columns 'DS' and 'SALES'.")

        required_cols = {"DS", "SALES"}
        missing = required_cols - set(y.columns)
        if missing:
            raise ValueError(f"`y` is missing required columns: {missing}")

        if not isinstance(th, int) or th <= 0:
            raise ValueError("`th` must be a positive integer.")

        if not (0 < conf < 1):
            raise ValueError("`conf` must be between 0 and 1, e.g. 0.95.")

        if len(y) < 1:
            raise ValueError("At least 1 observation is required.")

        return y.copy()

    @staticmethod
    def _prepare_series(y: pd.DataFrame, freq: str = "7D") -> pd.Series:
        df = y.copy()
        df["DS"] = pd.to_datetime(df["DS"])
        df = df.sort_values("DS").drop_duplicates(subset="DS")
        df["SALES"] = pd.to_numeric(df["SALES"], errors="coerce")
        df = df.fillna(0)

        ts = df.set_index("DS")["SALES"].asfreq(freq)
        ts = ts.interpolate(method="time").ffill().bfill()
        return ts

    @staticmethod
    def _build_result(
        ts: pd.Series,
        forecast: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        th: int,
        freq: str = "7D",
    ) -> pd.DataFrame:
        last_date = ts.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(freq),
            periods=th,
            freq=freq,
        )
        return pd.DataFrame(
            {"DS": future_dates, "forecast": forecast, "lower": lower, "upper": upper}
        )

    @abstractmethod
    def predict(self, y: pd.DataFrame, th: int = 1, conf: float = 0.95) -> pd.DataFrame:
        pass


class NaiveForecastModel(BaseForecastModel):
    name = "naive"

    def predict(self, y: pd.DataFrame, th: int = 1, conf: float = 0.95) -> pd.DataFrame:
        df = self._validate_inputs(y, th, conf)
        ts = self._prepare_series(df)

        mean_sales = ts.mean()
        sd_sales = ts.std() if ts.shape[0] >= 2 else 0
        z = NormalDist().inv_cdf((1 + conf) / 2)

        forecast = np.full(th, mean_sales, dtype=float)
        upper = np.full(th, mean_sales + z * sd_sales, dtype=float)
        lower = np.full(th, max(0, mean_sales - z * sd_sales), dtype=float)
        return self._build_result(ts, forecast, lower, upper, th)


class ArimaForecastModel(BaseForecastModel):
    name = "arima"

    def __init__(
        self,
        order: Optional[Tuple[int, int, int]] = None,
        max_p: int = 3,
        max_d: int = 2,
        max_q: int = 3,
    ) -> None:
        self.order = order
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q

    @staticmethod
    def _validate_inputs(y: pd.DataFrame, th: int, conf: float) -> pd.DataFrame:
        df = BaseForecastModel._validate_inputs(y, th, conf)
        if len(df) < 12:
            raise ValueError("At least 12 observations are needed to fit an ARIMA model.")
        return df

    def predict(self, y: pd.DataFrame, th: int = 1, conf: float = 0.95) -> pd.DataFrame:
        df = self._validate_inputs(y, th, conf)
        ts = self._prepare_series(df)
        best_model = self._fit_model(ts)

        alpha = 1 - conf
        forecast_res = best_model.get_forecast(steps=th)
        forecast_mean = forecast_res.predicted_mean.values
        conf_int = forecast_res.conf_int(alpha=alpha)
        lower_col, upper_col = conf_int.columns[0], conf_int.columns[1]

        return self._build_result(
            ts=ts,
            forecast=forecast_mean,
            lower=conf_int[lower_col].values,
            upper=conf_int[upper_col].values,
            th=th,
        )

    def _fit_model(self, ts: pd.Series):
        if ARIMA is None:
            raise ImportError(
                "statsmodels is required for ArimaForecastModel. "
                "Install it with `pip install statsmodels`."
            )

        if self.order is not None:
            return ARIMA(ts, order=self.order).fit()

        best_aic = np.inf
        best_model = None
        warnings.filterwarnings("ignore")
        try:
            for p in range(self.max_p + 1):
                for d in range(self.max_d + 1):
                    for q in range(self.max_q + 1):
                        try:
                            fitted = ARIMA(ts, order=(p, d, q)).fit()
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_model = fitted
                        except Exception:
                            continue
        finally:
            warnings.resetwarnings()

        if best_model is None:
            raise RuntimeError("Could not fit any ARIMA model. Try providing `order` manually.")
        return best_model


class TsbForecastModel(BaseForecastModel):
    name = "tsb"

    def __init__(self, alpha_p: float = 0.2, alpha_z: float = 0.2) -> None:
        if not (0 < alpha_p <= 1):
            raise ValueError("`alpha_p` must be in (0, 1].")
        if not (0 < alpha_z <= 1):
            raise ValueError("`alpha_z` must be in (0, 1].")
        self.alpha_p = alpha_p
        self.alpha_z = alpha_z

    def predict(self, y: pd.DataFrame, th: int = 1, conf: float = 0.95) -> pd.DataFrame:
        df = self._validate_inputs(y, th, conf)
        ts = self._prepare_series(df)
        values = ts.to_numpy(dtype=float)

        demand_occurrence = (values > 0).astype(float)
        positive_values = values[values > 0]

        # Initialization: probability of non-zero demand and mean non-zero size.
        p_hat = float(np.clip(demand_occurrence.mean(), 1e-6, 1.0 - 1e-6))
        z_hat = float(positive_values.mean()) if positive_values.size > 0 else 0.0

        # TSB recursive updates.
        for value in values:
            occurred = 1.0 if value > 0 else 0.0
            p_hat = p_hat + self.alpha_p * (occurred - p_hat)
            if occurred > 0:
                z_hat = z_hat + self.alpha_z * (value - z_hat)

        # Point forecast for each horizon step.
        mean_forecast = max(0.0, p_hat * z_hat)
        forecast = np.full(th, mean_forecast, dtype=float)

        # Approximate prediction interval:
        # Var(D) for intermittent demand D ~= Bernoulli(p_hat) * Size
        # with Size mean z_hat and empirical variance from non-zero values.
        size_var = float(np.var(positive_values, ddof=1)) if positive_values.size >= 2 else 0.0
        demand_var = max(0.0, p_hat * size_var + p_hat * (1.0 - p_hat) * (z_hat**2))
        demand_sd = np.sqrt(demand_var)
        z_score = NormalDist().inv_cdf((1 + conf) / 2)

        upper = np.full(th, mean_forecast + z_score * demand_sd, dtype=float)
        lower = np.full(th, max(0.0, mean_forecast - z_score * demand_sd), dtype=float)
        return self._build_result(ts, forecast, lower, upper, th)


class ForecastModels:
    _registry: Dict[str, Type[BaseForecastModel]] = {}

    @classmethod
    def register(cls, model_cls: Type[BaseForecastModel]) -> None:
        cls._registry[model_cls.name] = model_cls

    @classmethod
    def available_models(cls):
        return sorted(cls._registry.keys())

    @classmethod
    def create(cls, model_name: str, **model_kwargs) -> BaseForecastModel:
        model_cls = cls._registry.get(model_name)
        if model_cls is None:
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {cls.available_models()}"
            )
        return model_cls(**model_kwargs)

    @classmethod
    def predict(
        cls,
        model_name: str,
        y: pd.DataFrame,
        th: int = 1,
        conf: float = 0.95,
        **model_kwargs,
    ) -> pd.DataFrame:
        model = cls.create(model_name, **model_kwargs)
        return model.predict(y=y, th=th, conf=conf)


ForecastModels.register(NaiveForecastModel)
ForecastModels.register(ArimaForecastModel)
ForecastModels.register(TsbForecastModel)
