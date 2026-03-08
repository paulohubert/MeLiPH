from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from forecast_models import ForecastModels


def _metric_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _metric_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _metric_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    nonzero_mask = y_true != 0
    if not np.any(nonzero_mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100.0)


def _metric_mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    if y_train.shape[0] < 2:
        return float("nan")

    naive_denom = np.mean(np.abs(np.diff(y_train)))
    if naive_denom == 0 or np.isnan(naive_denom):
        return float("nan")

    mae = _metric_mae(y_true, y_pred)
    return float(mae / naive_denom)


def _metric_interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    if y_true.shape[0] == 0 or lower.shape[0] != y_true.shape[0] or upper.shape[0] != y_true.shape[0]:
        return float("nan")
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def _nan_metrics() -> Dict[str, float]:
    return {"MAPE": float("nan"), "MAE": float("nan"), "RMSE": float("nan"), "MASE": float("nan")}


def _nan_interval_metrics() -> Dict[str, float]:
    return {"CI_COVERAGE": float("nan")}


def backtest_models(
    y: pd.DataFrame,
    th: int = 1,
    leave_out_k: int = 1,
    conf: float = 0.95,
    model_names: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Backtest all registered forecast models on a single time series.

    Parameters
    ----------
    y : pd.DataFrame
        DataFrame containing at least:
        - 'DS': date/time column
        - 'SALES': target values
    th : int, default=1
        Forecast horizon used for each model prediction.
    leave_out_k : int, default=1
        Number of latest points to hold out as test data.
    conf : float, default=0.95
        Confidence level passed to forecast models.

    Returns
    -------
    dict | None
        - None if there are not enough data points.
        - Otherwise:
          {
              "th": ...,
              "leave_out_k": ...,
              "n_obs": ...,
              "results": {
                  "<model_name>": {"MAPE": ..., "MAE": ..., "RMSE": ..., "MASE": ...},
                  ...
              }
          }
    """
    if not isinstance(y, pd.DataFrame):
        raise TypeError("`y` must be a pandas DataFrame with columns 'DS' and 'SALES'.")
    if not isinstance(th, int) or th <= 0:
        raise ValueError("`th` must be a positive integer.")
    if not isinstance(leave_out_k, int) or leave_out_k <= 0:
        raise ValueError("`leave_out_k` must be a positive integer.")
    if not (0 < conf < 1):
        raise ValueError("`conf` must be between 0 and 1.")

    required_cols = {"DS", "SALES"}
    missing = required_cols - set(y.columns)
    if missing:
        raise ValueError(f"`y` is missing required columns: {missing}")

    series = y[["DS", "SALES"]].copy()
    series["DS"] = pd.to_datetime(series["DS"])
    series["SALES"] = pd.to_numeric(series["SALES"], errors="coerce")
    series = series.sort_values("DS").drop_duplicates(subset="DS")
    series = series.dropna(subset=["SALES"])

    n_obs = len(series)
    if n_obs <= leave_out_k:
        return None

    y_train = series.iloc[:-leave_out_k].copy()
    y_test = series.iloc[-leave_out_k:].copy()

    if len(y_train) < 2:
        return None

    eval_horizon = min(th, len(y_test))
    if eval_horizon <= 0:
        return None

    y_true = y_test["SALES"].to_numpy(dtype=float)[:eval_horizon]
    y_train_values = y_train["SALES"].to_numpy(dtype=float)

    point_results: Dict[str, Dict[str, float]] = {}
    interval_results: Dict[str, Dict[str, float]] = {}

    selected_models = model_names if model_names is not None else ForecastModels.available_models()
    for model_name in selected_models:
        try:
            prediction = ForecastModels.predict(model_name, y_train, th=th, conf=conf)
            y_pred = prediction["forecast"].to_numpy(dtype=float)[:eval_horizon]
            lower = prediction["lower"].to_numpy(dtype=float)[:eval_horizon]
            upper = prediction["upper"].to_numpy(dtype=float)[:eval_horizon]

            if y_pred.shape[0] != eval_horizon:
                point_results[model_name] = _nan_metrics()
                interval_results[model_name] = _nan_interval_metrics()
                continue

            point_results[model_name] = {
                "MAPE": _metric_mape(y_true, y_pred),
                "MAE": _metric_mae(y_true, y_pred),
                "RMSE": _metric_rmse(y_true, y_pred),
                "MASE": _metric_mase(y_true, y_pred, y_train_values),
            }
            interval_results[model_name] = {
                "CI_COVERAGE": _metric_interval_coverage(y_true, lower, upper)
            }
        except Exception:
            point_results[model_name] = _nan_metrics()
            interval_results[model_name] = _nan_interval_metrics()

    return {
        "th": th,
        "leave_out_k": leave_out_k,
        "n_obs": n_obs,
        "point_results": point_results,
        "interval_results": interval_results,
    }


def backtest_all_series(
    df: pd.DataFrame,
    th: int = 1,
    leave_out_k: int = 1,
    conf: float = 0.95,
    model_names: Optional[List[str]] = None,
    inventory_col: str = "INVENTORY_ID",
    region_col: str = "REGION",
    ds_col: str = "DS",
    sales_col: str = "SALES",
    n_jobs: int = -1,
) -> Dict[str, Dict[str, Optional[Dict[str, Dict[str, Dict[str, float]]]]]]:
    """
    Run backtesting in parallel for each (INVENTORY_ID, REGION) series.

    Returns
    -------
    dict
        Nested mapping:
        {
            "<inventory_id>": {
                "<region>": {
                    "point_results": {
                        "<model_name>": {"MAPE": ..., "MAE": ..., "RMSE": ..., "MASE": ...},
                        ...
                    },
                    "interval_results": {
                        "<model_name>": {"CI_COVERAGE": ...},
                        ...
                    }
                } | None
            },
            ...
        }
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")
    if not isinstance(th, int) or th <= 0:
        raise ValueError("`th` must be a positive integer.")
    if not isinstance(leave_out_k, int) or leave_out_k <= 0:
        raise ValueError("`leave_out_k` must be a positive integer.")
    if not (0 < conf < 1):
        raise ValueError("`conf` must be between 0 and 1.")

    required_cols = {inventory_col, region_col, ds_col, sales_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"`df` is missing required columns: {missing}")

    work_df = df[[inventory_col, region_col, ds_col, sales_col]].copy()

    grouped = list(work_df.groupby([inventory_col, region_col], dropna=False, sort=False))
    if len(grouped) == 0:
        return {}

    max_workers = None if n_jobs == -1 else max(1, n_jobs)

    def _run_one(item: Tuple[Tuple[Any, Any], pd.DataFrame]):
        (inventory_id, region), series_df = item
        one_series = series_df.rename(columns={ds_col: "DS", sales_col: "SALES"})
        bt = backtest_models(
            one_series,
            th=th,
            leave_out_k=leave_out_k,
            conf=conf,
            model_names=model_names,
        )
        if bt is None:
            payload = None
        else:
            payload = {
                "point_results": bt["point_results"],
                "interval_results": bt["interval_results"],
            }
        return inventory_id, region, payload

    nested_results: Dict[str, Dict[str, Optional[Dict[str, Dict[str, Dict[str, float]]]]]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for inventory_id, region, result in executor.map(_run_one, grouped):
            inv_key = str(inventory_id)
            region_key = str(region)
            if inv_key not in nested_results:
                nested_results[inv_key] = {}
            nested_results[inv_key][region_key] = result

    return nested_results


def classify_time_series(
    df: pd.DataFrame,
    inventory_col: str = "INVENTORY_ID",
    region_col: str = "REGION",
    sales_col: str = "SALES",
    class_col: str = "SERIES_CLASS",
    low_data_max_points: int = 5,
    zero_inflated_min_ratio: float = 0.40,
) -> pd.DataFrame:
    """
    Classify each (inventory, region) series and append the class to all rows.

    Rules
    -----
    1) Low data: n_points <= low_data_max_points
    2) Zero-inflated: n_points > low_data_max_points and zero_ratio >= zero_inflated_min_ratio
    3) Full series: n_points > low_data_max_points and zero_ratio < zero_inflated_min_ratio
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")

    required_cols = {inventory_col, region_col, sales_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"`df` is missing required columns: {missing}")

    work_df = df.copy()
    work_df[sales_col] = pd.to_numeric(work_df[sales_col], errors="coerce")

    summary = (
        work_df.groupby([inventory_col, region_col], dropna=False, sort=False)[sales_col]
        .agg(n_points="count", zero_ratio=lambda s: np.mean(s.fillna(0) == 0))
        .reset_index()
    )

    summary[class_col] = np.where(
        summary["n_points"] <= low_data_max_points,
        "Low data",
        np.where(
            summary["zero_ratio"] >= zero_inflated_min_ratio,
            "Zero-inflated",
            "Full series",
        ),
    )

    return work_df.merge(
        summary[[inventory_col, region_col, class_col]],
        on=[inventory_col, region_col],
        how="left",
    )


def backtest_all_series_by_class(
    df: pd.DataFrame,
    th: int = 1,
    leave_out_k: int = 1,
    conf: float = 0.95,
    inventory_col: str = "INVENTORY_ID",
    region_col: str = "REGION",
    ds_col: str = "DS",
    sales_col: str = "SALES",
    class_col: str = "SERIES_CLASS",
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Run class-specific backtests and return one unified object.

    Model applicability rules:
    - Low data: naive only
    - Full series: all models
    - Zero-inflated: naive and tsb
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")

    classified_df = classify_time_series(
        df=df,
        inventory_col=inventory_col,
        region_col=region_col,
        sales_col=sales_col,
        class_col=class_col,
    )

    class_to_models = {
        "Low data": ["naive"],
        "Full series": ForecastModels.available_models(),
        "Zero-inflated": ["naive", "tsb"],
    }

    pairs = (
        classified_df[[inventory_col, region_col, class_col]]
        .drop_duplicates()
        .to_dict("records")
    )
    max_workers = None if n_jobs == -1 else max(1, n_jobs)

    def _run_one(pair: Dict[str, Any]):
        inventory_id = pair[inventory_col]
        region = pair[region_col]
        series_class = pair[class_col]
        models_for_class = class_to_models.get(series_class, ForecastModels.available_models())

        one_series = classified_df.loc[
            (classified_df[inventory_col] == inventory_id)
            & (classified_df[region_col] == region),
            [ds_col, sales_col],
        ].rename(columns={ds_col: "DS", sales_col: "SALES"})

        bt = backtest_models(
            one_series,
            th=th,
            leave_out_k=leave_out_k,
            conf=conf,
            model_names=models_for_class,
        )
        if bt is None:
            metrics = None
        else:
            metrics = {
                "point_results": bt["point_results"],
                "interval_results": bt["interval_results"],
            }
        return str(inventory_id), str(region), str(series_class), metrics

    results_by_class: Dict[str, Dict[str, Dict[str, Optional[Dict[str, Any]]]]] = {
        "Low data": {},
        "Full series": {},
        "Zero-inflated": {},
    }
    unified_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for inv_key, region_key, series_class, metrics in executor.map(_run_one, pairs):
            if inv_key not in results_by_class[series_class]:
                results_by_class[series_class][inv_key] = {}
            results_by_class[series_class][inv_key][region_key] = metrics

            if inv_key not in unified_results:
                unified_results[inv_key] = {}
            unified_results[inv_key][region_key] = {
                "series_class": series_class,
                "results": metrics,
            }

    return {
        "class_model_map": class_to_models,
        "results_by_class": results_by_class,
        "results": unified_results,
    }
