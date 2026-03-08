import pandas as pd
import numpy as np

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
