"""Data quality validation module using PySpark aggregation without collecting to driver."""

from typing import Any, Dict
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from workforce_risk.config import get_config


def validate_cleaned_data(df: DataFrame) -> Dict[str, Any]:
    """Perform dataset-level, target-level, and text-level validation checks in PySpark.

    Args:
        df: Cleaned PySpark DataFrame.

    Returns:
        Structured dictionary containing data quality metrics and validation statuses.
    """
    config = get_config()
    total_rows = df.count()
    cols = df.columns
    total_cols = len(cols)

    # 1. Single-pass null counts across all columns
    null_exprs = [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in cols]
    null_row = df.select(null_exprs).first()
    null_summary = {c: int(null_row[c]) for c in cols} if null_row else {}

    # 2. Employee ID uniqueness check
    unique_emp_ids = df.select("employee_id").distinct().count()
    duplicate_emp_ids = total_rows - unique_emp_ids
    emp_id_is_unique = (duplicate_emp_ids == 0)

    # 3. Target distribution (left_company)
    target_col = config.targets.structured_target
    if target_col not in cols:
        raise ValueError(f"Target column '{target_col}' missing from cleaned dataset.")

    target_counts_df = df.groupBy(target_col).count().collect()
    target_distribution = {}
    for r in target_counts_df:
        k = str(r[target_col])
        cnt = int(r["count"])
        pct = round((cnt / total_rows) * 100, 4) if total_rows > 0 else 0.0
        target_distribution[k] = {"count": cnt, "percentage": pct}

    # Verify binary boolean values only
    valid_target_values = set(target_distribution.keys()).issubset({"True", "False", "true", "false", "1", "0"})
    if not valid_target_values:
        raise ValueError(f"Invalid target values detected in '{target_col}': {list(target_distribution.keys())}")

    # 4. Text column validation (recent_feedback)
    text_col = config.features.text_column
    if text_col not in cols:
        raise ValueError(f"Text column '{text_col}' missing from cleaned dataset.")

    text_metrics_row = df.select(
        F.count(F.when(F.col(text_col).isNotNull() & (F.col(text_col) != ""), text_col)).alias("valid_text"),
        F.count(F.when(F.col(text_col) == "", text_col)).alias("empty_text"),
        F.count_distinct(F.col(text_col)).alias("unique_text"),
    ).first()

    text_summary = {
        "text_column": text_col,
        "valid_count": int(text_metrics_row["valid_text"]) if text_metrics_row else 0,
        "empty_count": int(text_metrics_row["empty_text"]) if text_metrics_row else 0,
        "unique_count": int(text_metrics_row["unique_text"]) if text_metrics_row else 0,
    }

    # 5. Verify all mandatory leakage columns are preserved
    leakage_cols = config.features.leakage_exclusions
    missing_leakage_cols = [c for c in leakage_cols if c not in cols]
    if missing_leakage_cols:
        raise ValueError(f"Mandatory leakage columns missing from cleaned DataFrame: {missing_leakage_cols}")

    report = {
        "dataset_validation": {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "columns": cols,
            "employee_id_unique": emp_id_is_unique,
            "duplicate_employee_ids": duplicate_emp_ids,
            "null_summary": null_summary,
        },
        "target_validation": {
            "target_column": target_col,
            "is_valid_binary": valid_target_values,
            "distribution": target_distribution,
        },
        "text_validation": text_summary,
        "leakage_columns_preserved": {
            "expected": leakage_cols,
            "all_present": len(missing_leakage_cols) == 0,
        },
    }

    return report
