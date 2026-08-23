"""Data cleaning and normalization module using PySpark DataFrame operations."""

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, StringType


def clean_workforce_data(df: DataFrame) -> DataFrame:
    """Perform deterministic cleaning and schema normalization on the raw workforce DataFrame.

    Operations:
    1. Trims whitespace across all string columns.
    2. Normalizes empty strings in categorical columns to 'Unknown' or clean representation.
    3. Preserves `recent_feedback` natural language text with whitespace trimmed.
    4. Ensures array columns (skills) are non-null structured arrays.
    5. Deduplicates exact identical rows.
    6. Ensures strict type casting for all numeric, boolean, and categorical features.

    Args:
        df: Raw PySpark DataFrame conforming to get_raw_schema().

    Returns:
        Cleaned PySpark DataFrame.
    """
    cleaned_df = df

    # 1. String columns to trim
    string_cols = [
        "employee_id",
        "role",
        "job_level",
        "department",
        "recent_feedback",
        "communication_patterns",
        "turnover_reason",
        "risk_factors_summary",
        "persona_name",
    ]

    for col_name in string_cols:
        if col_name in cleaned_df.columns:
            cleaned_df = cleaned_df.withColumn(col_name, F.trim(F.col(col_name)))

    # 2. Normalize empty roles to 'Unknown' if whitespace-only was trimmed to empty
    if "role" in cleaned_df.columns:
        cleaned_df = cleaned_df.withColumn(
            "role",
            F.when((F.col("role") == "") | F.col("role").isNull(), F.lit("Unknown")).otherwise(
                F.col("role")
            ),
        )

    # 3. Array fields safety (ensure non-null arrays)
    array_cols = ["technical_skills", "soft_skills"]
    for col_name in array_cols:
        if col_name in cleaned_df.columns:
            cleaned_df = cleaned_df.withColumn(
                col_name,
                F.when(F.col(col_name).isNull(), F.array().cast(ArrayType(StringType()))).otherwise(
                    F.col(col_name)
                ),
            )

    # 4. Strict Boolean casting on target
    if "left_company" in cleaned_df.columns:
        cleaned_df = cleaned_df.withColumn(
            "left_company", F.col("left_company").cast("boolean")
        )

    # 5. Deduplicate exact duplicate rows
    cleaned_df = cleaned_df.dropDuplicates()

    return cleaned_df
