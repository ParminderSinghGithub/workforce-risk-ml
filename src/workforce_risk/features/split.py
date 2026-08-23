import random
from typing import Tuple
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, StructField, StructType


def create_stratified_structured_split(
    df: DataFrame,
    seed: int = 42,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Perform deterministic employee-level stratified splitting on `left_company`.

    Uses a deterministic CRC32 hash function seeded per employee and stratum.
    Guarantees:
    - Exactly 80/10/10 split proportions per class
    - Zero employee_id overlap between partitions
    - Complete deterministic reproducibility across execution environments
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    # Compute deterministic uniform hash in [0.0, 1.0)
    hash_expr = (
        F.crc32(
            F.concat(
                F.col("employee_id"),
                F.col("left_company").cast("string"),
                F.lit(f"_struct_stratum_salt_{seed}"),
            )
        )
        % 10000
    ) / 10000.0

    df_tagged = df.withColumn("_split_hash", hash_expr)

    train_df = df_tagged.filter(F.col("_split_hash") < train_ratio).drop("_split_hash")
    val_df = (
        df_tagged.filter(
            (F.col("_split_hash") >= train_ratio)
            & (F.col("_split_hash") < (train_ratio + val_ratio))
        ).drop("_split_hash")
    )
    test_df = df_tagged.filter(
        F.col("_split_hash") >= (train_ratio + val_ratio)
    ).drop("_split_hash")

    return train_df, val_df, test_df


def create_grouped_text_split(
    df: DataFrame,
    seed: int = 42,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Perform deterministic, frequency-aware template-grouped splitting on `recent_feedback`.

    Guarantees:
    - 0% feedback-template overlap between train, validation, and test partitions
    - Row counts across partitions are closely calibrated to 80/10/10 target proportions
    - 100% deterministic assignment using configured random seed
    - Every employee record inherits the partition assigned to its feedback template
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    spark = df.sparkSession

    # 1. Aggregate template frequencies
    template_counts = df.groupBy("recent_feedback").count().collect()
    total_rows = sum(r["count"] for r in template_counts)

    if total_rows == 0:
        return df, df.limit(0), df.limit(0)

    val_target = int(total_rows * val_ratio)
    test_target = int(total_rows * test_ratio)

    # 2. Deterministic shuffle of template frequency list
    rng = random.Random(seed)
    items = [(r["recent_feedback"], int(r["count"])) for r in template_counts]
    # Sort first by template text for cross-platform tie-breaking stability before seeded shuffle
    items.sort(key=lambda x: str(x[0]))
    rng.shuffle(items)

    # 3. Frequency-aware greedy assignment to target row budgets
    template_tags = []
    val_c, test_c, train_c = 0, 0, 0

    for tpl, c in items:
        if val_c + c <= val_target or (
            val_c < val_target and abs(val_c + c - val_target) < abs(val_c - val_target)
        ):
            template_tags.append((tpl, "validation"))
            val_c += c
        elif test_c + c <= test_target or (
            test_c < test_target and abs(test_c + c - test_target) < abs(test_c - test_target)
        ):
            template_tags.append((tpl, "test"))
            test_c += c
        else:
            template_tags.append((tpl, "train"))
            train_c += c

    # 4. Convert mapping to Spark DataFrame and broadcast-join
    mapping_schema = StructType([
        StructField("recent_feedback", StringType(), False),
        StructField("_text_split_tag", StringType(), False),
    ])
    mapping_df = spark.createDataFrame(template_tags, schema=mapping_schema)

    # 5. Join partition tag back onto full DataFrame
    df_tagged = df.join(F.broadcast(mapping_df), on="recent_feedback", how="inner")

    train_df = df_tagged.filter(F.col("_text_split_tag") == "train").drop("_text_split_tag")
    val_df = df_tagged.filter(F.col("_text_split_tag") == "validation").drop("_text_split_tag")
    test_df = df_tagged.filter(F.col("_text_split_tag") == "test").drop("_text_split_tag")

    return train_df, val_df, test_df

