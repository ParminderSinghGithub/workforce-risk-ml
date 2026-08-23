"""Feature transformation and partition-aware encoding pipelines in PySpark."""

from typing import Any, Dict, List, Tuple
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, StringIndexerModel
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, IntegerType

from workforce_risk.features.definitions import (
    EXCLUDED_LEAKAGE_COLUMNS,
    FEATURE_DEFINITIONS,
)


def extract_structured_features(df: DataFrame) -> DataFrame:
    """Extract and derive all legitimate structured predictor features in PySpark.

    Explicitly excludes all leakage columns, target variables, and text.
    """
    # 1. Base numeric transformations and safe array metrics
    fe_df = df.withColumn(
        "num_technical_skills",
        F.size(F.coalesce(F.col("technical_skills"), F.array())).cast(DoubleType()),
    ).withColumn(
        "num_soft_skills",
        F.size(F.coalesce(F.col("soft_skills"), F.array())).cast(DoubleType()),
    ).withColumn(
        "tenure_years",
        (F.col("tenure_months") / 12.0).cast(DoubleType()),
    ).withColumn(
        "workload_stress_interaction",
        (F.col("workload_score") * F.col("stress_level")).cast(DoubleType()),
    ).withColumn(
        "satisfaction_workload_gap",
        (F.col("satisfaction_score") - F.col("workload_score")).cast(DoubleType()),
    ).withColumn(
        "overtime_intensity",
        (F.col("overtime_hours") / (F.col("overtime_hours") + 40.0)).cast(DoubleType()),
    ).withColumn(
        "engagement_score",
        (
            (F.col("slack_activity") + F.col("meeting_participation") + F.col("collaboration_score"))
            / 3.0
        ).cast(DoubleType()),
    )

    # 2. Binary target integer normalization
    if "left_company" in fe_df.columns:
        fe_df = fe_df.withColumn(
            "left_company",
            F.when(F.col("left_company") == True, 1).otherwise(0).cast(IntegerType()),
        )

    # 3. Cast numeric features to DoubleType
    raw_numeric_cols = [
        "tenure_months",
        "salary",
        "performance_score",
        "satisfaction_score",
        "workload_score",
        "team_sentiment",
        "project_completion_rate",
        "overtime_hours",
        "training_participation",
        "collaboration_score",
        "email_sentiment",
        "slack_activity",
        "meeting_participation",
        "goal_achievement_rate",
        "stress_level",
        "role_complexity_score",
        "career_progression_score",
    ]
    for c in raw_numeric_cols:
        if c in fe_df.columns:
            fe_df = fe_df.withColumn(c, F.col(c).cast(DoubleType()))

    return fe_df


def build_feature_pipeline(
    train_df: DataFrame,
    val_df: DataFrame,
    test_df: DataFrame,
) -> Tuple[DataFrame, DataFrame, DataFrame, Dict[str, Any]]:
    """Fit categorical StringIndexers exclusively on the training partition and transform all partitions.

    Guarantees no data leakage from validation or test partitions into categorical encodings.
    """
    categorical_cols = [
        "department",
        "job_level",
        "role",
        "communication_patterns",
        "persona_name",
    ]

    indexers = [
        StringIndexer(
            inputCol=col,
            outputCol=f"{col}_idx",
            handleInvalid="keep",
            stringOrderType="frequencyDesc",
        )
        for col in categorical_cols
    ]

    pipeline = Pipeline(stages=indexers)
    pipeline_model = pipeline.fit(train_df)

    # Transform all splits using the model fitted on train only
    train_trans = pipeline_model.transform(train_df)
    val_trans = pipeline_model.transform(val_df)
    test_trans = pipeline_model.transform(test_df)

    # Select final ordered schema: identifier, target, and all 29 structured features
    structured_feature_names = [
        # 17 Raw Numeric Features
        "tenure_months",
        "salary",
        "performance_score",
        "satisfaction_score",
        "workload_score",
        "team_sentiment",
        "project_completion_rate",
        "overtime_hours",
        "training_participation",
        "collaboration_score",
        "email_sentiment",
        "slack_activity",
        "meeting_participation",
        "goal_achievement_rate",
        "stress_level",
        "role_complexity_score",
        "career_progression_score",
        # 7 Derived Numeric Features
        "num_technical_skills",
        "num_soft_skills",
        "tenure_years",
        "workload_stress_interaction",
        "satisfaction_workload_gap",
        "overtime_intensity",
        "engagement_score",
        # 5 Categorical Indexed Features
        "department_idx",
        "job_level_idx",
        "role_idx",
        "communication_patterns_idx",
        "persona_name_idx",
    ]

    final_columns = ["employee_id", "left_company"] + structured_feature_names

    # Leakage assertion: ensure no forbidden columns are present in structured_feature_names
    forbidden_in_features = set(structured_feature_names).intersection(
        set(EXCLUDED_LEAKAGE_COLUMNS)
    )
    if forbidden_in_features:
        raise ValueError(
            f"LEAKAGE VIOLATION: Forbidden columns found in structured features: {forbidden_in_features}"
        )

    train_final = train_trans.select(final_columns)
    val_final = val_trans.select(final_columns)
    test_final = test_trans.select(final_columns)

    # Extract category vocabulary sizes from fitted stages
    vocab_metadata = {}
    for stage in pipeline_model.stages:
        if isinstance(stage, StringIndexerModel):
            input_c = stage.getInputCol()
            vocab_metadata[input_c] = {
                "num_classes": len(stage.labels),
                "labels": stage.labels[:20],  # Sample of top labels
            }

    metadata = {
        "structured_feature_names": structured_feature_names,
        "feature_count": len(structured_feature_names),
        "categorical_vocabularies": vocab_metadata,
    }

    return train_final, val_final, test_final, metadata


def prepare_text_dataset(df: DataFrame, burnout_threshold: float = 0.75) -> DataFrame:
    """Prepare clean text dataset records with binary burnout risk target label.

    Output Schema:
    - employee_id: StringType
    - recent_feedback: StringType
    - burnout_risk: DoubleType (continuous ground truth for auditing)
    - high_burnout_risk: IntegerType (1 if burnout_risk >= threshold else 0)
    """
    return df.select(
        F.col("employee_id"),
        F.col("recent_feedback"),
        F.col("burnout_risk").cast(DoubleType()),
        F.when(F.col("burnout_risk") >= burnout_threshold, 1)
        .otherwise(0)
        .cast(IntegerType())
        .alias("high_burnout_risk"),
    )
