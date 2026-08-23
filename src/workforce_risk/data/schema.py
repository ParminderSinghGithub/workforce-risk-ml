"""Explicit PySpark StructType schema definition for raw workforce dataset."""

from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


def get_raw_schema() -> StructType:
    """Return the explicit PySpark schema for raw workforce JSON data.

    31 columns covering identifiers, demographics, compensation, performance,
    sentiment, multi-label skills, qualitative feedback, and outcome labels.
    """
    return StructType([
        StructField("employee_id", StringType(), nullable=False),
        StructField("role", StringType(), nullable=True),
        StructField("job_level", StringType(), nullable=True),
        StructField("department", StringType(), nullable=True),
        StructField("tenure_months", IntegerType(), nullable=True),
        StructField("salary", DoubleType(), nullable=True),
        StructField("performance_score", DoubleType(), nullable=True),
        StructField("satisfaction_score", DoubleType(), nullable=True),
        StructField("workload_score", DoubleType(), nullable=True),
        StructField("team_sentiment", DoubleType(), nullable=True),
        StructField("recent_feedback", StringType(), nullable=True),
        StructField("communication_patterns", StringType(), nullable=True),
        StructField("project_completion_rate", DoubleType(), nullable=True),
        StructField("overtime_hours", DoubleType(), nullable=True),
        StructField("training_participation", DoubleType(), nullable=True),
        StructField("collaboration_score", DoubleType(), nullable=True),
        StructField("technical_skills", ArrayType(StringType(), containsNull=True), nullable=True),
        StructField("soft_skills", ArrayType(StringType(), containsNull=True), nullable=True),
        StructField("email_sentiment", DoubleType(), nullable=True),
        StructField("slack_activity", DoubleType(), nullable=True),
        StructField("meeting_participation", DoubleType(), nullable=True),
        StructField("goal_achievement_rate", DoubleType(), nullable=True),
        StructField("stress_level", DoubleType(), nullable=True),
        StructField("burnout_risk", DoubleType(), nullable=True),
        StructField("left_company", BooleanType(), nullable=False),
        StructField("turnover_reason", StringType(), nullable=True),
        StructField("risk_factors_summary", StringType(), nullable=True),
        StructField("turnover_probability_generated", DoubleType(), nullable=True),
        StructField("persona_name", StringType(), nullable=True),
        StructField("role_complexity_score", DoubleType(), nullable=True),
        StructField("career_progression_score", DoubleType(), nullable=True),
    ])
