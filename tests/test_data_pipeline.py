"""Unit tests for PySpark schema enforcement, cleaning, and validation on small in-memory DataFrames."""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from workforce_risk.data.schema import get_raw_schema
from workforce_risk.data.clean import clean_workforce_data
from workforce_risk.data.validate import validate_cleaned_data
from workforce_risk.data.ingest import get_spark_session


@pytest.fixture(scope="session")
def spark_session():
    """Create a shared local SparkSession for unit tests."""
    spark = get_spark_session(app_name="WorkforceRiskUnitTests", driver_memory="2g", shuffle_partitions=2)
    yield spark
    spark.stop()


def test_raw_schema_structure():
    """Verify that get_raw_schema returns exactly 31 expected fields with correct types."""
    schema = get_raw_schema()
    assert isinstance(schema, StructType)
    assert len(schema.fields) == 31

    field_names = [f.name for f in schema.fields]
    assert "employee_id" in field_names
    assert "left_company" in field_names
    assert "recent_feedback" in field_names
    assert "technical_skills" in field_names
    assert "soft_skills" in field_names
    assert "turnover_reason" in field_names
    assert "turnover_probability_generated" in field_names
    assert "risk_factors_summary" in field_names
    assert "burnout_risk" in field_names

    # Verify specific types
    assert schema["employee_id"].dataType == StringType()
    assert schema["left_company"].dataType == BooleanType()
    assert schema["salary"].dataType == DoubleType()
    assert schema["tenure_months"].dataType == IntegerType()
    assert isinstance(schema["technical_skills"].dataType, ArrayType)
    assert isinstance(schema["soft_skills"].dataType, ArrayType)


def test_cleaning_transformations_on_synthetic_data(spark_session: SparkSession):
    """Verify string trimming, empty role normalization, array safety, and deduplication on a small DataFrame."""
    schema = get_raw_schema()

    # Small synthetic rows (including untrimmed whitespace, empty role, duplicate row)
    data = [
        {
            "employee_id": " SYN_001 ",
            "role": "   ",  # whitespace only
            "job_level": " Mid ",
            "department": " Engineering ",
            "tenure_months": 24,
            "salary": 95000.0,
            "performance_score": 0.8,
            "satisfaction_score": 0.7,
            "workload_score": 0.6,
            "team_sentiment": 0.8,
            "recent_feedback": "  Great team culture and strong growth.  ",
            "communication_patterns": " Direct ",
            "project_completion_rate": 0.9,
            "overtime_hours": 5.0,
            "training_participation": 0.5,
            "collaboration_score": 0.8,
            "technical_skills": ["Python", "Spark"],
            "soft_skills": None,  # should normalize to empty list
            "email_sentiment": 0.7,
            "slack_activity": 0.8,
            "meeting_participation": 0.8,
            "goal_achievement_rate": 0.85,
            "stress_level": 0.4,
            "burnout_risk": 0.3,
            "left_company": False,
            "turnover_reason": " Not Applicable ",
            "risk_factors_summary": " Low Risk ",
            "turnover_probability_generated": 0.15,
            "persona_name": " SteadyEddy ",
            "role_complexity_score": 0.5,
            "career_progression_score": 0.8,
        },
        # Duplicate of row 1 (should be removed by dropDuplicates)
        {
            "employee_id": " SYN_001 ",
            "role": "   ",
            "job_level": " Mid ",
            "department": " Engineering ",
            "tenure_months": 24,
            "salary": 95000.0,
            "performance_score": 0.8,
            "satisfaction_score": 0.7,
            "workload_score": 0.6,
            "team_sentiment": 0.8,
            "recent_feedback": "  Great team culture and strong growth.  ",
            "communication_patterns": " Direct ",
            "project_completion_rate": 0.9,
            "overtime_hours": 5.0,
            "training_participation": 0.5,
            "collaboration_score": 0.8,
            "technical_skills": ["Python", "Spark"],
            "soft_skills": None,
            "email_sentiment": 0.7,
            "slack_activity": 0.8,
            "meeting_participation": 0.8,
            "goal_achievement_rate": 0.85,
            "stress_level": 0.4,
            "burnout_risk": 0.3,
            "left_company": False,
            "turnover_reason": " Not Applicable ",
            "risk_factors_summary": " Low Risk ",
            "turnover_probability_generated": 0.15,
            "persona_name": " SteadyEddy ",
            "role_complexity_score": 0.5,
            "career_progression_score": 0.8,
        },
        # Distinct row 2
        {
            "employee_id": "SYN_002",
            "role": "Software Engineer",
            "job_level": "Senior",
            "department": "Research & Development",
            "tenure_months": 60,
            "salary": 145000.0,
            "performance_score": 0.9,
            "satisfaction_score": 0.4,
            "workload_score": 0.9,
            "team_sentiment": 0.5,
            "recent_feedback": "Heavy overtime and unrealistic deadlines.",
            "communication_patterns": "Reserved",
            "project_completion_rate": 0.8,
            "overtime_hours": 20.0,
            "training_participation": 0.2,
            "collaboration_score": 0.5,
            "technical_skills": ["PyTorch", "Transformers"],
            "soft_skills": ["Leadership"],
            "email_sentiment": 0.3,
            "slack_activity": 0.5,
            "meeting_participation": 0.5,
            "goal_achievement_rate": 0.75,
            "stress_level": 0.9,
            "burnout_risk": 0.85,
            "left_company": True,
            "turnover_reason": "Burnout / Work-Life Balance",
            "risk_factors_summary": "Severe Burnout Risk",
            "turnover_probability_generated": 0.78,
            "persona_name": "OverachievingSprinter",
            "role_complexity_score": 0.9,
            "career_progression_score": 0.7,
        },
    ]

    raw_df = spark_session.createDataFrame(data, schema=schema)
    assert raw_df.count() == 3

    cleaned_df = clean_workforce_data(raw_df)

    # 1. Exact duplicate removed
    assert cleaned_df.count() == 2

    # 2. Check row values
    rows = {r["employee_id"]: r for r in cleaned_df.collect()}

    # SYN_001 checks
    r1 = rows["SYN_001"]
    assert r1["employee_id"] == "SYN_001"  # trimmed
    assert r1["role"] == "Unknown"          # empty whitespace normalized to Unknown
    assert r1["department"] == "Engineering" # trimmed
    assert r1["recent_feedback"] == "Great team culture and strong growth." # trimmed
    assert r1["soft_skills"] == []         # None normalized to empty list
    assert r1["left_company"] is False

    # SYN_002 checks
    r2 = rows["SYN_002"]
    assert r2["role"] == "Software Engineer"
    assert r2["soft_skills"] == ["Leadership"]
    assert r2["left_company"] is True


def test_validation_report_generation(spark_session: SparkSession):
    """Verify that validate_cleaned_data generates a complete report and catches errors."""
    schema = get_raw_schema()
    data = [
        {
            "employee_id": f"SYN_{i:03d}",
            "role": "Engineer",
            "job_level": "Mid",
            "department": "Engineering",
            "tenure_months": 12 * (i + 1),
            "salary": 80000.0 + (i * 5000),
            "performance_score": 0.7,
            "satisfaction_score": 0.8,
            "workload_score": 0.5,
            "team_sentiment": 0.8,
            "recent_feedback": "Standard feedback text.",
            "communication_patterns": "Direct",
            "project_completion_rate": 0.9,
            "overtime_hours": 0.0,
            "training_participation": 0.4,
            "collaboration_score": 0.7,
            "technical_skills": ["Python"],
            "soft_skills": ["Teamwork"],
            "email_sentiment": 0.8,
            "slack_activity": 0.7,
            "meeting_participation": 0.7,
            "goal_achievement_rate": 0.8,
            "stress_level": 0.3,
            "burnout_risk": 0.2,
            "left_company": (i % 2 == 1),
            "turnover_reason": "Not Applicable" if (i % 2 == 0) else "Career Opportunity",
            "risk_factors_summary": "Low Risk",
            "turnover_probability_generated": 0.2,
            "persona_name": "SteadyEddy",
            "role_complexity_score": 0.4,
            "career_progression_score": 0.8,
        }
        for i in range(4)
    ]

    df = spark_session.createDataFrame(data, schema=schema)
    report = validate_cleaned_data(df)

    assert report["dataset_validation"]["total_rows"] == 4
    assert report["dataset_validation"]["total_columns"] == 31
    assert report["dataset_validation"]["employee_id_unique"] is True
    assert report["target_validation"]["is_valid_binary"] is True
    assert report["target_validation"]["distribution"]["True"]["count"] == 2
    assert report["target_validation"]["distribution"]["False"]["count"] == 2
    assert report["text_validation"]["valid_count"] == 4
    assert report["leakage_columns_preserved"]["all_present"] is True


def test_validation_catches_missing_target(spark_session: SparkSession):
    """Verify that validation raises an error if the target column is missing."""
    schema = get_raw_schema()
    data = [{"employee_id": "SYN_001", "role": "Engineer", "salary": 80000.0}]
    minimal_schema = StructType([
        StructField("employee_id", StringType(), False),
        StructField("role", StringType(), True),
        StructField("salary", DoubleType(), True),
    ])
    df = spark_session.createDataFrame(data, schema=minimal_schema)
    with pytest.raises(ValueError, match="Target column 'left_company' missing"):
        validate_cleaned_data(df)
