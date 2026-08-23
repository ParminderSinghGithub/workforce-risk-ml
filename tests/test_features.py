"""Unit tests for feature engineering, leakage prevention, and dataset splitting on small synthetic DataFrames."""

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
from workforce_risk.features.definitions import (
    EXCLUDED_LEAKAGE_COLUMNS,
    FEATURE_DEFINITIONS,
    generate_feature_manifest,
)
from workforce_risk.features.engineer import (
    build_feature_pipeline,
    extract_structured_features,
    prepare_text_dataset,
)
from workforce_risk.features.split import (
    create_grouped_text_split,
    create_stratified_structured_split,
)
from workforce_risk.data.ingest import get_spark_session


@pytest.fixture(scope="session")
def spark_session():
    """Create a shared local SparkSession for unit tests."""
    spark = get_spark_session(
        app_name="WorkforceRiskFeaturesTests",
        driver_memory="2g",
        shuffle_partitions=2,
    )
    yield spark
    spark.stop()


def test_feature_manifest_structure():
    """Verify feature manifest contains exact counts, allowable definitions, and leakage exclusions."""
    manifest = generate_feature_manifest()
    assert manifest["total_structured_features"] == 29
    assert manifest["feature_count_by_type"]["numeric"] == 24
    assert manifest["feature_count_by_type"]["categorical"] == 5
    assert len(manifest["features"]) == 29

    for col in [
        "employee_id",
        "turnover_reason",
        "turnover_probability_generated",
        "risk_factors_summary",
        "burnout_risk",
        "left_company",
        "recent_feedback",
    ]:
        assert col in manifest["excluded_columns"]["columns"]


def test_extract_structured_features(spark_session: SparkSession):
    """Verify derived interaction calculations, skill counting, and target integer normalization."""
    schema = get_raw_schema()
    data = [
        {
            "employee_id": "EMP_001",
            "role": "Software Engineer",
            "job_level": "Senior",
            "department": "Engineering",
            "tenure_months": 36,
            "salary": 120000.0,
            "performance_score": 0.8,
            "satisfaction_score": 0.6,
            "workload_score": 0.8,
            "team_sentiment": 0.7,
            "recent_feedback": "Great company.",
            "communication_patterns": "Direct",
            "project_completion_rate": 0.9,
            "overtime_hours": 10.0,
            "training_participation": 0.5,
            "collaboration_score": 0.8,
            "technical_skills": ["Python", "Spark", "PyTorch"],
            "soft_skills": ["Leadership", "Communication"],
            "email_sentiment": 0.6,
            "slack_activity": 0.7,
            "meeting_participation": 0.9,
            "goal_achievement_rate": 0.85,
            "stress_level": 0.75,
            "burnout_risk": 0.65,
            "left_company": True,
            "turnover_reason": "Career Growth",
            "risk_factors_summary": "High Workload",
            "turnover_probability_generated": 0.6,
            "persona_name": "TechLead",
            "role_complexity_score": 0.8,
            "career_progression_score": 0.7,
        }
    ]

    df = spark_session.createDataFrame(data, schema=schema)
    fe_df = extract_structured_features(df)
    row = fe_df.first()

    # Verify derived features
    assert row["num_technical_skills"] == 3.0
    assert row["num_soft_skills"] == 2.0
    assert abs(row["tenure_years"] - 3.0) < 1e-5
    assert abs(row["workload_stress_interaction"] - (0.8 * 0.75)) < 1e-5
    assert abs(row["satisfaction_workload_gap"] - (0.6 - 0.8)) < 1e-5
    assert abs(row["overtime_intensity"] - (10.0 / 50.0)) < 1e-5
    assert abs(row["engagement_score"] - ((0.7 + 0.9 + 0.8) / 3.0)) < 1e-5
    assert row["left_company"] == 1


def test_build_feature_pipeline_categorical_indexing(spark_session: SparkSession):
    """Verify StringIndexer fitting on training partition transforms all splits without leakage."""
    schema = get_raw_schema()
    train_data = [
        {
            "employee_id": f"EMP_{i:03d}",
            "role": "Engineer" if i % 2 == 0 else "Manager",
            "job_level": "Senior",
            "department": "Engineering",
            "tenure_months": 24,
            "salary": 100000.0,
            "performance_score": 0.8,
            "satisfaction_score": 0.7,
            "workload_score": 0.6,
            "team_sentiment": 0.7,
            "recent_feedback": "Text",
            "communication_patterns": "Direct",
            "project_completion_rate": 0.8,
            "overtime_hours": 5.0,
            "training_participation": 0.4,
            "collaboration_score": 0.7,
            "technical_skills": ["Python"],
            "soft_skills": ["Teamwork"],
            "email_sentiment": 0.7,
            "slack_activity": 0.6,
            "meeting_participation": 0.7,
            "goal_achievement_rate": 0.8,
            "stress_level": 0.5,
            "burnout_risk": 0.4,
            "left_company": (i % 3 == 0),
            "turnover_reason": "None",
            "risk_factors_summary": "Low",
            "turnover_probability_generated": 0.2,
            "persona_name": "Steady",
            "role_complexity_score": 0.6,
            "career_progression_score": 0.7,
        }
        for i in range(10)
    ]

    val_data = [dict(train_data[0], employee_id="VAL_001", role="Engineer")]
    test_data = [dict(train_data[0], employee_id="TST_001", role="Manager")]

    train_df = extract_structured_features(spark_session.createDataFrame(train_data, schema=schema))
    val_df = extract_structured_features(spark_session.createDataFrame(val_data, schema=schema))
    test_df = extract_structured_features(spark_session.createDataFrame(test_data, schema=schema))

    t_final, v_final, te_final, meta = build_feature_pipeline(train_df, val_df, test_df)

    assert meta["feature_count"] == 29
    assert len(t_final.columns) == 31  # employee_id + left_company + 29 features
    assert "department_idx" in t_final.columns
    assert "role_idx" in t_final.columns

    # Verify no raw leakage columns exist in final feature set
    for forbidden in ["turnover_reason", "burnout_risk", "turnover_probability_generated"]:
        assert forbidden not in t_final.columns


def test_prepare_text_dataset_threshold(spark_session: SparkSession):
    """Verify that high_burnout_risk is correctly derived based on burnout_threshold >= 0.75."""
    schema = get_raw_schema()
    data = [
        dict(
            {f.name: None for f in schema.fields},
            employee_id="E1",
            recent_feedback="Burnout is severe.",
            burnout_risk=0.85,
            left_company=True,
        ),
        dict(
            {f.name: None for f in schema.fields},
            employee_id="E2",
            recent_feedback="Work is fine.",
            burnout_risk=0.74,
            left_company=False,
        ),
        dict(
            {f.name: None for f in schema.fields},
            employee_id="E3",
            recent_feedback="High stress.",
            burnout_risk=0.75,
            left_company=False,
        ),
    ]

    df = spark_session.createDataFrame(data, schema=schema)
    text_df = prepare_text_dataset(df, burnout_threshold=0.75)
    rows = {r["employee_id"]: r["high_burnout_risk"] for r in text_df.collect()}

    assert rows["E1"] == 1  # 0.85 >= 0.75
    assert rows["E2"] == 0  # 0.74 < 0.75
    assert rows["E3"] == 1  # 0.75 >= 0.75


def test_stratified_structured_split_disjointness(spark_session: SparkSession):
    """Verify structured train/val/test splits are mutually disjoint and preserve all rows."""
    schema = get_raw_schema()
    data = [
        dict(
            {f.name: None for f in schema.fields},
            employee_id=f"EMP_{i:04d}",
            left_company=(i % 4 == 0),
        )
        for i in range(100)
    ]
    df = spark_session.createDataFrame(data, schema=schema)
    train_df, val_df, test_df = create_stratified_structured_split(
        df, seed=42, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )

    t_ids = {r["employee_id"] for r in train_df.collect()}
    v_ids = {r["employee_id"] for r in val_df.collect()}
    te_ids = {r["employee_id"] for r in test_df.collect()}

    assert len(t_ids) + len(v_ids) + len(te_ids) == 100
    assert len(t_ids.intersection(v_ids)) == 0
    assert len(t_ids.intersection(te_ids)) == 0
    assert len(v_ids.intersection(te_ids)) == 0


def test_grouped_text_split_no_template_overlap(spark_session: SparkSession):
    """Verify text template-grouped split produces strictly 0 template overlap across partitions."""
    schema = get_raw_schema()
    # 10 unique templates repeated across 100 employees
    templates = [f"Template {i}" for i in range(10)]
    data = [
        dict(
            {f.name: None for f in schema.fields},
            employee_id=f"EMP_{i:04d}",
            recent_feedback=templates[i % 10],
            burnout_risk=0.5,
            left_company=False,
        )
        for i in range(100)
    ]

    df = spark_session.createDataFrame(data, schema=schema)
    t_train, t_val, t_test = create_grouped_text_split(
        df, seed=42, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )

    train_tpls = {r["recent_feedback"] for r in t_train.collect()}
    val_tpls = {r["recent_feedback"] for r in t_val.collect()}
    test_tpls = {r["recent_feedback"] for r in t_test.collect()}

    assert len(train_tpls.intersection(val_tpls)) == 0
    assert len(train_tpls.intersection(test_tpls)) == 0
    assert len(val_tpls.intersection(test_tpls)) == 0
    assert t_train.count() + t_val.count() + t_test.count() == 100


def test_grouped_text_split_proportions_and_determinism(spark_session: SparkSession):
    """Verify that frequency-aware text splitting is deterministic and row-balanced close to 80/10/10."""
    schema = get_raw_schema()
    # 50 templates with variable frequency across 500 rows
    data = [
        dict(
            {f.name: None for f in schema.fields},
            employee_id=f"EMP_{i:04d}",
            recent_feedback=f"Feedback template {i % 50}",
            burnout_risk=0.6,
            left_company=False,
        )
        for i in range(500)
    ]

    df = spark_session.createDataFrame(data, schema=schema)

    # First run
    tr1, val1, te1 = create_grouped_text_split(
        df, seed=42, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
    tr1_count, val1_count, te1_count = tr1.count(), val1.count(), te1.count()

    # Verify row proportions close to 80/10/10
    assert tr1_count + val1_count + te1_count == 500
    assert abs((tr1_count / 500) - 0.80) <= 0.05
    assert abs((val1_count / 500) - 0.10) <= 0.05
    assert abs((te1_count / 500) - 0.10) <= 0.05

    # Second run (verify exact deterministic identity)
    tr2, val2, te2 = create_grouped_text_split(
        df, seed=42, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
    tr2_count, val2_count, te2_count = tr2.count(), val2.count(), te2.count()

    assert tr1_count == tr2_count
    assert val1_count == val2_count
    assert te1_count == te2_count

    tr1_ids = {r["employee_id"] for r in tr1.collect()}
    tr2_ids = {r["employee_id"] for r in tr2.collect()}
    assert tr1_ids == tr2_ids

