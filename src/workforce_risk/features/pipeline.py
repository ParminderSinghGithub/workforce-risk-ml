"""Orchestration pipeline for feature engineering, leakage verification, dataset splitting, and manifest export."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from workforce_risk.config import get_config
from workforce_risk.data.ingest import get_spark_session
from workforce_risk.features.definitions import (
    EXCLUDED_LEAKAGE_COLUMNS,
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


def run_feature_pipeline(
    cleaned_parquet_path: Optional[str | Path] = None,
    output_splits_dir: Optional[str | Path] = None,
    reports_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Execute the full feature engineering, splitting, and manifest generation pipeline in PySpark.

    Args:
        cleaned_parquet_path: Path to cleaned Parquet dataset.
        output_splits_dir: Target directory for partitioned train/val/test Parquet files.
        reports_dir: Directory for feature and dataset manifests.

    Returns:
        Summary dictionary with execution metrics and dataset manifests.
    """
    config = get_config()
    t0 = time.time()

    # Resolve paths
    if cleaned_parquet_path is None:
        cleaned_parquet_path = Path(config.paths.data_processed_dir) / "cleaned_workforce.parquet"
    if output_splits_dir is None:
        output_splits_dir = Path(config.paths.data_splits_dir)
    if reports_dir is None:
        reports_dir = Path(config.paths.reports_dir) / "features"

    cleaned_parquet_path = Path(cleaned_parquet_path).resolve()
    output_splits_dir = Path(output_splits_dir).resolve()
    reports_dir = Path(reports_dir).resolve()

    output_splits_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("[Features] Initializing PySpark Session...")
    spark = get_spark_session(app_name="WorkforceRiskFeaturePipeline")

    print(f"[Features] Reading cleaned workforce dataset from: {cleaned_parquet_path}")
    cleaned_df = spark.read.parquet(str(cleaned_parquet_path))
    total_employees = cleaned_df.count()

    # =========================================================================
    # 1. STRUCTURED MODALITY: Extraction, Stratified Split, and Partition-Aware Encoding
    # =========================================================================
    print("[Features] Extracting structured features and derived interactions...")
    struct_extracted = extract_structured_features(cleaned_df)

    print("[Features] Performing deterministic stratified split on 'left_company'...")
    raw_s_train, raw_s_val, raw_s_test = create_stratified_structured_split(
        struct_extracted,
        seed=config.project.seed,
        train_ratio=config.splits.train_ratio,
        val_ratio=config.splits.val_ratio,
        test_ratio=config.splits.test_ratio,
    )

    print("[Features] Fitting categorical encoders on training partition only...")
    s_train_df, s_val_df, s_test_df, encoding_meta = build_feature_pipeline(
        raw_s_train, raw_s_val, raw_s_test
    )

    # =========================================================================
    # 2. TEXT MODALITY: Preparation and Feedback-Template Grouped Split
    # =========================================================================
    print("[Features] Preparing text dataset records with binary burnout labels...")
    text_records = prepare_text_dataset(
        cleaned_df, burnout_threshold=config.targets.burnout_threshold
    )

    print("[Features] Performing feedback-template grouped split...")
    t_train_df, t_val_df, t_test_df = create_grouped_text_split(
        text_records,
        seed=config.project.seed,
        train_ratio=config.splits.train_ratio,
        val_ratio=config.splits.val_ratio,
        test_ratio=config.splits.test_ratio,
    )

    # =========================================================================
    # 3. COMPREHENSIVE VALIDATION AND LEAKAGE AUDIT
    # =========================================================================
    print("[Features] Validating split integrity and auditing leakage...")

    # Structured row counts
    s_train_count = s_train_df.count()
    s_val_count = s_val_df.count()
    s_test_count = s_test_df.count()
    assert (
        s_train_count + s_val_count + s_test_count == total_employees
    ), f"Structured split rows ({s_train_count + s_val_count + s_test_count}) != total ({total_employees})"

    # Structured target distribution per split
    def get_struct_target_dist(df: DataFrame, total: int) -> Dict[str, Any]:
        rows = df.groupBy("left_company").count().collect()
        dist = {}
        for r in rows:
            lbl = str(r["left_company"])
            cnt = int(r["count"])
            pct = round((cnt / total) * 100, 4) if total > 0 else 0.0
            dist[lbl] = {"count": cnt, "percentage": pct}
        return dist

    s_train_dist = get_struct_target_dist(s_train_df, s_train_count)
    s_val_dist = get_struct_target_dist(s_val_df, s_val_count)
    s_test_dist = get_struct_target_dist(s_test_df, s_test_count)

    # Leakage assertions
    for split_name, df in [("train", s_train_df), ("val", s_val_df), ("test", s_test_df)]:
        forbidden_cols = set(df.columns).intersection(set(EXCLUDED_LEAKAGE_COLUMNS))
        # employee_id and left_company are in the DataFrame as ID & Target, but NOT as feature inputs
        forbidden_leakage = forbidden_cols.difference({"employee_id", "left_company"})
        if forbidden_leakage:
            raise ValueError(
                f"Leakage detected in structured {split_name} split: {forbidden_leakage}"
            )

    # Text row counts and template uniqueness
    t_train_count = t_train_df.count()
    t_val_count = t_val_df.count()
    t_test_count = t_test_df.count()
    assert (
        t_train_count + t_val_count + t_test_count == total_employees
    ), f"Text split rows ({t_train_count + t_val_count + t_test_count}) != total ({total_employees})"

    t_train_templates = t_train_df.select("recent_feedback").distinct().count()
    t_val_templates = t_val_df.select("recent_feedback").distinct().count()
    t_test_templates = t_test_df.select("recent_feedback").distinct().count()

    # Template disjointness check (0 overlap)
    train_val_overlap = (
        t_train_df.select("recent_feedback")
        .intersect(t_val_df.select("recent_feedback"))
        .count()
    )
    train_test_overlap = (
        t_train_df.select("recent_feedback")
        .intersect(t_test_df.select("recent_feedback"))
        .count()
    )
    val_test_overlap = (
        t_val_df.select("recent_feedback")
        .intersect(t_test_df.select("recent_feedback"))
        .count()
    )

    if train_val_overlap != 0 or train_test_overlap != 0 or val_test_overlap != 0:
        raise ValueError(
            f"TEMPLATE OVERLAP DETECTED! Train/Val: {train_val_overlap}, "
            f"Train/Test: {train_test_overlap}, Val/Test: {val_test_overlap}"
        )

    # Text target distribution per split
    def get_text_target_dist(df: DataFrame, total: int) -> Dict[str, Any]:
        rows = df.groupBy("high_burnout_risk").count().collect()
        dist = {}
        for r in rows:
            lbl = str(r["high_burnout_risk"])
            cnt = int(r["count"])
            pct = round((cnt / total) * 100, 4) if total > 0 else 0.0
            dist[lbl] = {"count": cnt, "percentage": pct}
        return dist

    t_train_dist = get_text_target_dist(t_train_df, t_train_count)
    t_val_dist = get_text_target_dist(t_val_df, t_val_count)
    t_test_dist = get_text_target_dist(t_test_df, t_test_count)

    # =========================================================================
    # 4. WRITE DOWNSTREAM PARQUET SPLITS
    # =========================================================================
    print(f"[Features] Writing split Parquet datasets to: {output_splits_dir}")

    s_train_path = output_splits_dir / "structured_train.parquet"
    s_val_path = output_splits_dir / "structured_validation.parquet"
    s_test_path = output_splits_dir / "structured_test.parquet"

    t_train_path = output_splits_dir / "text_train.parquet"
    t_val_path = output_splits_dir / "text_validation.parquet"
    t_test_path = output_splits_dir / "text_test.parquet"

    s_train_df.write.mode("overwrite").parquet(str(s_train_path))
    s_val_df.write.mode("overwrite").parquet(str(s_val_path))
    s_test_df.write.mode("overwrite").parquet(str(s_test_path))

    t_train_df.write.mode("overwrite").parquet(str(t_train_path))
    t_val_df.write.mode("overwrite").parquet(str(t_val_path))
    t_test_df.write.mode("overwrite").parquet(str(t_test_path))

    # =========================================================================
    # 5. GENERATE FEATURE & DATASET MANIFESTS
    # =========================================================================
    print(f"[Features] Generating manifests in: {reports_dir}")

    feature_manifest = generate_feature_manifest()
    feature_manifest_path = reports_dir / "feature_manifest.json"
    with open(feature_manifest_path, "w", encoding="utf-8") as f:
        json.dump(feature_manifest, f, indent=2)

    dataset_manifest = {
        "creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_employee_records": total_employees,
        "random_seed": config.project.seed,
        "structured_modality": {
            "strategy": "stratified_by_target",
            "target_variable": "left_company",
            "feature_count": encoding_meta["feature_count"],
            "features": encoding_meta["structured_feature_names"],
            "splits": {
                "train": {
                    "count": s_train_count,
                    "percentage": round((s_train_count / total_employees) * 100, 4),
                    "distribution": s_train_dist,
                    "parquet_path": str(s_train_path),
                },
                "validation": {
                    "count": s_val_count,
                    "percentage": round((s_val_count / total_employees) * 100, 4),
                    "distribution": s_val_dist,
                    "parquet_path": str(s_val_path),
                },
                "test": {
                    "count": s_test_count,
                    "percentage": round((s_test_count / total_employees) * 100, 4),
                    "distribution": s_test_dist,
                    "parquet_path": str(s_test_path),
                },
            },
        },
        "text_modality": {
            "strategy": "grouped_by_feedback_template",
            "text_field": "recent_feedback",
            "target_variable": "high_burnout_risk",
            "burnout_threshold": config.targets.burnout_threshold,
            "template_overlap_between_splits": 0,
            "splits": {
                "train": {
                    "count": t_train_count,
                    "percentage": round((t_train_count / total_employees) * 100, 4),
                    "unique_templates": t_train_templates,
                    "distribution": t_train_dist,
                    "parquet_path": str(t_train_path),
                },
                "validation": {
                    "count": t_val_count,
                    "percentage": round((t_val_count / total_employees) * 100, 4),
                    "unique_templates": t_val_templates,
                    "distribution": t_val_dist,
                    "parquet_path": str(t_val_path),
                },
                "test": {
                    "count": t_test_count,
                    "percentage": round((t_test_count / total_employees) * 100, 4),
                    "unique_templates": t_test_templates,
                    "distribution": t_test_dist,
                    "parquet_path": str(t_test_path),
                },
            },
        },
        "leakage_exclusions": {
            "excluded_columns": EXCLUDED_LEAKAGE_COLUMNS,
            "status": "VERIFIED_EXCLUDED",
        },
    }

    dataset_manifest_path = reports_dir / "dataset_manifest.json"
    with open(dataset_manifest_path, "w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, indent=2)

    elapsed_time = round(time.time() - t0, 2)
    print(
        f"[Features] Feature pipeline completed successfully in {elapsed_time}s. "
        f"Manifests saved to: {reports_dir}"
    )

    return {
        "status": "SUCCESS",
        "elapsed_seconds": elapsed_time,
        "dataset_manifest": dataset_manifest,
        "feature_manifest": feature_manifest,
    }


if __name__ == "__main__":
    run_feature_pipeline()
