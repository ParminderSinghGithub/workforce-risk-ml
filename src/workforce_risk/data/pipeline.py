"""End-to-end PySpark ingestion, cleaning, validation, and Parquet export pipeline."""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from workforce_risk.config import get_config
from workforce_risk.data.clean import clean_workforce_data
from workforce_risk.data.ingest import get_spark_session, ingest_raw_data
from workforce_risk.data.validate import validate_cleaned_data


def run_data_pipeline(
    raw_path: Optional[str | Path] = None,
    output_parquet_path: Optional[str | Path] = None,
    report_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Execute the full PySpark ingestion, cleaning, validation, and Parquet writing pipeline.

    Args:
        raw_path: Path to raw workforce JSON file.
        output_parquet_path: Destination path for cleaned Parquet dataset.
        report_path: Destination path for JSON data quality report.

    Returns:
        Summary dictionary containing execution metrics and validation report.
    """
    config = get_config()
    t0 = time.time()

    # Resolve paths
    if raw_path is None:
        raw_path = Path(config.paths.data_raw_dir) / config.dataset.raw_file
    if output_parquet_path is None:
        output_parquet_path = Path(config.paths.data_processed_dir) / "cleaned_workforce.parquet"
    if report_path is None:
        report_path = Path(config.paths.reports_dir) / "data_quality" / "data_quality_report.json"

    raw_path = Path(raw_path).resolve()
    output_parquet_path = Path(output_parquet_path).resolve()
    report_path = Path(report_path).resolve()

    # Ensure parent output directories exist
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Pipeline] Initializing SparkSession...")
    spark = get_spark_session(app_name="WorkforceRiskDataPipeline")

    print(f"[Pipeline] Ingesting raw JSON from: {raw_path}")
    raw_df = ingest_raw_data(spark, raw_path)

    print(f"[Pipeline] Applying cleaning transformations...")
    cleaned_df = clean_workforce_data(raw_df)

    print(f"[Pipeline] Validating cleaned DataFrame in PySpark...")
    val_report = validate_cleaned_data(cleaned_df)

    print(f"[Pipeline] Writing cleaned DataFrame to Parquet: {output_parquet_path}")
    # Write as partitioned / multi-part Parquet (native Spark write)
    cleaned_df.write.mode("overwrite").parquet(str(output_parquet_path))

    elapsed_time = round(time.time() - t0, 2)
    pipeline_summary = {
        "execution_status": "SUCCESS",
        "elapsed_seconds": elapsed_time,
        "raw_input_path": str(raw_path),
        "cleaned_output_path": str(output_parquet_path),
        "data_quality_report_path": str(report_path),
        "validation_report": val_report,
    }

    # Write JSON quality report
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_summary, f, indent=2)

    print(f"[Pipeline] Completed successfully in {elapsed_time}s. Report saved to: {report_path}")
    return pipeline_summary


if __name__ == "__main__":
    run_data_pipeline()
