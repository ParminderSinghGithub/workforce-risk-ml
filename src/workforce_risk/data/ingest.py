"""Data ingestion module for reading raw workforce JSON into PySpark DataFrames."""

import os
import sys
from pathlib import Path
from typing import Optional
from pyspark.sql import DataFrame, SparkSession

from workforce_risk.config import get_config
from workforce_risk.data.schema import get_raw_schema


def _ensure_environment() -> None:
    """Ensure JAVA_HOME, HADOOP_HOME, and PySpark python executables are configured."""
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    # 1. JAVA_HOME setup
    if "JAVA_HOME" not in os.environ:
        candidate_paths = [
            Path.home() / ".jdk" / "jdk-17",
            Path.home() / ".jdk" / "microsoft-jdk-17.0.14-windows-x64" / "jdk-17.0.14+7",
            Path("C:/Program Files/Eclipse Adoptium/jdk-17.0.20.8-hotspot"),
            Path("C:/Program Files/Microsoft/jdk-17"),
        ]
        jdk_base = Path.home() / ".jdk"
        if jdk_base.exists():
            for p in jdk_base.glob("**/bin/java.exe"):
                candidate_paths.insert(0, p.parent.parent)

        for p in candidate_paths:
            if p.exists() and (p / "bin" / "java.exe").exists():
                os.environ["JAVA_HOME"] = str(p)
                os.environ["PATH"] = f"{p / 'bin'};{os.environ.get('PATH', '')}"
                break

    # 2. HADOOP_HOME setup (for Windows winutils.exe and native dll)
    if "HADOOP_HOME" not in os.environ:
        candidate_hadoop = [
            Path.home() / ".hadoop",
        ]
        for hp in candidate_hadoop:
            if hp.exists() and (hp / "bin" / "winutils.exe").exists():
                os.environ["HADOOP_HOME"] = str(hp)
                os.environ["PATH"] = f"{hp / 'bin'};{os.environ.get('PATH', '')}"
                break


def get_spark_session(
    app_name: str = "WorkforceRiskPipeline",
    driver_memory: str = "4g",
    shuffle_partitions: int = 4,
) -> SparkSession:
    """Create and configure a local SparkSession with conservative developer machine defaults.

    Args:
        app_name: Name for the Spark application.
        driver_memory: Maximum memory allocated to the local Spark driver.
        shuffle_partitions: Default number of partitions for shuffles.

    Returns:
        Configured SparkSession instance.
    """
    _ensure_environment()

    return (
        SparkSession.builder.appName(app_name)
        .master("local[2]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.default.parallelism", str(shuffle_partitions))
        .config("spark.ui.enabled", "false")
        .config("spark.python.worker.reuse", "true")
        .getOrCreate()
    )


def ingest_raw_data(
    spark: SparkSession,
    raw_path: Optional[str | Path] = None,
) -> DataFrame:
    """Ingest raw workforce JSON data with strict explicit schema enforcement.

    Args:
        spark: Active SparkSession.
        raw_path: Path to the raw JSON file. Defaults to data/raw/synthetic-employee-dataset.json.

    Returns:
        PySpark DataFrame conforming to explicit raw schema.
    """
    if raw_path is None:
        config = get_config()
        raw_path = Path(config.paths.data_raw_dir) / config.dataset.raw_file

    path_str = str(Path(raw_path).resolve())
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Raw data file not found at: {path_str}")

    schema = get_raw_schema()

    # Read JSON array with multiLine=True and explicit schema
    df = (
        spark.read.schema(schema)
        .option("multiLine", "true")
        .option("mode", "FAILFAST")
        .json(path_str)
    )

    return df
