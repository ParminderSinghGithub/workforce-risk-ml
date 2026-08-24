"""Unit tests for TF-IDF + Logistic Regression text baseline and text audit utilities."""

from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from workforce_risk.nlp.baseline import TfidfTextBaseline, train_text_baseline


def _create_synthetic_text_parquet(file_path: Path, num_rows: int = 50) -> None:
    """Create small synthetic Parquet file conforming to text schema."""
    positive_phrases = [
        "Awful management, extreme hours and repetitive tasks.",
        "Terrible burnout, low morale and non-stop pressure.",
        "Cons: toxic workplace, lack of sleep and heavy workload.",
    ]
    negative_phrases = [
        "Great benefits, helpful colleagues and flexible schedule.",
        "Good work life balance, nice cafeteria and plenty of perks.",
        "Easily the best company, great tuition assistance and leadership.",
    ]

    texts = []
    labels = []
    burnouts = []
    emp_ids = []

    for i in range(num_rows):
        emp_ids.append(f"EMP_TXT_{i:04d}")
        if i % 2 == 0:
            texts.append(positive_phrases[i % len(positive_phrases)])
            labels.append(1)
            burnouts.append(0.85)
        else:
            texts.append(negative_phrases[i % len(negative_phrases)])
            labels.append(0)
            burnouts.append(0.30)

    table = pa.Table.from_pydict({
        "employee_id": emp_ids,
        "recent_feedback": texts,
        "burnout_risk": burnouts,
        "high_burnout_risk": labels,
    })
    pq.write_table(table, str(file_path))


def test_tfidf_baseline_fit_and_predict():
    """Verify TfidfTextBaseline trains, outputs calibrated probabilities, and extracts top n-grams."""
    train_texts = [
        "awful burnout long hours toxic",
        "terrible workload pressure repetitive",
        "great benefits good balance helpful",
        "flexible schedule nice perks easily",
    ]
    train_labels = [1, 1, 0, 0]

    model = TfidfTextBaseline(max_features=100, ngram_range=(1, 2))
    model.fit(train_texts, train_labels)
    assert model.is_fitted

    test_texts = [
        "terrible burnout and toxic environment",
        "great flexible benefits and good balance",
    ]
    probs = model.predict_proba(test_texts)
    assert len(probs) == 2
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    # High burnout phrase should have higher probability than healthy phrase
    assert probs[0] > probs[1]

    top_features = model.get_top_features(top_n=2)
    assert len(top_features["top_positive"]) == 2
    assert len(top_features["top_negative"]) == 2


def test_tfidf_baseline_save_and_load(tmp_path: Path):
    """Verify saved TfidfTextBaseline reloads and reproduces exact predictions."""
    train_texts = ["severe burnout and stress", "great team and great perks"]
    train_labels = [1, 0]

    model = TfidfTextBaseline(max_features=50)
    model.fit(train_texts, train_labels)

    test_input = ["burnout", "perks"]
    original_probs = model.predict_proba(test_input)

    save_path = tmp_path / "baseline_model.joblib"
    model.save(save_path)

    reloaded_model = TfidfTextBaseline.load(save_path)
    assert reloaded_model.is_fitted
    reloaded_probs = reloaded_model.predict_proba(test_input)

    assert np.allclose(original_probs, reloaded_probs, atol=1e-6)


def test_train_text_baseline_pipeline(tmp_path: Path):
    """Verify end-to-end text baseline training pipeline execution and artifact generation."""
    train_p = tmp_path / "text_train.parquet"
    val_p = tmp_path / "text_val.parquet"
    test_p = tmp_path / "text_test.parquet"
    art_dir = tmp_path / "artifacts"

    _create_synthetic_text_parquet(train_p, num_rows=40)
    _create_synthetic_text_parquet(val_p, num_rows=20)
    _create_synthetic_text_parquet(test_p, num_rows=20)

    summary = train_text_baseline(
        train_path=train_p,
        val_path=val_p,
        test_path=test_p,
        artifacts_dir=art_dir,
        max_features=50,
    )

    assert summary["model_type"] == "Tfidf_LogisticRegression_Baseline"
    assert (art_dir / "tfidf_baseline.joblib").exists()
    assert (art_dir / "text_baseline_summary.json").exists()
    assert summary["validation_metrics"]["roc_auc"] >= 0.5
    assert summary["test_metrics"]["roc_auc"] >= 0.5
