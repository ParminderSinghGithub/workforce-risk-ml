"""Unit tests for DistilBERT + PEFT/LoRA text classification infrastructure."""

from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from workforce_risk.nlp.dataset import FeedbackTextDataset, create_text_data_loaders
from workforce_risk.nlp.evaluate import evaluate_text_transformer
from workforce_risk.nlp.model import (
    create_lora_text_classifier,
    get_trainable_parameters_summary,
    load_lora_text_model,
    save_lora_text_model,
)


def _create_synthetic_text_parquet(file_path: Path, num_rows: int = 20) -> None:
    """Create small synthetic Parquet file conforming to text schema."""
    positive_phrases = [
        "Awful management, extreme hours and repetitive tasks.",
        "Terrible burnout, low morale and non-stop pressure.",
    ]
    negative_phrases = [
        "Great benefits, helpful colleagues and flexible schedule.",
        "Good work life balance, nice cafeteria and plenty of perks.",
    ]

    texts, labels, burnouts, emp_ids = [], [], [], []
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


def test_feedback_text_dataset_tokenization():
    """Verify FeedbackTextDataset tokenizes texts with correct shapes and labels."""
    texts = ["Great job and flexible team.", "Terrible stress and toxic environment."]
    labels = [0, 1]

    dataset = FeedbackTextDataset(texts=texts, labels=labels, max_length=64)
    assert len(dataset) == 2

    item0 = dataset[0]
    assert "input_ids" in item0 and "attention_mask" in item0 and "labels" in item0
    assert item0["input_ids"].shape == (64,)
    assert item0["attention_mask"].shape == (64,)
    assert item0["labels"].item() == 0.0


def test_lora_model_initialization_and_trainable_parameters():
    """Verify LoRA wraps DistilBERT with < 2% trainable parameters."""
    model = create_lora_text_classifier(r=8, lora_alpha=16)
    summary = get_trainable_parameters_summary(model)

    assert summary["trainable_percent"] < 2.0
    assert summary["trainable_params"] > 0
    assert summary["all_params"] > 60_000_000


def test_lora_model_forward_pass():
    """Verify forward pass returns single logit and sigmoid yields calibrated probabilities."""
    model = create_lora_text_classifier(r=8, lora_alpha=16)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    inputs = tokenizer(["Good balance.", "Severe burnout."], padding="max_length", max_length=32, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits)

    assert logits.shape == (2, 1)
    assert probs.shape == (2, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_lora_model_training_step():
    """Verify forward, loss computation, and backward step update adapter parameters."""
    model = create_lora_text_classifier(r=8, lora_alpha=16)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Find one LoRA adapter weight to track updates
    lora_weight = None
    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad:
            lora_weight = param
            break
    assert lora_weight is not None

    initial_weight = lora_weight.clone()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    inputs = tokenizer(["Sample text for training."], padding="max_length", max_length=16, return_tensors="pt")
    labels = torch.tensor([[1.0]], dtype=torch.float32)

    optimizer.zero_grad()
    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    loss = criterion(outputs.logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()

    updated_weight = lora_weight
    assert not torch.equal(initial_weight, updated_weight), "LoRA weights did not update during training step"


def test_lora_model_save_and_reload_identity(tmp_path: Path):
    """Verify saved LoRA model reloads and reproduces exact predictions."""
    model = create_lora_text_classifier(r=8, lora_alpha=16)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model.eval()

    inputs = tokenizer(["Check consistency."], padding="max_length", max_length=32, return_tensors="pt")
    with torch.no_grad():
        orig_logits = model(**inputs).logits

    save_dir = tmp_path / "lora_saved"
    save_lora_text_model(model, tokenizer, save_dir)

    reloaded_model, reloaded_tok = load_lora_text_model(save_dir, device="cpu")
    with torch.no_grad():
        reloaded_logits = reloaded_model(**inputs).logits

    assert torch.allclose(orig_logits, reloaded_logits, atol=1e-5)


def test_evaluate_text_transformer_loop(tmp_path: Path):
    """Verify evaluation loop computes finite loss and valid classification metrics."""
    test_parquet = tmp_path / "test_eval.parquet"
    _create_synthetic_text_parquet(test_parquet, num_rows=10)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = FeedbackTextDataset.from_parquet(test_parquet, tokenizer=tokenizer, max_length=32)
    loader = torch.utils.data.DataLoader(dataset, batch_size=5)

    model = create_lora_text_classifier(r=8, lora_alpha=16)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cpu")

    loss, metrics, y_true, y_prob = evaluate_text_transformer(model, loader, criterion, device)
    assert loss >= 0.0
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert len(y_true) == 10
    assert len(y_prob) == 10
