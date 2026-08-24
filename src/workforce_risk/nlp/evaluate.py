"""Evaluation loop and metric calculations for transformer text classifier."""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from workforce_risk.models.evaluate import calculate_classification_metrics


def evaluate_text_transformer(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """Execute evaluation loop over text DataLoader and compute quantitative metrics."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch_size, 1]
            loss = criterion(logits, targets)

            batch_sz = input_ids.size(0)
            total_loss += loss.item() * batch_sz
            total_samples += batch_sz

            probs = torch.sigmoid(logits)

            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)
    y_true = np.vstack(all_targets).ravel()
    y_prob = np.vstack(all_probs).ravel()

    metrics = calculate_classification_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = round(avg_loss, 4)

    return avg_loss, metrics, y_true, y_prob
