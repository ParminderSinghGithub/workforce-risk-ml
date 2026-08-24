"""DistilBERT sequence classifier with PEFT/LoRA adaptation for text burnout risk modeling."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)


def create_lora_text_classifier(
    base_model_name: str = "distilbert-base-uncased",
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    num_labels: int = 1,
) -> PeftModel:
    """Initialize DistilBERT sequence classifier wrapped with LoRA adapters."""
    if target_modules is None:
        target_modules = ["q_lin", "v_lin"]

    # Initialize base sequence classifier (single logit output for BCEWithLogitsLoss)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )

    peft_model = get_peft_model(base_model, lora_config)
    return peft_model


def get_trainable_parameters_summary(model: nn.Module) -> Dict[str, Any]:
    """Calculate and return trainable vs frozen parameter counts."""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    return {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percent": round(100.0 * trainable_params / max(all_params, 1), 4),
    }


def save_lora_text_model(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str | Path,
    training_config: Optional[Dict[str, Any]] = None,
    base_model_name: str = "distilbert-base-uncased",
) -> Path:
    """Save LoRA adapter weights, base configuration, tokenizer, and metadata."""
    save_path = Path(output_dir).resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    # Save PEFT adapter weights and tokenizer
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    # Save metadata JSON
    metadata = {
        "base_model": base_model_name,
        "peft_type": "LORA",
        "parameter_summary": get_trainable_parameters_summary(model),
        "training_config": training_config or {},
    }
    with open(save_path / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return save_path


def load_lora_text_model(
    model_dir: str | Path,
    device: str | torch.device = "cpu",
    base_model_name: str = "distilbert-base-uncased",
) -> Tuple[PeftModel, PreTrainedTokenizerBase]:
    """Load persisted LoRA adapters and tokenizer from directory."""
    path = Path(model_dir).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model directory not found at: {path}")

    metadata_path = path / "model_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                base_model_name = meta.get("base_model", base_model_name)
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(str(path))
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,
    )
    model = PeftModel.from_pretrained(base_model, str(path))
    model.to(device)
    model.eval()

    return model, tokenizer
