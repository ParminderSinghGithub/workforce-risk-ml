"""NLP package for employee feedback text modeling with DistilBERT and LoRA."""

from workforce_risk.nlp.baseline import TfidfTextBaseline, train_text_baseline
from workforce_risk.nlp.dataset import FeedbackTextDataset, create_text_data_loaders
from workforce_risk.nlp.evaluate import evaluate_text_transformer
from workforce_risk.nlp.model import (
    create_lora_text_classifier,
    get_trainable_parameters_summary,
    load_lora_text_model,
    save_lora_text_model,
)
from workforce_risk.nlp.train import train_text_transformer

__all__ = [
    "TfidfTextBaseline",
    "train_text_baseline",
    "FeedbackTextDataset",
    "create_text_data_loaders",
    "create_lora_text_classifier",
    "get_trainable_parameters_summary",
    "save_lora_text_model",
    "load_lora_text_model",
    "evaluate_text_transformer",
    "train_text_transformer",
]
