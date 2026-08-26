"""NLP package for employee feedback text modeling with DistilBERT and LoRA."""

from workforce_risk.nlp.model import (
    create_lora_text_classifier,
    get_trainable_parameters_summary,
    load_lora_text_model,
    save_lora_text_model,
)

__all__ = [
    "create_lora_text_classifier",
    "get_trainable_parameters_summary",
    "save_lora_text_model",
    "load_lora_text_model",
]
