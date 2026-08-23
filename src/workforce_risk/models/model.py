"""PyTorch Multi-Layer Perceptron (MLP) architecture for structured tabular risk classification."""

from typing import List, Optional
import torch
import torch.nn as nn


class StructuredMLP(nn.Module):
    """Multi-Layer Perceptron for binary tabular risk prediction.

    Produces a single raw logit output for numerical stability with BCEWithLogitsLoss.
    Sigmoid is intentionally not applied in forward().
    """

    def __init__(
        self,
        input_dim: int = 29,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim

        # Final projection layer to 1 single classification logit
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing raw logit.

        Args:
            x: Input feature tensor of shape [batch_size, input_dim].

        Returns:
            Output logit tensor of shape [batch_size, 1].
        """
        return self.network(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute calibrated probabilities by applying sigmoid to raw logits."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    @property
    def total_parameters(self) -> int:
        """Return total number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
