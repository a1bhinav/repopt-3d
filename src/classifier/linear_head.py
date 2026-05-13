import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    """Plain Linear(d_in, num_classes). No softmax (cross-entropy handles it)."""

    def __init__(self, d_in: int, num_classes: int = 21):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(d_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
