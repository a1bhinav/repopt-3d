import torch
import torch.nn as nn


class ResidualMLPAdapter(nn.Module):
    """Two-layer residual MLP adapter applied per-point to fused 3D features.

    Forward computes ``x + dropout(fc2(relu(norm(fc1(x)))))``. The output
    dimension equals the input dimension so the residual addition is well
    defined and downstream code (text-feature dot product) is unchanged.

    fc2's weight AND bias are zero-initialized, so at initialization the
    adapter is exactly the identity map: ``adapter(x) == x``. This is
    load-bearing for the trust-region experiment — the penalty
    ``||adapter(x) - x||`` starts at zero and grows only as the adapter
    learns to deviate, which lets us interpret the trust-region weight as
    a direct constraint on how far we move from the OpenScene fusion
    baseline.
    """

    def __init__(self, d_in: int = 768, d_hidden: int = 768, dropout: float = 0.0):
        super().__init__()
        self.d_in = d_in
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.norm = nn.LayerNorm(d_hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_in)
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    @property
    def feature_dim(self) -> int:
        return self.d_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.fc2(self.act(self.norm(self.fc1(x)))))
