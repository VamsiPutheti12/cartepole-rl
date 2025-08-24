import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    """
    A simple MLP that maps a CartPole state (R^4) to action logits (R^2).
    We return logits (unnormalized scores) so that the action distribution
    can be constructed with numerical stability via Categorical(logits=...).
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 120):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: shape (4,) or (batch, 4)
        returns: logits with shape (2,) or (batch, 2)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        logits = self.net(state)
        return logits.squeeze()