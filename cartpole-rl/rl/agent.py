from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from rl.model import PolicyNetwork

class ReinforceAgent:
    """
    Vanilla REINFORCE with return normalization and optional entropy regularization.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-2,
        gamma: float = 0.99,
        entropy_beta: float = 0.01,
        grad_clip: float = 1.0,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.grad_clip = grad_clip

        # Storage for one episode
        self.saved_log_probs: List[torch.Tensor] = []
        self.saved_entropies: List[torch.Tensor] = []
        self.rewards: List[float] = []

    def select_action(self, state) -> int:
        """
        Sample an action from the current policy pi_theta(a|s).
        Save log-prob and entropy for the policy gradient update.
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        logits = self.policy(state_t)                         # shape (2,)
        dist = Categorical(logits=logits)                     # softmax under the hood
        action = dist.sample()                                # sample (encourages exploration)
        self.saved_log_probs.append(dist.log_prob(action))    # log pi_theta(a_t|s_t)
        self.saved_entropies.append(dist.entropy())           # H[pi(.|s_t)]
        return int(action.item())

    def _compute_returns(self) -> torch.Tensor:
        """
        Compute discounted returns G_t = sum_{k=0}^∞ gamma^k r_{t+k}.
        """
        R = 0.0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.append(R)
        returns.reverse()
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        # Normalize to reduce variance (helps stability)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        return returns_t

    def update_policy(self) -> dict:
        """
        REINFORCE update:
        minimize loss = - sum_t log pi_theta(a_t|s_t) * G_t  -  beta * sum_t H[pi(.|s_t)]
        """
        returns_t = self._compute_returns()
        log_probs_t = torch.stack(self.saved_log_probs)       # shape (T,)
        entropies_t = torch.stack(self.saved_entropies)       # shape (T,)

        policy_loss = -(log_probs_t * returns_t).sum()
        entropy_bonus = self.entropy_beta * entropies_t.sum()
        loss = policy_loss - entropy_bonus                    # maximize entropy => subtract in loss

        self.optimizer.zero_grad()
        loss.backward()

        # Optional gradient clipping for stability
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

        # Clear episode storage
        self.saved_log_probs.clear()
        self.saved_entropies.clear()
        self.rewards.clear()

        # For logging
        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "entropy_bonus": float(entropy_bonus.item()),
            "return_mean": float(returns_t.mean().item()),
        }
