import os
import random
from typing import Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch

def set_global_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def plot_rewards(rewards: Iterable[float], filename: str = "results/plots/training_curve.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.figure()
    plt.plot(list(rewards))
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress (CartPole-v1)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def ensure_dirs():
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

def save_policy(model: torch.nn.Module, path: str = "results/models/policy.pth"):
    ensure_dirs()
    torch.save(model.state_dict(), path)

def load_policy(model: torch.nn.Module, path: str = "results/models/policy.pth") -> Tuple[bool, str]:
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return True, f"Loaded model from {path}"
    return False, f"No model found at {path}"
