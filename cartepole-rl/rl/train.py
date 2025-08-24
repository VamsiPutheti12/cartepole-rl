from typing import List, Dict
import gymnasium as gym

from rl.agent import ReinforceAgent
from rl.utils import plot_rewards, set_global_seeds, ensure_dirs, save_policy

def train_agent(
    episodes: int = 500,
    gamma: float = 0.99,
    lr: float = 1e-2,
    entropy_beta: float = 0.01,
    seed: int = 42,
    render: bool = False,
) -> Dict[str, float]:
    """
    Train REINFORCE on CartPole-v1.
    Returns a dict with final stats and saves a plot + model.
    """
    set_global_seeds(seed)
    ensure_dirs()

    env = gym.make("CartPole-v1")  # render_mode="human" only if you want to watch
    env.reset(seed=seed)
    env.action_space.seed(seed)

    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n             # 2

    agent = ReinforceAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        entropy_beta=entropy_beta,
    )

    rewards_history: List[float] = []
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0.0

        while not (done or truncated):
            if render:
                env.render()

            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            agent.rewards.append(float(reward))  # store for returns
            total_reward += float(reward)
            state = next_state

        stats = agent.update_policy()  # single gradient step per episode
        rewards_history.append(total_reward)

        if (ep + 1) % 50 == 0:
            print(
                f"Ep {ep+1:4d} | Return {total_reward:6.1f} | "
                f"Loss {stats['loss']:.3f} | Ent {stats['entropy_bonus']:.3f}"
            )

    env.close()
    plot_rewards(rewards_history)
    save_policy(agent.policy)  # -> results/models/policy.pth
    print("Saved trained policy to results/models/policy.pth")
    return {
        "episodes": episodes,
        "avg_return_last_100": float(sum(rewards_history[-100:]) / max(1, len(rewards_history[-100:]))),
        "best_return": float(max(rewards_history)),
    }
