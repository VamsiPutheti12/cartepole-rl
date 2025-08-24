import torch
import gymnasium as gym

from rl.model import PolicyNetwork

def demo(render_mode: str = "human", episodes: int = 5):
    env = gym.make("CartPole-v1", render_mode=render_mode)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim)
    policy.load_state_dict(torch.load("results/models/policy.pth", map_location="cpu"))
    policy.eval()

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0.0

        while not (done or truncated):
            with torch.no_grad():
                logits = policy(torch.as_tensor(state, dtype=torch.float32))
                action = int(torch.argmax(logits).item())  # greedy at test-time
            state, reward, done, truncated, _ = env.step(action)
            total_reward += float(reward)

        print(f"Demo Episode {ep+1}: total reward = {total_reward:.1f}")

    env.close()

if __name__ == "__main__":
    demo()
