from rl.train import train_agent

if __name__ == "__main__":
    stats = train_agent(
        episodes=500,
        gamma=0.99,
        lr=1e-2,
        entropy_beta=0.01,
        seed=42,
        render=False,  # set True to watch (slower)
    )
    print("Training finished:", stats)
