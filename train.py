"""Train the DQN agent inside SUMO.   Usage: python train.py [--steps 40000]"""
import argparse
from stable_baselines3 import DQN
from sumo_env import TrafficEnv

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=40_000, help="40k = quick test, 100k+ = final run")
args = ap.parse_args()

env = TrafficEnv(gui=False)
model = DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=50_000,
            learning_starts=1000, batch_size=64, gamma=0.95,
            exploration_fraction=0.4, exploration_final_eps=0.05,
            target_update_interval=500, verbose=1)
model.learn(total_timesteps=args.steps)
model.save("models/dqn_traffic")
env.close()
print("Saved models/dqn_traffic.zip")
