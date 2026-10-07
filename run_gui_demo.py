"""Watch the trained agent control the intersection in SUMO-GUI.  Usage: python run_gui_demo.py"""
from stable_baselines3 import DQN
from sumo_env import TrafficEnv

env = TrafficEnv(gui=True, label="gui")
model = DQN.load("models/dqn_traffic")
obs, _ = env.reset(seed=999)
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, _, done, info = env.step(action)
print("Episode finished. Avg waiting:", info.get("avg_wait"))
