"""Compare DQN against Fixed-time and Longest-queue-first on unseen traffic patterns.
Usage: python evaluate.py [--episodes 8]
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from sumo_env import TrafficEnv, MAX_COUNT, NAMES


def run(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    done, t, info = False, 0, {}
    while not done:
        obs, _, _, done, info = env.step(policy(obs, t))
        t += 1
    return info["avg_wait"]


ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=8)
args = ap.parse_args()

env = TrafficEnv(gui=False)
model = DQN.load("models/dqn_traffic")

fixed = lambda obs, t: (t // 6) % env.n_phases        # rotate roughly every 30 s


def longest_queue(obs, t):                           # serve the phase whose approaches hold most cars
    scores = [sum(obs[NAMES.index(a)] for a in ap_) for ap_ in env.phase_approaches]
    return int(np.argmax(scores))


agent = lambda obs, t: int(model.predict(obs, deterministic=True)[0])

seeds = range(100, 100 + args.episodes)               # seeds never used in training
res = {"Fixed-time": [], "Longest-queue": [], "DQN": []}
for s in seeds:
    res["Fixed-time"].append(run(env, fixed, s))
    res["Longest-queue"].append(run(env, longest_queue, s))
    res["DQN"].append(run(env, agent, s))
    print(f"seed {s}:", {k: round(v[-1], 1) for k, v in res.items()})

means = {k: np.mean(v) for k, v in res.items()}
print("\n=== Average total waiting time (lower is better) ===")
for k, v in means.items():
    print(f"{k:15s} {v:8.1f}")
print(f"DQN improvement over Fixed-time: {100 * (means['Fixed-time'] - means['DQN']) / means['Fixed-time']:.1f}%")

plt.figure(figsize=(6, 4))
plt.bar(means.keys(), means.values(), yerr=[np.std(v) for v in res.values()],
        color=["gray", "orange", "green"], capsize=5)
plt.ylabel("Avg total waiting time (s)")
plt.title("Signal control comparison")
plt.tight_layout()
plt.savefig("results/result.png", dpi=150)
print("Saved results/result.png")
