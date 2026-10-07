"""SUMO single-intersection environment (Gymnasium API) for adaptive traffic signal control.

State  : [vehicle count on N, E, S, W approaches (scaled 0-1)] + one-hot of current green phase
Action : index of the green phase to serve next (a yellow phase is inserted on every switch)
Reward : decrease in total waiting time on the approaches (less waiting -> higher reward)
"""
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Locate SUMO: works with the installed SUMO (SUMO_HOME) or the pip package `eclipse-sumo`.
if "SUMO_HOME" not in os.environ:
    try:
        import sumo
        os.environ["SUMO_HOME"] = os.path.dirname(sumo.__file__)
    except ImportError:
        pass
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci
import sumolib

BASE = os.path.dirname(os.path.abspath(__file__))
NET = os.path.join(BASE, "sumo", "intersection.net.xml")
ROUTES = os.path.join(BASE, "sumo", "routes.rou.xml")

IN_EDGES = ["N_C", "E_C", "S_C", "W_C"]       # fixed order: N, E, S, W
NAMES = ["N", "E", "S", "W"]
TLS = "C"
MAX_PHASES = 8                                 # observation size is padded to this
YELLOW, STEP_GREEN, MAX_COUNT = 3, 5, 20       # seconds, seconds, count normaliser


def gen_routes(seed, sim_len=1800, path=ROUTES):
    """Random demand: 3 segments per episode, each with a different traffic level per approach."""
    rng = np.random.default_rng(seed)
    out, k = ["<routes>"], 0
    for seg in range(3):
        b, e = seg * (sim_len // 3), (seg + 1) * (sim_len // 3)
        level = {d: rng.uniform(0.04, 0.2) for d in NAMES}      # vehicles / second / approach
        for o in NAMES:
            for d in NAMES:
                if o != d:
                    out.append(f'<flow id="f{k}" begin="{b}" end="{e}" '
                               f'probability="{level[o] / 3:.4f}" from="{o}_C" to="C_{d}"/>')
                    k += 1
    out.append("</routes>")
    with open(path, "w") as f:
        f.write("\n".join(out))


class TrafficEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, gui=False, sim_len=1800, label="sim0"):
        super().__init__()
        self.gui, self.sim_len, self.label, self.running = gui, sim_len, label, False
        self._start(seed=0)                                   # probe the signal program once
        logic = traci.trafficlight.getAllProgramLogics(TLS)[0]
        self.green_states, self.yellow_states = [], []
        for p in logic.phases:
            s = p.state
            if ("G" in s or "g" in s) and "y" not in s.lower():
                self.green_states.append(s)
                self.yellow_states.append(s.replace("G", "y").replace("g", "y"))
        self.n_phases = len(self.green_states)
        self.phase_approaches = self._map_phases()
        self._close()
        print("Green phases found:")
        for i, (s, a) in enumerate(zip(self.green_states, self.phase_approaches)):
            print(f"  phase {i}: {s}  -> green for approaches {a}")
        self.action_space = spaces.Discrete(self.n_phases)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(4 + MAX_PHASES,), dtype=np.float32)

    # ---------- SUMO plumbing ----------
    def _map_phases(self):
        """Which approaches (N/E/S/W) get a green signal in each phase."""
        links = traci.trafficlight.getControlledLinks(TLS)
        res = []
        for s in self.green_states:
            ap = set()
            for i, group in enumerate(links):
                if s[i] in "Gg" and group:
                    ap.add(NAMES[IN_EDGES.index(group[0][0].rsplit("_", 1)[0])])
            res.append("".join(n for n in NAMES if n in ap))
        return res

    def _start(self, seed):
        gen_routes(seed, self.sim_len)
        exe = sumolib.checkBinary("sumo-gui" if self.gui else "sumo")
        cmd = [exe, "-n", NET, "-r", ROUTES, "--no-warnings", "--no-step-log",
               "--seed", str(seed), "--time-to-teleport", "-1"]
        if self.gui:
            cmd += ["--start", "--quit-on-end", "--delay", "100"]
        traci.start(cmd, label=self.label)
        traci.switch(self.label)
        self.running = True

    def _close(self):
        if self.running:
            traci.switch(self.label)
            traci.close()
            self.running = False

    def close(self):
        self._close()

    def _wait(self):
        return sum(traci.edge.getWaitingTime(e) for e in IN_EDGES)

    def _sim(self, n):
        for _ in range(n):
            traci.simulationStep()
            self.wait_log.append(self._wait())

    def counts(self):
        return [traci.edge.getLastStepVehicleNumber(e) for e in IN_EDGES]

    def _obs(self):
        c = [min(x / MAX_COUNT, 1.0) for x in self.counts()]
        onehot = np.zeros(MAX_PHASES)
        onehot[self.cur_idx] = 1
        return np.array(c + list(onehot), dtype=np.float32)

    # ---------- Gymnasium API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._close()
        self._start(seed if seed is not None else int(np.random.randint(1_000_000)))
        self.cur_idx, self.wait_log = 0, []
        traci.trafficlight.setRedYellowGreenState(TLS, self.green_states[0])
        self._sim(1)
        self.prev_wait = self._wait()
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        if action != self.cur_idx:                                   # switching -> yellow first
            traci.trafficlight.setRedYellowGreenState(TLS, self.yellow_states[self.cur_idx])
            self._sim(YELLOW)
            traci.trafficlight.setRedYellowGreenState(TLS, self.green_states[action])
            self.cur_idx = action
        self._sim(STEP_GREEN)
        w = self._wait()
        reward = (self.prev_wait - w) / 100.0
        self.prev_wait = w
        done = (traci.simulation.getTime() >= self.sim_len
                or traci.simulation.getMinExpectedNumber() <= 0)
        obs = self._obs()                      # read BEFORE closing SUMO
        info = {}
        if done:
            info = {"avg_wait": float(np.mean(self.wait_log))}
            self._close()
        return obs, reward, False, done, info


if __name__ == "__main__":      # quick self-check:  python sumo_env.py
    env = TrafficEnv()
    obs, _ = env.reset(seed=1)
    done, t = False, 0
    while not done:
        obs, r, _, done, info = env.step((t // 6) % env.n_phases)
        t += 1
    print("Self-check OK. Fixed-time avg waiting:", round(info["avg_wait"], 1))
