# AI-Based Adaptive Traffic Signal (YOLOv8 + DQN + SUMO)

Perception -> Decision -> Control:
- **Training/evaluation:** SUMO simulates an intersection; a DQN learns which phase to make green from per-approach vehicle counts.
- **Demo:** YOLOv8 counts vehicles per approach from video; the trained DQN recommends the green phase.

## Setup
1. Install **Python 3.10 or 3.11** (tick "Add to PATH" on Windows).
2. Run `setup.bat` (Windows) or `./setup.sh` (Linux/Mac). This creates `venv/` and installs everything,
   including SUMO via the `eclipse-sumo` pip package, so a separate SUMO install is optional.
   (For sumo-gui windows on Windows, installing the full SUMO from sumo.dlr.de is recommended; set `SUMO_HOME`.)
3. Activate the environment each session: `venv\Scripts\activate` (Windows) / `source venv/bin/activate`.
4. Check: `python sumo_env.py` -> should print "Self-check OK".

## Run order
| # | Command | What it does |
|---|---|---|
| 1 | `python sumo_env.py` | Sanity check of the simulator |
| 2 | `python train.py --steps 40000` | Quick train (10-20 min). Use `--steps 100000` for the final model |
| 3 | `python evaluate.py` | Fixed-time vs Longest-queue vs DQN, writes `results/result.png` |
| 4 | `python run_gui_demo.py` | Watch the agent control traffic in SUMO-GUI |
| 5 | Put a junction video at `videos/traffic.mp4` | Pexels/YouTube: elevated view of a 4-way junction |
| 6 | `python calibrate_zones.py videos/traffic.mp4` | Click 4 corners per approach (N,E,S,W), press `s` |
| 7 | `python yolo_counter.py videos/traffic.mp4` | Check counts look right |
| 8 | `python demo_live.py videos/traffic.mp4` | Full pipeline demo |

## Files
`sumo_env.py` environment | `train.py` | `evaluate.py` | `yolo_counter.py` | `demo_live.py` | `zones.json` zones | `sumo/` network files

## Troubleshooting
- **Connection / "Retrying in 1 seconds" lines:** normal while SUMO starts.
- **sumo-gui not found:** install full SUMO, or just skip GUI and use the plots.
- **Agent not beating baselines:** train longer (100k+ steps); check `ep_rew_mean` rises.
- **YOLO counts noisy/zero:** redo `calibrate_zones.py`, use `yolov8s.pt`, check video angle.
- **Counts far from SUMO range (0-20):** tune `--scale` in `demo_live.py`.

## Limitations (say these in your viva)
Sim-to-real gap (trained on simulated counts, run on video counts); single intersection; synthetic demand;
vehicles counted per frame (not tracked); demo recommends a phase rather than driving a real controller.
