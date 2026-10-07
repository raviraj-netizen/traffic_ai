#!/usr/bin/env bash
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
echo "Setup done. Next: python sumo_env.py (quick check), then python train.py"
