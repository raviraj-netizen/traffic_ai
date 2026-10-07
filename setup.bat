@echo off
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup done. Next: python sumo_env.py  (quick check), then python train.py
