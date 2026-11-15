@echo off
cd /d "%~dp0\.."
python run_exp1.py --mode paper --require-cuda %*
