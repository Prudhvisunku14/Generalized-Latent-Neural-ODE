@echo off
cd /d "%~dp0\.."
python run_exp3.py --mode paper --require-cuda %*
