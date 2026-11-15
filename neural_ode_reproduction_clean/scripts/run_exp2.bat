@echo off
cd /d "%~dp0\.."
python run_exp2.py --mode paper --require-cuda %*
