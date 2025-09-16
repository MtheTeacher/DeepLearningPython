@echo off
cd /d "%~dp0"
cd ..
cd ..
py -m deeplearning_python.cli demo %*
pause
