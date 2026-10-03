@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo 구성요소를 다시 설치합니다...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
pause
