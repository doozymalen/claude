@echo off
chcp 65001 > nul
cd /d "%~dp0"

where python > nul 2> nul
if errorlevel 1 (
    echo [오류] Python이 설치되어 있지 않습니다.
    echo python.org 에서 Python 3.11을 설치하세요. 설치할 때 "Add python.exe to PATH"를 꼭 체크하세요.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo 처음 실행: 필요한 구성요소를 설치합니다. 몇 분 걸릴 수 있습니다...
    python -m venv .venv
    if errorlevel 1 goto fail
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto fail
)

start "" ".venv\Scripts\pythonw.exe" launcher.py %*
exit /b 0

:fail
echo [오류] 설치 중 문제가 생겼습니다. 위 메시지를 확인하세요.
rmdir /s /q .venv 2> nul
pause
exit /b 1
