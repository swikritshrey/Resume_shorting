@echo off
cd /d "%~dp0"
where python >nul 2>nul || (echo Python is not installed. Get it from https://www.python.org/downloads/ and tick Add python to PATH during setup. & pause & exit /b 1)
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
python -m pip install -q -r requirements.txt || (echo Install failed. Check your internet connection and try again. & pause & exit /b 1)
echo Starting. Your browser will open in a few seconds. Keep this window open while you use the app.
start "" cmd /c "timeout /t 4 >nul & start http://127.0.0.1:5000"
python app.py
pause
