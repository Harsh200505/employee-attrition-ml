@echo off
cd /d "%~dp0"
if not exist ".mlenv\Scripts\python.exe" (
  echo The project Python environment is missing. Follow README.md to install it.
  pause
  exit /b 1
)
echo Open http://127.0.0.1:8501 in your browser after the demo starts.
".mlenv\Scripts\python.exe" demo.py
pause
