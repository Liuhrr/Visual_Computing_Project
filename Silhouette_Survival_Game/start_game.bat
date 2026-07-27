@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Game environment not found.
    echo Run: python -m venv .venv
    echo Then: .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)
".venv\Scripts\python.exe" silhouette_game.py
if errorlevel 1 pause
