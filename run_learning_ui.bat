@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo BLAD: brak lokalnego srodowiska .venv. Uruchom najpierw raz: .\run.bat
    pause
    exit /b 1
)
echo Uruchamiam TIMDR Learning UI...
.venv\Scripts\python.exe learning_ui.py
if errorlevel 1 pause
