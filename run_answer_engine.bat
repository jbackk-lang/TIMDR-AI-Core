@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" timdr_answer_ui.py
) else (
  python timdr_answer_ui.py
)
pause
