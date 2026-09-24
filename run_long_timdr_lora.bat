@echo off
setlocal
cd /d "%~dp0"
echo Preparing frozen source corpus...
.venv\Scripts\python.exe examples\prepare_timdr_lora_corpus.py
if errorlevel 1 goto :end
echo Starting bounded CPU LoRA. The run checks RAM between steps and stops after four hours.
.venv\Scripts\python.exe examples\train_timdr_lora_corpus_cpu.py --epochs 1 --max-training-seconds 14400
:end
echo.
pause
endlocal
