@echo off
setlocal
cd /d "%~dp0"

set "BASEPY="
python --version >nul 2>&1
if not errorlevel 1 set "BASEPY=python"
if defined BASEPY goto :python_found
py --version >nul 2>&1
if not errorlevel 1 set "BASEPY=py"
if defined BASEPY goto :python_found

echo BLAD: nie znaleziono Pythona.
echo Zainstaluj Python 3.10 lub nowszy i uruchom ten plik ponownie.
pause
exit /b 1

:python_found
if not exist ".venv\Scripts\python.exe" (
    echo Tworze lokalne srodowisko .venv...
    %BASEPY% -m venv .venv
    if errorlevel 1 goto :venv_failed
)

set "PY=.venv\Scripts\python.exe"
echo Uzywam: %PY%
%PY% --version

if /I "%~1"=="--tests" goto :install_dev
if /I "%~1"=="--graph" goto :build_graph
if /I "%~1"=="--learn" goto :learning_demo
if /I "%~1"=="--b4-boundary" goto :b4_boundary
if /I "%~1"=="--protect90-freeze" goto :protect90_freeze
if /I "%~1"=="--protect90" goto :protect90_train
if /I "%~1"=="--protect90-waveform-freeze" goto :protect90_waveform_freeze
if /I "%~1"=="--protect90-waveform" goto :protect90_waveform_train
if /I "%~1"=="--protect90-waveform-nn" goto :protect90_waveform_nn_train
if /I "%~1"=="--external-source" goto :external_source
if /I "%~1"=="--online-learn" goto :online_learn
if /I "%~1"=="--skill-learn" goto :skill_learn
if /I "%~1"=="--online-rank" goto :online_rank
if /I "%~1"=="--research-queue" goto :research_queue
if /I "%~1"=="--paderborn-freeze" goto :paderborn_freeze
if /I "%~1"=="--paderborn-schema" goto :paderborn_schema
if /I "%~1"=="--qwen4b-download" goto :qwen4b_download

echo.
echo Uruchamiam szybkie demo protokolu.
%PY% examples\protocol_demo.py
echo.
echo Pelne testy: run.bat --tests
pause
exit /b %ERRORLEVEL%

:build_graph
echo.
echo Buduje graf pochodzenia B4-Kitchen v0.3...
%PY% examples\build_b4_kitchen_graph.py
pause
exit /b %ERRORLEVEL%

:learning_demo
echo.
echo Uruchamiam sandbox uczenia bez dostepu do holdout...
%PY% examples\learning_demo.py
pause
exit /b %ERRORLEVEL%

:b4_boundary
echo.
echo Sprawdzam granice uczenia B4-Kitchen...
%PY% examples\inspect_b4_kitchen_learning_boundary.py
pause
exit /b %ERRORLEVEL%

:protect90_freeze
if "%~2"=="" goto :protect90_usage
echo.
echo Zamrazam prerejestracje PROTECT-90 przed pierwszym uczeniem...
%PY% examples\create_protect90_prereg.py "%~2"
pause
exit /b %ERRORLEVEL%

:protect90_train
if "%~2"=="" goto :protect90_usage
echo.
echo Uruchamiam wylacznie train/calibration zamrozonego planu PROTECT-90...
%PY% examples\train_protect90_multiclass.py "%~2"
pause
exit /b %ERRORLEVEL%

:protect90_usage
echo Uzycie:
echo   run.bat --protect90-freeze "C:\sciezka\TIMDR-Grid-Monitor"
echo   run.bat --protect90 "C:\sciezka\TIMDR-Grid-Monitor"
echo   run.bat --protect90-waveform-freeze "C:\sciezka\TIMDR-Grid-Monitor"
echo   run.bat --protect90-waveform "C:\sciezka\TIMDR-Grid-Monitor"
echo   run.bat --protect90-waveform-nn "C:\sciezka\TIMDR-Grid-Monitor"
pause
exit /b 2

:protect90_waveform_freeze
if "%~2"=="" goto :protect90_usage
echo.
echo Zamrazam osobna hipoteze cech z przebiegow EMT...
%PY% examples\create_protect90_waveform_prereg.py "%~2"
pause
exit /b %ERRORLEVEL%

:protect90_waveform_train
if "%~2"=="" goto :protect90_usage
%PY% -c "import numpy, pandas" >nul 2>&1
if errorlevel 1 goto :waveform_dependencies
echo.
echo Uruchamiam waveform train/calibration; holdout nie bedzie otwierany...
%PY% examples\train_protect90_waveform.py "%~2"
pause
exit /b %ERRORLEVEL%

:protect90_waveform_nn_train
if "%~2"=="" goto :protect90_usage
%PY% -c "import numpy" >nul 2>&1
if errorlevel 1 goto :waveform_dependencies
echo.
echo Uruchamiam siec MLP (numpy) na train/calibration; holdout nie bedzie otwierany...
%PY% examples\train_protect90_waveform_nn.py "%~2"
pause
exit /b %ERRORLEVEL%

:waveform_dependencies
echo Brakuje opcjonalnych zaleznosci numpy i pandas dla przebiegow EMT.
echo Zainstaluj je jawnie, a potem uruchom ponownie:
echo   .venv\Scripts\python.exe -m pip install -e ".[waveform]"
pause
exit /b 3

:external_source
if "%~3"=="" goto :external_source_usage
echo.
echo Pobieram tylko jawnie zadeklarowane i sumowane SHA-256 zrodlo...
%PY% examples\download_external_evidence.py "%~2" "%~3"
pause
exit /b %ERRORLEVEL%

:external_source_usage
echo Uzycie: run.bat --external-source "manifest.json" "source_id"
pause
exit /b 2

:online_learn
if "%~2"=="" goto :online_learn_usage
echo.
echo Uruchamiam automatyczny, ograniczony cykl uczenia z katalogow internetowych...
%PY% examples\online_learn.py "%~2"
pause
exit /b %ERRORLEVEL%

:online_learn_usage
echo Uzycie: run.bat --online-learn "online_catalogs.json"
pause
exit /b 2

:skill_learn
if "%~2"=="" goto :skill_learn_usage
echo.
echo Pobieram jawnie zadeklarowane zrodla i buduje kandydatow wedlug kapsuly TIMDR...
%PY% examples\skill_guided_learn.py "%~2"
pause
exit /b %ERRORLEVEL%

:skill_learn_usage
echo Uzycie: run.bat --skill-learn "online_catalogs.json"
pause
exit /b 2

:online_rank
echo.
echo Klasyfikuje zrodla jako kandydatow czterech galezi TIMDR...
%PY% examples\rank_online_sources.py
pause
exit /b %ERRORLEVEL%

:research_queue
echo.
echo Buduje kolejke kandydatow badawczych bez tworzenia hipotez...
%PY% examples\build_research_queue.py
pause
exit /b %ERRORLEVEL%

:paderborn_freeze
%PY% -c "import rarfile" >nul 2>&1
if errorlevel 1 goto :paderborn_dependencies
echo.
echo Zamrazam wybor Paderborn na podstawie samych nazw plikow archiwum...
%PY% examples\freeze_paderborn_dataset.py
pause
exit /b %ERRORLEVEL%

:paderborn_dependencies
echo Brakuje rarfile, potrzebnego tylko do spisu metadanych archiwum.
echo Zainstaluj: .venv\Scripts\python.exe -m pip install -e ".[paderborn]"
pause
exit /b 3

:paderborn_schema
%PY% -c "import numpy, scipy" >nul 2>&1
if errorlevel 1 goto :paderborn_analysis_dependencies
echo.
echo Sprawdzam kanal treningowy Paderborn; holdout jest blokowany...
%PY% examples\inspect_paderborn_train_schema.py
pause
exit /b %ERRORLEVEL%

:paderborn_analysis_dependencies
echo Brakuje numpy lub scipy dla technicznego odczytu MATLAB.
echo Zainstaluj: .venv\Scripts\python.exe -m pip install -e ".[paderborn-analysis]"
pause
exit /b 3

:qwen4b_download
%PY% -c "import huggingface_hub" >nul 2>&1
if errorlevel 1 goto :qwen4b_dependencies
echo.
echo Pobieram oficjalny Qwen3 4B Q4_K_M dla CPU i sprawdzam SHA-256...
%PY% examples\download_qwen3_4b_gguf.py
pause
exit /b %ERRORLEVEL%

:qwen4b_dependencies
echo Brakuje huggingface_hub, potrzebnego tylko do pobrania modelu.
echo Zainstaluj: .venv\Scripts\python.exe -m pip install huggingface_hub
pause
exit /b 3

:install_dev
%PY% -c "import pytest" >nul 2>&1
if not errorlevel 1 goto :run_tests
echo Instaluje zaleznosci testowe. Wymagane jest polaczenie z internetem...
%PY% -m pip install -e ".[dev]"
if errorlevel 1 goto :install_failed

:run_tests
%PY% -m pytest
pause
exit /b %ERRORLEVEL%

:venv_failed
echo BLAD: nie udalo sie utworzyc .venv.
pause
exit /b 1

:install_failed
echo BLAD: nie udalo sie zainstalowac pytest.
pause
exit /b 1
