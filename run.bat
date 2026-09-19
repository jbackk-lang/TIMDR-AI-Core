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
if /I "%~1"=="--external-source" goto :external_source

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
