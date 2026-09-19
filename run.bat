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
