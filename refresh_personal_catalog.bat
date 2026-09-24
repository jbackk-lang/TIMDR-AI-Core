@echo off
setlocal
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
set "CATALOG=%~1"
if "%CATALOG%"=="" set "CATALOG=online_catalogs.personal_repos.json"

if not exist "%PY%" (
    echo BLAD: brak lokalnego srodowiska .venv.
    echo Uruchom najpierw raz: .\run.bat
    pause
    exit /b 1
)

if not exist "%CATALOG%" (
    echo BLAD: brak katalogu %CATALOG%
    echo Utworz go poleceniem:
    echo   .venv\Scripts\python.exe examples\build_personal_repo_catalog.py
    pause
    exit /b 2
)

echo ============================================================
echo TIMDR-AI-Core - odswiezanie osobistego katalogu zrodel
echo Maksymalnie 4 porcje po 16 dokumentow, potem ranking i kolejka.
echo Nie tworzy hipotez, nie otwiera holdoutu, nie robi push.
echo ============================================================

for /L %%I in (1,1,4) do (
    echo.
    echo [Porcja %%I z 4]
    "%PY%" examples\skill_guided_learn.py "%CATALOG%"
    if errorlevel 1 goto :failed
)

echo.
echo [Ranking zrodel]
"%PY%" examples\rank_online_sources.py
if errorlevel 1 goto :failed

echo.
echo [Kolejka badawcza]
"%PY%" examples\build_research_queue.py
if errorlevel 1 goto :failed

echo.
echo Gotowe. Raporty sa w external_cache.
pause
exit /b 0

:failed
echo.
echo BLAD: etap zatrzymal sie. Poprzednie poprawnie zapisane porcje zostaja zachowane.
pause
exit /b 1
