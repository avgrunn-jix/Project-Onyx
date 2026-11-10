@echo off
setlocal
cd /d "%~dp0.."
set "LIMIT_ARG="
if not "%~1"=="" if not "%~1"=="-" set "LIMIT_ARG=--limit %~1"
set "TAG=%~2"
set "EXTRA="
shift
shift
:collect
if "%~1"=="" goto run
set "EXTRA=%EXTRA% %1"
shift
goto collect
:run
set "CACHE=%CD%\outputs\bench_cache"
if not exist "%CACHE%" mkdir "%CACHE%"
echo === bench_new%TAG%: default settings (symmetric clipping, aliases on)%EXTRA% ===
venv\Scripts\python.exe 1_data_extractor.py --mission Kepler --targets-file bench\targets_bench.csv %LIMIT_ARG% --workers 2 --cache-dir "%CACHE%" --output outputs\bench_new%TAG%\extracted_features.csv%EXTRA%
if errorlevel 1 exit /b 1
echo === bench_old%TAG%: --no-resolve-aliases%EXTRA% ===
venv\Scripts\python.exe 1_data_extractor.py --mission Kepler --targets-file bench\targets_bench.csv %LIMIT_ARG% --workers 2 --cache-dir "%CACHE%" --output outputs\bench_old%TAG%\extracted_features.csv --no-resolve-aliases%EXTRA%
if errorlevel 1 exit /b 1
endlocal
