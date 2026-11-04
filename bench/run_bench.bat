@echo off
setlocal
cd /d "%~dp0.."
set "LIMIT_ARG="
if not "%~1"=="" set "LIMIT_ARG=--limit %~1"
set "CACHE=%CD%\outputs\bench_cache"
if not exist "%CACHE%" mkdir "%CACHE%"
echo === bench_new: default settings (symmetric clipping, aliases on) ===
venv\Scripts\python.exe 1_data_extractor.py --mission Kepler --targets-file bench\targets_bench.csv %LIMIT_ARG% --workers 2 --cache-dir "%CACHE%" --output outputs\bench_new\extracted_features.csv
if errorlevel 1 exit /b 1
echo === bench_old: --no-resolve-aliases ===
venv\Scripts\python.exe 1_data_extractor.py --mission Kepler --targets-file bench\targets_bench.csv %LIMIT_ARG% --workers 2 --cache-dir "%CACHE%" --output outputs\bench_old\extracted_features.csv --no-resolve-aliases
if errorlevel 1 exit /b 1
endlocal
