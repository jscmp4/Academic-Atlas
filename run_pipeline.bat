@echo off
cd /d "%~dp0"

REM Kill anything on port 8050
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8050" ^| findstr "LISTENING"') do (
    if %%a NEQ 0 (
        taskkill /F /PID %%a >nul 2>&1
    )
)

REM Kill via PID file
if exist data\app.pid (
    set /p OLD_PID=<data\app.pid
    taskkill /F /PID %OLD_PID% >nul 2>&1
    del data\app.pid >nul 2>&1
)

timeout /t 1 /nobreak >nul
if exist __pycache__ rd /s /q __pycache__

echo Starting Research Landscape Map...
python app.py
pause
