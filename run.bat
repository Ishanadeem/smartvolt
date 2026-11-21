@echo off
title SmartVolt
echo ============================================
echo    Starting SmartVolt
echo ============================================
echo.

REM Free port 8000 if an old copy is still using it (single line - no block).
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1

echo Installing packages. The FIRST run can take a few minutes - please wait.
echo.
pip install -r requirements.txt

echo.
echo Opening your browser in a few seconds at http://127.0.0.1:8000
start "" cmd /c "timeout /t 7 >nul & start http://127.0.0.1:8000"

echo.
echo SmartVolt is running. Keep THIS window open while you use it.
echo Close this window to stop SmartVolt.
echo.
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

echo.
echo SmartVolt has stopped. Press any key to close.
pause >nul
