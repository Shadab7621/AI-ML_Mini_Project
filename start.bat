@echo off
rem Starts the app: the Python API (port 8000) and the web page (port 5173).
rem Run setup.bat once before the first start.
cd /d "%~dp0"

if not exist "backend\.venv\Scripts\python.exe" (
    echo The Python environment is missing. Run setup.bat first.
    pause
    exit /b 1
)
if not exist "frontend\node_modules" (
    echo The web packages are missing. Run setup.bat first.
    pause
    exit /b 1
)

start "Classification Visualiser - API" cmd /k "backend\.venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000"
start "Classification Visualiser - Web" cmd /k "cd frontend && npm run dev -- --open"

echo.
echo Two windows opened: the API and the web page.
echo The app opens in your browser at http://localhost:5173
echo To stop the app, close both windows.
