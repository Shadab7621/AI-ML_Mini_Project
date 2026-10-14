@echo off
rem One-time setup: creates the Python environment and installs the web packages.
rem Needs Python 3.10 or newer and Node.js 20.19 or newer.
cd /d "%~dp0"

echo [1/3] Creating the Python environment in backend\.venv ...
if exist "backend\.venv\Scripts\python.exe" goto :packages
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 -m venv backend\.venv
) else (
    python -m venv backend\.venv
)
if not exist "backend\.venv\Scripts\python.exe" goto :failed

:packages
echo [2/3] Installing the Python packages ...
backend\.venv\Scripts\python.exe -m pip install --disable-pip-version-check -r backend\requirements.txt
if errorlevel 1 goto :failed

echo [3/3] Installing the web packages ...
pushd frontend
call npm install
if errorlevel 1 (
    popd
    goto :failed
)
popd

echo.
echo Setup finished. Start the app with start.bat
pause
exit /b 0

:failed
echo.
echo Setup stopped because a step failed. Read the message above.
pause
exit /b 1
