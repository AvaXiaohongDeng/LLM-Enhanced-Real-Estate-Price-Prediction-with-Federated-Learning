@echo off
echo ========================================
echo   Random Forest Federated Learning Demo
echo ========================================
echo.

echo Starting Flower server for Random Forest...
start "RF Server" cmd /k "python server.py"

echo Waiting for server to initialize...
timeout /t 3 /nobreak >nul

echo Starting Client 1...
start "RF Client 1" cmd /k "python client1.py"

timeout /t 2 /nobreak >nul

echo Starting Client 2...
start "RF Client 2" cmd /k "python client2.py"

timeout /t 2 /nobreak >nul

echo Starting Client 3...
start "RF Client 3" cmd /k "python client3.py"

echo.
echo ========================================
echo   All clients started!
echo ========================================
echo.
echo Check the terminal windows:
echo   1. RF Server    - Coordinates training
echo   2. RF Client 1  - Uses data1.csv
echo   3. RF Client 2  - Uses data2.csv
echo   4. RF Client 3  - Uses data3.csv
echo.
echo Note: Random Forest FL exchanges hyperparameters only.
echo       Each client trains its own forest independently.
echo.
pause