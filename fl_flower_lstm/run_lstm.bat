@echo off
echo ========================================
echo   LSTM Federated Learning Demo
echo ========================================
echo.

echo Starting Flower server for LSTM...
start cmd /k "python server_lstm.py"

timeout /t 3 /nobreak >nul

echo Starting Client 1...
start cmd /k "python client_lstm1.py"

timeout /t 2 /nobreak >nul

echo Starting Client 2...
start cmd /k "python client_lstm2.py"

timeout /t 2 /nobreak >nul

echo Starting Client 3...
start cmd /k "python client_lstm3.py"

echo.
echo ========================================
echo   All clients started!
echo ========================================
echo.
echo Check the terminal windows:
echo   1. LSTM Server    - Coordinates training (5 rounds)
echo   2. LSTM Client 1  - Uses data1.csv with time series
echo   3. LSTM Client 2  - Uses data2.csv with time series
echo   4. LSTM Client 3  - Uses data3.csv with time series
echo.
echo Note: LSTM FL uses proper FedAvg weight averaging.
echo       Neural network weights are shared and aggregated.
echo       Each round improves the global LSTM model.
echo.
pause