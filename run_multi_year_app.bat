@echo off
echo ===============================================
echo   Egyptian High School Results Analyzer 2025
echo   محلل نتائج الثانوية العامة المصرية
echo ===============================================
echo.
echo 🚀 Starting application with multi-year support...
echo 📅 Default Year: 2025 (Total: 320)
echo 📊 Available Years: 2025, 2024, 2023
echo.

cd /d "%~dp0"

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Start Streamlit app
echo 🌐 Opening browser at http://localhost:8505
python -m streamlit run app.py --server.port 8505 --server.headless false

pause
