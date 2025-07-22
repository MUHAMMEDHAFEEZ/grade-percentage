@echo off
echo ===============================================
echo   Egyptian High School Results Analyzer 2025
echo   محلل نتائج الثانوية العامة المصرية
echo ===============================================
echo.
echo 🚀 Auto-Loading Student Data...
echo 📊 Ready for immediate student search!
echo 🔍 Just enter any student ID to get results
echo.

cd /d "%~dp0"

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Start Streamlit app with auto-loaded data
echo 🌐 Opening browser at http://localhost:8507
echo 📋 Data will be loaded automatically from results.csv
echo.
python -m streamlit run app.py --server.port 8507 --server.headless false

pause
