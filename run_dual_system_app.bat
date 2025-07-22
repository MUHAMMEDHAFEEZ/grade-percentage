@echo off
echo ===============================================
echo   Egyptian High School Results Analyzer 2025
echo   محلل نتائج الثانوية العامة المصرية
echo ===============================================
echo.
echo 🎓 Two Education Systems Available:
echo 🆕 New System (النظام الجديد): 320 degrees
echo 📚 Old System (النظام القديم): 410 degrees
echo.
echo 🚀 Features:
echo ✅ Choose between New/Old systems
echo ✅ Multiple years for each system
echo ✅ Auto-loaded student data
echo ✅ Instant student search
echo.

cd /d "%~dp0"

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Start Streamlit app
echo 🌐 Opening browser at http://localhost:8509
echo 📊 Choose your education system and start searching!
echo.
python -m streamlit run app.py --server.port 8509 --server.headless false

pause
