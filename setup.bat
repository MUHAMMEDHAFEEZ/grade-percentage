@echo off
echo Egyptian High School Results Analyzer Setup
echo ==========================================

echo.
echo Installing required Python packages...
echo.

pip install pandas>=1.5.0
pip install matplotlib>=3.6.0
pip install seaborn>=0.12.0
pip install numpy>=1.24.0
pip install streamlit>=1.28.0
pip install plotly>=5.15.0
pip install jupyter>=1.0.0
pip install ipywidgets>=8.0.0

echo.
echo Installation completed!
echo.
echo Available applications:
echo   Web App (Recommended): run_web_app.bat or python -m streamlit run streamlit_app.py
echo   Console Version: run_console.bat or python simple_analyzer.py
echo   Command Line: python analyze_results.py
echo.
pause
