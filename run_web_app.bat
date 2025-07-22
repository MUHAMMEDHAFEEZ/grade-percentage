@echo off
echo Starting Egyptian High School Results Analyzer (Web Version)...
echo ===============================================================

echo.
echo Launching Streamlit web application...
echo Your browser will open automatically at: http://localhost:8501
echo Press Ctrl+C to stop the application.
echo.

B:/projects/Gradepersent/.venv/Scripts/python.exe -m streamlit run streamlit_app.py --server.port 8501 --server.headless false
