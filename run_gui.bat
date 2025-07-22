@echo off
echo Starting Egyptian High School Results Analyzer...
echo.

if exist "grade_analyzer_gui.py" (
    echo Launching GUI version...
    python grade_analyzer_gui.py
) else (
    echo Error: grade_analyzer_gui.py not found!
    echo Please ensure you're in the correct directory.
    pause
)
