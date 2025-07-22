@echo off
echo ====================================================================
echo Egyptian High School Results Analyzer with Student Search
echo محلل نتائج الثانوية العامة مع البحث عن الطلاب
echo ====================================================================

echo.
echo Features / المميزات:
echo - Complete statistical analysis / تحليل إحصائي شامل
echo - Student search by National ID / البحث بالرقم القومي
echo - University qualification check / فحص التأهل للجامعات
echo - Rank and percentile calculation / حساب الترتيب والمئينية
echo - Interactive visualizations / رسوم بيانية تفاعلية
echo.

echo Starting Web Application...
echo Opening browser at: http://localhost:8501
echo.

B:/projects/Gradepersent/.venv/Scripts/python.exe -m streamlit run app.py --server.port 8501 --server.headless false
