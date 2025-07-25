# Egyptian High School Results Analyzer
# محلل نتائج الثانوية العامة المصرية

A comprehensive GUI application for analyzing Egyptian high school student results with Arabic numeral support.


## Screenshots / لقطات الشاشة

The application provides a comprehensive visual interface for analyzing Egyptian high school results:

### Main Interface / الواجهة الرئيسية
<table>
<tr>
<td width="50%">

![Main Application Interface](image.png)

**Features shown:**
- Clean, intuitive GUI layout
- File selection and loading area
- Statistical summary panel
- Analysis control buttons
- Arabic text support

</td>
<td width="50%">

![Data Loading and Statistics](image-1.png)

**Key elements:**
- CSV file browser integration
- Real-time data validation
- Comprehensive statistics display
- Student count and score metrics
- Mean, median, and distribution data

</td>
</tr>
</table>

### Analysis Visualizations / المخططات التحليلية
<table>
<tr>
<td width="33%">

![Grade Distribution Chart](image-2.png)

**Grade Distribution Analysis:**
- Histogram showing score frequency
- Normal distribution curve overlay
- Clear grade ranges and bins
- Statistical markers (mean, median)
- Color-coded visualization

</td>
<td width="33%">

![Box Plot Analysis](image-3.png)

**Box Plot Statistics:**
- Quartile distribution (Q1, Q2, Q3)
- Outlier detection and marking
- Median line highlighting
- Whiskers showing data range
- Statistical summary visualization

</td>
<td width="33%">

![University Thresholds](image-4.png)

**University Admission Analysis:**
- Threshold lines for each faculty
- Student distribution by admission level
- Medicine, Pharmacy, Engineering cuts
- Commerce and Arts requirements
- Eligibility percentage calculations

</td>
</tr>
</table>

### Detailed Features / المميزات التفصيلية

**Interface Components:**
- **File Management**: Easy CSV file selection with format validation
- **Statistics Panel**: Real-time calculation of key metrics including mean (المتوسط), median (الوسيط), and standard deviation (الانحراف المعياري)
- **Visualization Controls**: One-click generation of multiple chart types
- **Export Functionality**: High-quality PNG output and comprehensive reports

**Analysis Capabilities:**
- **Distribution Analysis**: Detailed histogram with statistical overlays showing how grades are distributed across the student population
- **Quartile Analysis**: Box plot visualization revealing data spread, outliers, and quartile boundaries
- **Threshold Evaluation**: University admission analysis with clear cutoff lines for different faculties
- **Performance Tracking**: Top student identification and ranking system

**Arabic Support Features:**
- Full RTL (Right-to-Left) text rendering
- Arabic numeral conversion (٠١٢٣٤٥٦٧٨٩ → 0123456789)
- Bilingual interface labels
- UTF-8 encoding support for Arabic names and data
## Features / المميزات

- **GUI Interface**: User-friendly graphical interface with Arabic support
- **Arabic Numerals**: Automatic conversion of Arabic numerals (٠١٢٣٤٥٦٧٨٩) to English
- **Statistical Analysis**: Mean, median, standard deviation, percentiles
- **University Thresholds**: Analysis based on typical admission requirements
- **Visualizations**: Histograms, box plots, bar charts, pie charts
- **Export Options**: Save plots as PNG and generate comprehensive reports
- **Top Students**: Display and export top performers

## University Thresholds / حدود القبول الجامعي

- Medicine (طب): 404 degrees
- Pharmacy (صيدلة): 397 degrees  
- Engineering (هندسة): 382 degrees
- Commerce (تجارة): 340 degrees
- Arts (آداب): 305 degrees

## Installation / التثبيت

### Prerequisites
- Python 3.8 or higher
- Windows/Linux/macOS

### Setup
1. Clone or download this project
2. Install required packages:
```bash
pip install -r requirements.txt
```

### Alternative installation:
```bash
pip install pandas matplotlib seaborn numpy
```

## Usage / الاستخدام

### GUI Version (Recommended)
```bash
python grade_analyzer_gui.py
```

### Command Line Version
```bash
python analyze_results.py
```

## CSV File Format / تنسيق ملف CSV

Your CSV file should have the following columns:
```
seating_no,arabic_name,total_degree
1001660,محمد ابو الحسن حسن مصطفى,163٫5
1001661,محمد احمد محمد ابو زيد,187٫5
```

- `seating_no`: Student ID number
- `arabic_name`: Student's Arabic name
- `total_degree`: Total score (supports Arabic numerals like ٣٨٢٫٥)

## Features Breakdown / تفصيل المميزات

### Statistical Analysis
- Total number of students
- Mean, median, mode
- Standard deviation
- Quartiles (25th, 75th percentile)
- Min/Max scores

### Visualizations
1. **Grade Distribution Histogram**: Shows how grades are distributed
2. **Box Plot**: Displays quartiles and outliers
3. **Top Students Chart**: Bar chart of highest performers
4. **University Thresholds**: Analysis of admission eligibility

### Export Options
- High-quality PNG plots (300 DPI)
- Comprehensive text reports
- Top students CSV export
- Automatic output folder creation

## GUI Instructions / تعليمات الواجهة

1. **Launch the application**: Run `python grade_analyzer_gui.py`
2. **Select CSV file**: Click "Browse" to choose your results file
3. **Load data**: Click "Load & Analyze" 
4. **View statistics**: Check the left panel for detailed statistics
5. **Generate plots**: Use the analysis buttons to create visualizations
6. **Save results**: Click "Save All Plots" to export everything

## Output Files / ملفات الخرج

The application creates an `analysis_output` folder containing:
- `grade_distribution_analysis.png`: Distribution plots
- `top_students.png`: Top performers chart
- `university_thresholds.png`: Threshold analysis
- `analysis_report.txt`: Comprehensive text report
- `top_students.csv`: Top 50 students data

## Troubleshooting / حل المشاكل

### Common Issues:

1. **File encoding errors**: Ensure your CSV file is saved as UTF-8
2. **Missing packages**: Run `pip install -r requirements.txt`
3. **Arabic text not displaying**: The app automatically handles Arabic fonts
4. **Large files**: The application can handle hundreds of thousands of records

### Performance Tips:
- For very large files (>500K records), use the command-line version
- Close other applications to free up memory
- Save plots in smaller batches if memory is limited

## Customization / التخصيص

### Modify University Thresholds:
Edit the `thresholds` dictionary in either script:
```python
thresholds = {
    'Medicine': 404,
    'Pharmacy': 397, 
    'Engineering': 382,
    'Commerce': 340,
    'Arts': 305
}
```

### Change Plot Colors:
Modify the `colors` list in the plotting functions:
```python
colors = ['red', 'orange', 'green', 'blue', 'purple']
```

## Technical Details / التفاصيل التقنية

- **Arabic Numeral Conversion**: Automatic conversion of ٠١٢٣٤٥٦٧٨٩ and ٫ (decimal)
- **Delimiter Detection**: Supports comma, tab, and semicolon delimiters
- **Error Handling**: Robust error handling for malformed data
- **Memory Efficient**: Optimized for large datasets
- **Unicode Support**: Full Arabic text support

## Sample Data Analysis / تحليل البيانات النموذجي

With the provided sample data (810,981 students), typical results show:
- Mean score: ~185-190
- Medicine eligibility: ~0.1-0.5%
- Engineering eligibility: ~2-5%
- Top scores: 400+ degrees

## Support / الدعم

For issues or questions:
1. Check the console output for error messages
2. Ensure your CSV file follows the required format
3. Verify all required packages are installed
4. Check that your Python version is 3.8+

## License / الرخصة

This project is open source and available for educational and research purposes.

---

**Created for analyzing Egyptian high school (Thanaweya Amma) results**
**تم إنشاؤه لتحليل نتائج الثانوية العامة المصرية**
