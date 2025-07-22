#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Egyptian High School Results Analyzer - Simple Console Version
Works without GUI dependencies - perfect for immediate analysis
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Set matplotlib to use Arabic-compatible font
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def convert_arabic_numerals(text):
    """Convert Arabic numerals to English numerals"""
    if pd.isna(text):
        return text
        
    # Arabic to English digit mapping
    arabic_to_english = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
        '٫': '.'  # Arabic decimal separator
    }
    
    result = str(text)
    for arabic, english in arabic_to_english.items():
        result = result.replace(arabic, english)
    
    try:
        return float(result)
    except (ValueError, TypeError):
        return text

def load_and_process_data(file_path="results.csv"):
    """Load and process the CSV file"""
    print(f"🔄 Loading data from {file_path}...")
    
    try:
        # Try different delimiters
        for delimiter in [',', '\t', ';']:
            try:
                df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
                if len(df.columns) >= 3:
                    print(f"✅ Successfully loaded with delimiter: '{delimiter}'")
                    break
            except:
                continue
        else:
            # If no delimiter works, try auto-detection
            df = pd.read_csv(file_path, encoding='utf-8')
            print("✅ Loaded with auto-detected delimiter")
        
        print(f"📊 Initial data shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Convert Arabic numerals in total_degree column
        if 'total_degree' in df.columns:
            print("🔄 Converting Arabic numerals to English...")
            df['total_degree'] = df['total_degree'].apply(convert_arabic_numerals)
            df['total_degree'] = pd.to_numeric(df['total_degree'], errors='coerce')
            
            # Remove rows with invalid grades
            initial_count = len(df)
            df = df.dropna(subset=['total_degree'])
            df = df[df['total_degree'] > 0]  # Remove zero or negative grades
            
            if initial_count > len(df):
                print(f"⚠️  Removed {initial_count - len(df)} invalid records")
            
            print(f"✅ Final data shape: {df.shape}")
            
        return df
        
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found.")
        print("📁 Please ensure the CSV file is in the same directory as this script.")
        return None
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def analyze_data(df):
    """Perform comprehensive data analysis"""
    if df is None:
        return
    
    # University thresholds
    thresholds = {
        'Medicine': 404,
        'Pharmacy': 397, 
        'Engineering': 382,
        'Commerce': 340,
        'Arts': 305
    }
    
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE ANALYSIS RESULTS")
    print("="*70)
    
    # Basic statistics
    stats = {
        'count': len(df),
        'mean': df['total_degree'].mean(),
        'median': df['total_degree'].median(),
        'std': df['total_degree'].std(),
        'min': df['total_degree'].min(),
        'max': df['total_degree'].max(),
        'q25': df['total_degree'].quantile(0.25),
        'q75': df['total_degree'].quantile(0.75)
    }
    
    print(f"\n📈 BASIC STATISTICS - الإحصائيات الأساسية")
    print("-" * 50)
    print(f"👥 Total Students: {stats['count']:,}")
    print(f"📊 Mean Score: {stats['mean']:.2f}")
    print(f"📊 Median Score: {stats['median']:.2f}")
    print(f"📊 Standard Deviation: {stats['std']:.2f}")
    print(f"📊 Minimum Score: {stats['min']:.1f}")
    print(f"📊 Maximum Score: {stats['max']:.1f}")
    print(f"📊 25th Percentile: {stats['q25']:.2f}")
    print(f"📊 75th Percentile: {stats['q75']:.2f}")
    
    # University thresholds analysis
    print(f"\n🎯 UNIVERSITY THRESHOLDS - حدود القبول الجامعي")
    print("-" * 60)
    
    threshold_data = {}
    for field, threshold in thresholds.items():
        count = len(df[df['total_degree'] >= threshold])
        percentage = (count / len(df)) * 100
        threshold_data[field] = {'count': count, 'percentage': percentage, 'threshold': threshold}
        print(f"🏛️  {field:12} (≥{threshold:3}): {count:6,} students ({percentage:5.2f}%)")
    
    # Top students
    print(f"\n🏆 TOP 10 STUDENTS - أفضل 10 طلاب")
    print("-" * 60)
    
    top_students = df.nlargest(10, 'total_degree')[['seating_no', 'arabic_name', 'total_degree']]
    for idx, (_, student) in enumerate(top_students.iterrows(), 1):
        name = student['arabic_name'][:30] + "..." if len(student['arabic_name']) > 30 else student['arabic_name']
        print(f"{idx:2d}. {name:<35} - {student['total_degree']:6.1f}")
    
    # Grade distribution
    print(f"\n📊 GRADE DISTRIBUTION - توزيع الدرجات")
    print("-" * 40)
    ranges = [(0, 200), (200, 250), (250, 300), (300, 350), (350, 380), (380, 400), (400, 410)]
    for start, end in ranges:
        count = len(df[(df['total_degree'] >= start) & (df['total_degree'] < end)])
        percentage = (count / len(df)) * 100
        print(f"📈 {start:3}-{end:3}: {count:6,} students ({percentage:5.2f}%)")
    
    return stats, threshold_data, top_students

def create_visualizations(df, stats, threshold_data):
    """Create and save visualization plots"""
    print(f"\n🎨 CREATING VISUALIZATIONS...")
    print("-" * 40)
    
    # Create output directory
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    scores = df['total_degree']
    
    # 1. Distribution Analysis
    print("📊 Creating distribution analysis...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Egyptian High School Results - Grade Distribution Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Histogram with statistics
    ax1.hist(scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["mean"]:.1f}')
    ax1.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f'Median: {stats["median"]:.1f}')
    ax1.set_xlabel('Total Degree')
    ax1.set_ylabel('Number of Students')
    ax1.set_title('Grade Distribution Histogram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(scores, vert=True, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax2.set_ylabel('Total Degree')
    ax2.set_title('Grade Distribution Box Plot')
    ax2.grid(True, alpha=0.3)
    
    # Histogram with thresholds
    ax3.hist(scores, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    thresholds = {'Medicine': 404, 'Pharmacy': 397, 'Engineering': 382, 'Commerce': 340, 'Arts': 305}
    for i, (field, threshold) in enumerate(thresholds.items()):
        ax3.axvline(threshold, color=colors[i % len(colors)], linestyle='--', 
                   linewidth=2, label=f'{field}: {threshold}')
    ax3.set_xlabel('Total Degree')
    ax3.set_ylabel('Number of Students')
    ax3.set_title('Grade Distribution with University Thresholds')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_scores = np.sort(scores)
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100
    ax4.plot(sorted_scores, cumulative, color='purple', linewidth=2)
    ax4.set_xlabel('Total Degree')
    ax4.set_ylabel('Cumulative Percentage')
    ax4.set_title('Cumulative Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'grade_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'grade_distribution_analysis.png'}")
    
    # 2. Top Students
    print("🏆 Creating top students chart...")
    top_students = df.nlargest(10, 'total_degree')
    
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(top_students))
    bars = plt.barh(y_pos, top_students['total_degree'], color='gold', alpha=0.8, edgecolor='black')
    
    plt.yticks(y_pos, [f"{name[:25]}..." if len(name) > 25 else name 
                      for name in top_students['arabic_name']])
    plt.xlabel('Total Degree')
    plt.title('Top 10 Students - أفضل 10 طلاب', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, top_students['total_degree'])):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{score:.1f}', ha='left', va='center', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'top_students.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'top_students.png'}")
    
    # 3. Threshold Analysis
    print("🎯 Creating threshold analysis...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('University Admission Threshold Analysis', fontsize=16, fontweight='bold')
    
    fields = list(threshold_data.keys())
    counts = [threshold_data[field]['count'] for field in fields]
    percentages = [threshold_data[field]['percentage'] for field in fields]
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    # Bar chart
    bars = ax1.bar(fields, counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Number of Students')
    ax1.set_title('Students Meeting University Thresholds')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Pie chart
    wedges, texts, autotexts = ax2.pie(percentages, labels=fields, autopct='%1.1f%%', 
                                      colors=colors, startangle=90)
    ax2.set_title('Percentage Distribution by University Thresholds')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'university_thresholds.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir / 'university_thresholds.png'}")

def save_comprehensive_report(df, stats, threshold_data, top_students):
    """Save comprehensive analysis report"""
    print(f"\n💾 SAVING COMPREHENSIVE REPORT...")
    print("-" * 40)
    
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate report
    from datetime import datetime
    report_path = output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("EGYPTIAN HIGH SCHOOL RESULTS ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data file: results.csv\n\n")
        
        # Basic statistics
        f.write("BASIC STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Students: {stats['count']:,}\n")
        f.write(f"Mean Score: {stats['mean']:.2f}\n")
        f.write(f"Median Score: {stats['median']:.2f}\n")
        f.write(f"Standard Deviation: {stats['std']:.2f}\n")
        f.write(f"Minimum Score: {stats['min']:.1f}\n")
        f.write(f"Maximum Score: {stats['max']:.1f}\n")
        f.write(f"25th Percentile: {stats['q25']:.2f}\n")
        f.write(f"75th Percentile: {stats['q75']:.2f}\n\n")
        
        # University thresholds
        f.write("UNIVERSITY ADMISSION ANALYSIS:\n")
        f.write("-" * 35 + "\n")
        for field, data in threshold_data.items():
            f.write(f"{field} (≥{data['threshold']}): {data['count']:,} students ({data['percentage']:.2f}%)\n")
        f.write("\n")
        
        # Top students
        f.write("TOP 20 STUDENTS:\n")
        f.write("-" * 20 + "\n")
        top_20 = df.nlargest(20, 'total_degree')
        for idx, (_, student) in enumerate(top_20.iterrows(), 1):
            f.write(f"{idx:2d}. {student['arabic_name']} (ID: {student['seating_no']}) - {student['total_degree']:.1f}\n")
    
    # Save top students as CSV
    top_students_path = output_dir / 'top_students.csv'
    df.nlargest(50, 'total_degree').to_csv(top_students_path, index=False, encoding='utf-8')
    
    print(f"✅ Report saved: {report_path}")
    print(f"✅ Top students CSV saved: {top_students_path}")

def main():
    """Main analysis function"""
    print("🎓 EGYPTIAN HIGH SCHOOL RESULTS ANALYZER")
    print("="*60)
    print("محلل نتائج الثانوية العامة المصرية")
    print("="*60)
    
    # Check if file exists
    file_path = "results.csv"
    if not Path(file_path).exists():
        print(f"❌ File '{file_path}' not found in current directory.")
        print("📁 Available files:")
        for f in Path(".").glob("*.csv"):
            print(f"   - {f.name}")
        
        # Try to find any CSV file
        csv_files = list(Path(".").glob("*.csv"))
        if csv_files:
            file_path = csv_files[0]
            print(f"🔄 Using: {file_path}")
        else:
            print("❌ No CSV files found. Please add your results.csv file to this directory.")
            input("Press Enter to exit...")
            return
    
    # Load and analyze data
    df = load_and_process_data(file_path)
    if df is not None:
        stats, threshold_data, top_students = analyze_data(df)
        
        # Create visualizations
        create_visualizations(df, stats, threshold_data)
        
        # Save comprehensive report
        save_comprehensive_report(df, stats, threshold_data, top_students)
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("📁 Check the 'analysis_output' folder for:")
        print("   📊 grade_distribution_analysis.png")
        print("   🏆 top_students.png") 
        print("   🎯 university_thresholds.png")
        print("   📄 analysis_report_[timestamp].txt")
        print("   📊 top_students.csv")
        print("="*70)
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
