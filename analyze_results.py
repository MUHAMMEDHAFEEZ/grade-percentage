#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Egyptian High School Results Analyzer - Command Line Version
Simple script for analyzing Egyptian high school student results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import re

# Set matplotlib to use Arabic-compatible font and better styling
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Tahoma', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

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

def load_and_process_data(file_path):
    """Load and process the CSV file"""
    print(f"Loading data from {file_path}...")
    
    # Try different delimiters
    for delimiter in [',', '\t', ';']:
        try:
            df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
            if len(df.columns) >= 3:
                print(f"Successfully loaded with delimiter: '{delimiter}'")
                break
        except:
            continue
    else:
        # If no delimiter works, try auto-detection
        df = pd.read_csv(file_path, encoding='utf-8')
        print("Loaded with auto-detected delimiter")
    
    print(f"Initial data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Convert Arabic numerals in total_degree column
    if 'total_degree' in df.columns:
        print("Converting Arabic numerals to English...")
        df['total_degree'] = df['total_degree'].apply(convert_arabic_numerals)
        df['total_degree'] = pd.to_numeric(df['total_degree'], errors='coerce')
        
        # Remove rows with invalid grades
        initial_count = len(df)
        df = df.dropna(subset=['total_degree'])
        df = df[df['total_degree'] > 0]  # Remove zero or negative grades
        
        print(f"Removed {initial_count - len(df)} invalid records")
        print(f"Final data shape: {df.shape}")
        
    return df

def calculate_statistics(df):
    """Calculate and display basic statistics"""
    print("\n" + "="*60)
    print("BASIC STATISTICS - الإحصائيات الأساسية")
    print("="*60)
    
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
    
    print(f"Total Students: {stats['count']:,}")
    print(f"Mean Score: {stats['mean']:.2f}")
    print(f"Median Score: {stats['median']:.2f}")
    print(f"Standard Deviation: {stats['std']:.2f}")
    print(f"Minimum Score: {stats['min']:.1f}")
    print(f"Maximum Score: {stats['max']:.1f}")
    print(f"25th Percentile: {stats['q25']:.2f}")
    print(f"75th Percentile: {stats['q75']:.2f}")
    
    return stats

def analyze_university_thresholds(df):
    """Analyze university admission thresholds"""
    thresholds = {
        'Medicine': 404,
        'Pharmacy': 397, 
        'Engineering': 382,
        'Commerce': 340,
        'Arts': 305
    }
    
    print("\n" + "="*60)
    print("UNIVERSITY THRESHOLDS ANALYSIS - تحليل حدود القبول الجامعي")
    print("="*60)
    
    threshold_data = {}
    for field, threshold in thresholds.items():
        count = len(df[df['total_degree'] >= threshold])
        percentage = (count / len(df)) * 100
        threshold_data[field] = {'count': count, 'percentage': percentage, 'threshold': threshold}
        print(f"{field} (≥{threshold}): {count:,} students ({percentage:.2f}%)")
    
    return threshold_data, thresholds

def show_top_students(df, n=10):
    """Display top N students"""
    print(f"\n" + "="*60)
    print(f"TOP {n} STUDENTS - أفضل {n} طلاب")
    print("="*60)
    
    top_students = df.nlargest(n, 'total_degree')[['seating_no', 'arabic_name', 'total_degree']]
    
    for idx, (_, student) in enumerate(top_students.iterrows(), 1):
        print(f"{idx:2d}. {student['arabic_name']} (ID: {student['seating_no']}) - {student['total_degree']:.1f}")
    
    return top_students

def plot_distribution_analysis(df, stats, thresholds, save_plots=True):
    """Create histogram and distribution plots"""
    print("\nCreating distribution analysis plots...")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Egyptian High School Results - Grade Distribution Analysis', 
                 fontsize=16, fontweight='bold')
    
    scores = df['total_degree']
    
    # 1. Histogram with statistics
    n, bins, patches = ax1.hist(scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["mean"]:.1f}')
    ax1.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f'Median: {stats["median"]:.1f}')
    ax1.set_xlabel('Total Degree')
    ax1.set_ylabel('Number of Students')
    ax1.set_title('Grade Distribution Histogram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot
    bp = ax2.boxplot(scores, vert=True, patch_artist=True, 
                     boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax2.set_ylabel('Total Degree')
    ax2.set_title('Grade Distribution Box Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. Histogram with thresholds
    ax3.hist(scores, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    for i, (field, threshold) in enumerate(thresholds.items()):
        ax3.axvline(threshold, color=colors[i % len(colors)], linestyle='--', 
                   linewidth=2, label=f'{field}: {threshold}')
    ax3.set_xlabel('Total Degree')
    ax3.set_ylabel('Number of Students')
    ax3.set_title('Grade Distribution with University Thresholds')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Density plot
    ax4.hist(scores, bins=50, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
    scores.plot.density(ax=ax4, color='darkgreen', linewidth=2)
    ax4.set_xlabel('Total Degree')
    ax4.set_ylabel('Density')
    ax4.set_title('Grade Density Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        output_dir = Path("analysis_output")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'grade_distribution_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / 'grade_distribution_analysis.png'}")
    
    plt.show()

def plot_top_students(df, n=10, save_plots=True):
    """Create bar chart of top students"""
    print(f"\nCreating top {n} students plot...")
    
    top_students = df.nlargest(n, 'total_degree')
    
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(top_students))
    bars = plt.barh(y_pos, top_students['total_degree'], color='gold', alpha=0.8, edgecolor='black')
    
    # Customize plot
    plt.yticks(y_pos, [f"{name[:25]}..." if len(name) > 25 else name 
                      for name in top_students['arabic_name']])
    plt.xlabel('Total Degree')
    plt.title(f'Top {n} Students - أفضل {n} طلاب', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, top_students['total_degree'])):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{score:.1f}', ha='left', va='center', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.gca().invert_yaxis()  # Top student at the top
    plt.tight_layout()
    
    if save_plots:
        output_dir = Path("analysis_output")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'top_students.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / 'top_students.png'}")
    
    plt.show()

def plot_threshold_analysis(threshold_data, save_plots=True):
    """Create plots for university threshold analysis"""
    print("\nCreating university threshold analysis plots...")
    
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
    
    if save_plots:
        output_dir = Path("analysis_output")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'university_thresholds.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / 'university_thresholds.png'}")
    
    plt.show()

def save_analysis_report(df, stats, threshold_data, top_students):
    """Save comprehensive analysis report to file"""
    print("\nSaving analysis report...")
    
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / 'analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("EGYPTIAN HIGH SCHOOL RESULTS ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
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
    
    print(f"Report saved: {report_path}")
    print(f"Top students CSV saved: {top_students_path}")

def main():
    """Main analysis function"""
    # File path - modify this to point to your CSV file
    file_path = "results.csv"
    
    print("Egyptian High School Results Analyzer")
    print("="*50)
    
    try:
        # Load and process data
        df = load_and_process_data(file_path)
        
        # Calculate statistics
        stats = calculate_statistics(df)
        
        # Analyze university thresholds
        threshold_data, thresholds = analyze_university_thresholds(df)
        
        # Show top students
        top_students = show_top_students(df, 10)
        
        print("\nGenerating visualizations...")
        
        # Create plots
        plot_distribution_analysis(df, stats, thresholds)
        plot_top_students(df, 10)
        plot_threshold_analysis(threshold_data)
        
        # Save comprehensive report
        save_analysis_report(df, stats, threshold_data, top_students)
        
        print("\n" + "="*60)
        print("Analysis completed successfully!")
        print("Check the 'analysis_output' folder for saved plots and reports.")
        print("="*60)
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        print("Please ensure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
