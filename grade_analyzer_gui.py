#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Egyptian High School Results Analyzer
A comprehensive GUI application for analyzing Egyptian high school student results
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as plt_figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
import numpy as np
from pathlib import Path
import re
import threading
from datetime import datetime
import os

# Set matplotlib to use Arabic-compatible font
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Tahoma', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class GradeAnalyzer:
    def __init__(self):
        self.df = None
        self.stats = {}
        self.thresholds = {
            'Medicine': 404,
            'Pharmacy': 397, 
            'Engineering': 382,
            'Commerce': 340,
            'Arts': 305
        }
        
    def convert_arabic_numerals(self, text):
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
    
    def load_csv(self, file_path):
        """Load and process the CSV file"""
        try:
            # Try different delimiters
            for delimiter in [',', '\t', ';']:
                try:
                    df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
                    if len(df.columns) >= 3:
                        break
                except:
                    continue
            else:
                # If no delimiter works, try auto-detection
                df = pd.read_csv(file_path, encoding='utf-8')
            
            # Convert Arabic numerals in total_degree column
            if 'total_degree' in df.columns:
                df['total_degree'] = df['total_degree'].apply(self.convert_arabic_numerals)
                df['total_degree'] = pd.to_numeric(df['total_degree'], errors='coerce')
                
                # Remove rows with invalid grades
                df = df.dropna(subset=['total_degree'])
                df = df[df['total_degree'] > 0]  # Remove zero or negative grades
                
            self.df = df
            self.calculate_statistics()
            return True, f"Successfully loaded {len(df)} student records"
            
        except Exception as e:
            return False, f"Error loading file: {str(e)}"
    
    def calculate_statistics(self):
        """Calculate basic statistics"""
        if self.df is not None and 'total_degree' in self.df.columns:
            self.stats = {
                'count': len(self.df),
                'mean': self.df['total_degree'].mean(),
                'median': self.df['total_degree'].median(),
                'std': self.df['total_degree'].std(),
                'min': self.df['total_degree'].min(),
                'max': self.df['total_degree'].max(),
                'q25': self.df['total_degree'].quantile(0.25),
                'q75': self.df['total_degree'].quantile(0.75)
            }
    
    def get_threshold_counts(self):
        """Get count of students meeting each threshold"""
        if self.df is None:
            return {}
            
        threshold_counts = {}
        for field, threshold in self.thresholds.items():
            count = len(self.df[self.df['total_degree'] >= threshold])
            percentage = (count / len(self.df)) * 100
            threshold_counts[field] = {
                'count': count,
                'percentage': percentage,
                'threshold': threshold
            }
        return threshold_counts
    
    def get_top_students(self, n=10):
        """Get top N students"""
        if self.df is None:
            return pd.DataFrame()
        
        return self.df.nlargest(n, 'total_degree')[['seating_no', 'arabic_name', 'total_degree']]


class GradeAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Egyptian High School Results Analyzer - محلل نتائج الثانوية العامة")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize analyzer
        self.analyzer = GradeAnalyzer()
        
        # Variables
        self.file_path = tk.StringVar()
        
        # Setup GUI
        self.setup_gui()
        
        # Set icon (if available)
        try:
            self.root.iconbitmap('icon.ico')
        except:
            pass
    
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(2, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Egyptian High School Results Analyzer", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="File Selection - اختيار الملف", padding=10)
        file_frame.grid(row=1, column=0, columnspan=3, sticky='ew', pady=(0, 10))
        file_frame.grid_columnconfigure(1, weight=1)
        
        # File path entry
        ttk.Label(file_frame, text="CSV File:").grid(row=0, column=0, sticky='w')
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_path, width=50)
        self.file_entry.grid(row=0, column=1, sticky='ew', padx=(10, 10))
        
        # Browse button
        self.browse_btn = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        self.browse_btn.grid(row=0, column=2, sticky='w')
        
        # Load button
        self.load_btn = ttk.Button(file_frame, text="Load & Analyze", command=self.load_and_analyze)
        self.load_btn.grid(row=0, column=3, sticky='w', padx=(10, 0))
        
        # Main content area
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=2, column=0, columnspan=3, sticky='nsew')
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=2)
        
        # Left panel - Statistics and Controls
        left_panel = ttk.Frame(content_frame, width=400)
        left_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        left_panel.grid_propagate(False)
        left_panel.grid_rowconfigure(1, weight=1)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(left_panel, text="Statistics - الإحصائيات", padding=10)
        stats_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        
        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=15, width=45, wrap=tk.WORD)
        self.stats_text.grid(row=0, column=0, sticky='nsew')
        
        # Controls frame
        controls_frame = ttk.LabelFrame(left_panel, text="Analysis Controls - أدوات التحليل", padding=10)
        controls_frame.grid(row=1, column=0, sticky='ew', pady=(0, 10))
        
        # Analysis buttons
        self.plot_histogram_btn = ttk.Button(controls_frame, text="Plot Histogram", 
                                           command=self.plot_histogram, state='disabled')
        self.plot_histogram_btn.grid(row=0, column=0, sticky='ew', pady=2)
        
        self.plot_top_students_btn = ttk.Button(controls_frame, text="Plot Top Students", 
                                              command=self.plot_top_students, state='disabled')
        self.plot_top_students_btn.grid(row=1, column=0, sticky='ew', pady=2)
        
        self.plot_thresholds_btn = ttk.Button(controls_frame, text="Plot Thresholds", 
                                            command=self.plot_thresholds, state='disabled')
        self.plot_thresholds_btn.grid(row=2, column=0, sticky='ew', pady=2)
        
        self.save_plots_btn = ttk.Button(controls_frame, text="Save All Plots", 
                                       command=self.save_all_plots, state='disabled')
        self.save_plots_btn.grid(row=3, column=0, sticky='ew', pady=2)
        
        self.export_report_btn = ttk.Button(controls_frame, text="Export Report", 
                                          command=self.export_report, state='disabled')
        self.export_report_btn.grid(row=4, column=0, sticky='ew', pady=2)
        
        # Configure controls frame grid
        controls_frame.grid_columnconfigure(0, weight=1)
        
        # Right panel - Plots
        right_panel = ttk.Frame(content_frame)
        right_panel.grid(row=0, column=1, sticky='nsew')
        right_panel.grid_rowconfigure(0, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)
        
        # Notebook for plots
        self.plot_notebook = ttk.Notebook(right_panel)
        self.plot_notebook.grid(row=0, column=0, sticky='nsew')
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - اختر ملف CSV للبدء")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=3, column=0, columnspan=3, sticky='ew', pady=(10, 0))
    
    def browse_file(self):
        """Browse for CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=os.getcwd()
        )
        if file_path:
            self.file_path.set(file_path)
    
    def load_and_analyze(self):
        """Load and analyze the CSV file"""
        if not self.file_path.get():
            messagebox.showerror("Error", "Please select a CSV file first")
            return
        
        # Disable buttons during loading
        self.load_btn.configure(state='disabled')
        self.status_var.set("Loading file... جاري تحميل الملف...")
        self.root.update()
        
        # Load in a separate thread to prevent GUI freezing
        def load_thread():
            success, message = self.analyzer.load_csv(self.file_path.get())
            
            # Update GUI in main thread
            self.root.after(0, lambda: self.after_load(success, message))
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    def after_load(self, success, message):
        """Handle after loading completion"""
        self.load_btn.configure(state='normal')
        
        if success:
            self.status_var.set(f"Loaded successfully - {message}")
            self.update_statistics()
            self.enable_analysis_buttons()
            messagebox.showinfo("Success", message)
        else:
            self.status_var.set("Error loading file")
            messagebox.showerror("Error", message)
    
    def update_statistics(self):
        """Update the statistics display"""
        if self.analyzer.df is None:
            return
        
        stats = self.analyzer.stats
        threshold_counts = self.analyzer.get_threshold_counts()
        top_students = self.analyzer.get_top_students(5)
        
        stats_text = f"""BASIC STATISTICS - الإحصائيات الأساسية
{'='*50}
Total Students: {stats['count']:,}
Mean Score: {stats['mean']:.2f}
Median Score: {stats['median']:.2f}
Standard Deviation: {stats['std']:.2f}
Minimum Score: {stats['min']:.1f}
Maximum Score: {stats['max']:.1f}
25th Percentile: {stats['q25']:.2f}
75th Percentile: {stats['q75']:.2f}

UNIVERSITY THRESHOLDS - حدود القبول الجامعي
{'='*50}
"""
        
        for field, data in threshold_counts.items():
            stats_text += f"{field} (≥{data['threshold']}): {data['count']:,} students ({data['percentage']:.2f}%)\n"
        
        stats_text += f"\nTOP 5 STUDENTS - أفضل 5 طلاب\n{'='*50}\n"
        for idx, (_, student) in enumerate(top_students.iterrows(), 1):
            stats_text += f"{idx}. {student['arabic_name']} - {student['total_degree']:.1f}\n"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
    
    def enable_analysis_buttons(self):
        """Enable analysis buttons after successful data load"""
        buttons = [self.plot_histogram_btn, self.plot_top_students_btn, 
                  self.plot_thresholds_btn, self.save_plots_btn, self.export_report_btn]
        for btn in buttons:
            btn.configure(state='normal')
    
    def plot_histogram(self):
        """Plot histogram of grades distribution"""
        if self.analyzer.df is None:
            return
        
        # Create figure
        fig = plt_figure.Figure(figsize=(10, 8), dpi=100)
        
        # Main histogram
        ax1 = fig.add_subplot(211)
        scores = self.analyzer.df['total_degree']
        
        n, bins, patches = ax1.hist(scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(self.analyzer.stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {self.analyzer.stats["mean"]:.1f}')
        ax1.axvline(self.analyzer.stats['median'], color='green', linestyle='--', linewidth=2, label=f'Median: {self.analyzer.stats["median"]:.1f}')
        
        # Add threshold lines
        for field, threshold in self.analyzer.thresholds.items():
            ax1.axvline(threshold, color='orange', linestyle=':', alpha=0.7, label=f'{field}: {threshold}')
        
        ax1.set_xlabel('Total Degree')
        ax1.set_ylabel('Number of Students')
        ax1.set_title('Distribution of Student Grades - توزيع درجات الطلاب')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2 = fig.add_subplot(212)
        ax2.boxplot(scores, vert=False, patch_artist=True, 
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax2.set_xlabel('Total Degree')
        ax2.set_title('Grade Distribution Box Plot - الرسم الصندوقي للدرجات')
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        # Add to notebook
        self.add_plot_to_notebook(fig, "Histogram")
    
    def plot_top_students(self):
        """Plot top 10 students"""
        if self.analyzer.df is None:
            return
        
        top_students = self.analyzer.get_top_students(10)
        
        fig = plt_figure.Figure(figsize=(12, 8), dpi=100)
        ax = fig.add_subplot(111)
        
        # Create bar plot
        y_pos = np.arange(len(top_students))
        bars = ax.barh(y_pos, top_students['total_degree'], color='gold', alpha=0.8, edgecolor='black')
        
        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{name[:20]}..." if len(name) > 20 else name 
                           for name in top_students['arabic_name']], fontsize=10)
        ax.set_xlabel('Total Degree')
        ax.set_title('Top 10 Students - أفضل 10 طلاب', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, top_students['total_degree'])):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                   f'{score:.1f}', ha='left', va='center', fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()  # Top student at the top
        
        fig.tight_layout()
        
        # Add to notebook
        self.add_plot_to_notebook(fig, "Top Students")
    
    def plot_thresholds(self):
        """Plot threshold analysis"""
        if self.analyzer.df is None:
            return
        
        threshold_counts = self.analyzer.get_threshold_counts()
        
        fig = plt_figure.Figure(figsize=(12, 10), dpi=100)
        
        # Bar chart of students meeting thresholds
        ax1 = fig.add_subplot(211)
        fields = list(threshold_counts.keys())
        counts = [threshold_counts[field]['count'] for field in fields]
        colors = ['red', 'orange', 'green', 'blue', 'purple']
        
        bars = ax1.bar(fields, counts, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Number of Students')
        ax1.set_title('Students Meeting University Thresholds - الطلاب المؤهلون للكليات')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Pie chart of percentages
        ax2 = fig.add_subplot(212)
        percentages = [threshold_counts[field]['percentage'] for field in fields]
        wedges, texts, autotexts = ax2.pie(percentages, labels=fields, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax2.set_title('Percentage Distribution by University Thresholds - التوزيع النسبي')
        
        fig.tight_layout()
        
        # Add to notebook
        self.add_plot_to_notebook(fig, "Thresholds")
    
    def add_plot_to_notebook(self, figure, tab_name):
        """Add a plot to the notebook"""
        # Create frame for the plot
        plot_frame = ttk.Frame(self.plot_notebook)
        
        # Create canvas
        canvas = FigureCanvasTkAgg(figure, plot_frame)
        canvas.draw()
        
        # Create toolbar
        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()
        
        # Pack widgets
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add tab to notebook
        self.plot_notebook.add(plot_frame, text=tab_name)
        self.plot_notebook.select(plot_frame)
    
    def save_all_plots(self):
        """Save all plots as PNG files"""
        if self.analyzer.df is None:
            messagebox.showerror("Error", "No data loaded")
            return
        
        # Create output directory
        output_dir = Path("analysis_output")
        output_dir.mkdir(exist_ok=True)
        
        try:
            # Generate and save all plots
            self.status_var.set("Saving plots... جاري حفظ الرسوم البيانية...")
            self.root.update()
            
            # Histogram
            fig1 = plt_figure.Figure(figsize=(12, 10), dpi=150)
            self.create_histogram_plot(fig1)
            fig1.savefig(output_dir / 'histogram_distribution.png', bbox_inches='tight', dpi=150)
            
            # Top students
            fig2 = plt_figure.Figure(figsize=(12, 8), dpi=150)
            self.create_top_students_plot(fig2)
            fig2.savefig(output_dir / 'top_students.png', bbox_inches='tight', dpi=150)
            
            # Thresholds
            fig3 = plt_figure.Figure(figsize=(12, 10), dpi=150)
            self.create_thresholds_plot(fig3)
            fig3.savefig(output_dir / 'university_thresholds.png', bbox_inches='tight', dpi=150)
            
            self.status_var.set("Plots saved successfully")
            messagebox.showinfo("Success", f"All plots saved to {output_dir.absolute()}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error saving plots: {str(e)}")
    
    def create_histogram_plot(self, fig):
        """Create histogram plot for saving"""
        ax1 = fig.add_subplot(211)
        scores = self.analyzer.df['total_degree']
        
        n, bins, patches = ax1.hist(scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(self.analyzer.stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {self.analyzer.stats["mean"]:.1f}')
        ax1.axvline(self.analyzer.stats['median'], color='green', linestyle='--', linewidth=2, label=f'Median: {self.analyzer.stats["median"]:.1f}')
        
        for field, threshold in self.analyzer.thresholds.items():
            ax1.axvline(threshold, color='orange', linestyle=':', alpha=0.7, label=f'{field}: {threshold}')
        
        ax1.set_xlabel('Total Degree')
        ax1.set_ylabel('Number of Students')
        ax1.set_title('Distribution of Student Grades - توزيع درجات الطلاب')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(212)
        ax2.boxplot(scores, vert=False, patch_artist=True, 
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax2.set_xlabel('Total Degree')
        ax2.set_title('Grade Distribution Box Plot - الرسم الصندوقي للدرجات')
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
    
    def create_top_students_plot(self, fig):
        """Create top students plot for saving"""
        top_students = self.analyzer.get_top_students(10)
        ax = fig.add_subplot(111)
        
        y_pos = np.arange(len(top_students))
        bars = ax.barh(y_pos, top_students['total_degree'], color='gold', alpha=0.8, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{name[:25]}..." if len(name) > 25 else name 
                           for name in top_students['arabic_name']], fontsize=10)
        ax.set_xlabel('Total Degree')
        ax.set_title('Top 10 Students - أفضل 10 طلاب', fontsize=14, fontweight='bold')
        
        for i, (bar, score) in enumerate(zip(bars, top_students['total_degree'])):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                   f'{score:.1f}', ha='left', va='center', fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        fig.tight_layout()
    
    def create_thresholds_plot(self, fig):
        """Create thresholds plot for saving"""
        threshold_counts = self.analyzer.get_threshold_counts()
        
        ax1 = fig.add_subplot(211)
        fields = list(threshold_counts.keys())
        counts = [threshold_counts[field]['count'] for field in fields]
        colors = ['red', 'orange', 'green', 'blue', 'purple']
        
        bars = ax1.bar(fields, counts, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Number of Students')
        ax1.set_title('Students Meeting University Thresholds - الطلاب المؤهلون للكليات')
        
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        ax1.grid(True, alpha=0.3, axis='y')
        
        ax2 = fig.add_subplot(212)
        percentages = [threshold_counts[field]['percentage'] for field in fields]
        ax2.pie(percentages, labels=fields, autopct='%1.1f%%', 
               colors=colors, startangle=90)
        ax2.set_title('Percentage Distribution by University Thresholds - التوزيع النسبي')
        
        fig.tight_layout()
    
    def export_report(self):
        """Export comprehensive analysis report"""
        if self.analyzer.df is None:
            messagebox.showerror("Error", "No data loaded")
            return
        
        try:
            output_dir = Path("analysis_output")
            output_dir.mkdir(exist_ok=True)
            
            # Generate report
            report_path = output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("EGYPTIAN HIGH SCHOOL RESULTS ANALYSIS REPORT\n")
                f.write("="*60 + "\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Data file: {self.file_path.get()}\n\n")
                
                # Basic statistics
                stats = self.analyzer.stats
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
                threshold_counts = self.analyzer.get_threshold_counts()
                f.write("UNIVERSITY ADMISSION ANALYSIS:\n")
                f.write("-" * 35 + "\n")
                for field, data in threshold_counts.items():
                    f.write(f"{field} (≥{data['threshold']}): {data['count']:,} students ({data['percentage']:.2f}%)\n")
                f.write("\n")
                
                # Top students
                top_students = self.analyzer.get_top_students(20)
                f.write("TOP 20 STUDENTS:\n")
                f.write("-" * 20 + "\n")
                for idx, (_, student) in enumerate(top_students.iterrows(), 1):
                    f.write(f"{idx:2d}. {student['arabic_name']} (ID: {student['seating_no']}) - {student['total_degree']:.1f}\n")
                
                # Grade distribution
                f.write("\nGRADE DISTRIBUTION:\n")
                f.write("-" * 25 + "\n")
                ranges = [(0, 200), (200, 250), (250, 300), (300, 350), (350, 380), (380, 400), (400, 410)]
                for start, end in ranges:
                    count = len(self.analyzer.df[(self.analyzer.df['total_degree'] >= start) & 
                                                (self.analyzer.df['total_degree'] < end)])
                    percentage = (count / len(self.analyzer.df)) * 100
                    f.write(f"{start}-{end}: {count:,} students ({percentage:.2f}%)\n")
            
            # Also export top students as CSV
            top_students_path = output_dir / 'top_students.csv'
            self.analyzer.get_top_students(50).to_csv(top_students_path, index=False, encoding='utf-8')
            
            self.status_var.set("Report exported successfully")
            messagebox.showinfo("Success", f"Report exported to {report_path.absolute()}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting report: {str(e)}")


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = GradeAnalyzerGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()
