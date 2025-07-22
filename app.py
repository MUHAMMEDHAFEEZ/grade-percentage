#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Egyptian High School Results Analyzer - Streamlit Web App (Deployment Ready)
Optimized for Streamlit Cloud deployment
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io
from datetime import datetime
import base64

# Configure page
st.set_page_config(
    page_title="Egyptian High School Results Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitGradeAnalyzer:
    def __init__(self):
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
    
    def load_csv(self, uploaded_file):
        """Load and process the uploaded CSV file"""
        try:
            # Try different delimiters
            content = uploaded_file.getvalue().decode('utf-8')
            
            # Try different delimiters
            for delimiter in [',', '\t', ';']:
                try:
                    df = pd.read_csv(io.StringIO(content), delimiter=delimiter)
                    if len(df.columns) >= 3:
                        break
                except:
                    continue
            else:
                # If no delimiter works, try auto-detection
                df = pd.read_csv(io.StringIO(content))
            
            # Convert Arabic numerals in total_degree column
            if 'total_degree' in df.columns:
                df['total_degree'] = df['total_degree'].apply(self.convert_arabic_numerals)
                df['total_degree'] = pd.to_numeric(df['total_degree'], errors='coerce')
                
                # Remove rows with invalid grades
                initial_count = len(df)
                df = df.dropna(subset=['total_degree'])
                df = df[df['total_degree'] > 0]  # Remove zero or negative grades
                
                if len(df) < initial_count:
                    st.warning(f"Removed {initial_count - len(df)} invalid records")
                
            return df, True, f"Successfully loaded {len(df)} student records"
            
        except Exception as e:
            return None, False, f"Error loading file: {str(e)}"
    
    def calculate_statistics(self, df):
        """Calculate basic statistics"""
        if df is not None and 'total_degree' in df.columns:
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
            return stats
        return {}
    
    def get_threshold_counts(self, df):
        """Get count of students meeting each threshold"""
        if df is None:
            return {}
            
        threshold_counts = {}
        for field, threshold in self.thresholds.items():
            count = len(df[df['total_degree'] >= threshold])
            percentage = (count / len(df)) * 100
            threshold_counts[field] = {
                'count': count,
                'percentage': percentage,
                'threshold': threshold
            }
        return threshold_counts
    
    def get_top_students(self, df, n=10):
        """Get top N students"""
        if df is None:
            return pd.DataFrame()
        
        return df.nlargest(n, 'total_degree')[['seating_no', 'arabic_name', 'total_degree']]

def create_distribution_plot(df, stats, thresholds):
    """Create interactive distribution plot using Plotly"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Grade Distribution Histogram',
            'Box Plot',
            'Grade Distribution with Thresholds',
            'Cumulative Distribution'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    scores = df['total_degree']
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=scores, nbinsx=50, name='Grade Distribution', 
                    marker_color='lightblue', opacity=0.7),
        row=1, col=1
    )
    
    # Add mean and median lines
    fig.add_vline(x=stats['mean'], line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {stats['mean']:.1f}", row=1, col=1)
    fig.add_vline(x=stats['median'], line_dash="dash", line_color="green",
                  annotation_text=f"Median: {stats['median']:.1f}", row=1, col=1)
    
    # Box plot
    fig.add_trace(
        go.Box(y=scores, name='Grade Distribution', marker_color='lightcoral'),
        row=1, col=2
    )
    
    # Histogram with thresholds
    fig.add_trace(
        go.Histogram(x=scores, nbinsx=50, name='Grades', 
                    marker_color='lightgreen', opacity=0.7),
        row=2, col=1
    )
    
    # Add threshold lines
    colors = ['red', 'orange', 'purple', 'blue', 'brown']
    for i, (field, threshold) in enumerate(thresholds.items()):
        fig.add_vline(x=threshold, line_dash="dot", 
                      line_color=colors[i % len(colors)],
                      annotation_text=f"{field}: {threshold}", row=2, col=1)
    
    # Cumulative distribution
    sorted_scores = np.sort(scores)
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100
    
    fig.add_trace(
        go.Scatter(x=sorted_scores, y=cumulative, mode='lines',
                  name='Cumulative %', line_color='purple'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Grade Distribution Analysis - تحليل توزيع الدرجات",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Total Degree", row=1, col=1)
    fig.update_xaxes(title_text="Grade", row=1, col=2)
    fig.update_xaxes(title_text="Total Degree", row=2, col=1)
    fig.update_xaxes(title_text="Total Degree", row=2, col=2)
    
    fig.update_yaxes(title_text="Number of Students", row=1, col=1)
    fig.update_yaxes(title_text="Total Degree", row=1, col=2)
    fig.update_yaxes(title_text="Number of Students", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Percentage", row=2, col=2)
    
    return fig

def create_top_students_plot(top_students):
    """Create top students bar chart"""
    fig = px.bar(
        top_students.iloc[::-1],  # Reverse order for top-to-bottom display
        x='total_degree',
        y='arabic_name',
        orientation='h',
        title='Top 10 Students - أفضل 10 طلاب',
        labels={'total_degree': 'Total Degree', 'arabic_name': 'Student Name'},
        color='total_degree',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    # Add value labels
    for i, score in enumerate(top_students['total_degree'].iloc[::-1]):
        fig.add_annotation(
            x=score + 5,
            y=i,
            text=f"{score:.1f}",
            showarrow=False,
            font=dict(color="black", size=12)
        )
    
    return fig

def create_threshold_analysis_plot(threshold_counts):
    """Create threshold analysis plots"""
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Students Meeting Thresholds', 'Percentage Distribution'),
        specs=[[{"type": "bar"}, {"type": "pie"}]]
    )
    
    fields = list(threshold_counts.keys())
    counts = [threshold_counts[field]['count'] for field in fields]
    percentages = [threshold_counts[field]['percentage'] for field in fields]
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    # Bar chart
    fig.add_trace(
        go.Bar(x=fields, y=counts, name='Student Count',
               marker_color=colors, text=counts, textposition='outside'),
        row=1, col=1
    )
    
    # Pie chart
    fig.add_trace(
        go.Pie(labels=fields, values=percentages, name="Percentage",
               marker_colors=colors),
        row=1, col=2
    )
    
    fig.update_layout(
        height=500,
        title_text="University Threshold Analysis - تحليل حدود القبول الجامعي"
    )
    
    fig.update_xaxes(title_text="University Field", row=1, col=1)
    fig.update_yaxes(title_text="Number of Students", row=1, col=1)
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("🎓 Egyptian High School Results Analyzer")
    st.markdown("### محلل نتائج الثانوية العامة المصرية")
    st.markdown("---")
    
    # Initialize analyzer
    analyzer = StreamlitGradeAnalyzer()
    
    # Sidebar for file upload and controls
    st.sidebar.header("📁 File Upload - تحميل الملف")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your Egyptian high school results CSV file"
    )
    
    if uploaded_file is not None:
        # Load and process data
        with st.spinner('Loading and processing data... جاري تحميل وتجهيز البيانات...'):
            df, success, message = analyzer.load_csv(uploaded_file)
        
        if success:
            st.success(message)
            
            # Calculate statistics
            stats = analyzer.calculate_statistics(df)
            threshold_counts = analyzer.get_threshold_counts(df)
            top_students = analyzer.get_top_students(df, 10)
            
            # Display data info
            st.sidebar.markdown("### 📊 Data Overview")
            st.sidebar.metric("Total Students", f"{len(df):,}")
            st.sidebar.metric("Average Score", f"{stats['mean']:.2f}")
            st.sidebar.metric("Highest Score", f"{stats['max']:.1f}")
            st.sidebar.metric("Lowest Score", f"{stats['min']:.1f}")
            
            # Main content tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Distribution Analysis", 
                "🏆 Top Students", 
                "🎯 University Thresholds",
                "📋 Statistics",
                "💾 Export Data"
            ])
            
            with tab1:
                st.subheader("Grade Distribution Analysis - تحليل توزيع الدرجات")
                
                # Interactive plot
                dist_fig = create_distribution_plot(df, stats, analyzer.thresholds)
                st.plotly_chart(dist_fig, use_container_width=True)
                
                # Basic statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Score", f"{stats['mean']:.2f}")
                with col2:
                    st.metric("Median Score", f"{stats['median']:.2f}")
                with col3:
                    st.metric("Standard Deviation", f"{stats['std']:.2f}")
                with col4:
                    st.metric("Range", f"{stats['max']:.1f} - {stats['min']:.1f}")
            
            with tab2:
                st.subheader("Top Students - أفضل الطلاب")
                
                # Number of top students to show
                n_students = st.slider("Number of top students to display", 5, 50, 10)
                top_n = analyzer.get_top_students(df, n_students)
                
                # Interactive plot
                if len(top_n) > 0:
                    top_fig = create_top_students_plot(top_n.head(10))
                    st.plotly_chart(top_fig, use_container_width=True)
                    
                    # Display table
                    st.subheader("Top Students Table")
                    st.dataframe(top_n, use_container_width=True)
            
            with tab3:
                st.subheader("University Threshold Analysis - تحليل حدود القبول الجامعي")
                
                # Interactive plot
                threshold_fig = create_threshold_analysis_plot(threshold_counts)
                st.plotly_chart(threshold_fig, use_container_width=True)
                
                # Threshold details
                st.subheader("Detailed Threshold Analysis")
                
                for field, data in threshold_counts.items():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{field}", f"{data['count']:,} students")
                    with col2:
                        st.metric("Percentage", f"{data['percentage']:.2f}%")
                    with col3:
                        st.metric("Threshold", f"≥{data['threshold']}")
                    st.markdown("---")
            
            with tab4:
                st.subheader("Comprehensive Statistics - الإحصائيات الشاملة")
                
                # Basic statistics
                st.markdown("#### Basic Statistics - الإحصائيات الأساسية")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    - **Total Students**: {stats['count']:,}
                    - **Mean Score**: {stats['mean']:.2f}
                    - **Median Score**: {stats['median']:.2f}
                    - **Standard Deviation**: {stats['std']:.2f}
                    """)
                
                with col2:
                    st.markdown(f"""
                    - **Minimum Score**: {stats['min']:.1f}
                    - **Maximum Score**: {stats['max']:.1f}
                    - **25th Percentile**: {stats['q25']:.2f}
                    - **75th Percentile**: {stats['q75']:.2f}
                    """)
                
                # Grade distribution by ranges
                st.markdown("#### Grade Distribution by Ranges")
                ranges = [(0, 200), (200, 250), (250, 300), (300, 350), (350, 380), (380, 400), (400, 410)]
                range_data = []
                
                for start, end in ranges:
                    count = len(df[(df['total_degree'] >= start) & (df['total_degree'] < end)])
                    percentage = (count / len(df)) * 100
                    range_data.append({
                        'Range': f"{start}-{end}",
                        'Count': count,
                        'Percentage': f"{percentage:.2f}%"
                    })
                
                range_df = pd.DataFrame(range_data)
                st.dataframe(range_df, use_container_width=True)
            
            with tab5:
                st.subheader("Export Data - تصدير البيانات")
                
                # Generate comprehensive report
                if st.button("📊 Generate Comprehensive Report"):
                    report_text = f"""EGYPTIAN HIGH SCHOOL RESULTS ANALYSIS REPORT
{'='*60}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Data file: {uploaded_file.name}

BASIC STATISTICS:
{'-'*30}
Total Students: {stats['count']:,}
Mean Score: {stats['mean']:.2f}
Median Score: {stats['median']:.2f}
Standard Deviation: {stats['std']:.2f}
Minimum Score: {stats['min']:.1f}
Maximum Score: {stats['max']:.1f}
25th Percentile: {stats['q25']:.2f}
75th Percentile: {stats['q75']:.2f}

UNIVERSITY ADMISSION ANALYSIS:
{'-'*35}
"""
                    for field, data in threshold_counts.items():
                        report_text += f"{field} (≥{data['threshold']}): {data['count']:,} students ({data['percentage']:.2f}%)\n"
                    
                    report_text += f"\nTOP 20 STUDENTS:\n{'-'*20}\n"
                    top_20 = analyzer.get_top_students(df, 20)
                    for idx, (_, student) in enumerate(top_20.iterrows(), 1):
                        report_text += f"{idx:2d}. {student['arabic_name']} (ID: {student['seating_no']}) - {student['total_degree']:.1f}\n"
                    
                    # Display report
                    st.text_area("Analysis Report", report_text, height=400)
                    
                    # Download button for report
                    st.download_button(
                        label="📥 Download Report",
                        data=report_text,
                        file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                # Download options
                st.markdown("#### Download Options")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📥 Download Top 50 Students"):
                        top_50 = analyzer.get_top_students(df, 50)
                        csv = top_50.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="top_50_students.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    if st.button("📥 Download Full Dataset"):
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="processed_results.csv",
                            mime="text/csv"
                        )
        
        else:
            st.error(message)
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload a CSV file using the sidebar to begin analysis")
        
        st.markdown("""
        ### How to Use - كيفية الاستخدام
        
        1. **Upload your CSV file** in the sidebar - قم بتحميل ملف CSV من الشريط الجانبي
        2. **Wait for processing** - انتظر حتى يتم تجهيز البيانات
        3. **Explore the analysis tabs** - استكشف علامات التبويب للتحليل
        4. **Export your results** - صدر النتائج
        
        ### CSV File Format - تنسيق ملف CSV
        
        Your CSV file should contain:
        - `seating_no`: Student ID number
        - `arabic_name`: Student's Arabic name
        - `total_degree`: Total score (supports Arabic numerals like ٣٨٢٫٥)
        
        ### University Thresholds - حدود القبول الجامعي
        - **Medicine (طب)**: 404 degrees
        - **Pharmacy (صيدلة)**: 397 degrees
        - **Engineering (هندسة)**: 382 degrees
        - **Commerce (تجارة)**: 340 degrees
        - **Arts (آداب)**: 305 degrees
        """)

if __name__ == "__main__":
    main()
