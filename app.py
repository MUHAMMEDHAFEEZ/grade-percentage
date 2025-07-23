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
import os
from datetime import datetime
import base64
from functools import lru_cache

# Configure page with performance optimizations
st.set_page_config(
    page_title="Egyptian High School Results Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Performance optimizations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_default_data():
    """Load default CSV data with caching"""
    try:
        default_file_path = "results.csv"
        if os.path.exists(default_file_path):
            # Try different delimiters
            for delimiter in [',', '\t', ';']:
                try:
                    df = pd.read_csv(default_file_path, delimiter=delimiter, encoding='utf-8')
                    if len(df.columns) >= 3:
                        break
                except:
                    continue
            else:
                df = pd.read_csv(default_file_path, encoding='utf-8')
            return df
    except Exception as e:
        st.error(f"Error loading default data: {str(e)}")
        return None
    return None

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def process_csv_data(df):
    """Process CSV data with caching"""
    if df is None:
        return None, False, "No data provided"
    
    try:
        # Convert Arabic numerals in total_degree column
        if 'total_degree' in df.columns:
            def convert_arabic_numerals(text):
                if pd.isna(text):
                    return text
                
                arabic_to_english = {
                    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
                    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
                    '٫': '.'
                }
                
                result = str(text)
                for arabic, english in arabic_to_english.items():
                    result = result.replace(arabic, english)
                
                try:
                    return float(result)
                except (ValueError, TypeError):
                    return text
            
            df['total_degree'] = df['total_degree'].apply(convert_arabic_numerals)
            df['total_degree'] = pd.to_numeric(df['total_degree'], errors='coerce')
            
            # Remove rows with invalid grades
            initial_count = len(df)
            df = df.dropna(subset=['total_degree'])
            df = df[df['total_degree'] > 0]
            
            return df, True, f"Successfully processed {len(df)} student records"
    except Exception as e:
        return None, False, f"Error processing data: {str(e)}"

@st.cache_data(ttl=1800)  # Cache for 30 minutes  
def calculate_statistics(df):
    """Calculate statistics with caching"""
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

class StreamlitGradeAnalyzer:
    def __init__(self):
        # System-specific configurations (cached in session state)
        if 'system_configs' not in st.session_state:
            st.session_state.system_configs = {
                'new_system': {
                    'name': 'النظام الجديد (New System)',
                    'name_en': 'New System',
                    'total_score': 320,
                    'years': [2025, 2024],
                    'default_year': 2025,
                    'thresholds': {
                        2025: {
                            'Medicine': 315,      # طب
                            'Pharmacy': 310,      # صيدلة  
                            'Engineering': 298,   # هندسة
                            'Commerce': 265,      # تجارة
                            'Arts': 238           # آداب
                        },
                        2024: {
                            'Medicine': 312,
                            'Pharmacy': 307,
                            'Engineering': 295,
                            'Commerce': 262,
                            'Arts': 235
                        }
                    },
                    'ranges': [(0, 150), (150, 200), (200, 240), (240, 270), (270, 300), (300, 315), (315, 320)],
                    'grade_categories': {
                        'ممتاز': (300, 320),
                        'جيد جداً': (270, 300),
                        'جيد': (240, 270),
                        'مقبول': (200, 240),
                        'ضعيف': (0, 200)
                    }
                },
                'old_system': {
                    'name': 'النظام القديم (Old System)',
                    'name_en': 'Old System',
                    'total_score': 410,
                    'years': [2023, 2022, 2021, 2020],
                    'default_year': 2023,
                    'thresholds': {
                        2023: {
                            'Medicine': 404,
                            'Pharmacy': 397,
                            'Engineering': 382,
                            'Commerce': 340,
                            'Arts': 305
                        },
                        2022: {
                            'Medicine': 401,
                            'Pharmacy': 394,
                            'Engineering': 379,
                            'Commerce': 337,
                            'Arts': 302
                        },
                        2021: {
                            'Medicine': 398,
                            'Pharmacy': 391,
                            'Engineering': 376,
                            'Commerce': 334,
                            'Arts': 299
                        },
                        2020: {
                            'Medicine': 395,
                            'Pharmacy': 388,
                            'Engineering': 373,
                            'Commerce': 331,
                            'Arts': 296
                        }
                    },
                    'ranges': [(0, 200), (200, 250), (250, 300), (300, 350), (350, 380), (380, 400), (400, 410)],
                    'grade_categories': {
                        'ممتاز': (380, 410),
                        'جيد جداً': (340, 380),
                        'جيد': (300, 340),
                        'مقبول': (250, 300),
                        'ضعيف': (0, 250)
                    }
                }
            }
        
        self.system_configs = st.session_state.system_configs
        
        # Initialize session state for current system settings
        if 'current_system' not in st.session_state:
            st.session_state.current_system = 'new_system'
        if 'current_year' not in st.session_state:
            st.session_state.current_year = self.system_configs['new_system']['default_year']
        
        self.current_system = st.session_state.current_system
        self.current_year = st.session_state.current_year
        self.update_current_settings()
    
    def update_current_settings(self):
        """Update current settings based on selected system and year"""
        config = self.system_configs[self.current_system]
        self.total_score = config['total_score']
        self.thresholds = config['thresholds'][self.current_year]
        self.ranges = config['ranges']
        self.grade_categories = config['grade_categories']
        self.system_name = config['name']
        self.system_name_en = config['name_en']
    
    def set_system(self, system):
        """Set the education system (new or old) with session state"""
        if system in self.system_configs:
            self.current_system = system
            st.session_state.current_system = system
            self.current_year = self.system_configs[system]['default_year']
            st.session_state.current_year = self.current_year
            self.update_current_settings()
            return True
        return False
    
    def set_year(self, year):
        """Set the analysis year within current system with session state"""
        if year in self.system_configs[self.current_system]['years']:
            self.current_year = year
            st.session_state.current_year = year
            self.update_current_settings()
            return True
        return False
    
    def get_available_systems(self):
        """Get list of available education systems"""
        return [
            {'key': 'new_system', 'name': 'النظام الجديد (320 درجة)', 'name_en': 'New System (320)'},
            {'key': 'old_system', 'name': 'النظام القديم (410 درجة)', 'name_en': 'Old System (410)'}
        ]
    
    def get_available_years(self):
        """Get list of available years for current system"""
        return sorted(self.system_configs[self.current_system]['years'], reverse=True)
    
    @lru_cache(maxsize=128)
    def get_grade_category(self, score):
        """Get grade category for a score with caching"""
        for category, (min_score, max_score) in self.grade_categories.items():
            if min_score <= score < max_score:
                return category
        return 'غير محدد'
    
    @st.cache_data(ttl=1800)
    def get_threshold_counts(_self, df):
        """Get count of students meeting each threshold with caching"""
        if df is None:
            return {}
            
        threshold_counts = {}
        for field, threshold in _self.thresholds.items():
            count = len(df[df['total_degree'] >= threshold])
            percentage = (count / len(df)) * 100
            threshold_counts[field] = {
                'count': count,
                'percentage': percentage,
                'threshold': threshold
            }
        return threshold_counts
    
    @st.cache_data(ttl=1800)
    def get_top_students(_self, df, n=10):
        """Get top N students with caching"""
        if df is None:
            return pd.DataFrame()
        
        return df.nlargest(n, 'total_degree')[['seating_no', 'arabic_name', 'total_degree']]
    
    @st.cache_data(ttl=900)  # Cache for 15 minutes
    def search_student_by_id(_self, df, student_id):
        """Search for a student by their seating number with caching"""
        if df is None:
            return None, "No data loaded"
        
        # Convert student_id to string for comparison
        student_id_str = str(student_id).strip()
        
        # Search for the student
        student_row = df[df['seating_no'].astype(str) == student_id_str]
        
        if student_row.empty:
            return None, f"Student with ID {student_id} not found"
        
        student_data = student_row.iloc[0]
        
        # Calculate rank
        rank = len(df[df['total_degree'] > student_data['total_degree']]) + 1
        total_students = len(df)
        percentile = ((total_students - rank) / total_students) * 100
        
        # Check available universities
        available_unis = []
        for field, threshold in _self.thresholds.items():
            if student_data['total_degree'] >= threshold:
                available_unis.append(field)
        
        # Get grade category
        grade_category = _self.get_grade_category(student_data['total_degree'])
        
        result = {
            'seating_no': student_data['seating_no'],
            'name': student_data['arabic_name'],
            'total_degree': student_data['total_degree'],
            'rank': rank,
            'total_students': total_students,
            'percentile': percentile,
            'available_universities': available_unis,
            'grade_category': grade_category,
            'system': _self.system_name,
            'year': _self.current_year
        }
        
        return result, "Student found successfully"

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def create_distribution_plot(df, stats, thresholds):
    """Create interactive distribution plot using Plotly with caching"""
    
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

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def create_top_students_plot(top_students):
    """Create top students bar chart with caching"""
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

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def create_threshold_analysis_plot(threshold_counts):
    """Create threshold analysis plots with caching"""
    
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
    
    # Initialize analyzer
    analyzer = StreamlitGradeAnalyzer()
    
    # Title and description
    st.title("🎓 Egyptian High School Results Analyzer")
    st.markdown("### محلل نتائج الثانوية العامة المصرية")
    
    # Display current system and year info
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.markdown(f"**🎓 Education System**: {analyzer.system_name}")
    with col2:
        st.markdown(f"**📅 Year**: {analyzer.current_year}")
    with col3:
        st.markdown(f"**📊 Total Score**: {analyzer.total_score}")
    with col4:
        system_emoji = "🆕" if analyzer.current_system == 'new_system' else "📚"
        st.markdown(f"**{system_emoji} Type**: {'New' if analyzer.current_system == 'new_system' else 'Old'}")
    
    # Auto-data loading info with system-specific message
    if analyzer.current_system == 'new_system':
        st.info("🆕 **النظام الجديد محمل!** Ready to analyze new system results (320 total). Search immediately!")
    else:
        st.info("📚 **النظام القديم محمل!** Ready to analyze old system results (410 total). Search immediately!")
    
    st.markdown("---")
    
    # Sidebar for system selection and file upload
    st.sidebar.header("⚙️ System Settings - إعدادات النظام")
    
    # Education System selection
    st.sidebar.markdown("### 🎓 Education System - نظام التعليم")
    available_systems = analyzer.get_available_systems()
    
    # Create system selection
    system_options = [sys['name'] for sys in available_systems]
    system_keys = [sys['key'] for sys in available_systems]
    
    current_system_index = system_keys.index(analyzer.current_system)
    selected_system_name = st.sidebar.selectbox(
        "Choose Education System",
        options=system_options,
        index=current_system_index,
        help="اختر نظام التعليم (جديد أم قديم)"
    )
    
    # Get selected system key
    selected_system_key = system_keys[system_options.index(selected_system_name)]
    
    # Update analyzer with selected system
    if analyzer.set_system(selected_system_key):
        if selected_system_key == 'new_system':
            st.sidebar.success(f"🆕 النظام الجديد - {analyzer.total_score} درجة")
        else:
            st.sidebar.info(f"� النظام القديم - {analyzer.total_score} درجة")
    
    # Year selection within the chosen system
    st.sidebar.markdown("### 📅 Academic Year - السنة الدراسية")
    available_years = analyzer.get_available_years()
    selected_year = st.sidebar.selectbox(
        "Select Academic Year",
        options=available_years,
        index=0,  # Default to most recent year
        help="اختر السنة الدراسية"
    )
    
    # Update analyzer with selected year
    if analyzer.set_year(selected_year):
        st.sidebar.success(f"📊 Using {selected_year} thresholds")
    
    st.sidebar.markdown("---")
    st.sidebar.header("📁 File Upload - تحميل الملف")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file (Optional - ملف اختياري)",
        type=['csv'],
        help="Upload your Egyptian high school results CSV file or use default data"
    )
    
    # Auto-load default data if no file uploaded with improved performance
    df = None
    success = False
    message = ""
    
    if uploaded_file is not None:
        # Load and process uploaded data
        with st.spinner('Loading uploaded file... جاري تحميل الملف المرفوع...'):
            # Use cached processing function
            try:
                content = uploaded_file.getvalue().decode('utf-8')
                
                # Try different delimiters
                for delimiter in [',', '\t', ';']:
                    try:
                        raw_df = pd.read_csv(io.StringIO(content), delimiter=delimiter)
                        if len(raw_df.columns) >= 3:
                            break
                    except:
                        continue
                else:
                    raw_df = pd.read_csv(io.StringIO(content))
                
                df, success, message = process_csv_data(raw_df)
            except Exception as e:
                success = False
                message = f"❌ Error loading uploaded file: {str(e)}"
    else:
        # Try to load default results.csv file with caching
        with st.spinner('Loading default data... جاري تحميل البيانات الافتراضية...'):
            raw_df = load_default_data()
            if raw_df is not None:
                df, success, message = process_csv_data(raw_df)
                if success:
                    st.sidebar.success("📊 Using cached default dataset")
                else:
                    st.sidebar.error("❌ Error processing default data")
            else:
                success = False
                message = "❌ Default results.csv file not found. Please upload a CSV file."
                st.sidebar.error("📁 No default data available")
    
    if success and df is not None:
            st.success(message)
            
            # Calculate statistics with caching
            stats = calculate_statistics(df)
            threshold_counts = analyzer.get_threshold_counts(df)
            top_students = analyzer.get_top_students(df, 10)
            
            # Display data info
            st.sidebar.markdown("### 📊 Data Overview")
            st.sidebar.metric("Total Students", f"{len(df):,}")
            st.sidebar.metric("Average Score", f"{stats['mean']:.2f}")
            st.sidebar.metric("Highest Score", f"{stats['max']:.1f}")
            st.sidebar.metric("Lowest Score", f"{stats['min']:.1f}")
            
            # Quick student search in sidebar
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🔍 Quick Student Search")
            sidebar_student_id = st.sidebar.text_input(
                "Student ID",
                placeholder="e.g., 1001660",
                key="sidebar_search"
            )
            
            if st.sidebar.button("Search Student", key="sidebar_search_btn"):
                if sidebar_student_id:
                    student_data, message = analyzer.search_student_by_id(df, sidebar_student_id)
                    if student_data:
                        st.sidebar.success("✅ Student Found!")
                        st.sidebar.markdown(f"**Name:** {student_data['name'][:20]}...")
                        st.sidebar.markdown(f"**Score:** {student_data['total_degree']:.1f}")
                        st.sidebar.markdown(f"**Rank:** {student_data['rank']:,}")
                        if student_data['available_universities']:
                            st.sidebar.markdown(f"**Universities:** {len(student_data['available_universities'])}")
                        st.sidebar.info("👆 Check the 'Student Search' tab for full details")
                    else:
                        st.sidebar.error("❌ Student not found")
            
            # Main content tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "🔍 Student Search", 
                "📈 Distribution Analysis", 
                "🏆 Top Students", 
                "🎯 University Thresholds",
                "📋 Statistics",
                "💾 Export Data"
            ])
            
            with tab1:
                st.subheader("🔍 Student Search - البحث عن طالب")
                st.markdown("#### Search by Seating Number - البحث برقم الجلوس")
                
                # Welcome message
                if uploaded_file is None:
                    st.success(f"✅ **Ready!** {len(df):,} student records loaded automatically. Enter a student ID below to search!")
                else:
                    st.info(f"📊 Using uploaded data: {len(df):,} student records")
                
                # Student search input
                col1, col2 = st.columns([3, 1])
                with col1:
                    student_id = st.text_input(
                        "Enter Student ID (Seating Number)",
                        placeholder="Example: 1001660",
                        help="Enter the student's seating number to view their results"
                    )
                with col2:
                    search_button = st.button("🔍 Search", type="primary")
                
                if search_button and student_id:
                    student_data, message = analyzer.search_student_by_id(df, student_id)
                    
                    if student_data:
                        # Display student information
                        st.success("✅ Student Found!")
                        
                        # Student details card
                        with st.container():
                            st.markdown("### 👤 Student Information - معلومات الطالب")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"""
                                **🆔 Student ID:** {student_data['seating_no']}
                                
                                **👤 Name:** {student_data['name']}
                                
                                **📊 Total Score:** {student_data['total_degree']:.1f} / {analyzer.total_score}
                                """)
                            
                            with col2:
                                st.markdown(f"""
                                **🏆 Rank:** {student_data['rank']:,} out of {student_data['total_students']:,}
                                
                                **📈 Percentile:** Top {100-student_data['percentile']:.1f}%
                                
                                **📊 Better than:** {student_data['percentile']:.1f}% of students
                                """)
                            
                            with col3:
                                grade_emoji = "🌟" if student_data['grade_category'] == "ممتاز" else "📈" if student_data['grade_category'] == "جيد جداً" else "📊" if student_data['grade_category'] == "جيد" else "📋" if student_data['grade_category'] == "مقبول" else "📉"
                                system_emoji = "🆕" if analyzer.current_system == 'new_system' else "📚"
                                st.markdown(f"""
                                **{grade_emoji} Grade Category:** {student_data['grade_category']}
                                
                                **{system_emoji} System:** {student_data['system']}
                                
                                **📅 Year:** {student_data['year']}
                                """)
                        
                        # Performance visualization
                        st.markdown("### 📊 Performance Analysis - تحليل الأداء")
                        
                        # Create performance chart
                        performance_data = {
                            'Metric': ['Your Score', 'Class Average', 'Top Student', 'Minimum Score'],
                            'Value': [
                                student_data['total_degree'],
                                stats['mean'],
                                stats['max'],
                                stats['min']
                            ],
                            'Color': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                        }
                        
                        performance_df = pd.DataFrame(performance_data)
                        fig_performance = px.bar(
                            performance_df,
                            x='Metric',
                            y='Value',
                            color='Color',
                            title=f"Performance Comparison for {student_data['name']}"
                        )
                        fig_performance.update_layout(showlegend=False, height=400)
                        st.plotly_chart(fig_performance, use_container_width=True)
                        
                        # Available universities
                        st.markdown("### 🎓 Available Universities - الجامعات المتاحة")
                        
                        if student_data['available_universities']:
                            st.success(f"🎉 Congratulations! You qualify for {len(student_data['available_universities'])} university programs:")
                            
                            for i, uni in enumerate(student_data['available_universities'], 1):
                                threshold = analyzer.thresholds[uni]
                                excess_points = student_data['total_degree'] - threshold
                                
                                # University info with styling
                                with st.container():
                                    col1, col2, col3 = st.columns([2, 1, 1])
                                    with col1:
                                        st.markdown(f"**{i}. {uni}** 🏛️")
                                    with col2:
                                        st.markdown(f"**Required:** {threshold}")
                                    with col3:
                                        st.markdown(f"**Excess:** +{excess_points:.1f}")
                                
                                # Progress bar showing how much above threshold
                                progress = min(excess_points / 20, 1.0)  # Max 20 points excess for full bar
                                st.progress(progress)
                                st.markdown("---")
                        else:
                            st.warning("📚 Unfortunately, your score doesn't meet the minimum requirements for the tracked university programs.")
                            st.markdown("**However, there might be other opportunities:**")
                            st.markdown("- Private universities with different requirements")
                            st.markdown("- Technical institutes and colleges")
                            st.markdown("- Alternative educational paths")
                        
                        # Improvement suggestions
                        if student_data['available_universities']:
                            next_target = None
                            for field, threshold in analyzer.thresholds.items():
                                if field not in student_data['available_universities']:
                                    if next_target is None or threshold < analyzer.thresholds[next_target]:
                                        next_target = field
                            
                            if next_target:
                                points_needed = analyzer.thresholds[next_target] - student_data['total_degree']
                                st.info(f"💡 **Next Goal:** To qualify for {next_target}, you would need {points_needed:.1f} more points!")
                        
                        # Position in grade distribution
                        st.markdown("### 📍 Your Position in Grade Distribution - موقعك في توزيع الدرجات")
                        
                        # Create distribution plot with student position
                        fig_dist = go.Figure()
                        
                        # Add histogram of all scores
                        fig_dist.add_trace(go.Histogram(
                            x=df['total_degree'],
                            nbinsx=50,
                            name='All Students',
                            opacity=0.7,
                            marker_color='lightblue'
                        ))
                        
                        # Add vertical line for student score
                        fig_dist.add_vline(
                            x=student_data['total_degree'],
                            line_dash="dash",
                            line_color="red",
                            line_width=3,
                            annotation_text=f"Your Score: {student_data['total_degree']:.1f}"
                        )
                        
                        # Add university thresholds
                        colors = ['green', 'orange', 'purple', 'blue', 'brown']
                        for i, (field, threshold) in enumerate(analyzer.thresholds.items()):
                            fig_dist.add_vline(
                                x=threshold,
                                line_dash="dot",
                                line_color=colors[i % len(colors)],
                                annotation_text=f"{field}: {threshold}"
                            )
                        
                        fig_dist.update_layout(
                            title="Your Position in Grade Distribution",
                            xaxis_title="Total Degree",
                            yaxis_title="Number of Students",
                            height=400
                        )
                        
                        st.plotly_chart(fig_dist, use_container_width=True)
                        
                    else:
                        st.error(f"❌ {message}")
                        st.info("💡 Please check the Student ID and try again. Make sure to enter the exact seating number.")
                
                elif search_button and not student_id:
                    st.warning("⚠️ Please enter a Student ID to search.")
                
                # Quick search examples
                st.markdown("### 💡 Quick Search Examples")
                st.markdown("""
                Try searching for these student IDs to see the feature in action:
                - **1001660** - محمد ابو الحسن حسن مصطفى (163.5)
                - **1001670** - محمد مختار محمد محمد عبد الموجود (288.0)
                - **1001677** - ملك كمال محمود محمد (250.5)
                """)
            
            with tab2:
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
            
            with tab3:
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
            
            with tab4:
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
            
            with tab5:
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
                st.markdown(f"#### Grade Distribution by Ranges (out of {analyzer.total_score})")
                ranges = analyzer.ranges
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
            
            with tab6:
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
        # Display error message if data loading failed
        st.error(message)
        
        # Instructions when no data is available
        st.info("👆 Please upload a CSV file using the sidebar or ensure results.csv exists")
        
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
        
        ### University Thresholds - حدود القبول الجامعي (2025)
        - **Medicine (طب)**: 315 degrees
        - **Pharmacy (صيدلة)**: 310 degrees
        - **Engineering (هندسة)**: 298 degrees
        - **Commerce (تجارة)**: 265 degrees
        - **Arts (آداب)**: 238 degrees
        
        **Note**: Select different academic years from the sidebar to see historical thresholds.
        """)

if __name__ == "__main__":
    main()
