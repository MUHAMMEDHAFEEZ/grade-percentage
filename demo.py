#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script to test Arabic numeral conversion and basic functionality
"""

import pandas as pd

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

def test_conversion():
    """Test the Arabic numeral conversion"""
    print("Testing Arabic Numeral Conversion")
    print("=" * 40)
    
    test_cases = [
        "163٫5",  # 163.5
        "187٫5",  # 187.5
        "168",    # 168
        "212",    # 212
        "٣٨٢٫٥", # 382.5
        "٤٠٤",   # 404
    ]
    
    for case in test_cases:
        converted = convert_arabic_numerals(case)
        print(f"'{case}' -> {converted} (type: {type(converted).__name__})")

def test_csv_loading():
    """Test loading the CSV file"""
    try:
        print("\nTesting CSV File Loading")
        print("=" * 40)
        
        # Try to read the first few rows
        df = pd.read_csv("results.csv", nrows=5)
        print(f"CSV columns: {list(df.columns)}")
        print(f"First 5 rows:")
        print(df)
        
        # Test conversion on total_degree column
        if 'total_degree' in df.columns:
            print(f"\nOriginal total_degree values:")
            print(df['total_degree'].tolist())
            
            print(f"\nConverted total_degree values:")
            df['total_degree_converted'] = df['total_degree'].apply(convert_arabic_numerals)
            print(df['total_degree_converted'].tolist())
            
        return True
        
    except FileNotFoundError:
        print("Error: results.csv file not found in current directory")
        return False
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return False

def main():
    print("Egyptian High School Results Analyzer - Demo")
    print("=" * 50)
    
    # Test Arabic numeral conversion
    test_conversion()
    
    # Test CSV loading
    csv_loaded = test_csv_loading()
    
    print("\n" + "=" * 50)
    if csv_loaded:
        print("✓ Demo completed successfully!")
        print("✓ Arabic numeral conversion working")
        print("✓ CSV file loading working")
        print("\nYou can now run the full analysis:")
        print("  GUI: python grade_analyzer_gui.py")
        print("  CLI: python analyze_results.py")
    else:
        print("⚠ CSV file not found")
        print("Please ensure 'results.csv' is in the same directory")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
