#!/usr/bin/env python3
"""
Test script to verify all modules import correctly and basic functionality works.
Run this before running the main analysis to catch any issues early.
"""

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import requests
        print("✓ requests imported")
    except ImportError as e:
        print(f"✗ requests failed: {e}")
        return False
    
    try:
        from bs4 import BeautifulSoup
        print("✓ BeautifulSoup imported")
    except ImportError as e:
        print(f"✗ BeautifulSoup failed: {e}")
        return False
    
    try:
        import nltk
        print("✓ NLTK imported")
    except ImportError as e:
        print(f"✗ NLTK failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported")
    except ImportError as e:
        print(f"✗ NumPy failed: {e}")
        return False
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("✓ Scikit-learn imported")
    except ImportError as e:
        print(f"✗ Scikit-learn failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas imported")
    except ImportError as e:
        print(f"✗ Pandas failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported")
    except ImportError as e:
        print(f"✗ Matplotlib failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✓ Seaborn imported")
    except ImportError as e:
        print(f"✗ Seaborn failed: {e}")
        return False
    
    try:
        from wordcloud import WordCloud
        print("✓ WordCloud imported")
    except ImportError as e:
        print(f"✗ WordCloud failed: {e}")
        return False
    
    try:
        import networkx as nx
        print("✓ NetworkX imported")
    except ImportError as e:
        print(f"✗ NetworkX failed: {e}")
        return False
    
    return True

def test_module_imports():
    """Test that our custom modules can be imported."""
    print("\nTesting custom module imports...")
    
    try:
        from scraper import WikipediaScraper
        print("✓ WikipediaScraper imported")
    except ImportError as e:
        print(f"✗ WikipediaScraper failed: {e}")
        return False
    
    try:
        from text_processor import TextProcessor
        print("✓ TextProcessor imported")
    except ImportError as e:
        print(f"✗ TextProcessor failed: {e}")
        return False
    
    try:
        from similarity_analyser import SimilarityAnalyser
        print("✓ SimilarityAnalyser imported")
    except ImportError as e:
        print(f"✗ SimilarityAnalyser failed: {e}")
        return False
    
    try:
        from visualisation import Visualiser
        print("✓ Visualiser imported")
    except ImportError as e:
        print(f"✗ Visualiser failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of each module."""
    print("\nTesting basic functionality...")
    
    try:
        from scraper import WikipediaScraper
        scraper = WikipediaScraper()
        print("✓ WikipediaScraper instantiated")
    except Exception as e:
        print(f"✗ WikipediaScraper instantiation failed: {e}")
        return False
    
    try:
        from text_processor import TextProcessor
        processor = TextProcessor()
        # test text cleaning
        test_text = "This is a TEST text with [1] citations and http://example.com URLs!"
        cleaned = processor.clean_text(test_text)
        assert len(cleaned) > 0
        print("✓ TextProcessor text cleaning works")
    except Exception as e:
        print(f"✗ TextProcessor failed: {e}")
        return False
    
    try:
        from similarity_analyser import SimilarityAnalyser
        analyser = SimilarityAnalyser()
        print("✓ SimilarityAnalyser instantiated")
    except Exception as e:
        print(f"✗ SimilarityAnalyser instantiation failed: {e}")
        return False
    
    try:
        from visualisation import Visualiser
        visualiser = Visualiser()
        print("✓ Visualiser instantiated")
    except Exception as e:
        print(f"✗ Visualiser instantiation failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Starting code tests...\n")
    
    success = True
    success &= test_imports()
    success &= test_module_imports()
    success &= test_basic_functionality()
    
    print("\n" + "="*50)
    if success:
        print("🎉 All tests passed! Code is ready to run.")
        print("\nNext steps:")
        print("1. Run: jupyter notebook main_analysis.ipynb")
        print("2. Execute cells in order")
        print("3. First run will download NLTK data automatically")
    else:
        print("❌ Some tests failed. Please install missing packages:")
        print("pip install -r requirements.txt")
    print("="*50)

if __name__ == "__main__":
    main()