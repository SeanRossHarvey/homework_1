# Text data collection and analysis

This project implements a complete pipeline for collecting, processing, and analysing text data from Wikipedia across different topics.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the analysis:
```bash
jupyter notebook main_analysis.ipynb
```

## Project structure

- `scraper.py` - Wikipedia article scraper
- `text_processor.py` - Text preprocessing and DTM creation
- `similarity_analyser.py` - Document similarity and clustering analysis
- `visualisation.py` - Plotting utilities
- `main_analysis.ipynb` - Main analysis notebook
- `data/` - Directory for scraped and processed data

## Pipeline steps

1. **Data collection**: Scrapes 50 articles per topic from Wikipedia
2. **Text processing**: Tokenisation, stopword removal, TF-IDF vectorisation
3. **Similarity analysis**: Cosine similarity calculation and threshold-based clustering
4. **Visualisation**: Network graphs, confusion matrices, and word clouds

## Topics analysed

- Politics
- Technology (computer science)
- Science (physics)
- History (ancient history)