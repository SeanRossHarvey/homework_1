import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import List, Dict, Tuple
import json
import os
import re

# download required nltk data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class TextProcessor:
    """Process text data for analysis."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # convert to lowercase for consistency
        text = text.lower()
        
        # remove urls
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # remove wikipedia citations like [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenise(self, text: str) -> List[str]:
        """Tokenise text and remove stopwords."""
        # clean text first
        text = self.clean_text(text)
        
        # tokenise into words
        tokens = word_tokenize(text)
        
        # remove stopwords and short tokens (length <= 2)
        tokens = [token for token in tokens 
                 if token not in self.stop_words 
                 and len(token) > 2]
        
        return tokens
    
    def process_documents(self, documents: List[Dict]) -> Tuple[List[str], List[str], List[str]]:
        """Process a list of documents."""
        processed_texts = []
        titles = []
        topics = []
        
        for doc in documents:
            # combine title and first paragraph for better context
            full_text = f"{doc['title']} {doc['first_paragraph']}"
            
            # tokenise and rejoin for vectorisation
            tokens = self.tokenise(full_text)
            processed_text = ' '.join(tokens)
            
            if processed_text:  # only add non-empty documents
                processed_texts.append(processed_text)
                titles.append(doc['title'])
                topics.append(doc['topic'])
        
        return processed_texts, titles, topics
    
    def create_dtm(self, texts: List[str], method: str = 'tfidf') -> Tuple[np.ndarray, List[str]]:
        """Create Document-Term Matrix using specified method."""
        if method == 'tfidf':
            vectoriser = TfidfVectorizer(
                max_features=1000,  # limit vocabulary size
                min_df=2,           # ignore rare terms
                max_df=0.8,         # ignore very common terms
                ngram_range=(1, 2)  # include unigrams and bigrams
            )
        elif method == 'count':
            vectoriser = CountVectorizer(
                max_features=1000,
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2)
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        dtm = vectoriser.fit_transform(texts)
        feature_names = vectoriser.get_feature_names_out()
        
        return dtm, feature_names
    
    def load_and_process_data(self, filepath: str = 'data/scraped_data.json') -> Dict:
        """Load scraped data and process it."""
        # load scraped data from json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # flatten all documents across topics
        all_documents = []
        for topic, articles in data.items():
            all_documents.extend(articles)
        
        print(f"Total documents: {len(all_documents)}")
        
        # process documents (clean, tokenise, etc.)
        texts, titles, topics = self.process_documents(all_documents)
        
        print(f"Processed documents: {len(texts)}")
        
        # create document-term matrix using tf-idf
        dtm_tfidf, features = self.create_dtm(texts, method='tfidf')
        
        print(f"DTM shape: {dtm_tfidf.shape}")
        
        return {
            'texts': texts,
            'titles': titles,
            'topics': topics,
            'dtm_tfidf': dtm_tfidf,
            'features': features,
            'raw_documents': all_documents
        }
    
    def save_processed_data(self, processed_data: Dict, filename: str = 'processed_data.npz'):
        """Save processed data for later use."""
        os.makedirs('data', exist_ok=True)
        filepath = os.path.join('data', filename)
        
        # save numpy arrays for efficient loading
        np.savez(filepath,
                 dtm_tfidf=processed_data['dtm_tfidf'].toarray(),
                 features=processed_data['features'],
                 topics=processed_data['topics'],
                 titles=processed_data['titles'])
        
        # save texts separately as json for readability
        texts_filepath = os.path.join('data', 'processed_texts.json')
        with open(texts_filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'texts': processed_data['texts'],
                'titles': processed_data['titles'],
                'topics': processed_data['topics']
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Processed data saved to {filepath} and {texts_filepath}")


if __name__ == "__main__":
    processor = TextProcessor()
    processed_data = processor.load_and_process_data()
    processor.save_processed_data(processed_data)