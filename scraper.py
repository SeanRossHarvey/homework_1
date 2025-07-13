import requests
from bs4 import BeautifulSoup
import json
import time
from typing import List, Dict
import os

class WikipediaScraper:
    """Scraper for Wikipedia articles across different topics."""
    
    def __init__(self):
        self.base_url = "https://en.wikipedia.org/wiki/"
        self.category_url = "https://en.wikipedia.org/wiki/Category:"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
    def get_category_pages(self, category: str, limit: int = 50) -> List[str]:
        """Get article URLs from a Wikipedia category."""
        url = f"{self.category_url}{category}"
        pages = []
        
        # scrape category pages until we have enough articles
        while len(pages) < limit:
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # find pages in category section
            pages_div = soup.find('div', {'id': 'mw-pages'})
            if pages_div:
                links = pages_div.find_all('a')
                for link in links:
                    if len(pages) >= limit:
                        break
                    href = link.get('href')
                    # only add actual article links, not special pages
                    if href and href.startswith('/wiki/') and ':' not in href:
                        pages.append(f"https://en.wikipedia.org{href}")
            
            # check for pagination
            next_link = soup.find('a', text='next page')
            if next_link and len(pages) < limit:
                url = f"https://en.wikipedia.org{next_link.get('href')}"
                time.sleep(0.5)  # rate limiting to be polite
            else:
                break
                
        return pages[:limit]
    
    def scrape_article(self, url: str) -> Dict[str, str]:
        """Scrape title and first paragraph from a Wikipedia article."""
        try:
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # extract article title
            title = soup.find('h1', {'class': 'firstHeading'}).text.strip()
            
            # extract first meaningful paragraph
            content_div = soup.find('div', {'id': 'mw-content-text'})
            paragraphs = content_div.find_all('p', recursive=False)
            
            first_para = ""
            for p in paragraphs:
                text = p.text.strip()
                # skip empty paragraphs and coordinate info
                if text and not text.startswith('Coordinates:'):
                    first_para = text
                    break
            
            return {
                'url': url,
                'title': title,
                'first_paragraph': first_para
            }
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
    
    def scrape_topics(self, topics: Dict[str, str], articles_per_topic: int = 50) -> Dict[str, List[Dict]]:
        """Scrape articles for multiple topics."""
        all_data = {}
        
        for topic_name, category_name in topics.items():
            print(f"\nScraping {topic_name} articles...")
            
            # get article urls from category page
            urls = self.get_category_pages(category_name, articles_per_topic)
            print(f"Found {len(urls)} URLs for {topic_name}")
            
            # scrape each article in the category
            articles = []
            for i, url in enumerate(urls):
                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(urls)}")
                
                article_data = self.scrape_article(url)
                if article_data:
                    article_data['topic'] = topic_name
                    articles.append(article_data)
                
                time.sleep(0.5)  # rate limiting
            
            all_data[topic_name] = articles
            print(f"Scraped {len(articles)} articles for {topic_name}")
        
        return all_data
    
    def save_data(self, data: Dict[str, List[Dict]], filename: str = 'scraped_data.json'):
        """Save scraped data to JSON file."""
        os.makedirs('data', exist_ok=True)
        filepath = os.path.join('data', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\nData saved to {filepath}")
        
        # print summary statistics
        total_articles = sum(len(articles) for articles in data.values())
        print(f"Total articles scraped: {total_articles}")
        for topic, articles in data.items():
            print(f"  {topic}: {len(articles)} articles")


if __name__ == "__main__":
    # define distinct topics for comparison
    topics = {
        'Politics': 'Politics',
        'Technology': 'Computer_science',
        'Science': 'Physics',
        'History': 'Ancient_history'
    }
    
    scraper = WikipediaScraper()
    data = scraper.scrape_topics(topics, articles_per_topic=50)
    scraper.save_data(data)