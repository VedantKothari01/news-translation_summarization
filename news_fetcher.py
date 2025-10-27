import requests
from datetime import datetime
import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

def fetch_latest_news(category='general', count=5, api_key=None) -> List[Dict]:
    """
    Fetch latest news articles from NewsAPI.
    
    Args:
        category: News category
        count: Number of articles to fetch
        api_key: NewsAPI key
    
    Returns:
        List of article dictionaries
    """
    
    if api_key is None:
        api_key = os.getenv('NEWS_API_KEY')
    
    if not api_key or api_key == 'YOUR_API_KEY_HERE':
        raise ValueError("NEWS_API_KEY not found in environment. Please add it to .env file.")
    
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        'apiKey': api_key,
        'category': category,
        'language': 'en',
        'pageSize': count
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] != 'ok':
            raise Exception(f"API Error: {data.get('message', 'Unknown error')}")
        
        articles = []
        for article in data.get('articles', []):
            # Filter out invalid articles
            if article.get('title') == '[Removed]' or not article.get('title'):
                continue
            if not article.get('content') and not article.get('description'):
                continue
            
            articles.append({
                'title': article.get('title', 'No title'),
                'content': article.get('content') or article.get('description', 'No content available'),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'url': article.get('url', ''),
                'published_at': article.get('publishedAt', datetime.now().isoformat()),
                'image_url': article.get('urlToImage')
            })
            
            if len(articles) >= count:
                break
        
        if not articles:
            raise Exception("No valid articles retrieved from API")
        
        return articles
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {e}")
    except Exception as e:
        raise Exception(f"Error fetching news: {e}")
