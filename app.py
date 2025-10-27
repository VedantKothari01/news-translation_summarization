import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from news_fetcher import fetch_latest_news
from translate import TranslationPipeline
from summarize import SummarizationPipeline
from language_detector import detect_language

st.set_page_config(
    page_title="News Hub",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'articles' not in st.session_state:
    st.session_state.articles = []
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0
if 'translator' not in st.session_state:
    st.session_state.translator = None
if 'summarizer' not in st.session_state:
    st.session_state.summarizer = None
if 'processed_articles' not in st.session_state:
    st.session_state.processed_articles = {}
if 'auto_fetched' not in st.session_state:
    st.session_state.auto_fetched = False
if 'target_lang' not in st.session_state:
    st.session_state.target_lang = 'hi'

# Load models once
if st.session_state.translator is None:
    with st.spinner("Loading translation model..."):
        st.session_state.translator = TranslationPipeline()
if st.session_state.summarizer is None:
    with st.spinner("Loading summarization model..."):
        st.session_state.summarizer = SummarizationPipeline()

st.title("🌐 Multilingual News Hub")
st.caption("AI-Powered Translation & Summarization | Real-Time News")

st.markdown("---")

with st.sidebar:
    st.header("⚙️ Settings")
    
    target_lang = st.selectbox(
        "Target Language",
        options=['en', 'hi', 'es', 'fr', 'de', 'zh', 'ar', 'ja', 'pt', 'ru', 'ta', 'te', 'mr', 'gu'],
        format_func=lambda x: {
            'en': 'English', 'hi': 'Hindi', 'es': 'Spanish', 'fr': 'French',
            'de': 'German', 'zh': 'Chinese', 'ar': 'Arabic', 'ja': 'Japanese',
            'pt': 'Portuguese', 'ru': 'Russian', 'ta': 'Tamil', 'te': 'Telugu',
            'mr': 'Marathi', 'gu': 'Gujarati'
        }[x],
        index=1
    )
    
    # Only clear cache when language actually changes
    if target_lang != st.session_state.target_lang:
        st.session_state.target_lang = target_lang
        # Keep cache for current language only
        st.session_state.processed_articles = {
            k: v for k, v in st.session_state.processed_articles.items() 
            if k.endswith(f'_{target_lang}')
        }
    
    st.markdown("---")
    
    if st.button("🔄 Refresh News", type="primary", use_container_width=True):
        st.session_state.auto_fetched = False
        st.session_state.articles = []
        st.session_state.current_idx = 0
        st.session_state.processed_articles = {}
    
    st.markdown("---")
    st.markdown("**Translation:** mBART-50 (610M params)")
    st.markdown("**Summarization:** BART-CNN (406M params)")

# Auto-fetch latest news on first load
if not st.session_state.auto_fetched and len(st.session_state.articles) == 0:
    with st.spinner("Fetching latest trending news..."):
        try:
            raw_articles = fetch_latest_news(category='general', count=5)
            if raw_articles:
                st.session_state.articles = raw_articles
                st.session_state.auto_fetched = True
                st.rerun()
            else:
                st.error("No articles retrieved")
        except Exception as e:
            st.error(f"Unable to fetch news: {e}")

if st.session_state.articles:
    article = st.session_state.articles[st.session_state.current_idx]
    
    # Navigation buttons
    col1, col2, col3 = st.columns([2, 3, 2])
    
    with col1:
        if st.button("⏮️ Previous", disabled=st.session_state.current_idx == 0):
            st.session_state.current_idx -= 1
            st.rerun()
    
    with col2:
        st.markdown(f"**Article {st.session_state.current_idx + 1} of {len(st.session_state.articles)}**")
    
    with col3:
        if st.button("Next ⏭️", disabled=st.session_state.current_idx >= len(st.session_state.articles) - 1):
            st.session_state.current_idx += 1
            st.rerun()
    
    st.markdown("---")
    
    # Create unique cache key using article title + language
    cache_key = f"{article['title'][:100]}_{target_lang}"
    
    # Check if already processed
    if cache_key in st.session_state.processed_articles:
        processed = st.session_state.processed_articles[cache_key]
    else:
        # Process the article
        with st.spinner("Translating and summarizing..."):
            try:
                source_lang = detect_language(article['title'])
                
                article_content = article.get('content', article.get('description', 'No content'))
                
                # Translate title
                try:
                    translated_title = st.session_state.translator.translate(
                        article['title'],
                        source_lang,
                        target_lang
                    )
                except:
                    translated_title = article['title']
                
                try:
                    translated_content = st.session_state.translator.translate(
                        article_content,
                        source_lang,
                        target_lang,
                        max_length=1024
                    )
                except:
                    translated_content = article_content
                
                try:
                    summary_english = st.session_state.summarizer.summarize(
                        article_content,
                        max_length=200,
                        min_length=50
                    )
                    
                    if target_lang != 'en' and summary_english and len(summary_english) > 20:
                        summary_translated = st.session_state.translator.translate(
                            summary_english,
                            'en',
                            target_lang
                        )
                    else:
                        summary_translated = summary_english or article_content[:200]
                except:
                    summary_translated = translated_content[:200]
                
                processed = {
                    'title': translated_title,
                    'summary': summary_translated,
                    'content': translated_content,
                    'source_lang': source_lang
                }
                
                # Cache the result
                st.session_state.processed_articles[cache_key] = processed
                
            except Exception as e:
                st.error(f"Processing error: {e}")
                processed = {
                    'title': article['title'],
                    'summary': article.get('content', article.get('description', ''))[:200],
                    'content': article.get('content', article.get('description', '')),
                    'source_lang': 'en'
                }
    
    # Display the article
    st.markdown(f"### {processed['title']}")
    st.caption(f"📰 {article.get('source', 'Unknown')}  |  🕐 {article.get('published_at', '')[:10]}  |  🌍 {processed['source_lang'].upper()}")
    
    # Display image if available
    image_url = article.get('image_url')
    if image_url:
        try:
            st.image(image_url, width=600)
        except Exception:
            pass
    
    st.markdown("---")
    
    # Display summary
    st.markdown("#### Summary")
    st.write(processed['summary'])
    
    st.markdown("---")
    
    # Full article in expander
    with st.expander("📖 Read Full Translated Article"):
        st.markdown(processed['content'])
        if article.get('url'):
            st.markdown(f"[🔗 View Original Article]({article['url']})")
    
    st.markdown("---")
    
else:
    st.info("👈 Click 'Refresh News' to fetch articles")

st.caption("Deep Learning & NLP | Multilingual News Translation & Summarization")