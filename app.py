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
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

if st.session_state.translator is None:
    st.session_state.translator = TranslationPipeline()
if st.session_state.summarizer is None:
    st.session_state.summarizer = SummarizationPipeline()

st.title("🌍 Multilingual News Hub")
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
    
    news_category = st.selectbox(
        "Category",
        ['general', 'technology', 'business', 'science', 'health', 'sports', 'entertainment']
    )
    
    num_articles = st.slider("Articles", 3, 10, 5)
    
    st.markdown("---")
    
    if st.button("🔄 Refresh News", type="primary", use_container_width=True):
        st.session_state.auto_fetched = False
        st.session_state.articles = []
        st.session_state.current_idx = 0
    
    st.markdown("---")
    st.markdown("**Translation:** mBART-50")
    st.markdown("**Summarization:** BART-Large-CNN")

if not st.session_state.auto_fetched and len(st.session_state.articles) == 0:
    with st.spinner("Fetching latest news..."):
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
    
    cache_key = f"{st.session_state.current_idx}_{target_lang}"
    
    if cache_key in st.session_state.processed_articles:
        processed = st.session_state.processed_articles[cache_key]
    else:
        with st.spinner("Processing article..."):
            try:
                source_lang = detect_language(article['title'])
                
                article_content = article.get('content', article.get('description', 'No content'))
                
                translated_title = st.session_state.translator.translate(
                    article['title'],
                    source_lang,
                    target_lang
                )
                
                translated_content = st.session_state.translator.translate(
                    article_content,
                    source_lang,
                    target_lang
                )
                
                summary_text = article_content if source_lang == target_lang else translated_content
                
                summary = st.session_state.summarizer.summarize(
                    summary_text,
                    max_length=300,
                    min_length=80
                )
                
                processed = {
                    'title': translated_title,
                    'summary': summary,
                    'content': translated_content,
                    'source_lang': source_lang
                }
                
                st.session_state.processed_articles[cache_key] = processed
            except Exception as e:
                st.error(f"Processing error: {e}")
                processed = {
                    'title': article['title'],
                    'summary': article.get('content', article.get('description', ''))[:150],
                    'content': article.get('content', article.get('description', '')),
                    'source_lang': 'en'
                }
    
    st.markdown(f"### {processed['title']}")
    st.caption(f"📍 {article.get('source', 'Unknown')}  |  🕒 {article.get('published_at', '')[:10]}  |  🌐 {processed['source_lang'].upper()}")
    
    image_url = article.get('image_url')
    if image_url:
        try:
            st.image(image_url, width=600)
        except Exception:
            pass
    
    st.markdown("---")
    
    st.markdown("#### Summary")
    st.write(processed['summary'])
    
    st.markdown("---")
    
    with st.expander("📖 Read Full Article"):
        st.write(processed['content'])
        if article.get('url'):
            st.markdown(f"[🔗 Original Article]({article['url']})")
    
    st.markdown("---")
    
else:
    st.info("👈 Click 'Refresh News' to fetch articles")

st.caption("Deep Learning & NLP | Multilingual News Translation & Summarization")
