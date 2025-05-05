import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST, TimeFrame, APIError
import logging
import requests
import tweepy
from transformers import pipeline
from dotenv import load_dotenv

# === .env laden ===
load_dotenv()

# === Logging Setup ===
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# === API-Konfiguration ===
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"

api = REST(API_KEY, API_SECRET, BASE_URL)

# === Sentiment-Model vorbereiten (z. B. FinBERT, RoBERTa) ===
sentiment_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# === Twitter Setup ===
twitter_bearer = os.getenv("TWITTER_BEARER")
twitter_headers = {"Authorization": f"Bearer {twitter_bearer}"}

def fetch_tweets(query, max_results=10):
    url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results={max_results}&tweet.fields=text"
    resp = requests.get(url, headers=twitter_headers)
    if resp.status_code == 200:
        return [tweet['text'] for tweet in resp.json().get('data', [])]
    return []

# === NewsAPI Setup ===
newsapi_key = os.getenv("NEWSAPI_KEY")

def fetch_news(query):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&apiKey={newsapi_key}"
    resp = requests.get(url)
    if resp.status_code == 200:
        articles = resp.json().get("articles", [])
        return [a["title"] for a in articles]
    return []

# === Finnhub Setup ===
finnhub_key = os.getenv("FINNHUB_API_KEY")

def fetch_finnhub_news():
    url = f"https://finnhub.io/api/v1/news?category=general&token={finnhub_key}"
    resp = requests.get(url)
    if resp.status_code == 200:
        return [a['headline'] for a in resp.json()]
    return []

# === CoinGecko Setup ===
coingecko_base = "https://api.coingecko.com/api/v3"

def fetch_coingecko_news(asset_id="bitcoin"):
    url = f"{coingecko_base}/coins/{asset_id}/status_updates"
    resp = requests.get(url)
    if resp.status_code == 200:
        updates = resp.json().get("status_updates", [])
        return [u['description'] for u in updates]
    return []

def analyze_sentiment(texts):
    results = sentiment_model(texts)
    scores = [1 if r['label'] == 'positive' else -1 if r['label'] == 'negative' else 0 for r in results]
    return sum(scores) / len(scores) if scores else 0

# Beispiel GUI-Block zur Anzeige
st.sidebar.markdown("### 🧠 Sentiment-Analyse")
asset = st.sidebar.text_input("🧪 Asset für Sentiment-Suche (z. B. AAPL, Bitcoin):", "AAPL")

if st.sidebar.button("Analysiere Sentiment"):
    with st.spinner("Tweets laden..."):
        tweets = fetch_tweets(asset)
    with st.spinner("News laden..."):
        news = fetch_news(asset)
        finnhub = fetch_finnhub_news()
        coingecko = fetch_coingecko_news(asset.lower())

    combined_texts = tweets + news + finnhub + coingecko
    if combined_texts:
        sentiment_score = analyze_sentiment(combined_texts[:20])
        st.sidebar.success(f"📈 Durchschnittliches Sentiment: {sentiment_score:.2f}")
    else:
        st.sidebar.warning("Keine Texte gefunden für Analyse.")

# Rest deines Codes bleibt unverändert …
