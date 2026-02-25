from fetchers.news_fetcher import OptimizedNewsFetcher
import json

fetcher = OptimizedNewsFetcher("config/rss_sources.json")

# Проверка одного источника вручную
test_url = "https://www.finam.ru/analysis/conews/rsspoint/"
try:
    import feedparser
    feed = feedparser.parse(test_url)
    print(f"Feed entries: {len(feed.entries)}")
    if feed.entries:
        print(f"First title: {feed.entries[0].get('title', '')}")
except Exception as e:
    print(f"Error: {e}")

# Проверка через fetcher
news = fetcher.get_last_news(limit=10)
print(f"Fetcher вернул: {len(news)} новостей")
# После существующего кода добавьте:
print("\n" + "="*50)
print("ТЕСТ SMART BROKER")
print("="*50)

from models.smart_broker import SmartPortfolioBroker
import json

with open('config/settings.json', 'r', encoding='utf-8') as f:
    settings = json.load(f)

broker = SmartPortfolioBroker(settings)
sentiment_data = broker.get_sentiment_history(limit=10)

print(f"\n✅ broker.get_sentiment_history() вернул {len(sentiment_data)} записей")
if sentiment_data:
    print(f"Пример: {sentiment_data[0]}")