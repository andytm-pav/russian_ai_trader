"""
Ядро новостной торговли - анализ RSS и генерация торговых сигналов
"""

import json
import time
import feedparser
import threading
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import re

from utils.logger import setup_logger
from models.trader_model import trader_model_instance

logger = setup_logger("NEWS_CORE")


class NewsTraderCore:
    """Ядро для торговли на основе новостей из RSS"""

    def __init__(self, config_path: str = "config/rss_sources.json"):
        self.config = self._load_config(config_path)
        self.model = trader_model_instance
        self.last_fetch_time = 0
        self.news_cache = defaultdict(list)
        self.running = False
        self.fetch_thread = None

        logger.info(f"Инициализировано ядро новостной торговли с {len(self.config['sources'])} источниками")

    def _load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации RSS"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации RSS: {e}")
            return {
                "sources": [],
                "update_interval_minutes": 5,
                "max_news_per_source": 20
            }

    def start_continuous_fetching(self):
        """Запуск непрерывного сбора новостей в отдельном потоке"""
        if self.running:
            return

        self.running = True
        self.fetch_thread = threading.Thread(
            target=self._fetch_loop,
            daemon=True,
            name = "NewsFetcherThread"
        )
        self.fetch_thread.start()
        logger.info("Запущен непрерывный сбор новостей")
        # +++ ДЕТАЛИ О ИНТЕРВАЛЕ +++
        interval = self.config.get('update_interval_minutes', 5)
        logger.debug(f"Интервал обновления: {interval} минут ({interval * 60} секунд)")

    def stop_continuous_fetching(self):
        """Остановка сбора новостей"""
        self.running = False
        if self.fetch_thread:
            self.fetch_thread.join(timeout=5)
        logger.info("Сбор новостей остановлен")

    def _fetch_loop(self):
        """Цикл сбора новостей"""
        update_interval = self.config.get('update_interval_minutes', 5) * 60

        # +++ СЧЕТЧИК ЦИКЛОВ И ЛОГИРОВАНИЕ +++
        cycle_count = 0
        logger.debug(f"Поток сбора новостей запущен, интервал: {update_interval} секунд")

        while self.running:
            try:
                cycle_count += 1
                logger.debug(f"Начало цикла сбора новостей #{cycle_count}")

                # +++ РЕЗУЛЬТАТ СБОРА +++
                news_result = self.fetch_all_news()
                total_news = sum(len(items) for items in news_result.values())
                logger.debug(f"Цикл #{cycle_count}: собрано {total_news} новостей из {len(news_result)} источников")

                time.sleep(update_interval)
            except Exception as e:
                logger.error(f"Ошибка в цикле сбора новостей: {e}")
                time.sleep(60)

    def fetch_all_news(self) -> Dict[str, List[Dict]]:
        """Сбор новостей со всех источников"""
        all_news = {}

        for source in self.config['sources']:
            if not source.get('enabled', True):
                continue

            try:
                news = self._fetch_source_news(source)
                if news:
                    all_news[source['name']] = news
                    logger.debug(f"Источник {source['name']}: собрано {len(news)} новостей")
            except Exception as e:
                logger.error(f"Ошибка сбора новостей из {source['name']}: {e}")

        self.last_fetch_time = time.time()
        return all_news

    def _fetch_source_news(self, source: Dict) -> List[Dict]:
        """Сбор новостей из одного источника"""
        try:
            feed = feedparser.parse(source['url'])
            news_items = []

            max_news = self.config.get('max_news_per_source', 20)

            for entry in feed.entries[:max_news]:
                # Пропускаем старые новости
                published = self._parse_date(entry.get('published', ''))
                if published and (time.time() - published.timestamp()) > 86400:  # Старше суток
                    continue

                news_item = {
                    'title': entry.get('title', ''),
                    'summary': entry.get('summary', ''),
                    'link': entry.get('link', ''),
                    'published': published.isoformat() if published else '',
                    'source': source['name'],
                    'category': source.get('category', 'general'),
                    'priority': source.get('priority', 5)
                }

                # Фильтрация по ключевым словам
                if self._filter_news(news_item):
                    news_items.append(news_item)

            return news_items

        except Exception as e:
            logger.error(f"Ошибка парсинга RSS {source['url']}: {e}")
            return []

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Парсинг даты из различных форматов"""
        try:
            # Пробуем разные форматы
            formats = [
                '%a, %d %b %Y %H:%M:%S %z',
                '%a, %d %b %Y %H:%M:%S %Z',
                '%Y-%m-%dT%H:%M:%S%z',
                '%Y-%m-%d %H:%M:%S'
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue

            return datetime.now()
        except:
            return datetime.now()

    def _filter_news(self, news_item: Dict) -> bool:
        """Фильтрация новостей по ключевым словам"""
        keywords = self.config.get('keywords_filter', {})
        include = keywords.get('include', [])
        exclude = keywords.get('exclude', [])

        text = f"{news_item['title']} {news_item['summary']}".lower()

        # Проверяем исключающие слова
        for word in exclude:
            if word.lower() in text:
                return False

        # Если есть включающие слова, проверяем их
        if include:
            for word in include:
                if word.lower() in text:
                    return True
            return False

        return True

    def analyze_news_sentiment(self, news_items: List[Dict]) -> Dict[str, float]:
        """Анализ тональности новостей"""
        sentiment_dict = self.config.get('sentiment_dictionary', {})
        positive_words = set(word.lower() for word in sentiment_dict.get('positive', []))
        negative_words = set(word.lower() for word in sentiment_dict.get('negative', []))

        ticker_sentiments = defaultdict(list)

        # Загружаем список тикеров для поиска
        try:
            with open('config/tickers.json', 'r', encoding='utf-8') as f:
                tickers_config = json.load(f)
                tickers = [item['ticker'] for item in tickers_config.get('watchlist', [])]
        except:
            tickers = []

        for news in news_items:
            text = f"{news['title']} {news['summary']}".lower()

            # Находим упоминания тикеров
            mentioned_tickers = []
            for ticker in tickers:
                if re.search(rf'\b{ticker}\b', text, re.IGNORECASE):
                    mentioned_tickers.append(ticker)

            # Анализ тональности
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)

            if pos_count + neg_count > 0:
                sentiment = (pos_count - neg_count) / (pos_count + neg_count)
            else:
                sentiment = 0.0

            # Учитываем приоритет источника
            sentiment *= (news.get('priority', 5) / 10.0)

            # Распределяем сентимент по тикерам
            if mentioned_tickers:
                for ticker in mentioned_tickers:
                    ticker_sentiments[ticker].append(sentiment)
            else:
                # Если тикеры не упомянуты, добавляем в общий рыночный
                ticker_sentiments['MARKET'].append(sentiment)

        # Усредняем сентименты
        result = {}
        for ticker, sentiments in ticker_sentiments.items():
            if sentiments:
                result[ticker] = sum(sentiments) / len(sentiments)

        return result

    def get_current_sentiment(self, ticker: str) -> float:
        """Получение текущего сентимента для тикера"""
        try:
            # 1. Проверяем кэш
            if hasattr(self, 'sentiment_cache'):
                cached = self.sentiment_cache.get(ticker)
                if cached and time.time() - cached['timestamp'] < 300:  # 5 минут
                    return cached['sentiment']

            # 2. Собираем свежие новости
            all_news = self.fetch_all_news()
            news_items = []
            for source_news in all_news.values():
                news_items.extend(source_news)

            # 3. Фильтруем новости по тикеру
            ticker_news = []
            for news in news_items:
                text = f"{news['title']} {news['summary']}".lower()
                # Ищем упоминание тикера
                if re.search(rf'\b{ticker}\b', text, re.IGNORECASE):
                    ticker_news.append(news)

            # 4. Анализируем сентимент
            if ticker_news:
                sentiments = self.analyze_news_sentiment(ticker_news)
                sentiment = sentiments.get(ticker, 0.0)
            else:
                # Если новостей нет, используем рыночный сентимент
                if news_items:
                    sentiments = self.analyze_news_sentiment(news_items)
                    sentiment = sentiments.get('MARKET', 0.0)
                else:
                    sentiment = 0.0

            # 5. Кэшируем результат
            if not hasattr(self, 'sentiment_cache'):
                self.sentiment_cache = {}
            self.sentiment_cache[ticker] = {
                'sentiment': sentiment,
                'timestamp': time.time(),
                'news_count': len(ticker_news)
            }

            return sentiment

        except Exception as e:
            logger.error(f"Ошибка получения сентимента для {ticker}: {e}")
            return 0.0

    def generate_trading_signals(self, prices: Dict[str, float]) -> List[Dict]:
        """Генерация торговых сигналов на основе новостей"""
        try:
            # 1. Сбор новостей
            all_news = self.fetch_all_news()

            # 2. Преобразуем в плоский список
            news_items = []
            for source_news in all_news.values():
                news_items.extend(source_news)

            if not news_items:
                return []

            # 3. Анализ тональности
            sentiments = self.analyze_news_sentiment(news_items)

            # 4. Генерация сигналов
            signals = []
            market_sentiment = sentiments.get('MARKET', 0.0)

            for ticker, price in prices.items():
                if ticker == 'MARKET':
                    continue

                ticker_sentiment = sentiments.get(ticker, market_sentiment)

                # Используем существующую модель для принятия решения
                news_texts = [n['title'] for n in news_items if ticker in n.get('title', '')]

                if news_texts:
                    # Кодируем новости через модель
                    news_features = self.model.encode_news(news_texts[:3])

                    # Строим вектор состояния
                    state = self.model.build_state_vector(
                        ticker=ticker,
                        price=price,
                        momentum=0.0,  # Можно получить из данных рынка
                        sentiment=ticker_sentiment,
                        news_features=news_features,
                        market_data={'volume': 0, 'spread': 0.01}
                    )

                    # Выбираем действие
                    action, confidence, _ = self.model.choose_action(state, ticker, price)

                    # Генерируем сигнал если уверенность высокая
                    if confidence > 0.7:
                        signal = {
                            'ticker': ticker,
                            'action': ['BUY', 'HOLD', 'SELL'][action],
                            'confidence': float(confidence),
                            'sentiment': float(ticker_sentiment),
                            'price': float(price),
                            'reason': 'news_analysis',
                            'timestamp': datetime.now().isoformat(),
                            'news_count': len(news_texts)
                        }

                        signals.append(signal)
                        logger.info(f"Сгенерирован сигнал: {ticker} {signal['action']} "
                                    f"(conf={confidence:.2f}, sent={ticker_sentiment:.2f})")

            return signals

        except Exception as e:
            logger.error(f"Ошибка генерации сигналов: {e}")
            return []

    def get_news_summary(self) -> Dict:
        """Получение сводки новостей"""
        return {
            'last_fetch': datetime.fromtimestamp(self.last_fetch_time).isoformat() if self.last_fetch_time else None,
            'sources_count': len([s for s in self.config['sources'] if s.get('enabled', True)]),
            'cache_size': sum(len(v) for v in self.news_cache.values())
        }