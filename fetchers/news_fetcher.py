"""
Сбор новостей с различных источников - ОПТИМИЗИРОВАННАЯ ВЕРСИЯ
"""

import json
import time
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import threading

try:
    from ahocorasick import Automaton
    AHOCORASICK_AVAILABLE = True
except ImportError:
    AHOCORASICK_AVAILABLE = False
    import re

from utils.logger import get_logger

logger = get_logger("NEWS_FETCHER_OPT")


class OptimizedNewsFetcher:
    """Оптимизированный класс для сбора новостей"""

    def __init__(self, config_path: str = "config/rss_sources.json"):
        self.config = self._load_config(config_path)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Кэш новостей (TTL из конфига)
        self.news_cache = []
        self.cache_ttl = self.config.get('cache_ttl_seconds', 300)  # 5 минут
        self.last_fetch_time = 0
        self.fetch_interval = self.config.get('update_interval_minutes', 5) * 60

        # 🔥 МНОГОПОТОЧНОСТЬ
        self.max_workers = self.config.get('max_workers', 5)

        # 🔥 БЫСТРАЯ ФИЛЬТРАЦИЯ (Aho-Corasick)
        self._build_fast_filters()

        # 🔥 КЭШ СЕНТИМЕНТА
        self.sentiment_cache = {}
        self.sentiment_cache_ttl = self.config.get('sentiment_cache_ttl', 3600)  # 1 час

        # 🔥 ПРЕДОБУЧЕННАЯ МОДЕЛЬ
        self.sentiment_model = None
        self.use_ml_model = self.config.get('use_ml_model', False)
        if self.use_ml_model:
            self._init_sentiment_model()

        # Статистика
        self.stats = {
            'total_fetches': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'filter_hits': 0,
            'avg_fetch_time': 0
        }

        logger.info(
            f"Инициализирован OptimizedNewsFetcher:\n"
            f"  - Источников: {len(self.config.get('sources', []))}\n"
            f"  - Многопоточность: {self.max_workers} workers\n"
            f"  - Фильтрация: {'Aho-Corasick' if AHOCORASICK_AVAILABLE else 'Regex'}\n"
            f"  - ML модель: {'Да' if self.use_ml_model else 'Нет'}"
        )

    def _load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Добавляем параметры оптимизации по умолчанию
            config.setdefault('max_workers', 5)
            config.setdefault('cache_ttl_seconds', 300)
            config.setdefault('sentiment_cache_ttl', 3600)
            config.setdefault('use_ml_model', False)
            config.setdefault('batch_size', 32)

            return config
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            return {
                'sources': [],
                'update_interval_minutes': 5,
                'max_workers': 5,
                'cache_ttl_seconds': 300,
                'sentiment_cache_ttl': 3600,
                'use_ml_model': False,
                'batch_size': 32
            }

    def _build_fast_filters(self):
        """Построение быстрых фильтров с Aho-Corasick"""
        keywords_config = self.config.get('keywords_filter', {})
        self.include_words = keywords_config.get('include', [])
        self.exclude_words = keywords_config.get('exclude', [])

        if AHOCORASICK_AVAILABLE and (self.include_words or self.exclude_words):
            # Строим автомат для быстрого поиска
            self.filter_automaton = Automaton()

            for word in self.include_words:
                self.filter_automaton.add_word(word.lower(), ('include', word))
            for word in self.exclude_words:
                self.filter_automaton.add_word(word.lower(), ('exclude', word))

            self.filter_automaton.make_automaton()
            self.use_automaton = True
            logger.debug(f"Построен Aho-Corasick автомат с {len(self.include_words) + len(self.exclude_words)} словами")
        else:
            # Fallback на регулярки
            self.use_automaton = False
            if self.include_words:
                self.include_pattern = re.compile(
                    '|'.join(map(re.escape, self.include_words)),
                    re.IGNORECASE
                )
            if self.exclude_words:
                self.exclude_pattern = re.compile(
                    '|'.join(map(re.escape, self.exclude_words)),
                    re.IGNORECASE
                )
            logger.debug("Использую regex для фильтрации (Aho-Corasick не доступен)")

    def _init_sentiment_model(self):
        """Инициализация предобученной модели"""
        try:
            from transformers import pipeline

            model_name = self.config.get('sentiment_model', 'mxlcw/rubert-tiny2-russian-financial-sentiment')

            # Загружаем модель один раз
            self.sentiment_model = pipeline(
                "text-classification",
                model=model_name,
                device=-1  # CPU, для GPU укажите 0
            )

            logger.info(f"Загружена модель сентимента: {model_name}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}, использую словарный метод")
            self.use_ml_model = False

    def get_last_news(self, limit: int = 50) -> List[Dict]:
        """Получение последних новостей с многопоточным сбором"""
        current_time = time.time()
        start_time = current_time

        # Статистика
        self.stats['total_fetches'] += 1

        # Проверка кэша
        if (current_time - self.last_fetch_time < self.fetch_interval and
                self.news_cache):
            self.stats['cache_hits'] += 1
            # logger.debug(f"Кэш HIT: {len(self.news_cache)} новостей")
            return self.news_cache[:limit]

        self.stats['cache_misses'] += 1

        try:
            sources = [s for s in self.config.get('sources', []) if s.get('enabled', True)]

            if not sources:
                return []

            # 🔥 МНОГОПОТОЧНЫЙ СБОР
            all_news = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Запускаем все задачи
                future_to_source = {
                    executor.submit(self._fetch_from_source, source): source
                    for source in sources
                }

                # Собираем результаты по мере готовности
                for future in as_completed(future_to_source):
                    source = future_to_source[future]
                    try:
                        source_news = future.result(timeout=15)
                        all_news.extend(source_news)
                        logger.debug(f"Источник {source['name']}: {len(source_news)} новостей")
                    except Exception as e:
                        logger.error(f"Ошибка в источнике {source['name']}: {e}")

            # Сортировка и дедупликация
            all_news.sort(key=lambda x: x.get('ts', 0), reverse=True)
            unique_news = self._fast_remove_duplicates(all_news)

            # Обновление кэша
            self.news_cache = unique_news
            self.last_fetch_time = current_time

            # Статистика времени
            elapsed = time.time() - start_time
            self.stats['avg_fetch_time'] = (self.stats['avg_fetch_time'] * 0.9 + elapsed * 0.1)

            logger.info(
                f"Собрано {len(unique_news)} новостей за {elapsed:.2f}с "
                f"(avg: {self.stats['avg_fetch_time']:.2f}с)"
            )

            return unique_news[:limit]

        except Exception as e:
            logger.error(f"Ошибка получения новостей: {e}")
            return self.news_cache[:limit] if self.news_cache else []

    def _fetch_from_source(self, source: Dict) -> List[Dict]:
        """Сбор новостей из одного источника (вызывается в потоках)"""
        source_type = source.get('type', 'rss')

        try:
            if source_type == 'rss':
                return self._fetch_rss(source)
            elif source_type == 'api':
                return self._fetch_api(source)
            elif source_type == 'web':
                return self._fetch_web(source)
            else:
                return []
        except Exception as e:
            logger.error(f"Ошибка в _fetch_from_source {source.get('name')}: {e}")
            return []

    def _fetch_rss(self, source: Dict) -> List[Dict]:
        """Оптимизированный парсинг RSS"""
        try:
            import feedparser

            feed = feedparser.parse(source['url'])
            news_items = []
            max_items = source.get('max_items', 20)

            # Текущее время в UTC
            now_aware = datetime.now(timezone.utc)

            for entry in feed.entries[:max_items]:
                # Парсинг даты
                published = self._parse_date(entry.get('published', ''))

                if published:
                    # Фильтр по времени (старше 3 дней)
                    if published.tzinfo is None:
                        published = published.replace(tzinfo=timezone.utc)

                    age_days = (now_aware - published).days
                    if age_days > 3:
                        continue

                news_item = {
                    'title': entry.get('title', ''),
                    'summary': entry.get('summary', '') or entry.get('description', ''),
                    'link': entry.get('link', ''),
                    'published': published.isoformat() if published else '',
                    'ts': self._get_timestamp(published),
                    'source': source['name'],
                    'category': source.get('category', 'general'),
                    'priority': source.get('priority', 5)
                }

                # 🔥 БЫСТРАЯ ФИЛЬТРАЦИЯ
                if self._fast_filter(news_item):
                    news_items.append(news_item)

            return news_items

        except Exception as e:
            logger.error(f"Ошибка RSS {source.get('url')}: {e}")
            return []

    def _fast_filter(self, news_item: Dict) -> bool:
        """Быстрая фильтрация через Aho-Corasick или regex"""
        text = f"{news_item['title']} {news_item.get('summary', '')}".lower()

        # 🔥 СТАТИСТИКА
        self.stats['filter_hits'] += 1

        # Проверка исключающих слов
        if self.exclude_words:
            if self.use_automaton:
                # Aho-Corasick
                for _, (word_type, word) in self.filter_automaton.iter(text):
                    if word_type == 'exclude':
                        return False
            else:
                # Regex
                if hasattr(self, 'exclude_pattern') and self.exclude_pattern.search(text):
                    return False

        # Если есть включающие слова
        if self.include_words:
            if self.use_automaton:
                # Aho-Corasick
                for _, (word_type, word) in self.filter_automaton.iter(text):
                    if word_type == 'include':
                        return True
                return False
            else:
                # Regex
                if hasattr(self, 'include_pattern') and self.include_pattern.search(text):
                    return True
                return False

        return True

    def analyze_sentiment_batch(self, news_items: List[Dict]) -> List[Dict]:
        """Пакетный анализ сентимента с кэшированием"""
        if not news_items:
            return []

        # Если используем ML модель
        if self.use_ml_model and self.sentiment_model:
            return self._analyze_sentiment_ml_batch(news_items)

        # Иначе используем словарный метод
        return self._analyze_sentiment_lexicon(news_items)

    def _analyze_sentiment_ml_batch(self, news_items: List[Dict]) -> List[Dict]:
        """ML сентимент с пакетной обработкой"""
        batch_size = self.config.get('batch_size', 32)

        # Подготовка текстов для анализа
        texts = []
        news_indices = []

        for i, news in enumerate(news_items):
            text = f"{news['title']} {news.get('summary', '')}"

            # Проверка кэша
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cached = self.sentiment_cache.get(text_hash)

            if cached and (time.time() - cached['timestamp'] < self.sentiment_cache_ttl):
                # Кэш hit
                news['sentiment'] = cached['sentiment']
                news['sentiment_score'] = cached['score']
            else:
                # Нужно обработать
                texts.append(text[:512])  # Обрезаем
                news_indices.append(i)

        # Пакетная обработка
        if texts:
            for start_idx in range(0, len(texts), batch_size):
                batch_texts = texts[start_idx:start_idx + batch_size]
                try:
                    results = self.sentiment_model(batch_texts)

                    for j, result in enumerate(results):
                        idx = news_indices[start_idx + j]
                        news_item = news_items[idx]

                        # Преобразование результата
                        label = result['label'].lower()
                        score = result['score']

                        if 'positive' in label:
                            sentiment = score
                        elif 'negative' in label:
                            sentiment = -score
                        else:
                            sentiment = 0.0

                        news_item['sentiment'] = sentiment
                        news_item['sentiment_score'] = score

                        # Сохраняем в кэш
                        text_hash = hashlib.md5(texts[start_idx + j].encode()).hexdigest()
                        self.sentiment_cache[text_hash] = {
                            'sentiment': sentiment,
                            'score': score,
                            'timestamp': time.time()
                        }

                except Exception as e:
                    logger.error(f"Ошибка ML batch {start_idx}: {e}")
                    for j in range(start_idx, min(start_idx + batch_size, len(texts))):
                        idx = news_indices[j]
                        news_items[idx]['sentiment'] = 0.0
                        news_items[idx]['sentiment_score'] = 0.0

        return news_items

    def _analyze_sentiment_lexicon(self, news_items: List[Dict]) -> List[Dict]:
        """Словарный анализ сентимента (быстрый fallback)"""
        sentiment_dict = self.config.get('sentiment_dictionary', {})
        positive_words = set(w.lower() for w in sentiment_dict.get('positive', []))
        negative_words = set(w.lower() for w in sentiment_dict.get('negative', []))

        for news in news_items:
            text = f"{news['title']} {news.get('summary', '')}".lower()

            pos_count = sum(1 for w in positive_words if w in text)
            neg_count = sum(1 for w in negative_words if w in text)

            if pos_count + neg_count > 0:
                sentiment = (pos_count - neg_count) / (pos_count + neg_count)
            else:
                sentiment = 0.0

            # Учитываем приоритет источника
            sentiment *= (news.get('priority', 5) / 10.0)

            news['sentiment'] = sentiment
            news['sentiment_score'] = abs(sentiment)

        return news_items

    def _fast_remove_duplicates(self, news_items: List[Dict]) -> List[Dict]:
        """Быстрое удаление дубликатов с использованием множества"""
        seen = set()
        unique = []

        for item in news_items:
            # Нормализация заголовка
            title = re.sub(r'[^\w\s]', '', item['title'].lower()).strip()
            if title and title not in seen:
                seen.add(title)
                unique.append(item)

        return unique

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Парсинг даты с кэшированием форматов"""
        if not date_str or date_str.strip() == '':
            # Если дата пустая - возвращаем текущее время
            logger.debug("Пустая дата в новости, использую текущее время")
            return datetime.now(timezone.utc)

        # Кэш форматов
        if not hasattr(self, '_date_formats'):
            self._date_formats = [
                '%a, %d %b %Y %H:%M:%S %z',
                '%a, %d %b %Y %H:%M:%S %Z',
                '%Y-%m-%dT%H:%M:%S%z',
                '%Y-%m-%d %H:%M:%S%z',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S',
                '%d.%m.%Y %H:%M',
                '%d/%m/%Y %H:%M',
                '%Y-%m-%d',
                '%d.%m.%Y'
            ]

        date_str = date_str.strip()

        for fmt in self._date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue

        # Если ни один формат не подошёл
        logger.debug(f"Не удалось распарсить дату: {date_str}, использую текущее время")
        return datetime.now(timezone.utc)

    def _get_timestamp(self, dt: Optional[datetime]) -> float:
        """Быстрое получение timestamp"""
        if dt is None:
            return time.time()
        return dt.timestamp()

    def search_news(self, query: str = '', ticker: str = '', limit: int = 20,
                    keywords: List[str] = None) -> List[Dict]:
        """Быстрый поиск новостей с поддержкой ключевых слов

        Args:
            query: поисковый запрос
            ticker: тикер (латиницей)
            limit: максимальное количество результатов
            keywords: список ключевых слов для поиска (русские названия)
        """
        all_news = self.get_last_news(limit=200)

        if not query and not ticker and not keywords:
            return all_news[:limit]

        query = query.lower() if query else ''
        ticker_upper = ticker.upper() if ticker else ''
        keywords_lower = [kw.lower() for kw in keywords] if keywords else []

        filtered = []

        for news in all_news:
            title = news.get('title', '')
            summary = news.get('summary', '')
            title_lower = title.lower()
            summary_lower = summary.lower()

            if ticker_upper:
                if ticker_upper not in title.upper() and ticker_upper not in summary.upper():
                    continue

            if keywords_lower:
                found = False
                for kw in keywords_lower:
                    if kw in title_lower or kw in summary_lower:
                        found = True
                        break
                if not found:
                    continue

            if query:
                if query not in title_lower and query not in summary_lower:
                    continue

            filtered.append(news)
            if len(filtered) >= limit:
                break

        return filtered

    def get_news_summary(self) -> Dict:
        """Быстрая сводка по новостям"""
        all_news = self.get_last_news(limit=100)

        # Группировка
        categories = defaultdict(int)
        sources = defaultdict(int)

        for news in all_news:
            categories[news.get('category', 'unknown')] += 1
            sources[news.get('source', 'unknown')] += 1

        # Топ тикеров из новостей
        ticker_mentions = defaultdict(int)
        try:
            with open('config/tickers.json', 'r', encoding='utf-8') as f:
                tickers_config = json.load(f)
                tickers = [item['ticker'] for item in tickers_config.get('watchlist', [])]

                for news in all_news[:50]:
                    title = news['title'].upper()
                    for ticker in tickers:
                        if ticker in title:
                            ticker_mentions[ticker] += 1
        except:
            pass

        top_tickers = sorted(ticker_mentions.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            'total_news': len(all_news),
            'last_fetch': datetime.fromtimestamp(self.last_fetch_time).isoformat() if self.last_fetch_time else None,
            'sources_active': len(sources),
            'categories': dict(categories),
            'sources': dict(sources),
            'top_tickers': top_tickers,
            'stats': self.stats,
            'cache_size': len(self.news_cache)
        }

    def force_refresh(self):
        """Принудительное обновление"""
        self.last_fetch_time = 0
        self.news_cache = []
        self.stats['cache_hits'] = 0
        self.stats['cache_misses'] = 0
        logger.info("Принудительное обновление новостей")