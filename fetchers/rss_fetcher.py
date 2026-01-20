"""
Специализированный RSS фетчер для новостей
"""
import json
import feedparser
import time
from datetime import datetime
from typing import Dict, List, Optional, Set
import re
from collections import defaultdict

from utils.logger import setup_logger

logger = setup_logger("RSS_FETCHER")


class RSSFetcher:
    """Специализированный класс для работы с RSS"""

    def __init__(self, config_path: str = "config/rss_sources.json"):
        self.config = self._load_config(config_path)
        self.feeds = {}
        self.last_update = {}
        self.news_by_ticker = defaultdict(list)
        self.ticker_patterns = self._load_ticker_patterns()
        self.encoding_fix_sources = {}

        logger.info(f"Инициализирован RSSFetcher с {len(self.config.get('sources', []))} источниками")

    def _load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # ✅ ДОБАВИТЬ: Автоматическое определение источников с проблемами кодировки
            for source in config.get('sources', []):
                url = source.get('url', '')
                # Анализируем домен для определения проблемных источников
                if any(domain in url for domain in ['vedomosti.ru', 'finam.ru']):
                    source['force_utf8'] = True  # Флаг для исправления кодировки
                else:
                    source['force_utf8'] = False

            return config

        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации RSS: {e}")
            return {'sources': []}

    def _load_ticker_patterns(self) -> Dict[str, re.Pattern]:
        """Загрузка паттернов для поиска тикеров"""
        patterns = {}

        try:
            with open('config/tickers.json', 'r', encoding='utf-8') as f:
                tickers_config = json.load(f)
                tickers = [item['ticker'] for item in tickers_config.get('watchlist', [])]

                for ticker in tickers:
                    # Создаем паттерн для поиска тикера как отдельного слова
                    pattern = re.compile(rf'\b{ticker}\b', re.IGNORECASE)
                    patterns[ticker] = pattern

        except Exception as e:
            logger.error(f"Ошибка загрузки тикеров: {e}")

        return patterns

    def _get_timestamp(self, dt: datetime) -> float:
        """Безопасное получение timestamp"""
        import time
        if dt is None:
            return time.time()
        return dt.timestamp()



    def update_all_feeds(self) -> Dict[str, List[Dict]]:
        """Обновление всех RSS-лент"""
        all_news = {}

        for source in self.config.get('sources', []):
            if not source.get('enabled', True):
                continue

            source_name = source['name']
            news = self.update_feed(source)

            if news:
                all_news[source_name] = news
                logger.debug(f"Обновлен {source_name}: {len(news)} новостей")

        # Анализ новостей по тикерам
        self._analyze_news_by_ticker(all_news)

        return all_news

    def fetch_all_news(self) -> List[Dict]:
        """Получить все новости из всех источников"""
        # Используем существующий метод для обновления
        all_news_by_source = self.update_all_feeds()

        # Собираем все новости в плоский список
        all_news = []
        for source_name, news_list in all_news_by_source.items():
            for news in news_list:
                # Добавляем имя источника в каждую новость
                news_with_source = news.copy()
                news_with_source['source_name'] = source_name
                all_news.append(news_with_source)

        logger.debug(f"Получено {len(all_news)} новостей из всех источников")
        return all_news

    def update_feed(self, source: Dict) -> List[Dict]:
        """Обновление одной RSS-ленты с исправлением кодировки"""
        try:
            url = source['url']
            source_name = source['name']

            # Проверяем, когда обновляли последний раз
            last_update = self.last_update.get(url, 0)
            update_interval = source.get('update_interval', 300)

            if time.time() - last_update < update_interval:
                cached = self.feeds.get(url)
                if cached:
                    return cached

            # ✅ УНИВЕРСАЛЬНОЕ ИСПРАВЛЕНИЕ КОДИРОВКИ
            import requests

            try:
                # Загружаем через requests для контроля кодировки
                response = requests.get(url, timeout=10)

                # Проверяем заголовки ответа
                content_type = response.headers.get('content-type', '').lower()

                # Исправляем некорректные декларации кодировки
                if 'charset=us-ascii' in content_type and 'utf-8' in content_type:
                    # Меняем объявленную кодировку на UTF-8
                    fixed_content_type = content_type.replace('charset=us-ascii', 'charset=utf-8')
                    response.headers['content-type'] = fixed_content_type

                # Создаем фиктивный файл для feedparser
                import io
                content_io = io.BytesIO(response.content)

                # Парсим с исправленной кодировкой
                feed = feedparser.parse(content_io)

            except requests.RequestException:
                # Fallback: оригинальный метод если requests не работает
                feed = feedparser.parse(url)

            if feed.bozo:  # Ошибка парсинга
                # ✅ БОЛЕЕ ЩАДЯЩАЯ ОБРАБОТКА ОШИБОК КОДИРОВКИ
                if hasattr(feed.bozo_exception, 'getMessage') and 'document declared as us-ascii' in str(
                        feed.bozo_exception):
                    # Игнорируем ошибку кодировки, если есть данные
                    if hasattr(feed, 'entries') and feed.entries:
                        logger.warning(f"Игнорируем ошибку кодировки для {url}, но данные получены")
                        # Продолжаем обработку
                    else:
                        logger.warning(f"Ошибка парсинга RSS {url}: {feed.bozo_exception}")
                        return []
                else:
                    logger.warning(f"Ошибка парсинга RSS {url}: {feed.bozo_exception}")
                    return []

            news_items = []
            max_items = source.get('max_items', 20)

            for entry in feed.entries[:max_items]:
                news_item = self._parse_entry(entry, source)
                if news_item:
                    news_items.append(news_item)

            # Кэширование
            self.feeds[url] = news_items
            self.last_update[url] = time.time()

            return news_items

        except Exception as e:
            logger.error(f"Ошибка обновления RSS {source.get('name', 'unknown')}: {e}")
            return []

    def _parse_entry(self, entry, source: Dict) -> Optional[Dict]:
        """Парсинг RSS записи"""
        try:
            # Заголовок
            title = entry.get('title', '').strip()
            if not title:
                return None

            # Описание/контент
            summary = entry.get('summary', '') or entry.get('description', '')

            # Ссылка
            link = entry.get('link', '')

            # Дата публикации
            published_parsed = entry.get('published_parsed')
            if published_parsed:
                published = datetime(*published_parsed[:6])
            else:
                published = datetime.now()

            # Автор
            author = entry.get('author', '')

            # Категории
            categories = []
            if 'tags' in entry:
                categories = [tag.get('term', '') for tag in entry.tags]
            elif 'categories' in entry:
                categories = [cat.get('term', '') for cat in entry.categories]

            news_item = {
                'title': title,
                'summary': summary[:500] if summary else '',  # Ограничиваем длину
                'link': link,
                'published': published.isoformat(),
                'timestamp': self._get_timestamp(published),
                'author': author,
                'categories': categories,
                'source': source['name'],
                'source_url': source['url'],
                'source_priority': source.get('priority', 5)
            }

            # Фильтрация
            if not self._filter_entry(news_item, source):
                return None

            return news_item

        except Exception as e:
            logger.debug(f"Ошибка парсинга RSS записи: {e}")
            return None

    def _filter_entry(self, news_item: Dict, source: Dict) -> bool:
        """Фильтрация RSS записей"""
        # Фильтры из конфигурации источника
        filters = source.get('filters', {})

        # По ключевым словам
        keywords = filters.get('keywords', [])
        if keywords:
            text = f"{news_item['title']} {news_item['summary']}".lower()
            has_keyword = any(keyword.lower() in text for keyword in keywords)
            if not has_keyword:
                return False

        # По категориям
        categories = filters.get('categories', [])
        if categories:
            item_categories = [cat.lower() for cat in news_item['categories']]
            has_category = any(cat.lower() in item_categories for cat in categories)
            if not has_category:
                return False

        # По времени (не старше N дней)
        max_age_days = filters.get('max_age_days', 3)
        if max_age_days:
            news_time = datetime.fromisoformat(news_item['published'].replace('Z', '+00:00'))
            age_days = (datetime.now() - news_time).days
            if age_days > max_age_days:
                return False

        return True

    def _analyze_news_by_ticker(self, all_news: Dict[str, List[Dict]]):
        """Анализ новостей по тикерам"""
        # Очищаем предыдущий анализ
        self.news_by_ticker.clear()

        # Собираем все новости в один список
        all_news_list = []
        for source_news in all_news.values():
            all_news_list.extend(source_news)

        # Анализируем каждую новость
        for news in all_news_list:
            title = news['title']
            summary = news.get('summary', '')
            text = f"{title} {summary}".upper()

            # Ищем упоминания тикеров
            for ticker, pattern in self.ticker_patterns.items():
                if pattern.search(text):
                    self.news_by_ticker[ticker].append(news)

    def get_news_for_ticker(self, ticker: str, limit: int = 5) -> List[Dict]:
        """Получение новостей по конкретному тикеру"""
        # Обновляем данные, если нужно
        if not self.news_by_ticker:
            self.update_all_feeds()

        return self.news_by_ticker.get(ticker, [])[:limit]

    def get_tickers_in_news(self, min_mentions: int = 1) -> List[str]:
        """Получение списка тикеров, упомянутых в новостях"""
        # Обновляем данные, если нужно
        if not self.news_by_ticker:
            self.update_all_feeds()

        tickers = []
        for ticker, news_list in self.news_by_ticker.items():
            if len(news_list) >= min_mentions:
                tickers.append(ticker)

        # Сортировка по количеству упоминаний
        tickers.sort(key=lambda t: len(self.news_by_ticker[t]), reverse=True)

        return tickers

    def get_news_summary(self) -> Dict:
        """Сводка по новостям"""
        # Обновляем данные
        all_news = self.update_all_feeds()

        summary = {
            'total_sources': len(all_news),
            'total_news': sum(len(news) for news in all_news.values()),
            'sources': {},
            'tickers_mentioned': len(self.news_by_ticker),
            'top_tickers': [],
            'last_update': datetime.now().isoformat()
        }

        # Статистика по источникам
        for source_name, news_list in all_news.items():
            summary['sources'][source_name] = len(news_list)

        # Топ тикеров
        ticker_mentions = [(ticker, len(news)) for ticker, news in self.news_by_ticker.items()]
        ticker_mentions.sort(key=lambda x: x[1], reverse=True)

        summary['top_tickers'] = ticker_mentions[:10]

        # Временной анализ
        now = datetime.now()
        recent_news = 0
        for news_list in all_news.values():
            for news in news_list:
                news_time = datetime.fromisoformat(news['published'].replace('Z', '+00:00'))
                if (now - news_time).hours < 24:
                    recent_news += 1

        summary['news_last_24h'] = recent_news

        return summary

    def search_news(self,
                    query: str = '',
                    ticker: str = '',
                    source: str = '',
                    limit: int = 20) -> List[Dict]:
        """Поиск новостей"""
        # Обновляем данные
        all_news = self.update_all_feeds()

        # Собираем все новости
        all_news_list = []
        for source_name, news_list in all_news.items():
            for news in news_list:
                news['_source'] = source_name
                all_news_list.append(news)

        # Фильтрация
        filtered_news = []

        for news in all_news_list:
            # Фильтр по источнику
            if source and news['_source'] != source:
                continue

            # Фильтр по тикеру
            if ticker:
                news_text = f"{news['title']} {news.get('summary', '')}".upper()
                if ticker.upper() not in news_text:
                    # Проверяем через паттерны
                    if ticker in self.ticker_patterns:
                        if not self.ticker_patterns[ticker].search(news_text):
                            continue
                    else:
                        continue

            # Фильтр по запросу
            if query:
                query_lower = query.lower()
                title_lower = news['title'].lower()
                summary_lower = news.get('summary', '').lower()

                if (query_lower not in title_lower and
                        query_lower not in summary_lower):
                    continue

            filtered_news.append(news)

            if len(filtered_news) >= limit:
                break

        # Удаляем служебное поле
        for news in filtered_news:
            if '_source' in news:
                del news['_source']

        return filtered_news

    def get_feed_status(self) -> Dict[str, Dict]:
        """Статус всех RSS-лент"""
        status = {}

        for source in self.config.get('sources', []):
            if not source.get('enabled', True):
                continue

            url = source['url']
            source_name = source['name']

            last_update = self.last_update.get(url, 0)
            last_update_str = datetime.fromtimestamp(last_update).isoformat() if last_update else 'Never'

            feed_data = self.feeds.get(url, [])

            status[source_name] = {
                'url': url,
                'last_update': last_update_str,
                'news_count': len(feed_data),
                'status': 'active' if feed_data else 'inactive',
                'update_interval': source.get('update_interval', 300),
                'priority': source.get('priority', 5)
            }

        return status

    def force_refresh_feed(self, source_name: str) -> bool:
        """Принудительное обновление конкретной ленты"""
        for source in self.config.get('sources', []):
            if source['name'] == source_name and source.get('enabled', True):
                url = source['url']

                # Сбрасываем кэш
                if url in self.feeds:
                    del self.feeds[url]
                if url in self.last_update:
                    del self.last_update[url]

                # Обновляем
                news = self.update_feed(source)

                logger.info(f"Принудительно обновлена лента {source_name}: {len(news)} новостей")
                return True

        logger.warning(f"Лента {source_name} не найдена или отключена")
        return False