"""
Сбор новостей с различных источников
"""

import json
import time
import requests
from datetime import datetime, timedelta, timezone  # ИМПОРТИРУЕМ timezone
from typing import Dict, List, Optional
import re

from utils.logger import get_logger

logger = get_logger("NEWS_FETCHER")


class NewsFetcher:
    """Класс для сбора новостей"""

    def __init__(self, config_path: str = "config/rss_sources.json"):
        self.config = self._load_config(config_path)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Кэш новостей
        self.news_cache = []
        self.last_fetch_time = 0
        self.fetch_interval = self.config.get('update_interval_minutes', 5) * 60

        logger.info(f"Инициализирован NewsFetcher с {len(self.config.get('sources', []))} источниками")

    def _load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации новостей: {e}")
            return {'sources': [], 'update_interval_minutes': 5}

    def get_last_news(self, limit: int = 50) -> List[Dict]:
        """Получение последних новостей"""
        current_time = time.time()

        # Используем кэш, если не прошло время интервала
        if (current_time - self.last_fetch_time < self.fetch_interval and
                self.news_cache):
            logger.debug(f"Использую кэшированные новости ({len(self.news_cache)} шт)")
            return self.news_cache[:limit]

        try:
            all_news = []

            # Сбор новостей со всех источников
            for source in self.config.get('sources', []):
                if not source.get('enabled', True):
                    continue

                try:
                    source_news = self._fetch_from_source(source)
                    all_news.extend(source_news)

                    logger.debug(f"Источник {source['name']}: {len(source_news)} новостей")

                except Exception as e:
                    logger.error(f"Ошибка сбора новостей из {source['name']}: {e}")

            # Сортировка по времени (новые сначала)
            all_news.sort(key=lambda x: x.get('ts', 0), reverse=True)

            # Фильтрация дубликатов
            unique_news = self._remove_duplicates(all_news)

            # Обновление кэша
            self.news_cache = unique_news
            self.last_fetch_time = current_time

            logger.info(f"Собрано {len(unique_news)} уникальных новостей")

            return unique_news[:limit]

        except Exception as e:
            logger.error(f"Ошибка получения новостей: {e}")
            return self.news_cache[:limit] if self.news_cache else []

    def _fetch_from_source(self, source: Dict) -> List[Dict]:
        """Сбор новостей из одного источника"""
        news_items = []

        # Определяем тип источника
        source_type = source.get('type', 'rss')

        if source_type == 'rss':
            news_items = self._fetch_rss(source)
        elif source_type == 'api':
            news_items = self._fetch_api(source)
        elif source_type == 'web':
            news_items = self._fetch_web(source)

        return news_items

    def _fetch_rss(self, source: Dict) -> List[Dict]:
        """Парсинг RSS с ИСПРАВЛЕННЫМ сравнением дат"""
        try:
            import feedparser

            feed = feedparser.parse(source['url'])
            news_items = []

            max_items = source.get('max_items', 20)

            for entry in feed.entries[:max_items]:
                # Парсинг даты
                published = self._parse_date(entry.get('published', ''))

                # Пропускаем старые новости (старше 3 дней)
                if published:
                    # ИСПРАВЛЕНИЕ: делаем обе даты aware (с часовым поясом)
                    if published.tzinfo is None:
                        # Если published naive, делаем его aware в UTC
                        published = published.replace(tzinfo=timezone.utc)

                    # Текущее время тоже делаем aware в UTC
                    now_aware = datetime.now(timezone.utc)
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
                    'category': source.get('category', 'general')
                }

                # Фильтрация по ключевым словам
                if self._filter_news_item(news_item):
                    news_items.append(news_item)

            return news_items

        except Exception as e:
            logger.error(f"Ошибка парсинга RSS {source['url']}: {e}")
            return []

    def _fetch_api(self, source: Dict) -> List[Dict]:
        """Получение новостей через API"""
        try:
            url = source['url']
            headers = source.get('headers', {})
            params = source.get('params', {})

            response = self.session.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Обработка в зависимости от структуры API
            news_items = []

            # Пример для разных API структур
            if 'articles' in data:
                articles = data['articles']
            elif 'news' in data:
                articles = data['news']
            elif 'results' in data:
                articles = data['results']
            else:
                articles = data if isinstance(data, list) else []

            for article in articles[:20]:  # Ограничиваем
                title = article.get('title') or article.get('headline', '')
                summary = article.get('summary') or article.get('description') or article.get('content', '')

                # Парсинг даты
                date_str = article.get('publishedAt') or article.get('date') or article.get('timestamp')
                published = self._parse_date(date_str) if date_str else datetime.now()

                news_item = {
                    'title': title,
                    'summary': summary[:500] if summary else '',  # Ограничиваем длину
                    'link': article.get('url') or article.get('link', ''),
                    'published': published.isoformat(),
                    'ts': self._get_timestamp(published),
                    'source': source['name'],
                    'category': source.get('category', 'general')
                }

                if self._filter_news_item(news_item):
                    news_items.append(news_item)

            return news_items

        except Exception as e:
            logger.error(f"Ошибка API {source['url']}: {e}")
            return []

    def _fetch_web(self, source: Dict) -> List[Dict]:
        """Парсинг веб-страниц"""
        try:
            from bs4 import BeautifulSoup

            response = self.session.get(source['url'], timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Конфигурация парсинга
            config = source.get('parsing_config', {})

            news_items = []

            # Поиск элементов новостей
            news_elements = soup.select(config.get('item_selector', '.news-item'))

            for element in news_elements[:20]:  # Ограничиваем
                try:
                    # Извлечение заголовка
                    title_selector = config.get('title_selector', '.title')
                    title_elem = element.select_one(title_selector)
                    title = title_elem.text.strip() if title_elem else ''

                    # Извлечение ссылки
                    link_selector = config.get('link_selector', 'a')
                    link_elem = element.select_one(link_selector)

                    if link_elem and link_elem.get('href'):
                        link = link_elem['href']
                        # Преобразование относительных ссылок
                        if link.startswith('/'):
                            base_url = '/'.join(source['url'].split('/')[:3])
                            link = base_url + link
                    else:
                        link = ''

                    # Извлечение даты
                    date_selector = config.get('date_selector', '.date')
                    date_elem = element.select_one(date_selector)
                    date_str = date_elem.text.strip() if date_elem else ''

                    published = self._parse_date(date_str) if date_str else datetime.now()

                    # Извлечение описания
                    summary_selector = config.get('summary_selector', '.summary')
                    summary_elem = element.select_one(summary_selector)
                    summary = summary_elem.text.strip() if summary_elem else ''

                    if title:  # Только если есть заголовок
                        news_item = {
                            'title': title,
                            'summary': summary[:300],
                            'link': link,
                            'published': published.isoformat(),
                            'ts': self._get_timestamp(published),
                            'source': source['name'],
                            'category': source.get('category', 'general')
                        }

                        if self._filter_news_item(news_item):
                            news_items.append(news_item)

                except Exception as e:
                    logger.debug(f"Ошибка парсинга элемента новости: {e}")
                    continue

            return news_items

        except Exception as e:
            logger.error(f"Ошибка веб-парсинга {source['url']}: {e}")
            return []

    def _get_timestamp(self, dt: datetime) -> float:
        """Безопасное получение timestamp с учётом часовых поясов"""
        if dt.tzinfo is not None:
            # Для aware datetime используем timestamp() (он всегда возвращает UTC timestamp)
            return dt.timestamp()
        else:
            # Для naive datetime создаём aware в локальном поясе, затем получаем UTC timestamp
            import time
            local_tz = time.localtime().tm_gmtoff if hasattr(time, 'localtime') else 0
            return dt.timestamp() + local_tz

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Парсинг даты из различных форматов с ИСПРАВЛЕНИЕМ часовых поясов"""
        try:
            if not date_str:
                return None

            # Удаляем лишние пробелы
            date_str = date_str.strip()

            # Список форматов для попыток парсинга
            formats = [
                '%a, %d %b %Y %H:%M:%S %z',    # RSS с часовым поясом
                '%a, %d %b %Y %H:%M:%S %Z',    # RSS с названием пояса
                '%Y-%m-%dT%H:%M:%S%z',         # ISO с часовым поясом
                '%Y-%m-%d %H:%M:%S%z',         # Альтернативный ISO
                '%Y-%m-%dT%H:%M:%S',           # ISO без пояса
                '%Y-%m-%d %H:%M:%S',           # Простой формат
                '%d.%m.%Y %H:%M',              # Российский формат
                '%d/%m/%Y %H:%M',              # Альтернативный
                '%Y-%m-%d',                    # Только дата
                '%d.%m.%Y',                    # Российская дата
                '%d/%m/%Y'                     # Альтернативная дата
            ]

            for fmt in formats:
                try:
                    dt = datetime.strptime(date_str, fmt)

                    # Если формат содержит %z (часовой пояс), то dt будет aware
                    # Если нет - сделаем его aware в UTC
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)

                    return dt
                except ValueError:
                    continue

            # Если не удалось распарсить, возвращаем текущую дату в UTC
            logger.warning(f"Не удалось распарсить дату: {date_str[:50]}...")
            return datetime.now(timezone.utc)

        except Exception as e:
            logger.error(f"Ошибка парсинга даты '{date_str[:50]}...': {e}")
            return datetime.now(timezone.utc)

    def _filter_news_item(self, news_item: Dict) -> bool:
        """Фильтрация новостей по ключевым словам"""
        keywords_config = self.config.get('keywords_filter', {})
        include_words = keywords_config.get('include', [])
        exclude_words = keywords_config.get('exclude', [])

        text = f"{news_item['title']} {news_item['summary']}".lower()

        # Проверка исключающих слов
        for word in exclude_words:
            if word.lower() in text:
                return False

        # Если указаны включающие слова, проверяем их
        if include_words:
            for word in include_words:
                if word.lower() in text:
                    return True
            return False  # Не содержит ни одного включающего слова

        return True  # Нет фильтров или только exclude фильтры

    def _remove_duplicates(self, news_items: List[Dict]) -> List[Dict]:
        """Удаление дубликатов новостей"""
        seen_titles = set()
        unique_news = []

        for item in news_items:
            title = item['title'].strip().lower()

            # Нормализация заголовка
            title_norm = re.sub(r'[^\w\s]', '', title)  # Удаляем пунктуацию

            if title_norm not in seen_titles:
                seen_titles.add(title_norm)
                unique_news.append(item)

        return unique_news

    def search_news(self, query: str, limit: int = 20) -> List[Dict]:
        """Поиск новостей по запросу"""
        all_news = self.get_last_news(limit=200)  # Берем больше для поиска

        if not query:
            return all_news[:limit]

        query_words = query.lower().split()

        # Ранжирование новостей по релевантности
        ranked_news = []

        for news in all_news:
            title = news['title'].lower()
            summary = news.get('summary', '').lower()
            text = f"{title} {summary}"

            # Подсчет релевантности
            relevance = 0

            for word in query_words:
                if word in title:
                    relevance += 3  # Больший вес для заголовка
                elif word in summary:
                    relevance += 1

            if relevance > 0:
                ranked_news.append((relevance, news))

        # Сортировка по релевантности
        ranked_news.sort(key=lambda x: x[0], reverse=True)

        # Возвращаем только новости
        return [news for _, news in ranked_news[:limit]]

    def get_news_by_ticker(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Получение новостей по конкретному тикеру"""
        all_news = self.get_last_news(limit=100)

        ticker_news = []
        ticker_upper = ticker.upper()

        for news in all_news:
            title = news['title'].upper()

            # Поиск упоминания тикера
            if (ticker_upper in title or
                    f" {ticker_upper} " in f" {title} " or  # Отдельное слово
                    ticker_upper in news.get('summary', '').upper()):

                ticker_news.append(news)

                if len(ticker_news) >= limit:
                    break

        return ticker_news

    def get_news_summary(self) -> Dict:
        """Получение сводки по новостям"""
        all_news = self.get_last_news(limit=50)

        summary = {
            'total_news': len(all_news),
            'last_fetch': datetime.fromtimestamp(self.last_fetch_time).isoformat()
            if self.last_fetch_time else None,
            'sources_active': len([s for s in self.config.get('sources', [])
                                   if s.get('enabled', True)]),
            'categories': {},
            'recent_tickers': set()
        }

        # Анализ категорий
        for news in all_news:
            category = news.get('category', 'unknown')
            summary['categories'][category] = summary['categories'].get(category, 0) + 1

        # Поиск упоминаний тикеров (простейшая реализация)
        common_tickers = ['SBER', 'GAZP', 'LKOH', 'ROSN', 'VTBR', 'GMKN', 'NVTK', 'YNDX']

        for news in all_news[:20]:  # Только первые 20
            title = news['title'].upper()
            for ticker in common_tickers:
                if ticker in title:
                    summary['recent_tickers'].add(ticker)

        summary['recent_tickers'] = list(summary['recent_tickers'])

        return summary

    def get_market_news_summary(self, limit: int = 100) -> Dict:
        """Сводка рыночных новостей для анализа настроения"""
        all_news = self.get_last_news(limit=limit)

        # Группировка по категориям
        categories = {}
        sources = {}

        for news in all_news:
            # Категории
            category = news.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1

            # Источники
            source = news.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1

        # Поиск ключевых тем
        import collections
        word_counter = collections.Counter()

        for news in all_news[:50]:  # Только первые 50
            title_words = news['title'].lower().split()
            for word in title_words:
                if len(word) > 3:  # Игнорируем короткие слова
                    word_counter[word] += 1

        top_themes = word_counter.most_common(10)

        return {
            'total_news': len(all_news),
            'categories': dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)),
            'sources': dict(sorted(sources.items(), key=lambda x: x[1], reverse=True)),
            'top_themes': top_themes,
            'last_update': datetime.now().isoformat()
        }

    def force_refresh(self):
        """Принудительное обновление новостей"""
        self.last_fetch_time = 0  # Сбрасываем время последнего сбора
        logger.info("Принудительное обновление новостей")